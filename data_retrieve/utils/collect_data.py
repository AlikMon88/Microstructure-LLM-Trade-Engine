### retrieve snap_order_book L1/L2
import pandas as pd
import numpy as np
import math
from datetime import datetime
import ccxt

import pandas as pd
from datetime import datetime, timedelta
import threading
import time


class OrderBookProvider:
    def __init__(self, exchanges: list[str], depth: int = 10):
        """
        exchanges : list of exchange names (as in ccxt)
        depth     : how many levels to fetch for L2 (default=10)
        """
        self.depth = depth
        self.exchanges = {ex: getattr(ccxt, ex)() for ex in exchanges}

    def _fetch_orderbook(self, exchange: str, symbol: str, level: str = "L1"):
        """
        Fetch order book for one exchange + one symbol.
        level : "L1" = top of book (best bid/ask)
                "L2" = full depth up to self.depth
        Returns DataFrame with MultiIndex (side, price) for L2
        or single-row DataFrame for L1.
        """
        ex = self.exchanges[exchange]
        ob = ex.fetch_order_book(symbol, limit=self.depth)

        ts = pd.to_datetime(ob["timestamp"], unit="ms") if ob["timestamp"] else datetime.utcnow()

        if level.upper() == "L1":
            best_bid = ob["bids"][0] if ob["bids"] else [None, None]
            best_ask = ob["asks"][0] if ob["asks"] else [None, None]
            data = {
                "datetime": ts,
                "bid_price": best_bid[0],
                "bid_size": best_bid[1],
                "ask_price": best_ask[0],
                "ask_size": best_ask[1],
            }
            return pd.DataFrame([data]).set_index("datetime")

        elif level.upper() == "L2":
            bids = pd.DataFrame(ob["bids"], columns=["price", "size"]).assign(side="bid")
            asks = pd.DataFrame(ob["asks"], columns=["price", "size"]).assign(side="ask")
            df = pd.concat([bids, asks])
            df["datetime"] = ts
            df = df.set_index(["datetime", "side", "price"]).sort_index()
            return df
        else:
            raise ValueError("level must be 'L1' or 'L2'")

    def fetch_multi(self, exchange: str = None, tokens: list[str] = None, level: str = "L1"):
        """
        Case 1: multiple exchanges + single token
        Case 2: single exchange + multiple tokens
        Returns MultiIndex DataFrame with (token, datetime) or (token, datetime, side, price) for L2
        """
        frames = []

        # Case 1: multiple exchanges, single token
        if exchange is None and len(self.exchanges) > 1 and tokens and len(tokens) == 1:
            token = tokens[0]
            for ex in self.exchanges:
                try:
                    df = self._fetch_orderbook(ex, token, level)
                    df["token"] = f"{token}@{ex}"
                    frames.append(df)
                except Exception as e:
                    print(f"Failed {ex}:{token} -> {e}")

        # Case 2: single exchange, multiple tokens
        elif exchange and tokens and len(tokens) > 1:
            for token in tokens:
                try:
                    df = self._fetch_orderbook(exchange, token, level)
                    df["token"] = token
                    frames.append(df)
                except Exception as e:
                    print(f"Failed {exchange}:{token} -> {e}")
        else:
            raise ValueError("Invalid input: must be (multi-exchange, single token) or (single exchange, multi-token)")

        if not frames:
            return pd.DataFrame()

        df_all = pd.concat(frames)

        # For L1 → (token, datetime) MultiIndex
        if level.upper() == "L1":
            df_all = df_all.reset_index().set_index(["token", "datetime"]).sort_index()

        # For L2 → (token, datetime, side, price) MultiIndex
        elif level.upper() == "L2":
            df_all = df_all.reset_index().set_index(["token", "datetime", "side", "price"]).sort_index()

        return df_all
    
    
### ------------------------------
### Rolling Order-Book Information > main_cell_jupyter_run
### ------------------------------

class RollingOrderBookStore:
    def __init__(self, provider: OrderBookProvider, exchange: str, tokens: list[str], 
                 level: str = "L2", window_minutes: int = 240, interval: int = 60):
        """
        provider       : OrderBookProvider instance
        exchange       : exchange name (single exchange, multi tokens case)
        tokens         : list of tokens to track
        level          : "L1" or "L2" (default L2)
        window_minutes : how much history to retain (default=240 min = 4h)
        interval       : fetch interval in seconds (default=60s)
        """
        self.provider = provider
        self.exchange = exchange
        self.tokens = tokens
        self.level = level
        self.window = timedelta(minutes=window_minutes)
        self.interval = interval

        # store as list of DataFrames
        self._data = []
        self._running = False
        self._thread = None

    def _loop(self):
        while self._running:
            try:
                df = self.provider.fetch_multi(exchange=self.exchange, tokens=self.tokens, level=self.level)
                self._data.append(df)
                self._cleanup()
            except Exception as e:
                print(f"Fetch error: {e}")
            time.sleep(self.interval)

    def _cleanup(self):
        """Remove snapshots older than window_minutes."""
        cutoff = datetime.utcnow() - self.window
        new_data = []
        for df in self._data:
            if df.index.get_level_values("datetime").max() >= cutoff:
                new_data.append(df)
        self._data = new_data

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def get_data(self, lookback: str = "30min"):
        """
        Retrieve concatenated DataFrame of history up to now.
        lookback : e.g. "30min", "1h", "4h"
        """
        delta = pd.Timedelta(lookback)
        cutoff = datetime.utcnow() - delta
        frames = []
        for df in self._data:
            mask = df.index.get_level_values("datetime") >= cutoff
            if mask.any():
                frames.append(df[mask])
        if frames:
            return pd.concat(frames).sort_index()
        else:
            return pd.DataFrame()


class CCXTDataProvider:
    def __init__(self, exchanges: list[str], timeframe: str = "1h", limit: int = 200):
        """
        exchanges : list of exchange names (as in ccxt)
        timeframe : OHLCV timeframe (e.g. '1m', '15m', '1h')
        limit     : number of candles to fetch
        """
        self.timeframe = timeframe
        self.limit = limit
        self.exchanges = {ex: getattr(ccxt, ex)() for ex in exchanges}

    def _fetch_ohlcv(self, exchange: str, symbol: str):
        """
        Fetch OHLCV for one exchange and one symbol.
        Returns DataFrame with datetime index.
        """
        ex = self.exchanges[exchange]
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=self.limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("datetime")
        return df.drop(columns=["timestamp"])

    def fetch_multi(self, exchange: str = None, tokens: list[str] = None):
        """
        Case 1: multiple exchanges + single token
        Case 2: single exchange + multiple tokens

        Returns MultiIndex DataFrame with index (token, datetime).
        """
        frames = []

        if exchange is None and len(self.exchanges) > 1 and tokens and len(tokens) == 1:
            # Case 1: multiple exchanges + single token
            token = tokens[0]
            for ex in self.exchanges:
                try:
                    df = self._fetch_ohlcv(ex, token)
                    df["token"] = f"{token}@{ex}"
                    frames.append(df)
                except Exception as e:
                    print(f"Failed {ex}:{token} -> {e}")

        elif exchange and tokens and len(tokens) > 1:
            # Case 2: single exchange + multiple tokens
            for token in tokens:
                try:
                    df = self._fetch_ohlcv(exchange, token)
                    df["token"] = token
                    frames.append(df)
                except Exception as e:
                    print(f"Failed {exchange}:{token} -> {e}")
        else:
            raise ValueError("Invalid input: must be (multi-exchange, single token) or (single exchange, multi-token)")

        if not frames:
            return pd.DataFrame()

        df_all = pd.concat(frames)
        df_all = df_all.reset_index().set_index(["token", "datetime"]).sort_index()
        return df_all


if __name__ == '__main__':
    print('snap-retireve')