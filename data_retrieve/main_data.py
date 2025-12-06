import random
import datetime
from typing import Dict, List, Tuple
import pandas as pd
import ccxt
import time

class MarketDataProvider():
    """
    Fetches market data using CCXT from multiple exchanges.
    """

    def __init__(self, exchanges: List[str], symbols: List[str]):
        self.exchanges = {ex: getattr(ccxt, ex)() for ex in exchanges}
        self.symbols = symbols

    def fetch_ohlcv(self, exchange: str, symbol: str, start: datetime.datetime, end: datetime.datetime):
        """
        Fetch OHLCV data from CCXT.
        """
        ex = self.exchanges[exchange]

        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_candles = []
        timeframe = '30m'
        limit = 500

        while since < end_ms:
            candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1

            if len(candles) < limit:
                break

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["exchange"] = exchange
        df["symbol"] = symbol
        return df.set_index("timestamp")


class MultiDataSnapshot():
    """
    Creates randomized time snapshots of markets across multiple exchanges.
    Supports:
      - Random snapshots across arbitrary history
      - Randomized exchange ensemble selection
      - Stress test snapshots for high volatility events
    """

    def __init__(self, data_provider: MarketDataProvider, min_days: int = 1, max_days: int = 30):
        self.data_provider = data_provider
        self.min_days = min_days
        self.max_days = max_days

    def _random_time_range(self, start_date: datetime.datetime, end_date: datetime.datetime):
        delta_days = random.randint(self.min_days, self.max_days)
        start = start_date + datetime.timedelta(days=random.randint(0, max(1, (end_date - start_date).days - delta_days)))
        end = start + datetime.timedelta(days=delta_days)
        return start, end

    def _random_exchanges(self):
        all_exchanges = list(self.data_provider.exchanges.keys())
        k = random.randint(1, len(all_exchanges))
        return random.sample(all_exchanges, k=k)

    def _fixed_exchanges_one(self):
        all_exchanges = list(self.data_provider.exchanges.keys())
        k = random.randint(1, len(all_exchanges) - 1)
        return [all_exchanges[k]]  
    
    def get_snapshot(self, start_date: datetime.datetime, end_date: datetime.datetime, is_fixed=True):
        start, end = self._random_time_range(start_date, end_date)
        if is_fixed:
            chosen_exchanges = self._fixed_exchanges_one()
        else:
            chosen_exchanges = self._random_exchanges()
            
        snapshot = []
        for ex in chosen_exchanges:
            for sym in self.data_provider.symbols:
                df = self.data_provider.fetch_ohlcv(ex, sym, start, end)
                if not df.empty:
                    snapshot.append(df)

        if not snapshot:
            return {}

        snapshot_df = pd.concat(snapshot).sort_index()
        if is_fixed:
            key = f"{start.strftime('%Y-%m-%d %H:%M')}|{end.strftime('%Y-%m-%d %H:%M')}|{chosen_exchanges[0]}"
        else: 
            key = f"{start.strftime('%Y-%m-%d %H:%M')}|{end.strftime('%Y-%m-%d %H:%M')}"
            
        return {key: snapshot_df}

    def get_multiple_snapshots(self, start_date: datetime.datetime, end_date: datetime.datetime, n = 5, is_fixed=True):
        snapshots = {}
        for _ in range(n):
            snap = self.get_snapshot(start_date, end_date, is_fixed=is_fixed)
            snapshots.update(snap)
        return snapshots

    def get_stress_test_snapshots(self, known_events: List[Tuple[datetime.datetime, datetime.datetime]], is_fixed=True):
        """
        Returns snapshots focusing on known high-volatility events (black swans).
        Provide a list of (start, end) tuples for stress periods.
        """
        snapshots = {}
        for (start, end) in known_events:
            if is_fixed:
                chosen_exchanges = self._random_exchanges()
            else:
                chosen_exchanges = self._fixed_exchanges_one()
                
            snapshot = []
            for ex in chosen_exchanges:
                for sym in self.data_provider.symbols:
                    df = self.data_provider.fetch_ohlcv(ex, sym, start, end)
                    if not df.empty:
                        snapshot.append(df)
            if snapshot:
                snapshot_df = pd.concat(snapshot).sort_index()
                if is_fixed:
                    key = f"{start.strftime('%Y-%m-%d %H:%M')}|{end.strftime('%Y-%m-%d %H:%M')}|{chosen_exchanges[-1]}|(Stress)"
                else:
                    key = f"{start.strftime('%Y-%m-%d %H:%M')}|{end.strftime('%Y-%m-%d %H:%M')}|(Stress)"
                    
                snapshots[key] = snapshot_df
        return snapshots

def classify_exchanges_by_volume(symbol: str = "BTC/USDT", top_n: int = 5):
    """
    Classify exchanges into high and low volume for a given trading pair.
    
    Parameters:
        symbol (str): trading pair to check (default 'BTC/USDT')
        top_n (int): number of exchanges to return for high/low classification
    
    Returns:
        Dict with 'high_volume' and 'low_volume' lists of exchanges
    """
    
    volumes = []

    for ex_id in ccxt.exchanges:
        try:
            exchange = getattr(ccxt, ex_id)()
            markets = exchange.load_markets()
            if symbol not in markets:
                continue

            ticker = exchange.fetch_ticker(symbol)
            vol = ticker.get("quoteVolume") or ticker.get("baseVolume")
            if vol:
                volumes.append((ex_id, vol))
        except Exception:
            # Skip exchanges that don't support fetch_ticker or are unavailable
            continue

    if not volumes:
        return {"high_volume": [], "low_volume": []}

    # Sort by volume descending
    volumes_sorted = sorted(volumes, key=lambda x: x[1], reverse=True)

    # Pick top_n and bottom_n
    high_volume = [ex for ex, _ in volumes_sorted[:top_n]]
    low_volume = [ex for ex, _ in volumes_sorted[-top_n:]]

    return {"high_volume": high_volume, "low_volume": low_volume} 

# Example Usage
if __name__ == "__main__":
    
    print('CCXT-Exchanges')
    print(ccxt.exchanges)
    
    provider = MarketDataProvider(exchanges=["binance", "kraken", "coinbase"], symbols=["BTC/USDT", "ETH/USDT"])
    snapshot_module = MultiDataSnapshot(provider)

    start_date = datetime.datetime(2018, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)

    # Random snapshots
    snapshots = snapshot_module.get_multiple_snapshots(start_date, end_date, n=3)
    for k, v in snapshots.items():
        print(f"Random Snapshot {k} -> {len(v)} rows")

    # Stress test snapshots (example: COVID crash March 2020, FTX collapse Nov 2022)
    stress_periods = [
        (datetime.datetime(2020, 3, 1), datetime.datetime(2020, 3, 31)),
        (datetime.datetime(2022, 11, 1), datetime.datetime(2022, 11, 15))
    ]
    stress_snaps = snapshot_module.get_stress_test_snapshots(stress_periods)
    for k, v in stress_snaps.items():
        print(f"Stress Snapshot {k} -> {len(v)} rows")
        
    result = snapshot_module.classify_exchanges_by_volume("BTC/USDT", top_n=3)
    print("High volume exchanges:", result["high_volume"])
    print("Low volume exchanges:", result["low_volume"])