import pandas as pd
import numpy as np
from IPython.display import display

class FeatureEngineMTF:
    """
    Calculates momentum and order-book style features from OHLCV data.
    Supports multi-timeframe aggregation: 5m -> 30m -> 2h
    """
    
    def __init__(self, ohclv, token):
        self.ohlcv = ohclv
        self.token = token        
    
    def calculate_features(self, frame):
        """
        Args:
            ohlcv: DataFrame with columns ['timestamp','open','high','low','close','volume']
            
        Returns:
            Dictionary of features for different frames
        """
        if self.ohlcv.empty:
            return {}
        
        features = {}
        last_candle = self.ohlcv.iloc[-1]
        
        # --- Current price ---
        features['current_price'] = last_candle['close']
        
        # --- Multi-frame price/volume features ---
        features = self._aggregate_frame_features(self.ohlcv, frame)
        
        # for frame in ['5min','30min','2H']:
        #     features.update(self._aggregate_frame_features(ohlcv, frame))
        
        return features
    
    # ---------------- Price / Volume Features ----------------
    def _aggregate_frame_features(self, ohlcv, frame: str):
        """
        Compute features for every timestamp in the ohlcv DataFrame.
        Returns:
            DataFrame indexed by timestamp with columns:
            [roc, vwap, price_vs_vwap, vol_acc, volatility]
        """

        vol_acc_dict = {
            '3min': ('1min', '3min'),
            '5min': ('1min', '5min'),
            '30min_intrabar': ('5min', '30min'),
            '15min_trend': ('15min', '75min'),
            '30min_trend': ('30min', '180min'),
            '2h_trend': ('2h', '8h'),
            '4h_trend': ('4h', '20h')
        }

        # store results here
        result = []

        # loop through each timestamp
        for ts in ohlcv.index:

            start_time = ts - pd.Timedelta('1440min')
            relevant = ohlcv[ohlcv.index >= start_time].loc[:ts]

            if relevant.empty or len(relevant) < 2:
                result.append({
                    'timestamp': ts,
                    f'roc': 0.0,
                    f'vwap': np.nan,
                    f'price_vs_vwap': np.nan,
                    f'vol_acc': 1.0,
                    f'volatility': 0.0
                })
                continue

            close_prices = relevant['close']
            typical_price = (relevant['high'] + relevant['low'] + relevant['close']) / 3
            volume = relevant['volume']

            roc = close_prices.iloc[-1] / close_prices.iloc[0] - 1
            vwap = (typical_price * volume).sum() / volume.sum()
            price_vs_vwap = close_prices.iloc[-1] - vwap
            vol_acc = self._volume_acceleration(relevant, s_l_window=vol_acc_dict[frame])
            volatility = close_prices.pct_change().std()

            result.append({
                'timestamp': ts,
                f'roc': roc,
                f'vwap': vwap,
                f'price_vs_vwap': price_vs_vwap,
                f'vol_acc': vol_acc,
                f'volatility': volatility
            })

        # convert to DataFrame
        df = pd.DataFrame(result).set_index('timestamp')
        return df
        
    def _volume_acceleration(self, df, s_l_window):
        """
        Compares recent volume to average volume
        """
        
        s_window, l_window = s_l_window
        short_window = pd.Timedelta(s_window)
        long_window = pd.Timedelta(l_window)
        end_time = df.index[-1]
        
        short_vol = df[df.index >= end_time - short_window]['volume'].sum()
        long_vol = df['volume'].sum()
        if long_vol == 0:
            return 1.0
        long_window_mins = long_window.total_seconds()/60
        short_window_mins = short_window.total_seconds()/60
        avg_long_per_short = long_vol * (short_window_mins / long_window_mins)
        if avg_long_per_short == 0:
            return 1.0
        return short_vol / avg_long_per_short
    
    
### ROLLING ORDERBOOKS

class OrderBookRollingFeatures:
    def __init__(self, df: pd.DataFrame):
        """
        df must be a MultiIndex DataFrame with levels (datetime, side, price).
        Columns should include 'size'.
        """
        self.df = df.sort_index()

    # -------------------------------
    # Helpers
    # -------------------------------
    def _group_sides(self, snap):
        bids = snap.xs("bid", level="side")
        asks = snap.xs("ask", level="side")
        return bids, asks

    def _midprice(self, bids, asks):
        best_bid = bids.index.max()#[-1]
        best_ask = asks.index.min()#[-1]
        return (best_bid + best_ask) / 2

    # -------------------------------
    # Core Features
    # -------------------------------
    def orderbook_imbalance(self, topn=5):
        out = {}
        for ts, snap in self.df.groupby(level="datetime"):
            bids, asks = self._group_sides(snap)
            bids, asks = bids.loc[ts], asks.loc[ts]
            bid_vol = bids.sort_index(ascending=False).head(topn)["size"].sum()
            ask_vol = asks.sort_index().head(topn)["size"].sum()
            obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            out[ts] = obi
        return pd.Series(out, name=f"OBI_top{topn}")

    def book_slope(self, topn=10):
        out = {}
        for ts, snap in self.df.groupby(level="datetime"):
            bids, asks = self._group_sides(snap)
            bids, asks = bids.loc[ts], asks.loc[ts]
            mid = self._midprice(bids, asks)
            bids = bids.assign(dist=(mid - bids.index) / mid)
            asks = asks.assign(dist=(asks.index - mid) / mid)

            def slope(side_df):
                side_df = side_df.head(topn)
                if len(side_df) < 2:
                    return np.nan
                X = np.log1p(side_df["dist"].values)
                y = np.log1p(side_df["size"].values)
                coef = np.polyfit(X, y, 1)[0]
                return coef

            out[ts] = {
                "slope_bid": slope(bids.sort_index(ascending=False)),
                "slope_ask": slope(asks.sort_index())
            }
        return pd.DataFrame.from_dict(out, orient="index")

    def microprice_drift(self):
        out = {}
        for ts, snap in self.df.groupby(level="datetime"):
            bids, asks = self._group_sides(snap)
            bids, asks = bids.loc[ts], asks.loc[ts]
            best_bid, bid_size = bids.index.max(), bids.loc[bids.index.max(), "size"]
            best_ask, ask_size = asks.index.min(), asks.loc[asks.index.min(), "size"]
            mid = (best_bid + best_ask) / 2
            micro = (best_ask * bid_size + best_bid * ask_size) / (bid_size + ask_size + 1e-9)
            drift = (micro - mid) / mid
            out[ts] = drift
        return pd.Series(out, name="Microprice_drift")

    def relative_spread(self):
        out = {}
        for ts, snap in self.df.groupby(level="datetime"):
            bids, asks = self._group_sides(snap)
            bids, asks = bids.loc[ts], asks.loc[ts]
            best_bid = bids.index.max()
            best_ask = asks.index.min()
            mid = (best_bid + best_ask) / 2
            rel_spread = (best_ask - best_bid) / mid
            out[ts] = rel_spread
        return pd.Series(out, name="RelSpread")

    def depth_concentration(self, topn=3):
        out = {}
        for ts, snap in self.df.groupby(level="datetime"):
            bids, asks = self._group_sides(snap)
            bids, asks = bids.loc[ts], asks.loc[ts]
            top_bids = bids.sort_index(ascending=False).head(topn)["size"].sum()
            top_asks = asks.sort_index().head(topn)["size"].sum()
            total = bids["size"].sum() + asks["size"].sum()
            frac = (top_bids + top_asks) / (total + 1e-9)
            out[ts] = frac
        return pd.Series(out, name=f"DepthConc_top{topn}")

    # -------------------------------
    # Aggregator with rolling comparisons
    # -------------------------------
    def extract_all_features(self):
        features = pd.concat([
            self.orderbook_imbalance(topn=5),
            self.book_slope(topn=10),
            self.microprice_drift(),
            self.relative_spread(),
            self.depth_concentration(topn=3),
        ], axis=1).sort_index()

        # Rolling means
        f_5m = features.rolling("5min").mean()
        f_30m = features.rolling("30min").mean()
        f_2h = features.rolling("2h").mean()

        # Add relative comparisons
        for col in ["OBI_top5", "slope_bid", "slope_ask", "Microprice_drift", "RelSpread", "DepthConc_top3"]:
            features[f"{col}_5m_vs_30m"] = f_5m[col] - f_30m[col]
            features[f"{col}_30m_vs_2h"] = f_30m[col] - f_2h[col]
            
            ## FIX: Straight-lines
            # Absolute “point estimate” comparisons (last window averages only)
            features[f"{col}_last5m"] = f_5m[col].iloc[-1]
            features[f"{col}_last30m"] = f_30m[col].iloc[-1]
            features[f"{col}_last2h"] = f_2h[col].iloc[-1]
            features[f"{col}_last5m_vs_30m"] = f_5m[col].iloc[-1] - f_30m[col].iloc[-1]
            features[f"{col}_last30m_vs_2h"] = f_30m[col].iloc[-1] - f_2h[col].iloc[-1]

        return features