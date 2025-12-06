import numpy as np
import math
import pandas as pd
import scipy

def __fixed_exchange_universe__():
    
    high_volume_futures = [
    "binance",    # Binance Futures
    "bybit",      # Bybit
    # "hyperliquid",
    "okx",        # OKX
    "bitget",     # Bitget
    "gateio"      # Gate.io
    ]

    low_volume_futures = [
        "deribit",    # Deribit (options heavy, futures smaller)
        "bitfinex",   # Bitfinex Futures
        "kucoinfutures", # KuCoin Futures
        "bingx",      # BingX
        "phemex"      # Phemex
    ]
    
    exchanges = high_volume_futures + low_volume_futures 
    return exchanges

def __fixed_top_liquids__():
    token_universe = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'ATOM/USDT', 'TRX/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'HYPE/USDT', 'PURR/USDT', 'MATIC/USDT', 'LINK/USDT', 'DOT/USDT', 'BLUR/USDT', 'EIGEN/USDT', 'CRV/USDT', 'APE/USDT']
    return token_universe

def __fixed_token_universe__():
    ## top and loww liquids
    token_universe = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'ATOM/USDT', 'OP/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'HYPE/USDT', 'PURR/USDT', 'ARB/USDT', 'SUI/USDT', 'CRV/USDT', 'BLUR/USDT', 'EIGEN/USDT', 'FTT/USDT', 'APE/USDT']
    return token_universe 

def __fixed_conv_div_universe__():
    token_universe = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'ATOM/USDT', 'OP/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'HYPE/USDT', 'PURR/USDT', 'SUSHI/USDT', 'FARTCOIN/USDT', 'kSHIB/USDT', 'kFLOKI/USDT', 'EIGEN/USDT', 'DODO/USDT', 'APE/USDT']
    return token_universe


def __vol_liq_universe_selection__(df, lookback="2h", top_n=10):

    """
    Selects top tokens based on liquidity (volume) and volatility.
    """

    end = df.index.get_level_values("datetime").max()
    start = end - pd.Timedelta(lookback)
    df_lb = df.loc[pd.IndexSlice[:, start:end], :]

    grouped = df_lb.groupby("token")

    # liquidity: average dollar volume (close * volume)
    liquidity = grouped.apply(lambda g: (g["close"] * g["volume"]).mean())

    # volatility: standard deviation of log returns
    logret = grouped.apply(lambda g: np.log(g["close"]).diff().std())

    liq_z = (liquidity - liquidity.mean()) / liquidity.std()
    vol_z = (logret - logret.mean()) / logret.std()

    ## FIX: Non-symmtric weighting
    ## Score
    vol_w = 0.85
    liq_w = 1 - vol_w
    
    # combined score
    score = (liq_w * liq_z) + (vol_w * vol_z)

    top_tokens = score.nlargest(top_n).index.tolist()
    return top_tokens

def __vol_liq_universe_selection_v2__(df, lookback="2h", top_n=10):
    """
    Selects top tokens based on volatility and liquidity, with rank-normalized scoring.
    """
    
    vol_weight = 0.85

    end = df.index.get_level_values("datetime").max()
    start = end - pd.Timedelta(lookback)
    df_lb = df.loc[pd.IndexSlice[:, start:end], :]

    grouped = df_lb.groupby("token")

    # Liquidity: median dollar volume
    liquidity = grouped.apply(lambda g: (g["close"] * g["volume"]).median())

    # Volatility: std of log returns
    logret = grouped.apply(lambda g: np.log(g["close"]).diff().std())

    # Rank-normalize to handle outliers (convert to percentiles, then to z-scores)
    liq_rank = liquidity.rank(pct=True)
    vol_rank = logret.rank(pct=True)
    liq_z = scipy.stats.norm.ppf(liq_rank.clip(1e-6, 1 - 1e-6))
    vol_z = scipy.stats.norm.ppf(vol_rank.clip(1e-6, 1 - 1e-6))

    # Weighted combination (normalized weights)
    liq_w = 1 - vol_weight
    score = pd.Series(vol_weight * vol_z + liq_w * liq_z, index=liquidity.index)

    top_tokens = score.nlargest(top_n).index.tolist()
    return top_tokens


def __vol_corr_universe_selection_v2__(
    multi_df,
    btc_df,
    lookback="2h",
    top_n=5,
    ext_corr_threshold=0.6,
    uncorr_threshold=0.15 
):
    """
    Returns 3 buckets of tokens:
    1. High correlation with BTC + high volatility
    2. Negative correlation with BTC + high volatility
    3. Uncorrelated with BTC + high volatility
    """

    # Compute lookback window
    end = multi_df.index.get_level_values("datetime").max()
    start = end - pd.Timedelta(lookback)

    df_lb = multi_df.loc[pd.IndexSlice[:, start:end], :]
    btc_df_lb = btc_df.loc[start:end]

    grouped = df_lb.groupby("token")

    # Returns
    btc_ret = btc_df_lb["close"].pct_change()

    # Correlation per token
    corr = grouped.apply(lambda g: btc_ret.corr(g["close"].pct_change()))

    # Volatility (log-return std)
    vol = grouped.apply(lambda g: np.log(g["close"]).diff().std())

    # Make sure vol is not NaN
    vol = vol.fillna(0)

    # ---- BUCKETS ---- #
    high_corr_mask = corr > ext_corr_threshold
    neg_corr_mask = corr < -ext_corr_threshold
    uncorr_mask = corr.abs() <= uncorr_threshold

    # Sort each bucket by volatility (descending)
    high_corr_tokens = vol[high_corr_mask].sort_values(ascending=False).head(top_n).index.tolist()
    neg_corr_tokens = vol[neg_corr_mask].sort_values(ascending=False).head(top_n).index.tolist()
    uncorr_tokens = vol[uncorr_mask].sort_values(ascending=False).head(top_n).index.tolist()

    return high_corr_tokens, neg_corr_tokens, uncorr_tokens 


if __name__ == '__main__':
    print()