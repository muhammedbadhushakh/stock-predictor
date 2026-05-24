"""
features.py
───────────
Builds the feature matrix used for model training.

Technical indicators (via `ta` library):
  • RSI-14
  • MACD & MACD Signal
  • Bollinger Band Width
  • OBV (On-Balance Volume)
  • SMA 10 / SMA 30
  • Daily return & volatility

Sentiment:
  • Daily sentiment score from fetch_data.fetch_news_sentiment_mock()
    (swap for real FinBERT scores once you have a NewsAPI key)

Target:
  • Binary label: 1 if next-day Close > today's Close, else 0
"""

import pandas as pd
import numpy as np
import ta


# ─────────────────────────────────────────────────────────────
# Technical Indicators
# ─────────────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to the price DataFrame."""
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    # Trend
    df["sma_10"]  = ta.trend.sma_indicator(close, window=10)
    df["sma_30"]  = ta.trend.sma_indicator(close, window=30)
    df["ema_12"]  = ta.trend.ema_indicator(close, window=12)

    macd_obj      = ta.trend.MACD(close)
    df["macd"]    = macd_obj.macd()
    df["macd_sig"]= macd_obj.macd_signal()
    df["macd_diff"]= macd_obj.macd_diff()

    # Momentum
    df["rsi"]     = ta.momentum.rsi(close, window=14)
    df["stoch_k"] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)

    # Volatility
    bb            = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_width"]= bb.bollinger_wband()
    df["bb_pct"]  = bb.bollinger_pband()
    df["atr"]     = ta.volatility.average_true_range(high, low, close, window=14)

    # Volume
    df["obv"]     = ta.volume.on_balance_volume(close, vol)
    df["vwap"]    = (close * vol).cumsum() / vol.cumsum()

    # Price-derived
    df["daily_return"] = close.pct_change()
    df["volatility_5"] = df["daily_return"].rolling(5).std()
    df["price_vs_sma10"] = close / df["sma_10"] - 1
    df["price_vs_sma30"] = close / df["sma_30"] - 1

    # Lag features (yesterday & 2 days ago returns)
    df["lag_1_return"] = df["daily_return"].shift(1)
    df["lag_2_return"] = df["daily_return"].shift(2)

    return df


# ─────────────────────────────────────────────────────────────
# Target Label
# ─────────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary classification target:
        1  →  next-day close > today's close  (price goes UP)
        0  →  next-day close ≤ today's close  (price goes DOWN / flat)
    """
    close = df["Close"].squeeze()
    df["target"] = (close.shift(-1) > close).astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# Assemble Full Feature Matrix
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "sma_10", "sma_30", "ema_12",
    "macd", "macd_sig", "macd_diff",
    "rsi", "stoch_k",
    "bb_width", "bb_pct", "atr",
    "obv", "vwap",
    "daily_return", "volatility_5",
    "price_vs_sma10", "price_vs_sma30",
    "lag_1_return", "lag_2_return",
    "sentiment",
]


def build_feature_matrix(
    price_df: pd.DataFrame,
    sentiment_series: pd.Series,
) -> pd.DataFrame:
    """
    Merges price features and sentiment into a clean feature matrix.

    Returns:
        DataFrame with FEATURE_COLS + 'target', NaN rows dropped.
    """
    df = price_df.copy()
    df = add_technical_indicators(df)
    df = add_target(df)

    # Merge sentiment (align by date index)
    df["sentiment"] = sentiment_series.reindex(df.index).fillna(0)

    # Drop rows where any feature or target is NaN
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    print(f"[Features] Feature matrix shape: {df[FEATURE_COLS].shape}")
    return df