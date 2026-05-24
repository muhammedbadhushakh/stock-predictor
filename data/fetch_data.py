"""
fetch_data.py
─────────────
Fetches historical OHLCV price data from Yahoo Finance.
Optionally fetches news headlines from NewsAPI (requires free API key).
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta


def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data for a given ticker symbol.

    Args:
        ticker: Stock symbol e.g. 'AAPL'
        start:  Start date string 'YYYY-MM-DD'
        end:    End date string 'YYYY-MM-DD'

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"[Data] Fetching price data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check symbol and date range.")

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Flatten MultiIndex columns if present (yfinance quirk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"[Data] Retrieved {len(df)} trading days.")
    return df


def fetch_news_sentiment_mock(ticker: str, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Generates mock sentiment scores for demonstration.
    Replace with a real NewsAPI / RSS feed + NLP pipeline in production.

    Returns:
        pd.Series of float sentiment scores in [-1, +1], indexed by date.

    ── Upgrade path ────────────────────────────────────────────────────────────
    1. Get a free key at https://newsapi.org
    2. Replace this function body with:

        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={ticker}&from={date}&sortBy=publishedAt"
            f"&language=en&apiKey={API_KEY}"
        )
        articles = requests.get(url).json().get("articles", [])
        # pipe headlines through FinBERT (see features.py for scorer)
    ────────────────────────────────────────────────────────────────────────────
    """
    import numpy as np

    rng = np.random.default_rng(seed=42)
    scores = pd.Series(
        rng.uniform(-0.6, 0.6, size=len(dates)),
        index=dates,
        name="sentiment",
    )
    return scores