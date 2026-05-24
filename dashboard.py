"""
dashboard.py
────────────
Interactive Plotly visualizations for the stock predictor.

Plots:
  1. Price chart with buy signals overlaid
  2. Backtest: strategy vs buy-and-hold cumulative returns
  3. Feature importances bar chart
  4. RSI & MACD subplots
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_price_with_signals(df: pd.DataFrame, y_pred: np.ndarray, ticker: str):
    """
    Candlestick chart with model's UP-prediction signals marked.
    """
    close = df["Close"].squeeze()
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"{ticker} Price + Model Signals", "Volume"],
        vertical_spacing=0.05,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(),
        high=df["High"].squeeze(),
        low=df["Low"].squeeze(),
        close=close,
        name="OHLC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Buy signals (where model predicted UP)
    signal_dates = df.index[y_pred == 1]
    signal_prices = close.loc[signal_dates]
    fig.add_trace(go.Scatter(
        x=signal_dates,
        y=signal_prices * 0.985,        # slightly below bar
        mode="markers",
        marker=dict(symbol="triangle-up", size=8, color="#00bcd4"),
        name="Predicted UP",
    ), row=1, col=1)

    # Volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"].squeeze(),
        name="Volume",
        marker_color="rgba(100,149,237,0.5)",
    ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Model Buy Signals",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        legend=dict(orientation="h", y=1.05),
    )
    fig.show()


def plot_backtest(bt: pd.DataFrame, ticker: str):
    """
    Cumulative return: strategy vs buy-and-hold.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bt.index,
        y=(bt["cumulative_market"] - 1) * 100,
        name="Buy & Hold",
        line=dict(color="#ef5350", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bt.index,
        y=(bt["cumulative_strategy"] - 1) * 100,
        name="Model Strategy",
        line=dict(color="#26a69a", width=2),
        fill="tonexty",
        fillcolor="rgba(38,166,154,0.1)",
    ))

    fig.update_layout(
        title=f"{ticker} — Strategy vs Buy & Hold (Test Period)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
    )
    fig.show()


def plot_feature_importance(fi_df: pd.DataFrame):
    """
    Horizontal bar chart of XGBoost feature importances.
    """
    fig = go.Figure(go.Bar(
        x=fi_df["importance"],
        y=fi_df["feature"],
        orientation="h",
        marker=dict(
            color=fi_df["importance"],
            colorscale="Viridis",
            showscale=True,
        ),
    ))
    fig.update_layout(
        title="Feature Importances (XGBoost)",
        xaxis_title="Importance Score",
        template="plotly_dark",
        height=550,
        yaxis=dict(autorange="reversed"),
    )
    fig.show()


def plot_technical_indicators(df: pd.DataFrame, ticker: str):
    """
    Multi-panel chart: Price + Bollinger Bands, RSI, MACD.
    """
    close = df["Close"].squeeze()
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=[
            f"{ticker} — Price & Bollinger Bands",
            "RSI (14)",
            "MACD",
        ],
        vertical_spacing=0.06,
    )

    # Price + BB
    fig.add_trace(go.Scatter(x=df.index, y=close, name="Close", line=dict(color="#90caf9")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_30"].squeeze(), name="SMA 30",
                             line=dict(color="#ffd54f", dash="dot")), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"].squeeze(), name="RSI",
                             line=dict(color="#ce93d8")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    macd = df["macd"].squeeze()
    macd_sig = df["macd_sig"].squeeze()
    macd_diff = df["macd_diff"].squeeze()

    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_diff]
    fig.add_trace(go.Bar(x=df.index, y=macd_diff, name="MACD Hist",
                         marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd, name="MACD",
                             line=dict(color="#80deea")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_sig, name="Signal",
                             line=dict(color="#ffab91")), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.08),
    )
    fig.show()