import pandas as pd
import numpy as np

from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

import plotly.graph_objects as go


print("Loading local stock data...")

df = pd.read_csv("stock_data.csv")

print(df.head())


# Indicators
df["SMA_10"] = SMAIndicator(
    close=df["Close"],
    window=5
).sma_indicator()

df["RSI"] = RSIIndicator(
    close=df["Close"],
    window=5
).rsi()


# Target
df["Target"] = df["Close"].shift(-1)

df.dropna(inplace=True)


# Features
X = df[[
    "Close",
    "Volume",
    "SMA_10",
    "RSI"
]]

y = df["Target"]


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)


# Model
model = XGBRegressor()

print("Training model...")

model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)


# RMSE
rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

print("RMSE:", rmse)


# Plot
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        y=y_test.values,
        mode="lines",
        name="Actual"
    )
)

fig.add_trace(
    go.Scatter(
        y=predictions,
        mode="lines",
        name="Predicted"
    )
)

fig.update_layout(
    title="Stock Price Prediction",
    xaxis_title="Time",
    yaxis_title="Price"
)

fig.show()