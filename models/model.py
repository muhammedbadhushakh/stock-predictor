"""
model.py
────────
XGBoost classifier that predicts next-day stock price direction.

Features:
  • Time-series aware train/test split (no lookahead bias)
  • Hyperparameter tuning via GridSearchCV
  • MLflow experiment tracking
  • Feature importance analysis
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

from features import FEATURE_COLS


# ─────────────────────────────────────────────────────────────
# Train / Test Split  (time-aware — no shuffling!)
# ─────────────────────────────────────────────────────────────

def train_test_split_ts(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Splits data into train and test sets chronologically.
    test_ratio: fraction of data reserved for testing (most recent).
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    print(f"[Model] Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


# ─────────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────────

def train_model(train_df: pd.DataFrame, tune: bool = True):
    """
    Train an XGBoost classifier with optional hyperparameter tuning.

    Args:
        train_df: training split from build_feature_matrix()
        tune:     if True, run GridSearchCV (slower but better params)

    Returns:
        (fitted_model, fitted_scaler)
    """
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    base_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    if tune:
        print("[Model] Running GridSearchCV (this may take ~1 min)...")
        param_grid = {
            "n_estimators":  [100, 300],
            "max_depth":     [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample":     [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        tscv = TimeSeriesSplit(n_splits=5)
        gs = GridSearchCV(
            XGBClassifier(**base_params),
            param_grid,
            cv=tscv,
            scoring="roc_auc",
            verbose=0,
            refit=True,
        )
        gs.fit(X_scaled, y_train)
        model = gs.best_estimator_
        print(f"[Model] Best params: {gs.best_params_}")
    else:
        model = XGBClassifier(
            **base_params,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        model.fit(X_scaled, y_train)

    return model, scaler


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, scaler, test_df: pd.DataFrame) -> dict:
    """
    Evaluate the model on the held-out test set.

    Returns dict of metrics: accuracy, f1, roc_auc.
    """
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_test = test_df["target"]

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc":  round(roc_auc_score(y_test, y_proba), 4),
    }

    print("\n" + "═" * 40)
    print("  MODEL EVALUATION")
    print("═" * 40)
    for k, v in metrics.items():
        print(f"  {k:<12} {v}")
    print("─" * 40)
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    return metrics, y_pred, y_proba


# ─────────────────────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────────────────────

def feature_importance_df(model) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    fi = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


# ─────────────────────────────────────────────────────────────
# Backtesting  (simple long/short strategy)
# ─────────────────────────────────────────────────────────────

def backtest(test_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Simulates a simple strategy:
        • Buy (long) if model predicts UP  (1)
        • Stay out if model predicts DOWN  (0)

    Returns DataFrame with cumulative strategy vs buy-and-hold returns.
    """
    bt = test_df[["Close"]].copy()
    bt = bt.copy()
    close = bt["Close"].squeeze()
    bt["market_return"]   = close.pct_change().fillna(0)
    bt["signal"]          = y_pred
    bt["strategy_return"] = bt["market_return"] * bt["signal"].shift(1).fillna(0)

    bt["cumulative_market"]   = (1 + bt["market_return"]).cumprod()
    bt["cumulative_strategy"] = (1 + bt["strategy_return"]).cumprod()

    final_market   = bt["cumulative_market"].iloc[-1]
    final_strategy = bt["cumulative_strategy"].iloc[-1]
    print(f"\n[Backtest] Buy-and-Hold Return:  {(final_market - 1)*100:.1f}%")
    print(f"[Backtest] Strategy Return:       {(final_strategy - 1)*100:.1f}%")

    return bt


# ─────────────────────────────────────────────────────────────
# MLflow Logging
# ─────────────────────────────────────────────────────────────

def log_to_mlflow(ticker: str, metrics: dict, model, params: dict = None):
    """Log a training run to MLflow for experiment tracking."""
    mlflow.set_experiment(f"stock_predictor_{ticker}")
    with mlflow.start_run():
        mlflow.log_params(params or model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")
    print("[MLflow] Run logged. Run `mlflow ui` to view experiments.")