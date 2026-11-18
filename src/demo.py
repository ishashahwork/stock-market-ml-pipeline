"""Deterministic synthetic data and a small, in-memory model for the public demo.

No prices in this module are observed market prices. Nothing is saved as a real
company model, and demo scores must not be presented as project performance.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBClassifier, XGBRegressor

from src.feature_engineering import build_features
from src.PySimFin import transform_share_prices

COMPANIES = {
    "AMZN": ("Amazon", 11, 130.0),
    "MSFT": ("Microsoft", 22, 240.0),
    "AAPL": ("Apple", 33, 145.0),
    "AAON": ("AAON", 44, 60.0),
    "ALL": ("Allstate", 55, 110.0),
}
DEMO_END = "2026-03-20"


@dataclass
class DemoAnalysis:
    ticker: str
    raw: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.Series
    training_end: pd.Timestamp
    evaluation_start: pd.Timestamp
    training_rows: int
    feature_count: int


def generate_demo_prices(ticker: str) -> pd.DataFrame:
    """Create reproducible OHLCV data; ticker names are illustrative labels only."""
    if ticker not in COMPANIES:
        raise ValueError(f"Unsupported demo ticker: {ticker}")
    _, seed, starting_price = COMPANIES[ticker]
    rng = np.random.default_rng(seed)
    count = 780
    phase = np.arange(count)
    daily_return = 0.0003 + 0.002 * np.sin(phase / 19) + rng.normal(0, 0.012, count)
    close = starting_price * np.exp(np.cumsum(daily_return))
    opening = close * np.exp(rng.normal(0, 0.004, count))
    spread = rng.uniform(0.002, 0.015, count)
    return pd.DataFrame({
        "Open": opening,
        "High": np.maximum(opening, close) * (1 + spread),
        "Low": np.minimum(opening, close) * (1 - spread),
        "Close": close,
        "Adj. Close": close,
        "Volume": rng.integers(1_000_000, 8_000_000, count),
        "Shares Outstanding": np.full(count, 1_000_000_000.0),
        "Dividend": np.zeros(count),
    }, index=pd.bdate_range(end=DEMO_END, periods=count, name="Date"))


def build_demo_analysis(ticker: str) -> DemoAnalysis:
    """Fit compact XGBoost models on synthetic history and predict a later period."""
    raw = generate_demo_prices(ticker)
    training = build_features(
        pl.from_pandas(raw.drop(columns="Dividend")).lazy().with_row_index("__row_id")
    ).collect().to_pandas()
    positions = training.pop("__row_id").to_numpy()
    training.index = raw.index.take(positions)
    features = [name for name in training if name not in {"Target Return Class", "Target Return Metric"}]
    # Leave a full row between the last training input and evaluation inputs.
    cutoff = 480
    fit = training.iloc[:cutoff - 1]
    classifier = XGBClassifier(
        n_estimators=60, max_depth=3, learning_rate=0.06, min_child_weight=4,
        subsample=0.9, colsample_bytree=0.9, objective="binary:logistic",
        eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=42,
    )
    regressor = XGBRegressor(
        n_estimators=60, max_depth=3, learning_rate=0.06,
        objective="reg:squarederror", tree_method="hist", n_jobs=1, random_state=42,
    )
    classifier.fit(fit[features], fit["Target Return Class"].astype(int))
    regressor.fit(fit[features], fit["Target Return Metric"])
    evaluation_start = training.index[cutoff]
    inputs = transform_share_prices(raw).loc[evaluation_start:]
    probabilities = classifier.predict_proba(inputs[features])
    predictions = raw.loc[inputs.index].copy()
    predictions["Prediction"] = classifier.predict(inputs[features])
    predictions["Probability Up"] = probabilities[:, 1]
    predictions["Prediction Confidence"] = probabilities.max(axis=1)
    predictions["Predicted Return"] = regressor.predict(inputs[features])
    predictions["Actual Next Return"] = raw["Adj. Close"].shift(-1).div(raw["Adj. Close"]).sub(1).reindex(inputs.index)
    return DemoAnalysis(
        ticker=ticker, raw=raw, predictions=predictions,
        feature_importance=pd.Series(classifier.feature_importances_, index=features).sort_values(ascending=False),
        training_end=fit.index[-1], evaluation_start=evaluation_start,
        training_rows=len(fit), feature_count=len(features),
    )
