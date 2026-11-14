"""Baseline and machine-learning model definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


class HistoricalAverageBaseline(RegressorMixin, BaseEstimator):
    """Expected usage from training averages for the same weekday/time slot."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalAverageBaseline":
        values = X[["day_of_week", "hour", "minute"]].copy()
        values["target"] = np.asarray(y, dtype=float)
        self.slot_means_ = values.groupby(
            ["day_of_week", "hour", "minute"], observed=True
        )["target"].mean()
        self.time_means_ = values.groupby(["hour", "minute"], observed=True)[
            "target"
        ].mean()
        self.global_mean_ = float(values["target"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "slot_means_"):
            raise ValueError("Baseline must be fitted before prediction")
        predictions: list[float] = []
        for row in X[["day_of_week", "hour", "minute"]].itertuples(index=False):
            slot_key = (row.day_of_week, row.hour, row.minute)
            time_key = (row.hour, row.minute)
            value = self.slot_means_.get(slot_key, np.nan)
            if pd.isna(value):
                value = self.time_means_.get(time_key, self.global_mean_)
            predictions.append(float(value))
        return np.asarray(predictions)


def build_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Return the baseline and the three requested ML models."""
    return {
        "baseline": HistoricalAverageBaseline(),
        "linear_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.04,
                        max_depth=3,
                        min_samples_leaf=3,
                        random_state=random_state,
                        loss="squared_error",
                    ),
                ),
            ]
        ),
    }

