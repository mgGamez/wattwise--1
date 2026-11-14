"""Reusable prediction and anomaly-analysis functions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features import make_features, modeling_rows


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "wattwise_model.joblib"


def _load_bundle(model_path: str | Path) -> dict:
    bundle = joblib.load(model_path)
    required = {
        "model",
        "model_name",
        "feature_columns",
        "anomaly_threshold_wh",
        "positive_calibration_residuals_wh",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing fields: {sorted(missing)}")
    return bundle


def _positive_residual_percentiles(
    residuals: np.ndarray, calibration_residuals: np.ndarray
) -> np.ndarray:
    positive_reference = np.sort(
        np.asarray(calibration_residuals, dtype=float)
    )
    if len(positive_reference) == 0:
        raise ValueError("Anomaly calibration has no positive residuals")
    scores = np.zeros(len(residuals), dtype=float)
    positive_mask = residuals > 0
    scores[positive_mask] = (
        np.searchsorted(
            positive_reference,
            residuals[positive_mask],
            side="right",
        )
        / len(positive_reference)
        * 100.0
    )
    return scores


def predict_expected_usage(
    data: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Predict expected usage for rows with at least seven days of history.

    ``data`` must contain cleaned WATTWISE columns (or the original UCI column
    names). Its current consumption is ignored by feature creation; only earlier
    readings contribute to lag and rolling features.
    """
    bundle = _load_bundle(model_path)
    features = modeling_rows(make_features(data))
    columns = bundle["feature_columns"]
    predictions = np.clip(bundle["model"].predict(features[columns]), 0, None)
    return pd.DataFrame(
        {
            "timestamp": features["timestamp"].to_numpy(),
            "predicted_consumption_wh": predictions,
            "model_name": bundle["model_name"],
        }
    )


def analyse_consumption(
    data: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    anomalies_only: bool = True,
) -> pd.DataFrame:
    """Compare actual and expected usage and identify unusually high periods."""
    bundle = _load_bundle(model_path)
    features = modeling_rows(make_features(data))
    columns = bundle["feature_columns"]
    predicted = np.clip(bundle["model"].predict(features[columns]), 0, None)
    actual = features["consumption_wh"].to_numpy(dtype=float)
    residual = actual - predicted
    percentage_above = np.divide(
        residual,
        predicted,
        out=np.full_like(residual, np.nan, dtype=float),
        where=predicted > 0,
    ) * 100.0
    percentile = _positive_residual_percentiles(
        residual,
        np.asarray(bundle["positive_calibration_residuals_wh"]),
    )
    result = pd.DataFrame(
        {
            "timestamp": features["timestamp"].to_numpy(),
            "predicted_consumption_wh": predicted,
            "actual_consumption_wh": actual,
            "residual_wh": residual,
            "percentage_above_prediction": percentage_above,
            "anomaly_percentile": percentile,
            "is_anomaly": residual > float(bundle["anomaly_threshold_wh"]),
        }
    )
    if anomalies_only:
        result = result.loc[result["is_anomaly"]]
    return result.reset_index(drop=True)

