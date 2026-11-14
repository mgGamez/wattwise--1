"""End-to-end training, evaluation, and anomaly-detection pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.data import clean_dataset
from ml.features import FEATURE_COLUMNS, make_features, modeling_rows
from ml.inference import _positive_residual_percentiles
from ml.models import build_models


MODEL_LABELS = {
    "baseline": "Baseline",
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)}")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_wh": float(mean_absolute_error(actual, predicted)),
        "rmse_wh": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def _chronological_split(
    frame: pd.DataFrame, train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    split_index = int(len(frame) * train_fraction)
    if split_index == 0 or split_index == len(frame):
        raise ValueError("Not enough rows for chronological train/test split")
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def _fit_and_predict(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    fitted_models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, float]] = {}
    y_train = train["consumption_wh"]
    y_test = test["consumption_wh"].to_numpy(dtype=float)

    for name, model in build_models().items():
        fitted = model.fit(train[FEATURE_COLUMNS], y_train)
        predicted = np.clip(fitted.predict(test[FEATURE_COLUMNS]), 0, None)
        fitted_models[name] = fitted
        predictions[name] = predicted
        model_metrics[name] = _metrics(y_test, predicted)
    return fitted_models, predictions, model_metrics


def _calibrate_anomaly_threshold(
    selected_model: Any,
    train: pd.DataFrame,
    calibration_fraction: float = 0.20,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Calibrate on a later training slice, never on the held-out test period."""
    fit_part, calibration_part = _chronological_split(
        train, 1.0 - calibration_fraction
    )
    calibration_model = clone(selected_model)
    calibration_model.fit(
        fit_part[FEATURE_COLUMNS], fit_part["consumption_wh"]
    )
    predicted = np.clip(
        calibration_model.predict(calibration_part[FEATURE_COLUMNS]), 0, None
    )
    residuals = calibration_part["consumption_wh"].to_numpy(dtype=float) - predicted
    positive_residuals = residuals[residuals > 0]
    if len(positive_residuals) == 0:
        raise ValueError("Calibration slice did not contain positive residuals")
    threshold = float(np.quantile(positive_residuals, 0.95))
    metadata = {
        "method": "95th percentile of positive residuals",
        "threshold_source": "latest 20% of the training period",
        "fit_period_start": fit_part["timestamp"].min(),
        "fit_period_end": fit_part["timestamp"].max(),
        "calibration_period_start": calibration_part["timestamp"].min(),
        "calibration_period_end": calibration_part["timestamp"].max(),
        "calibration_rows": int(len(calibration_part)),
        "positive_calibration_residuals": int(len(positive_residuals)),
        "threshold_wh": threshold,
    }
    return threshold, np.sort(positive_residuals), metadata


def _failure_analysis(
    test: pd.DataFrame,
    predicted: np.ndarray,
    model_name: str,
    output_path: Path,
) -> pd.DataFrame:
    analysis = test[
        [
            "timestamp",
            "consumption_wh",
            "hour",
            "day_of_week",
            "is_weekend",
            "previous_outdoor_temperature_c",
            "previous_outdoor_humidity_pct",
            "previous_hour_consumption",
            "previous_day_same_hour",
            "rolling_24h_average",
            "rolling_7day_average",
        ]
    ].copy()
    analysis["predicted_consumption_wh"] = predicted
    analysis["residual_wh"] = analysis["consumption_wh"] - predicted
    analysis["absolute_error_wh"] = analysis["residual_wh"].abs()
    analysis["model"] = MODEL_LABELS[model_name]
    analysis["what_model_can_observe"] = (
        "Calendar time, earlier outdoor temperature/humidity, and earlier consumption patterns"
    )
    analysis["what_model_cannot_observe"] = (
        "Occupancy, appliance state, events, maintenance, holidays, and sensor faults"
    )
    largest = analysis.nlargest(10, "absolute_error_wh").reset_index(drop=True)
    largest.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return largest


def run_pipeline(project_root: str | Path | None = None) -> dict[str, Any]:
    """Run all WATTWISE ML steps and return the metrics document."""
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    raw_path = root / "data" / "raw" / "energydata_complete.csv"
    cleaned_path = root / "data" / "processed" / "energy_cleaned.csv"
    quality_path = root / "data" / "processed" / "data_quality_report.json"
    features_path = root / "data" / "processed" / "energy_features.csv"
    results_dir = root / "results"
    model_dir = root / "model"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cleaned, quality_report = clean_dataset(raw_path, cleaned_path, quality_path)
    feature_frame = make_features(cleaned)
    model_frame = modeling_rows(feature_frame)
    feature_frame.to_csv(
        features_path, index=False, date_format="%Y-%m-%d %H:%M:%S"
    )

    train, test = _chronological_split(model_frame, 0.80)
    fitted_models, all_predictions, model_metrics = _fit_and_predict(train, test)
    best_model_name = min(
        model_metrics,
        key=lambda name: (
            model_metrics[name]["rmse_wh"],
            model_metrics[name]["mae_wh"],
        ),
    )
    best_model = fitted_models[best_model_name]
    best_predictions = all_predictions[best_model_name]

    anomaly_threshold, positive_calibration_residuals, calibration_metadata = (
        _calibrate_anomaly_threshold(build_models()[best_model_name], train)
    )
    actual = test["consumption_wh"].to_numpy(dtype=float)
    residuals = actual - best_predictions
    percentages = np.divide(
        residuals,
        best_predictions,
        out=np.full_like(residuals, np.nan, dtype=float),
        where=best_predictions > 0,
    ) * 100.0
    anomaly_percentiles = _positive_residual_percentiles(
        residuals, positive_calibration_residuals
    )
    anomaly_flags = residuals > anomaly_threshold

    predictions_output = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "actual_consumption_wh": actual,
            "baseline_prediction_wh": all_predictions["baseline"],
            "linear_regression_prediction_wh": all_predictions[
                "linear_regression"
            ],
            "random_forest_prediction_wh": all_predictions["random_forest"],
            "gradient_boosting_prediction_wh": all_predictions[
                "gradient_boosting"
            ],
            "selected_model": MODEL_LABELS[best_model_name],
            "predicted_consumption_wh": best_predictions,
            "residual_wh": residuals,
            "percentage_above_prediction": percentages,
            "anomaly_percentile": anomaly_percentiles,
            "is_anomaly": anomaly_flags,
        }
    )
    predictions_output.to_csv(
        results_dir / "predictions.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    anomaly_columns = [
        "timestamp",
        "predicted_consumption_wh",
        "actual_consumption_wh",
        "residual_wh",
        "percentage_above_prediction",
        "anomaly_percentile",
    ]
    anomalies_output = predictions_output.loc[anomaly_flags, anomaly_columns].copy()
    anomalies_output = anomalies_output.sort_values(
        "residual_wh", ascending=False
    ).reset_index(drop=True)
    anomalies_output.to_csv(
        results_dir / "anomalies.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    failures = _failure_analysis(
        test,
        best_predictions,
        best_model_name,
        results_dir / "failure_analysis.csv",
    )

    bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "model_label": MODEL_LABELS[best_model_name],
        "feature_columns": FEATURE_COLUMNS,
        "anomaly_threshold_wh": anomaly_threshold,
        "positive_calibration_residuals_wh": positive_calibration_residuals,
        "training_period_start": train["timestamp"].min().isoformat(),
        "training_period_end": train["timestamp"].max().isoformat(),
        "measurement_interval_minutes": 10,
        "target": "consumption_wh",
    }
    joblib.dump(bundle, model_dir / "wattwise_model.joblib")

    metadata = {
        key: value
        for key, value in bundle.items()
        if key not in {"model", "positive_calibration_residuals_wh"}
    }
    metadata["positive_calibration_residual_count"] = int(
        len(positive_calibration_residuals)
    )
    (model_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_ready), encoding="utf-8"
    )

    metrics_document: dict[str, Any] = {
        "dataset": {
            "name": "Appliances Energy Prediction",
            "source": "UCI Machine Learning Repository",
            "source_url": "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "raw_rows": quality_report["source_rows"],
            "cleaned_rows": quality_report["cleaned_rows"],
            "modeling_rows": int(len(model_frame)),
            "measurement_interval_minutes": 10,
            "energy_unit": "Wh per 10-minute interval",
            "date_range": quality_report["date_range"],
        },
        "split": {
            "method": "chronological 80/20 split; no shuffling",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_start": train["timestamp"].min(),
            "train_end": train["timestamp"].max(),
            "test_start": test["timestamp"].min(),
            "test_end": test["timestamp"].max(),
        },
        "models": {
            MODEL_LABELS[name]: values for name, values in model_metrics.items()
        },
        "selection_rule": "Lowest test RMSE; test MAE breaks an exact RMSE tie",
        "best_model": MODEL_LABELS[best_model_name],
        "anomaly_detection": {
            **calibration_metadata,
            "comparison": "positive residual strictly greater than threshold",
            "test_anomaly_count": int(anomaly_flags.sum()),
            "test_anomaly_rate": float(anomaly_flags.mean()),
            "interpretation": "Unexpectedly high consumption requiring investigation; not proof of energy waste.",
        },
        "failure_analysis": {
            "rows_saved": int(len(failures)),
            "ranking": "largest absolute errors in the held-out test period",
            "path": "results/failure_analysis.csv",
        },
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics_document, indent=2, default=_json_ready),
        encoding="utf-8",
    )
    return metrics_document


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2, default=_json_ready))

