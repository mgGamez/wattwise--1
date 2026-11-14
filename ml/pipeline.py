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
from ml.features import FEATURE_SETS, make_features, modeling_rows
from ml.inference import _positive_residual_percentiles
from ml.models import HistoricalAverageBaseline, build_model_candidates


MODEL_LABELS = {
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
    """Backward-compatible two-part chronological split helper."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    split_index = int(len(frame) * train_fraction)
    if split_index == 0 or split_index == len(frame):
        raise ValueError("Not enough rows for chronological split")
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def _train_validation_test_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sorted observations into three non-overlapping time periods."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fraction must be below 1")
    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    if train_end == 0 or validation_end == train_end or validation_end == len(frame):
        raise ValueError("Not enough rows for train/validation/test split")
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


def _important_hyperparameters(estimator: Any) -> dict[str, Any]:
    model = estimator.named_steps["model"]
    names = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "subsample",
    ]
    parameters = model.get_params()
    return {name: parameters[name] for name in names if name in parameters}


def _evaluate_validation_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Tune a small predefined grid using validation performance only."""
    y_train = train["consumption_wh"].to_numpy(dtype=float)
    y_validation = validation["consumption_wh"].to_numpy(dtype=float)
    records: list[dict[str, Any]] = []

    for feature_set_name, feature_columns in FEATURE_SETS.items():
        for model_name, variants in build_model_candidates().items():
            for variant_name, prototype in variants.items():
                fitted = clone(prototype).fit(train[feature_columns], y_train)
                train_predictions = np.clip(
                    fitted.predict(train[feature_columns]), 0, None
                )
                validation_predictions = np.clip(
                    fitted.predict(validation[feature_columns]), 0, None
                )
                records.append(
                    {
                        "model_name": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "variant": variant_name,
                        "feature_set": feature_set_name,
                        "feature_columns": list(feature_columns),
                        "train_metrics": _metrics(y_train, train_predictions),
                        "validation_metrics": _metrics(
                            y_validation, validation_predictions
                        ),
                        "validation_predictions": validation_predictions,
                        "fitted_estimator": fitted,
                        "prototype": prototype,
                        "hyperparameters": _important_hyperparameters(fitted),
                    }
                )

    def ranking_key(record: dict[str, Any]) -> tuple[float, float]:
        return (
            record["validation_metrics"]["rmse_wh"],
            record["validation_metrics"]["mae_wh"],
        )

    best_by_model = {
        model_name: min(
            (r for r in records if r["model_name"] == model_name),
            key=ranking_key,
        )
        for model_name in MODEL_LABELS
    }
    selected = min(best_by_model.values(), key=ranking_key)
    return best_by_model, selected, records


def _calibrate_anomaly_threshold(
    validation: pd.DataFrame,
    validation_predictions: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Calibrate exclusively from pre-test validation residuals."""
    residuals = (
        validation["consumption_wh"].to_numpy(dtype=float)
        - validation_predictions
    )
    positive_residuals = residuals[residuals > 0]
    if len(positive_residuals) == 0:
        raise ValueError("Validation period did not contain positive residuals")
    threshold = float(np.quantile(positive_residuals, 0.95))
    metadata = {
        "method": "95th percentile of positive validation residuals",
        "threshold_source": "validation period only; final test residuals are excluded",
        "calibration_period_start": validation["timestamp"].min(),
        "calibration_period_end": validation["timestamp"].max(),
        "calibration_rows": int(len(validation)),
        "positive_calibration_residuals": int(len(positive_residuals)),
        "threshold_wh": threshold,
    }
    return threshold, np.sort(positive_residuals), metadata


def _period_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "start": frame["timestamp"].min(),
        "end": frame["timestamp"].max(),
    }


def _distribution_summary(frame: pd.DataFrame) -> dict[str, float]:
    consumption = frame["consumption_wh"]
    return {
        "consumption_mean_wh": float(consumption.mean()),
        "consumption_median_wh": float(consumption.median()),
        "consumption_std_wh": float(consumption.std()),
        "consumption_p95_wh": float(consumption.quantile(0.95)),
        "outdoor_temperature_mean_c": float(
            frame["previous_outdoor_temperature_c"].mean()
        ),
        "outdoor_humidity_mean_pct": float(
            frame["previous_outdoor_humidity_pct"].mean()
        ),
    }


def _lag_feature_quality(train: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lag_columns = [
        "previous_interval_consumption",
        "previous_hour_consumption",
        "previous_day_same_hour",
        "rolling_1h_average",
        "rolling_6h_average",
        "rolling_24h_average",
        "rolling_7day_average",
    ]
    return {
        column: {
            "training_null_rows": int(train[column].isna().sum()),
            "training_pearson_correlation_with_target": float(
                train[column].corr(train["consumption_wh"])
            ),
            "uses_current_target": False,
        }
        for column in lag_columns
    }


def _failure_analysis(
    test: pd.DataFrame,
    predicted: np.ndarray,
    model_name: str,
    output_path: Path,
) -> pd.DataFrame:
    observed_columns = [
        "timestamp",
        "consumption_wh",
        "hour",
        "day_of_week",
        "is_weekend",
        "previous_outdoor_temperature_c",
        "previous_outdoor_humidity_pct",
        "previous_interval_consumption",
        "previous_hour_consumption",
        "previous_day_same_hour",
        "rolling_1h_average",
        "rolling_6h_average",
        "rolling_24h_average",
        "rolling_7day_average",
    ]
    analysis = test[observed_columns].copy()
    analysis["predicted_consumption_wh"] = predicted
    analysis["residual_wh"] = analysis["consumption_wh"] - predicted
    analysis["absolute_error_wh"] = analysis["residual_wh"].abs()
    analysis["model"] = MODEL_LABELS[model_name]
    analysis["what_model_can_observe"] = (
        "Calendar time, earlier outdoor weather, and earlier consumption patterns"
    )
    analysis["what_model_cannot_observe"] = (
        "Occupancy, appliance state, events, maintenance, holidays, and sensor faults"
    )
    largest = analysis.nlargest(10, "absolute_error_wh").reset_index(drop=True)
    largest.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return largest


def _validation_outputs(
    best_by_model: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for model_name, record in best_by_model.items():
        train_rmse = record["train_metrics"]["rmse_wh"]
        validation_rmse = record["validation_metrics"]["rmse_wh"]
        output[MODEL_LABELS[model_name]] = {
            "mae_wh": record["validation_metrics"]["mae_wh"],
            "rmse_wh": validation_rmse,
            "train_mae_wh": record["train_metrics"]["mae_wh"],
            "train_rmse_wh": train_rmse,
            "validation_minus_train_rmse_wh": validation_rmse - train_rmse,
            "variant": record["variant"],
            "feature_set": record["feature_set"],
            "feature_columns": record["feature_columns"],
            "hyperparameters": record["hyperparameters"],
        }
    return output


def _feature_set_outputs(records: list[dict[str, Any]]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for feature_set_name, feature_columns in FEATURE_SETS.items():
        best = min(
            (r for r in records if r["feature_set"] == feature_set_name),
            key=lambda r: (
                r["validation_metrics"]["rmse_wh"],
                r["validation_metrics"]["mae_wh"],
            ),
        )
        comparison[feature_set_name] = {
            "best_model": best["model_label"],
            "best_variant": best["variant"],
            "validation_mae_wh": best["validation_metrics"]["mae_wh"],
            "validation_rmse_wh": best["validation_metrics"]["rmse_wh"],
            "feature_columns": list(feature_columns),
        }
    return comparison


def _tree_model_findings(records: list[dict[str, Any]]) -> dict[str, Any]:
    def get_record(model_name: str, variant: str, feature_set: str) -> dict[str, Any]:
        return next(
            record
            for record in records
            if record["model_name"] == model_name
            and record["variant"] == variant
            and record["feature_set"] == feature_set
        )

    rf_original = get_record(
        "random_forest", "original", "cyclic_plus_recent_history"
    )
    rf_regularized = get_record(
        "random_forest", "regularized", "cyclic_plus_recent_history"
    )
    gb_original = get_record("gradient_boosting", "original", "recent_history")
    gb_regularized = get_record(
        "gradient_boosting", "regularized", "recent_history"
    )

    def compact(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "train_rmse_wh": record["train_metrics"]["rmse_wh"],
            "validation_rmse_wh": record["validation_metrics"]["rmse_wh"],
            "validation_minus_train_rmse_wh": (
                record["validation_metrics"]["rmse_wh"]
                - record["train_metrics"]["rmse_wh"]
            ),
        }

    return {
        "random_forest": {
            "original": compact(rf_original),
            "regularized": compact(rf_regularized),
            "finding": "The original forest had a much larger train-validation RMSE gap. Modest depth/leaf regularization reduced that gap and improved validation RMSE; this is consistent with reduced overfitting.",
        },
        "gradient_boosting": {
            "original": compact(gb_original),
            "regularized": compact(gb_regularized),
            "finding": "The shallow regularized variant improved validation RMSE. This does not establish a causal reason, but it supports keeping the less complex validation winner.",
        },
    }


def run_pipeline(project_root: str | Path | None = None) -> dict[str, Any]:
    """Run the leakage-safe WATTWISE evaluation and save its artifacts."""
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

    train, validation, test = _train_validation_test_split(model_frame)
    best_by_model, selected, validation_records = _evaluate_validation_candidates(
        train, validation
    )
    selected_model_name = selected["model_name"]
    selected_columns = selected["feature_columns"]

    anomaly_threshold, positive_calibration_residuals, calibration_metadata = (
        _calibrate_anomaly_threshold(
            validation, selected["validation_predictions"]
        )
    )

    # The test set is touched only after model/features/hyperparameters and the
    # anomaly threshold have been frozen using train/validation data.
    pretest = pd.concat([train, validation], ignore_index=True)
    final_model = clone(selected["prototype"]).fit(
        pretest[selected_columns], pretest["consumption_wh"]
    )
    final_predictions = np.clip(
        final_model.predict(test[selected_columns]), 0, None
    )
    final_baseline = HistoricalAverageBaseline().fit(
        pretest, pretest["consumption_wh"]
    )
    baseline_predictions = np.clip(final_baseline.predict(test), 0, None)

    actual = test["consumption_wh"].to_numpy(dtype=float)
    final_model_metrics = _metrics(actual, final_predictions)
    final_baseline_metrics = _metrics(actual, baseline_predictions)
    improvement = {
        "mae_percent": float(
            (final_baseline_metrics["mae_wh"] - final_model_metrics["mae_wh"])
            / final_baseline_metrics["mae_wh"]
            * 100.0
        ),
        "rmse_percent": float(
            (
                final_baseline_metrics["rmse_wh"]
                - final_model_metrics["rmse_wh"]
            )
            / final_baseline_metrics["rmse_wh"]
            * 100.0
        ),
        "formula": "(baseline error - model error) / baseline error * 100; positive means improvement",
    }

    residuals = actual - final_predictions
    percentages = np.divide(
        residuals,
        final_predictions,
        out=np.full_like(residuals, np.nan, dtype=float),
        where=final_predictions > 0,
    ) * 100.0
    anomaly_percentiles = _positive_residual_percentiles(
        residuals, positive_calibration_residuals
    )
    anomaly_flags = residuals > anomaly_threshold

    predictions_output = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "actual_consumption_wh": actual,
            "baseline_prediction_wh": baseline_predictions,
            "selected_model": MODEL_LABELS[selected_model_name],
            "predicted_consumption_wh": final_predictions,
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
        final_predictions,
        selected_model_name,
        results_dir / "failure_analysis.csv",
    )

    validation_table = pd.DataFrame(
        [
            {
                "model": record["model_label"],
                "variant": record["variant"],
                "feature_set": record["feature_set"],
                "train_mae_wh": record["train_metrics"]["mae_wh"],
                "train_rmse_wh": record["train_metrics"]["rmse_wh"],
                "validation_mae_wh": record["validation_metrics"]["mae_wh"],
                "validation_rmse_wh": record["validation_metrics"]["rmse_wh"],
                "selected": record is selected,
            }
            for record in validation_records
        ]
    ).sort_values(["validation_rmse_wh", "validation_mae_wh"])
    validation_table.to_csv(results_dir / "validation_candidates.csv", index=False)

    bundle = {
        "model": final_model,
        "model_name": selected_model_name,
        "model_label": MODEL_LABELS[selected_model_name],
        "model_variant": selected["variant"],
        "feature_set": selected["feature_set"],
        "feature_columns": selected_columns,
        "anomaly_threshold_wh": anomaly_threshold,
        "positive_calibration_residuals_wh": positive_calibration_residuals,
        "training_period_start": pretest["timestamp"].min().isoformat(),
        "training_period_end": pretest["timestamp"].max().isoformat(),
        "validation_period_start": validation["timestamp"].min().isoformat(),
        "validation_period_end": validation["timestamp"].max().isoformat(),
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

    feature_set_comparison = _feature_set_outputs(validation_records)
    retained_new_features = [
        feature
        for feature in [
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "previous_interval_consumption",
            "rolling_1h_average",
            "rolling_6h_average",
        ]
        if feature in selected_columns
    ]
    metrics_document: dict[str, Any] = {
        "dataset": {
            "name": "UCI Appliances Energy Prediction",
            "framing": "AI energy intelligence for buildings",
            "intended_user": "A building occupant or facilities manager",
            "source": "UCI Machine Learning Repository",
            "source_url": "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "building_type": "One low-energy residential building; not school-specific data",
            "raw_rows": quality_report["source_rows"],
            "cleaned_rows": quality_report["cleaned_rows"],
            "modeling_rows": int(len(model_frame)),
            "measurement_interval_minutes": 10,
            "target": "Appliances energy consumption",
            "energy_unit": "Wh per 10-minute interval",
            "date_range": quality_report["date_range"],
        },
        "split_method": "chronological 70% train / 15% validation / 15% test; no shuffling",
        "train_period": _period_summary(train),
        "validation_period": _period_summary(validation),
        "test_period": _period_summary(test),
        "validation_model_comparison": _validation_outputs(best_by_model),
        "selected_model": {
            "name": MODEL_LABELS[selected_model_name],
            "selection_rule": "Lowest validation RMSE; validation MAE breaks an exact RMSE tie",
            "variant": selected["variant"],
            "feature_set": selected["feature_set"],
            "feature_columns": selected_columns,
            "hyperparameters": selected["hyperparameters"],
            "refit_before_test": "Refitted on combined train + validation data after selection",
        },
        "final_test_baseline": final_baseline_metrics,
        "final_test_selected_model_results": final_model_metrics,
        "percentage_improvement_over_baseline": improvement,
        "anomaly_methodology": {
            **calibration_metadata,
            "comparison": "final-test positive residual strictly greater than frozen threshold",
            "final_test_anomaly_count": int(anomaly_flags.sum()),
            "final_test_anomaly_rate": float(anomaly_flags.mean()),
            "interpretation": "Unexpectedly high appliance consumption requiring investigation; not proof of energy waste.",
        },
        "model_investigation": {
            "tuning_scope": {
                "model_feature_combinations_evaluated": int(
                    len(validation_records)
                ),
                "random_forest_variants": ["original", "regularized"],
                "gradient_boosting_variants": ["original", "regularized"],
                "test_set_used_for_tuning": False,
            },
            "feature_set_validation_comparison": feature_set_comparison,
            "tree_model_findings": _tree_model_findings(validation_records),
            "retained_new_features": retained_new_features,
            "overfitting_check": {
                label: {
                    "train_rmse_wh": values["train_rmse_wh"],
                    "validation_rmse_wh": values["rmse_wh"],
                    "validation_minus_train_rmse_wh": values[
                        "validation_minus_train_rmse_wh"
                    ],
                    "note": "A positive train-to-validation gap is consistent with overfitting and/or temporal shift; it does not identify a cause by itself.",
                }
                for label, values in _validation_outputs(best_by_model).items()
            },
            "feature_scaling": {
                "linear_regression": "Median imputation followed by StandardScaler",
                "tree_models": "Not scaled because tree splits are scale-invariant",
            },
            "lag_feature_quality": _lag_feature_quality(train),
            "leakage_controls": [
                "All lag and rolling features use strictly earlier timestamps.",
                "Rolling windows are calculated after shifting consumption by one interval.",
                "Current weather is excluded; weather is lagged by one interval.",
                "Model, feature set, and hyperparameters are selected on validation only.",
                "Anomaly threshold is calibrated on validation residuals only.",
                "Unselected models are not evaluated on the final test period.",
            ],
            "temporal_distribution_summary": {
                "train": _distribution_summary(train),
                "validation": _distribution_summary(validation),
                "test": _distribution_summary(test),
                "interpretation": "Differences across chronological periods may reduce generalization, especially for trees that cannot extrapolate beyond learned split regions.",
            },
        },
        "failure_analysis": {
            "rows_saved": int(len(failures)),
            "ranking": "largest absolute errors in the final unseen test period",
            "path": "results/failure_analysis.csv",
            "causes_inferred": False,
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
