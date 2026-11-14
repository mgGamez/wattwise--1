"""Dataset loading, validation, and cleaning for WATTWISE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_COLUMNS = {
    "date": "timestamp",
    "Appliances": "consumption_wh",
    "T_out": "outdoor_temperature_c",
    "RH_out": "outdoor_humidity_pct",
}
EXPECTED_INTERVAL = pd.Timedelta(minutes=10)


def _to_python(value: Any) -> Any:
    """Convert NumPy values into JSON-safe Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return the four columns used by the model with consistent names."""
    frame = data.rename(columns=SOURCE_COLUMNS).copy()
    required = list(SOURCE_COLUMNS.values())
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame[required]


def clean_dataset(
    raw_path: str | Path,
    cleaned_path: str | Path,
    report_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the UCI data, remove unusable records, and save clean data.

    The function does not interpolate consumption or weather values. Interpolation
    can accidentally use future observations. Weather gaps, if any, are handled by
    a training-only imputer inside each model instead.
    """
    raw_path = Path(raw_path)
    cleaned_path = Path(cleaned_path)
    report_path = Path(report_path)

    raw = pd.read_csv(raw_path)
    missing_source_columns = [c for c in SOURCE_COLUMNS if c not in raw.columns]
    if missing_source_columns:
        raise ValueError(f"Raw dataset is missing columns: {missing_source_columns}")

    frame = standardize_columns(raw)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in (
        "consumption_wh",
        "outdoor_temperature_c",
        "outdoor_humidity_pct",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_timestamp_rows = int(frame["timestamp"].isna().sum())
    missing_consumption_rows = int(frame["consumption_wh"].isna().sum())
    negative_consumption_rows = int((frame["consumption_wh"] < 0).sum())
    non_finite_consumption_rows = int(
        (~np.isfinite(frame["consumption_wh"].dropna())).sum()
    )
    duplicate_timestamp_rows = int(
        frame.loc[frame["timestamp"].notna(), "timestamp"].duplicated().sum()
    )

    valid_mask = (
        frame["timestamp"].notna()
        & frame["consumption_wh"].notna()
        & np.isfinite(frame["consumption_wh"])
        & (frame["consumption_wh"] >= 0)
    )
    cleaned = frame.loc[valid_mask].copy()
    cleaned = cleaned.sort_values("timestamp", kind="stable")
    cleaned = cleaned.drop_duplicates(subset="timestamp", keep="first")
    cleaned = cleaned.reset_index(drop=True)

    timestamps = cleaned["timestamp"]
    deltas = timestamps.diff().dropna()
    unexpected_intervals = int((deltas != EXPECTED_INTERVAL).sum())
    expected_timestamps = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq=EXPECTED_INTERVAL,
    )
    missing_timestamps = expected_timestamps.difference(pd.DatetimeIndex(timestamps))

    interval_counts = {
        str(delta): int(count)
        for delta, count in deltas.value_counts().sort_index().items()
    }
    report: dict[str, Any] = {
        "source_file": raw_path.name,
        "source_rows": int(len(raw)),
        "cleaned_rows": int(len(cleaned)),
        "removed_rows": int(len(raw) - len(cleaned)),
        "date_range": {
            "start": timestamps.min().isoformat(),
            "end": timestamps.max().isoformat(),
        },
        "expected_interval_minutes": 10,
        "interval_counts": interval_counts,
        "checks": {
            "invalid_timestamp_rows": invalid_timestamp_rows,
            "duplicate_timestamp_rows": duplicate_timestamp_rows,
            "missing_consumption_rows": missing_consumption_rows,
            "negative_consumption_rows": negative_consumption_rows,
            "non_finite_consumption_rows": non_finite_consumption_rows,
            "missing_expected_timestamps": int(len(missing_timestamps)),
            "unexpected_interval_rows": unexpected_intervals,
            "missing_temperature_rows": int(
                cleaned["outdoor_temperature_c"].isna().sum()
            ),
            "missing_humidity_rows": int(
                cleaned["outdoor_humidity_pct"].isna().sum()
            ),
        },
        "consumption_wh_summary": {
            key: _to_python(value)
            for key, value in cleaned["consumption_wh"].describe().to_dict().items()
        },
        "cleaning_decisions": [
            "Parsed timestamps and sorted rows chronologically.",
            "Kept the first row for any duplicate timestamp.",
            "Removed rows with invalid timestamps, missing/non-finite consumption, or negative consumption.",
            "Did not interpolate values, because future-aware interpolation could leak information.",
        ],
    }

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return cleaned, report


def load_cleaned_data(path: str | Path) -> pd.DataFrame:
    """Load a cleaned WATTWISE CSV."""
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    return standardize_columns(frame).sort_values("timestamp").reset_index(drop=True)

