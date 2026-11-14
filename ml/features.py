"""Leakage-safe feature engineering for ten-minute energy observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data import EXPECTED_INTERVAL, standardize_columns


LEGACY_FEATURE_COLUMNS = [
    "hour",
    "minute",
    "day_of_week",
    "month",
    "is_weekend",
    "previous_outdoor_temperature_c",
    "previous_outdoor_humidity_pct",
    "previous_hour_consumption",
    "previous_day_same_hour",
    "rolling_24h_average",
    "rolling_7day_average",
]

CYCLIC_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month",
    "is_weekend",
    "previous_outdoor_temperature_c",
    "previous_outdoor_humidity_pct",
    "previous_hour_consumption",
    "previous_day_same_hour",
    "rolling_24h_average",
    "rolling_7day_average",
]

RECENT_HISTORY_FEATURES = [
    "previous_interval_consumption",
    "rolling_1h_average",
    "rolling_6h_average",
]

FEATURE_SETS = {
    "legacy": LEGACY_FEATURE_COLUMNS,
    "cyclic_calendar": CYCLIC_FEATURE_COLUMNS,
    "recent_history": LEGACY_FEATURE_COLUMNS + RECENT_HISTORY_FEATURES,
    "cyclic_plus_recent_history": CYCLIC_FEATURE_COLUMNS
    + RECENT_HISTORY_FEATURES,
}

# The pipeline stores the selected list in the model bundle. This default keeps
# imports backward compatible and represents every candidate feature available.
FEATURE_COLUMNS = list(dict.fromkeys(sum(FEATURE_SETS.values(), [])))


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create model features using only information available by each timestamp.

    Current consumption is the prediction target and is never used as a feature.
    Every consumption-derived feature is shifted to a strictly earlier timestamp.
    """
    frame = standardize_columns(data)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError("Feature input contains duplicate timestamps")

    timestamps = frame["timestamp"]
    frame["hour"] = timestamps.dt.hour.astype("int8")
    frame["minute"] = timestamps.dt.minute.astype("int8")
    frame["day_of_week"] = timestamps.dt.dayofweek.astype("int8")
    frame["month"] = timestamps.dt.month.astype("int8")
    frame["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype("int8")

    fractional_hour = timestamps.dt.hour + timestamps.dt.minute / 60.0
    frame["hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24.0)
    frame["day_of_week_sin"] = np.sin(
        2 * np.pi * timestamps.dt.dayofweek / 7.0
    )
    frame["day_of_week_cos"] = np.cos(
        2 * np.pi * timestamps.dt.dayofweek / 7.0
    )

    previous_timestamp = pd.DatetimeIndex(timestamps - EXPECTED_INTERVAL)
    indexed_temperature = frame.set_index("timestamp")["outdoor_temperature_c"]
    indexed_humidity = frame.set_index("timestamp")["outdoor_humidity_pct"]
    frame["previous_outdoor_temperature_c"] = indexed_temperature.reindex(
        previous_timestamp
    ).to_numpy()
    frame["previous_outdoor_humidity_pct"] = indexed_humidity.reindex(
        previous_timestamp
    ).to_numpy()

    indexed_consumption = frame.set_index("timestamp")["consumption_wh"]
    frame["previous_interval_consumption"] = indexed_consumption.reindex(
        previous_timestamp
    ).to_numpy()
    frame["previous_hour_consumption"] = indexed_consumption.reindex(
        pd.DatetimeIndex(timestamps - pd.Timedelta(hours=1))
    ).to_numpy()
    frame["previous_day_same_hour"] = indexed_consumption.reindex(
        pd.DatetimeIndex(timestamps - pd.Timedelta(days=1))
    ).to_numpy()

    past_only = indexed_consumption.shift(1)
    frame["rolling_1h_average"] = (
        past_only.rolling("1h", min_periods=6).mean().to_numpy()
    )
    frame["rolling_6h_average"] = (
        past_only.rolling("6h", min_periods=36).mean().to_numpy()
    )
    frame["rolling_24h_average"] = (
        past_only.rolling("24h", min_periods=144).mean().to_numpy()
    )
    frame["rolling_7day_average"] = (
        past_only.rolling("7D", min_periods=1008).mean().to_numpy()
    )

    frame.attrs["expected_interval"] = str(EXPECTED_INTERVAL)
    return frame


def modeling_rows(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Remove only rows lacking required consumption-history features."""
    history_features = [
        "previous_interval_consumption",
        "previous_hour_consumption",
        "previous_day_same_hour",
        "rolling_1h_average",
        "rolling_6h_average",
        "rolling_24h_average",
        "rolling_7day_average",
    ]
    return feature_frame.dropna(subset=history_features).reset_index(drop=True)

