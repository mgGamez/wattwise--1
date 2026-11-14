"""Leakage-safe feature engineering for ten-minute energy observations."""

from __future__ import annotations

import pandas as pd

from ml.data import EXPECTED_INTERVAL, standardize_columns


FEATURE_COLUMNS = [
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
    frame["previous_hour_consumption"] = indexed_consumption.reindex(
        pd.DatetimeIndex(timestamps - pd.Timedelta(hours=1))
    ).to_numpy()
    frame["previous_day_same_hour"] = indexed_consumption.reindex(
        pd.DatetimeIndex(timestamps - pd.Timedelta(days=1))
    ).to_numpy()

    past_only = indexed_consumption.shift(1)
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
        "previous_hour_consumption",
        "previous_day_same_hour",
        "rolling_24h_average",
        "rolling_7day_average",
    ]
    return feature_frame.dropna(subset=history_features).reset_index(drop=True)

