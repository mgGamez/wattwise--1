from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, make_features, modeling_rows
from ml.pipeline import _chronological_split


def synthetic_data(periods: int = 1_100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", periods=periods, freq="10min"
            ),
            "consumption_wh": np.arange(periods, dtype=float) + 10,
            "outdoor_temperature_c": np.linspace(5, 15, periods),
            "outdoor_humidity_pct": np.linspace(80, 60, periods),
        }
    )


def test_lags_and_rolling_windows_use_only_earlier_values() -> None:
    data = synthetic_data()
    featured = make_features(data)
    row_index = 1_050

    assert featured.loc[row_index, "previous_hour_consumption"] == data.loc[
        row_index - 6, "consumption_wh"
    ]
    assert featured.loc[row_index, "previous_day_same_hour"] == data.loc[
        row_index - 144, "consumption_wh"
    ]
    assert featured.loc[row_index, "rolling_24h_average"] == data.loc[
        row_index - 144 : row_index - 1, "consumption_wh"
    ].mean()
    assert featured.loc[row_index, "rolling_7day_average"] == data.loc[
        row_index - 1_008 : row_index - 1, "consumption_wh"
    ].mean()


def test_changing_current_and_future_targets_cannot_change_current_features() -> None:
    data = synthetic_data()
    cutoff = 1_050
    original = make_features(data)
    changed = data.copy()
    changed.loc[cutoff:, "consumption_wh"] = 999_999
    changed_features = make_features(changed)

    pd.testing.assert_series_equal(
        original.loc[cutoff, FEATURE_COLUMNS],
        changed_features.loc[cutoff, FEATURE_COLUMNS],
        check_names=False,
    )


def test_current_weather_is_not_used_for_current_prediction() -> None:
    data = synthetic_data()
    cutoff = 1_050
    original = make_features(data)
    changed = data.copy()
    changed.loc[cutoff:, "outdoor_temperature_c"] = -999
    changed.loc[cutoff:, "outdoor_humidity_pct"] = -999
    changed_features = make_features(changed)

    pd.testing.assert_series_equal(
        original.loc[cutoff, FEATURE_COLUMNS],
        changed_features.loc[cutoff, FEATURE_COLUMNS],
        check_names=False,
    )


def test_modeling_rows_require_full_seven_day_history() -> None:
    modeled = modeling_rows(make_features(synthetic_data()))
    assert modeled.iloc[0]["timestamp"] == pd.Timestamp("2026-01-08 00:00:00")


def test_chronological_split_never_shuffles() -> None:
    frame = modeling_rows(make_features(synthetic_data()))
    train, test = _chronological_split(frame, 0.8)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert train["timestamp"].is_monotonic_increasing
    assert test["timestamp"].is_monotonic_increasing

