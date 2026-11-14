from __future__ import annotations

import numpy as np
import pandas as pd

from ml.inference import _positive_residual_percentiles
from ml.pipeline import _calibrate_anomaly_threshold


def test_positive_residual_percentiles_use_calibration_distribution() -> None:
    residuals = np.array([-5.0, 0.0, 5.0, 15.0, 25.0])
    reference = np.array([5.0, 10.0, 20.0, 30.0])
    scores = _positive_residual_percentiles(residuals, reference)
    np.testing.assert_allclose(scores, [0.0, 0.0, 25.0, 50.0, 75.0])


def test_anomaly_threshold_uses_validation_residuals_only() -> None:
    validation = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="10min"),
            "consumption_wh": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    predictions = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    threshold, positive_residuals, metadata = _calibrate_anomaly_threshold(
        validation, predictions
    )
    np.testing.assert_allclose(positive_residuals, [5.0, 10.0, 15.0, 20.0])
    assert np.isclose(threshold, np.quantile(positive_residuals, 0.95))
    assert "test" in metadata["threshold_source"]

