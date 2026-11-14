from __future__ import annotations

import numpy as np

from ml.inference import _positive_residual_percentiles


def test_positive_residual_percentiles_use_calibration_distribution() -> None:
    residuals = np.array([-5.0, 0.0, 5.0, 15.0, 25.0])
    reference = np.array([5.0, 10.0, 20.0, 30.0])
    scores = _positive_residual_percentiles(residuals, reference)
    np.testing.assert_allclose(scores, [0.0, 0.0, 25.0, 50.0, 75.0])

