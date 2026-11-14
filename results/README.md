# Generated results

Run `python scripts/run_pipeline.py` from the repository root to rebuild these
files from the raw UCI CSV.

- `metrics.json`: dataset scope, chronological split, MAE/RMSE comparison,
  selected model, and anomaly-calibration details.
- `predictions.csv`: every held-out test timestamp, all four predictions, the
  selected prediction, residual, percentage difference, percentile, and flag.
- `anomalies.csv`: test observations above the training-calibrated positive
  residual threshold, sorted by residual.
- `failure_analysis.csv`: the ten largest absolute test errors, together with
  the features the model can observe and important information it cannot see.

An anomaly means unexpectedly high consumption compared with the model. It is
a period requiring investigation, not proof of a fault or avoidable energy use.

