# Generated results

Run `python scripts/run_pipeline.py` from the repository root to rebuild these
files from the raw UCI CSV.

- `metrics.json`: dataset scope, train/validation/test periods, validation model
  comparison, final test results, diagnostics, and anomaly methodology.
- `validation_candidates.csv`: the bounded validation-only feature and modest
  hyperparameter search; it contains no final-test metrics.
- `predictions.csv`: every final-test timestamp, baseline and selected-model
  predictions, residual, percentage difference, percentile, and flag.
- `anomalies.csv`: test observations above the validation-calibrated positive
  residual threshold, sorted by residual.
- `failure_analysis.csv`: the ten largest absolute test errors, together with
  the features the model can observe and important information it cannot see.

An anomaly means unexpectedly high consumption compared with the model. It is
a period requiring investigation, not proof of a fault or avoidable energy use.

