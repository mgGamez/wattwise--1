# WATTWISE

**Track:** AI + Climate

**Project:** Veritas AI International High School Hackathon 2026

WATTWISE is **AI energy intelligence for buildings**. It learns normal
electricity-consumption patterns, predicts expected usage, and highlights
unusually high observations for a building occupant or facilities manager to
investigate. A flagged observation is not proof of energy waste, a broken
appliance, or any particular cause.

This repository contains the validated ML foundation and a static Next.js
hackathon website that reads the generated result files directly.

## Research question and current result

> Can machine learning learn normal building energy-consumption patterns well
> enough to detect unexpectedly high electricity usage?

Model and feature selection use a chronological validation period. The final
test period remains unseen until those choices and the anomaly threshold are
frozen. Random Forest had the lowest validation RMSE and was selected.

| Method | Validation MAE (Wh) | Validation RMSE (Wh) |
|---|---:|---:|
| Linear Regression | 27.08 | 60.93 |
| **Random Forest** | 27.41 | **58.51** |
| Gradient Boosting | **25.89** | 59.63 |

The frozen final comparison on 2,810 unseen observations was:

| Method | Final test MAE (Wh) | Final test RMSE (Wh) |
|---|---:|---:|
| Baseline | 46.65 | 86.31 |
| **Selected Random Forest** | **27.20** | **59.56** |

That is a 41.70% MAE improvement and 30.99% RMSE improvement over the baseline.
Full-precision results and diagnostics are in `results/metrics.json`.

## Dataset

WATTWISE uses the public
[UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction),
listed by UCI under CC BY 4.0.

- 19,735 rows from one low-energy residential building
- 11 January through 27 May 2016
- one row every 10 minutes, with no missing intervals
- target: appliance energy consumption in Wh per 10-minute interval
- model weather inputs: lagged outdoor temperature and relative humidity

This is building data from one residence, not school-specific data. See
`data/DATASET.md` for provenance, definitions, quality findings, and limits.

## Evaluation methodology

1. Validate and clean timestamps, energy, temperature, and humidity.
2. Create calendar and strictly past-looking consumption features.
3. Drop the first seven days, which lack a full 7-day history.
4. Split chronologically into 70% train, 15% validation, and 15% test.
5. Compare a bounded set of model/feature candidates on validation only.
6. Freeze the lowest-validation-RMSE configuration and anomaly threshold.
7. Refit the winner on train + validation and evaluate once on final test.
8. Calculate `actual - predicted` and flag large positive residuals.

No rows are shuffled. Unselected models do not receive final-test scores.

### Selected input features

| Feature | Meaning | Leakage control |
|---|---|---|
| `hour_sin`, `hour_cos` | Cyclic time of day | Timestamp only |
| `day_of_week_sin`, `day_of_week_cos` | Cyclic weekday | Timestamp only |
| `month`, `is_weekend` | Calendar context | Timestamp only |
| `previous_outdoor_temperature_c` | Temperature 10 minutes earlier | Earlier sensor reading |
| `previous_outdoor_humidity_pct` | Humidity 10 minutes earlier | Earlier sensor reading |
| `previous_interval_consumption` | Consumption 10 minutes earlier | One row earlier |
| `previous_hour_consumption` | Consumption one hour earlier | Six rows earlier |
| `previous_day_same_hour` | Consumption one day earlier | 144 rows earlier |
| `rolling_1h_average` | Previous-hour mean | Shifted; current row excluded |
| `rolling_6h_average` | Previous-6-hour mean | Shifted; current row excluded |
| `rolling_24h_average` | Previous-day mean | Shifted; current row excluded |
| `rolling_7day_average` | Previous-week mean | Shifted; current row excluded |

The current observation's consumption is the target and never enters its own
features.

## Model investigation

The earlier tree models generalized poorly largely because they lacked strong
short-term history and the original Random Forest overfit its training period.
Validation evidence—not final-test tuning—supported two changes:

- Adding previous-interval, 1-hour, and 6-hour history reduced the best
  validation RMSE from about 81–82 Wh to about 59 Wh.
- The original forest's train/validation RMSE was 32.47/64.16 Wh. Modest depth
  and leaf regularization changed that to 53.49/58.51 Wh, greatly reducing the
  gap and improving validation performance.
- Cyclic calendar encoding improved the best recent-history configuration from
  58.98 to 58.51 Wh validation RMSE.
- Linear Regression uses median imputation and standardization. Tree models are
  not scaled because tree splits are scale-invariant.

These observations are consistent with overfitting and temporal distribution
shift; they do not prove a single cause. All 20 predefined validation candidates
are inspectable in `results/validation_candidates.csv`.

## Concepts to explain to judges

### What does the model predict?

It predicts expected appliance energy consumption in Wh for one 10-minute
interval. It does not predict a cause, cost, emissions, or whether usage was
necessary.

### What is a baseline?

A baseline is a simple reference method. WATTWISE averages pre-test readings
for the same weekday and 10-minute time slot, with time-of-day and overall
fallbacks. The ML model must be compared against it on the identical test period.

### Why train, validation, and test?

Training fits the models. Validation chooses features, modest hyperparameters,
and the model family. The later test period estimates performance only after
those choices are frozen. This prevents choosing a model because it happened to
look best on the final answers.

### What do MAE and RMSE mean?

- **MAE** is the average absolute prediction error. The selected model's final
  MAE is 27.20 Wh per 10-minute observation.
- **RMSE** penalizes large misses more strongly. The selected model's final RMSE
  is 59.56 Wh.

Lower is better for both.

### What is a residual?

```text
residual = actual consumption - predicted consumption
```

A positive residual means consumption was above prediction. A negative residual
means it was below prediction.

### How does anomaly detection work?

The threshold is the 95th percentile of the selected model's positive
**validation** residuals: 203.80 Wh. It is frozen before final testing, so test
residuals do not choose the threshold. It flags 48 of 2,810 final-test
observations (1.71%). Each anomaly also receives a percentile relative to the
positive validation residual distribution.

An anomaly is unexpectedly high consumption requiring investigation, not proof
of energy waste.

### What can the model not know?

The data does not reveal occupancy, appliance state, events, holidays,
maintenance, or sensor faults. The model cannot explain why an observation is
unusual or decide whether it was avoidable.

## Failure analysis

`results/failure_analysis.csv` contains the ten largest absolute errors in the
final unseen test period. The three largest were:

| Timestamp | Actual (Wh) | Predicted (Wh) | Absolute error (Wh) |
|---|---:|---:|---:|
| 2016-05-27 09:40 | 580 | 66.79 | 513.21 |
| 2016-05-16 17:40 | 630 | 130.78 | 499.22 |
| 2016-05-13 11:10 | 600 | 115.78 | 484.22 |

The file records what the model could and could not observe without inventing
causes.

## Run the project

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python -m pytest -q
```

## Reuse the trained model

The functions need chronological history. Seven days are required for the
7-day rolling feature.

```python
import pandas as pd
from ml.inference import analyse_consumption, predict_expected_usage

history = pd.read_csv(
    "data/processed/energy_cleaned.csv",
    parse_dates=["timestamp"],
)

expected = predict_expected_usage(history)
periods_to_investigate = analyse_consumption(history)
```

Set `anomalies_only=False` in `analyse_consumption(...)` to receive every
observation with its residual and flag.

## Repository structure

```text
data/       source, cleaned data, features, quality report, and dataset card
ml/         cleaning, feature, model, evaluation, and inference code
model/      selected trained model bundle and readable metadata
results/    metrics, validation candidates, predictions, anomalies, and failures
notebook/   reserved for explanation; pipeline logic remains in ml/
scripts/    command-line pipeline entry point
tests/      leakage, feature, split, calibration, and scoring tests
website/    static Next.js presentation site powered by generated ML results
```

## Reproducibility notes

- Random seed: 42 for Random Forest and Gradient Boosting.
- Split: earliest 70% train, next 15% validation, final 15% test.
- Selection: lowest validation RMSE; validation MAE breaks an exact tie.
- Tuning: 20 predefined combinations; no test-guided tuning.
- Threshold: 95th percentile of positive validation residuals, frozen pre-test.
- Model artifact: `model/wattwise_model.joblib`.
