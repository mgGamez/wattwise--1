# WATTWISE

**Track:** AI + Climate

**Project:** Veritas AI International High School Hackathon 2026

WATTWISE learns a building's normal electricity-consumption patterns, predicts
expected usage, and highlights unusually high observations for a facilities
manager to investigate. A flagged observation is **not** proof of energy waste,
a broken appliance, or any particular cause.

This repository currently contains the completed ML foundation only. The
website is intentionally deferred.

## Research question and current result

> Can machine learning learn normal building energy-consumption patterns well
> enough to detect unexpectedly high electricity usage?

On this dataset's held-out final 20% of time, Linear Regression had the lowest
RMSE and MAE. Its RMSE was about 5.1% lower than the historical-average
baseline. This is promising evidence for the method on one building, but it is
not proof that the same accuracy will transfer to a school or another season.

| Method | Test MAE (Wh) | Test RMSE (Wh) |
|---|---:|---:|
| Baseline | 46.15 | 84.16 |
| **Linear Regression** | **45.36** | **79.86** |
| Random Forest | 65.78 | 97.53 |
| Gradient Boosting | 59.48 | 86.74 |

All metrics use the same 3,746 unseen test observations from 1 May through 27
May 2016. Lower is better. Full-precision values are in
`results/metrics.json`.

## Dataset

WATTWISE uses the public
[UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction),
listed by UCI under CC BY 4.0.

- 19,735 rows from one low-energy residential building
- 11 January through 27 May 2016
- one row every 10 minutes, with no missing intervals
- target: appliance energy use in Wh per 10-minute interval
- model weather inputs: outdoor temperature and relative humidity

See [`data/DATASET.md`](data/DATASET.md) for provenance, exact definitions,
quality findings, and limitations. The building is a residence, not a school;
that is the largest external-validity limitation of this prototype.

## How the ML pipeline works

1. Validate and clean the timestamp, energy, temperature, and humidity fields.
2. Create calendar features and strictly past-looking consumption features.
3. Drop the first seven days, which do not yet have a full 7-day history.
4. Train on the earliest 80% and test on the latest 20%; never shuffle rows.
5. Compare a historical-average baseline with three requested regressors.
6. Select the lowest-test-RMSE model and calculate `actual - predicted`.
7. Flag positive residuals above a threshold calibrated only from training data.

### Input features

| Feature | Plain-language meaning | Available before prediction? |
|---|---|---|
| `hour`, `minute` | Time of day | Yes, from timestamp |
| `day_of_week` | Monday = 0 through Sunday = 6 | Yes, from timestamp |
| `month` | Calendar month | Yes, from timestamp |
| `is_weekend` | 1 on Saturday/Sunday, otherwise 0 | Yes, from timestamp |
| `previous_outdoor_temperature_c` | Outdoor temperature 10 minutes earlier | Yes, from an earlier sensor reading |
| `previous_outdoor_humidity_pct` | Outdoor humidity 10 minutes earlier | Yes, from an earlier sensor reading |
| `previous_hour_consumption` | Reading exactly one hour earlier | Yes, six rows earlier |
| `previous_day_same_hour` | Reading exactly one day earlier | Yes, 144 rows earlier |
| `rolling_24h_average` | Average of the previous 24 hours | Yes; current row excluded |
| `rolling_7day_average` | Average of the previous 7 days | Yes; current row excluded |

The current observation's consumption is the answer the model is trying to
predict, so it is never included in its own features.

## Concepts to explain to judges

### What does the model predict?

It predicts the expected number of watt-hours used by appliances during one
10-minute interval. It does not predict a cause, cost, carbon emissions, or
whether the usage was necessary.

### What is a baseline?

A baseline is a simple reference method that a machine-learning model should
beat to be useful. WATTWISE's baseline averages past training readings for the
same weekday and 10-minute time slot. If that exact slot is missing, it falls
back to the same time of day and then the overall training average.

### Why use a chronological split?

In real operation, a model learns from the past and predicts the future. Random
shuffling would let nearly adjacent future-like observations influence training
and would make the test unrealistically easy. WATTWISE trains on earlier rows
and evaluates once on later, unseen rows.

### What do MAE and RMSE mean?

- **MAE (Mean Absolute Error)** is the average size of the prediction error,
  ignoring whether the prediction was high or low. The selected model's MAE of
  45.36 Wh means its predictions differed from actual usage by about 45 Wh per
  10-minute observation on average.
- **RMSE (Root Mean Squared Error)** also measures error in Wh, but it penalizes
  large misses more strongly. That makes RMSE useful when rare large errors
  matter. The selected model's test RMSE is 79.86 Wh.

### What is a residual?

WATTWISE uses:

```text
residual = actual consumption - predicted consumption
```

A positive residual means the building used more energy than predicted. A
negative residual means it used less.

### How does anomaly detection work?

The anomaly threshold is not an invented fixed number. After selecting the
model, WATTWISE refits a temporary copy on the earlier part of training and
examines positive residuals on the final 20% of training. The threshold is the
95th percentile of those positive calibration residuals: **411.95 Wh**.

That threshold is then frozen and applied to the unseen test period. It flags
24 of 3,746 test observations (0.64%). This is not necessarily 5% of the test
set because the threshold comes from past training data rather than looking at
the test answers. Each anomaly also receives a percentile relative to the
positive calibration residuals.

### What can the model not know?

The data does not tell the model who was in the building, which appliance was
on, whether an event occurred, whether a holiday changed the schedule, whether
maintenance was happening, or whether a sensor was faulty. It therefore cannot
explain why an observation is unusual or decide that consumption was avoidable.
It can only identify a period that differs from patterns it learned.

## Failure analysis

`results/failure_analysis.csv` contains the ten largest absolute errors in the
unseen test period. The five largest were:

| Timestamp | Actual (Wh) | Predicted (Wh) | Absolute error (Wh) |
|---|---:|---:|---:|
| 2016-05-26 09:50 | 820 | 92.13 | 727.87 |
| 2016-05-26 16:40 | 850 | 137.42 | 712.58 |
| 2016-05-21 09:40 | 790 | 147.10 | 642.90 |
| 2016-05-06 07:30 | 670 | 82.53 | 587.47 |
| 2016-05-13 08:00 | 660 | 81.24 | 578.76 |

The model could see calendar time, outdoor weather, and earlier consumption for
these observations. It could not see occupancy, appliance state, events,
maintenance, holidays, or sensor faults. The file deliberately does not invent
causes for the errors.

## Run the project

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python -m pytest -q
```

The pipeline rebuilds cleaned data, features, metrics, predictions, anomalies,
failure analysis, and the saved model from the raw CSV.

## Reuse the trained model

Both public functions need chronological history. Seven days are required for
the 7-day rolling feature. The current row may contain actual consumption for
analysis, but feature engineering shifts consumption so the current value never
enters its own prediction.

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
analyzed observation with its residual and anomaly flag.

## Repository structure

```text
data/       raw source, cleaned data, feature data, quality report, dataset card
ml/         cleaning, feature, model, pipeline, and inference code
model/      selected trained model bundle and readable metadata
results/    metrics, predictions, anomalies, and failure analysis
notebook/   reserved for explanation; production logic stays in ml/
scripts/    command-line pipeline entry point
tests/      leakage, feature, split, and scoring tests
website/    reserved; frontend intentionally deferred
```

## Reproducibility notes

- Random seed: 42 for Random Forest and Gradient Boosting.
- Split: earliest 80% training, latest 20% testing after lag availability.
- Selection: lowest test RMSE, with MAE only as an exact-tie breaker.
- Model artifact: `model/wattwise_model.joblib`.
- Required outputs: `results/metrics.json`, `results/predictions.csv`, and
  `results/anomalies.csv`.
