# WATTWISE dataset card

## Source and license

- **Dataset:** Appliances Energy Prediction
- **Repository:** UCI Machine Learning Repository
- **Dataset page:** <https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction>
- **Downloaded archive:** <https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip>
- **License listed by UCI:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Original creators:** Luis M. Candanedo, Véronique Feldheim, and Dominique Deramaix

The raw CSV is retained unchanged at `data/raw/energydata_complete.csv`. The
downloaded UCI archive is retained beside it for provenance.

- Archive SHA-256: `2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a`
- CSV SHA-256: `2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d`

## What the data represents

| Property | Value |
|---|---|
| Building | One low-energy residential building in Belgium |
| Date range | 2016-01-11 17:00 through 2016-05-27 18:00 |
| Rows | 19,735 |
| Row meaning | One observation for a 10-minute interval |
| Measurement interval | 10 minutes, with no gaps in the source file |
| Prediction target | `Appliances`: appliance energy use |
| Target unit | Wh per 10-minute interval |
| Weather used | Previous 10-minute reading of outdoor temperature (°C) and relative humidity (%) |

The source contains a separate `lights` field. WATTWISE does **not** silently
combine that field with `Appliances`; the model target is the published
`Appliances` measurement only. It also excludes the source's `rv1` and `rv2`
random variables because they have no physical meaning for this project.

## Columns used by WATTWISE

| Cleaned column | Source column | Meaning |
|---|---|---|
| `timestamp` | `date` | Timestamp associated with the observation |
| `consumption_wh` | `Appliances` | Appliance energy use in Wh for the interval |
| `outdoor_temperature_c` | `T_out` | Outdoor temperature in degrees Celsius; model uses its previous 10-minute reading |
| `outdoor_humidity_pct` | `RH_out` | Outdoor relative humidity in percent; model uses its previous 10-minute reading |

## Data-quality findings

The reproducible report is in `data/processed/data_quality_report.json`.

- 0 invalid or missing timestamps
- 0 duplicate timestamps
- 0 missing energy values
- 0 negative or non-finite energy values
- 0 missing expected 10-minute timestamps
- 0 inconsistent intervals
- 0 missing outdoor temperature or humidity values

All 19,735 source rows are therefore present in `energy_cleaned.csv`. The
cleaning step standardizes names, parses timestamps, sorts chronologically, and
checks domain rules. It deliberately does not interpolate data, because an
interpolation that looks ahead could leak future information into a prediction.

## Suitability and limitations

This dataset is useful for a transparent prototype because it is public,
regularly sampled, small enough to reproduce locally, and contains weather.
However, it is one residence rather than a school. Its schedule, equipment,
occupancy, and climate may differ from another building. The results test the
WATTWISE method on this building; they do not establish performance for every
school or commercial building.

