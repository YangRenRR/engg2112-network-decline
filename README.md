# Where Will the Network Fail Next?

Forecasting regional mobile network decline in New South Wales to prioritise
network maintenance.

**ENGG2112 · Semester 2, 2026 · TUT05**
Renyu Yang · Jingwei Qiu · Ying Xiao · Xiangbin Chen

---

## Problem

Mobile network maintenance is mostly reactive: faults are fixed after they
appear, and maintenance attention tends to follow population density rather
than need. Regional and remote areas already have weaker coverage, and network
failures matter most during emergencies.

## What we build

A classifier that predicts, for each region, whether median mobile download
speed will fall by more than 20% in the following quarter. Regions are then
ranked by predicted probability to produce a maintenance priority list.

```
Ookla speeds + BoM weather + ABS population
        -> features per region per quarter
        -> Random Forest classifier
        -> probability of significant decline
        -> maintenance priority ranking
```

## Data

| Data | Source | Granularity | Licence |
|---|---|---|---|
| Mobile download/upload speed, latency, tests | [Ookla Open Data](https://github.com/teamookla/ookla-open-data) | Quarterly, ~600 m tile | CC BY-NC-SA 4.0 |
| Temperature, rainfall | [Open-Meteo historical archive](https://open-meteo.com/) | Daily, any coordinate | Open |
| Population | Australian Bureau of Statistics | Annual, by SA2 | Open |

Ookla tiles are available from **2019 Q1 to 2026 Q1** (28 quarters). Files are
queried remotely with DuckDB rather than downloaded in full.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Aggregate one quarter of NSW mobile performance into grid cells
python src/fetch_ookla.py --year 2026 --quarter 1 --out data/nsw_2026Q1.csv

# How often does speed fall by more than 10/20/30% between two quarters?
python src/decline_rate.py --from 2025 4 --to 2026 1

# Build the modelling table: one row per region per quarter, 2019 Q1 to 2026 Q1
python src/build_panel.py --start 2019 1 --end 2026 1 --out data/panel.csv

# Quarterly weather features for every region in the panel
python src/weather.py --panel data/panel.csv --out data/weather.csv

# Train on 2019-2023, test on 2024-2026
python src/model.py --panel data/panel.csv --split 2024 --weather data/weather.csv

# Compare feature sets on the same rows, scored on the ranking they produce
python src/compare.py

# Predict the size of the change instead of a yes/no label
python src/regress.py

# Score the model against a mean-reversion baseline
python src/baselines.py
python src/where_rf_wins.py

# Daily download speeds from Measurement Lab
python src/mlab.py --start 2026-06-01 --end 2026-08-31 --out data/mlab_daily.csv

# Ask the weather and forecasting questions again, a day at a time
python src/daily.py --daily data/mlab_daily.csv

# Rank every region for the next quarter and write the dashboard's input
python src/export_predictions.py --panel data/panel.csv --out ui
```

`panel.csv` is one row per region per quarter:

| region | lon | lat | year | quarter | down_mbps | tests | prev_down_mbps | pct_change | declined |
|---|---|---|---|---|---|---|---|---|---|

`region` is a zoom-12 quadkey prefix, roughly an 8 km cell, standing in for an
ABS SA2 area until the boundary files are wired in. Quadkey prefixes are used
rather than the `tile_x` / `tile_y` columns because those only exist from
2023 Q3 onward, and we need every quarter back to 2019.

## Findings so far

### The 20% threshold is workable

Aggregating NSW tiles into ~11 km cells (a stand-in for SA2), for
2025 Q4 -> 2026 Q1:

| Regions | Fell >10% | Fell >20% | Fell >30% | Median change |
|---|---|---|---|---|
| 142 | 35.9% | **19.7%** | 9.9% | -2.3% |

About one region in five is labelled a significant decline. That is frequent
enough to train on and rare enough to be worth flagging.

### The panel covers 29 quarters and is large enough to model

Building the full table from 2019 Q1 to 2026 Q1 gives:

| Rows | Regions | Quarters | Rows with a previous quarter | Declined |
|---|---|---|---|---|
| 8,873 | 768 | 29 | 7,237 | **14.2%** |

Over the full period 14.2% of region-quarters are labelled a significant
decline, against 19.7% for the single 2025 Q4 to 2026 Q1 transition. The
single-quarter figure was on the high side; 14.2% is the number to design
around.

Coverage also shrinks over time: about 330 regions per quarter in 2020-21
against 240-280 in 2025-26, with total tests roughly halving. Models trained
on older quarters are therefore fitted on denser data than they will see at
prediction time, which is worth stating as a limitation.

### First model results

Trained on 2019-2023 and tested on 2024-2026, predicting whether a region
declines in the **following** quarter:

| | Recall | Precision | Regions flagged |
|---|---|---|---|
| Baseline: never flag anything | 0.00 | – | 0% |
| Baseline: flag every region | 1.00 | 0.125 | 100% |
| **Random forest** | **0.485** | **0.367** | **16.5%** |

The model finds about half the regions that go on to decline, and when it
flags a region it is right roughly a third of the time. Against the base rate
of 12.5% that is close to a threefold improvement in precision, while
inspecting only a sixth of the state.

Neither baseline is usable: never flagging finds nothing, and flagging
everything is the reactive status quo with extra steps. The useful comparison
is the last row against the second, and it says the same budget of
inspections can be aimed about three times better.

The most useful features are the change in speed into the current quarter,
the number of tiles and tests in a region, and the current speed itself.
Weather features have not been added yet.

### A single fitted parameter recovers most of the signal

Our first baseline, assuming next quarter looks like this one, predicts no
change at all and is too weak to learn anything from. The forest's most
useful feature by a wide margin is how far a region sits from its own
long-run level, so the honest comparison is against mean reversion: predict
a fraction of the gap between a region's current speed and its own average,
with that fraction fitted on the training years.

| Predictor | MAE | Average precision | Recall @10% | @15% | @20% |
|---|---|---|---|---|---|
| No change | 0.193 | 0.120 | 11.6% | 17.4% | 23.2% |
| Seasonal average | 0.206 | 0.143 | 12.1% | 18.4% | 27.4% |
| **Mean reversion, one parameter** | 0.188 | **0.347** | **34.7%** | 46.3% | 56.3% |
| Random forest | **0.167** | **0.385** | 33.2% | **50.0%** | **58.9%** |

One line of arithmetic reaches 0.347 average precision against 0.120 for a
random ranking. The forest adds 0.038 on top of that. Put another way, about
86% of the predictable structure in this problem is mean reversion.

That is not a reason to drop the model, but it is a reason to be precise
about what it buys. The forest improves MAE by 11% and finds 3.7 percentage
points more declining regions within a 15% inspection budget. Whether that
justifies the extra complexity depends on what an unnecessary inspection
costs, which public data cannot tell us.

### The forest earns its keep on the large moves

| Size of the actual change | MAE, mean reversion | MAE, forest | Forest closer |
|---|---|---|---|
| Smallest quarter | 0.035 | 0.075 | 31% |
| Small | 0.090 | 0.102 | 53% |
| Large | 0.176 | 0.161 | 63% |
| **Largest quarter** | **0.453** | **0.330** | **81%** |

On quiet quarters the simple rule is better and the forest over-reacts. On
the largest moves the forest is 27% closer and wins four times out of five.
The gain is concentrated rather than spread thinly, and it sits exactly
where maintenance planning needs it: the regions about to fall a long way.

### Regression beats classification where it matters

| | Average precision | Recall @10% | @15% | @20% | MAE |
|---|---|---|---|---|---|
| Classifier, probability of a >20% drop | 0.393 | 34.7% | 44.7% | 55.8% | – |
| Regressor, size of the change | 0.385 | 33.2% | **50.0%** | **58.9%** | 0.167 |

Ranking quality is about the same, but at the inspection budgets that would
be used in practice the regressor finds more. It also drops the arbitrary
20% threshold and keeps the magnitude, so a region can be reported as
"expected to fall 35%" rather than "0.6 probability of decline".

### Multi-quarter history helps; weather does not

| Features | Average precision | Recall @15% |
|---|---|---|
| Current quarter only | 0.370 | 46.3% |
| **+ multi-quarter history** | **0.393** | 44.7% |
| + weather | 0.362 | 44.2% |
| + both | 0.388 | **47.4%** |

Four of the seven most useful features are the ones describing how a region
has been moving rather than where it is: its position relative to its own
long-run level, and its change over the last one, two and three quarters. A
region at 50 Mbit/s that fell from 80 and one that climbed from 30 face very
different next quarters, and a snapshot cannot tell them apart.

### Weather adds nothing at this granularity

| | Recall | Precision | Flagged |
|---|---|---|---|
| Without weather | 0.485 | 0.367 | 16.5% |
| With weather | 0.456 | 0.377 | 15.1% |

No weather feature appears in the eight most useful features. Hot days,
heavy rain days and quarterly rainfall totals carry almost no signal about
whether a region's speed will fall next quarter.

This is a result rather than a failure. Weather is fetched per two-degree
cell, roughly 200 km, and summarised to a whole quarter, so a storm that
knocks out a tower for two days is averaged away long before the model sees
it. The proposal assumed weather would matter; at the granularity open data
allows, it does not.

### The daily check confirms both quarterly conclusions

A quarter is a long time to average over, so weather could plausibly have been
hidden by it. Measurement Lab publishes individual speed tests with timestamps,
which allows the same two questions to be asked a day at a time. 92 days of
Australian tests were collected, 83,626 tests in total.

Weather still explains nothing. Nothing beats simply predicting the mean:

| model | MAE |
| --- | --- |
| predicting the mean | 7.24 Mbit/s |
| sample size + weekday | 7.48 Mbit/s |
| the same + weather | 7.89 Mbit/s |

The strongest weather correlation is wind speed at +0.197, and its sign is
wrong: higher wind goes with faster speeds. This is noise.

Tomorrow's change is more predictable than next quarter's, but for the same
reason as before:

| model | MAE |
| --- | --- |
| predict no change | 0.140 |
| repeat today | 0.240 |
| mean reversion (k=-0.773) | **0.105** |
| random forest | 0.108 |

The forest loses to a single fitted parameter here, where at quarterly
resolution it won narrowly. With 51 training days it has nothing to learn from.
The fitted k of -0.773 is far stronger than the quarterly 0.07, which suggests
much of the daily movement is sampling noise rebounding rather than real change:
daily test counts range from 177 to 5,152.

This series is Australia-wide rather than per region, so it cannot produce a
maintenance ranking and does not replace the panel. Its value is that the
quarterly conclusions survive a change of data source, time granularity and
target.

### The model fails where it matters most

| Area type | Regions | Base rate | Recall | Precision | Flagged |
|---|---|---|---|---|---|
| Urban | 185 | 11.3% | 0.462 | 0.400 | 13.1% |
| Outer | 54 | 20.9% | 0.442 | 0.322 | 28.6% |
| **Regional** | **10** | **5.9%** | **0.000** | **0.000** | 17.6% |

Bands are set by each region's median number of speed tests, standing in for
population until ABS figures are joined.

The regional band contains ten regions and seventeen labelled rows, and the
model catches none of their declines. This is the equity problem the proposal
predicted, measured: speed tests are run by people who choose to run them, so
sparsely populated areas produce too little data to model, and those are the
areas with the weakest coverage to begin with. Any deployment would have to
flag these regions for human review rather than trust a prediction, and a
single average score would have hidden this entirely.

### Predicting the current quarter is not a task

A first version scored perfect accuracy, which was the bug announcing itself.
The label is derived from `pct_change`, and `pct_change` was also a feature,
so the model was reading the answer rather than predicting it.

The target is now the **following** quarter's decline, and performance fell to
the numbers above. This is worth recording because the proposal did not
distinguish between describing a decline that has already happened and
forecasting one that has not.

### Aggregation is required, not optional

At the raw 600 m tile level, very few tiles have data in **both** quarters,
because a tile only appears when someone runs a speed test there:

| Minimum tests per tile | Tiles matched across both quarters |
|---|---|
| 10 | 481 |
| 30 | 86 |

86 samples cannot train a model. Aggregating to region level raises this to
142 stable regions, each with at least 50 tests in both quarters. This changed
our plan from tile-level to region-level modelling.

## Status

- [x] Confirm data availability and licence
- [x] Measure the base rate of decline at several thresholds
- [x] Build the region-quarter panel for 2019 Q1 to 2026 Q1
- [ ] Replace the coarse grid with a spatial join on ABS SA2 boundaries
- [x] Add quarterly weather features (Open-Meteo, in place of BoM)
- [x] Baselines and a first random forest, validated by time
- [x] Report recall separately for urban, outer suburban and regional areas
- [x] Evaluate against an inspection budget rather than a fixed threshold
- [x] Test the model against a mean-reversion baseline
- [x] Daily-resolution check of the weather question using M-Lab
- [ ] Variable region size so sparse areas have enough data to model
- [ ] Produce the maintenance priority ranking

## Notes

Speed data reflects users who choose to run a test, so coverage is uneven.
Regions with few tests are flagged rather than ranked.
