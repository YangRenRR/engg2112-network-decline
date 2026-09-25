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
| Temperature, rainfall | Bureau of Meteorology, Climate Data Online | Daily, by station | Open |
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
- [ ] Add BoM weather features aggregated to quarters
- [ ] Baseline: assume next quarter equals this quarter
- [ ] Train and tune the Random Forest classifier
- [ ] Time-based validation; report recall by area type
- [ ] Produce the maintenance priority ranking

## Notes

Speed data reflects users who choose to run a test, so coverage is uneven.
Regions with few tests are flagged rather than ranked.
