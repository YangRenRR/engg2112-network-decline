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
```

## Findings so far

### The 20% threshold is workable

Aggregating NSW tiles into ~11 km cells (a stand-in for SA2), for
2025 Q4 -> 2026 Q1:

| Regions | Fell >10% | Fell >20% | Fell >30% | Median change |
|---|---|---|---|---|
| 142 | 35.9% | **19.7%** | 9.9% | -2.3% |

About one region in five is labelled a significant decline. That is frequent
enough to train on and rare enough to be worth flagging.

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
- [ ] Replace the coarse grid with a spatial join on ABS SA2 boundaries
- [ ] Add BoM weather features aggregated to quarters
- [ ] Baseline: assume next quarter equals this quarter
- [ ] Train and tune the Random Forest classifier
- [ ] Time-based validation; report recall by area type
- [ ] Produce the maintenance priority ranking

## Notes

Speed data reflects users who choose to run a test, so coverage is uneven.
Regions with few tests are flagged rather than ranked.
