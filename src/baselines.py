"""
Test the model against a baseline that is actually hard to beat.

The first baseline we used, assuming next quarter looks like this one, is
weak: it predicts no change at all. The random forest's most useful feature
by a wide margin is how far a region sits from its own long-run level, which
suggests the model may mostly be learning mean reversion: regions currently
above their own average tend to fall back, and regions below it tend to
recover.

If one line of arithmetic captures most of that, the forest is adding little
and we should say so. This compares four predictors of next quarter's change:

  no change          predict zero
  seasonal           predict this region's average change for this quarter
  mean reversion     predict a fraction of the gap to the region's own mean,
                     with the fraction fitted on the training years only
  random forest      the model

Usage:
    python src/baselines.py --panel data/panel.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error

from compare import BUDGETS, recall_at_budget
from model import (FEATURES, LAG_FEATURES, add_area_type, add_history,
                   add_lags, add_target)
from regress import add_regression_target


def score_row(name, y_cls, y_chg, pred_change):
    rank = -np.asarray(pred_change)            # a bigger predicted drop ranks higher
    row = {"MAE": mean_absolute_error(y_chg, pred_change),
           "average_precision": average_precision_score(y_cls, rank)}
    for b in BUDGETS:
        row[f"recall@{int(b*100)}%"] = recall_at_budget(y_cls, rank, b)
    return name, row


def main(panel_path, split_year, trees, seed):
    df = pd.read_csv(panel_path)
    df = add_regression_target(add_area_type(add_lags(add_history(add_target(df)))))
    feats = FEATURES + LAG_FEATURES
    df = df.dropna(subset=feats + ["target", "target_change"])

    train, test = df[df.year < split_year], df[df.year >= split_year]
    y_cls, y_chg = test.target.astype(int), test.target_change
    print(f"{len(train):,} train rows, {len(test):,} test rows, "
          f"base rate {y_cls.mean():.3f}\n")

    rows = {}
    rows.update(dict([score_row("no change", y_cls, y_chg, np.zeros(len(test)))]))

    seasonal = train.groupby("quarter").target_change.mean()
    rows.update(dict([score_row("seasonal average", y_cls, y_chg,
                                test.quarter.map(seasonal).values)]))

    # mean reversion: next change is a fraction of the gap to the region's own mean
    gap_train = train.speed_vs_own_mean.values
    k = -np.linalg.lstsq(gap_train.reshape(-1, 1),
                         train.target_change.values, rcond=None)[0][0]
    rows.update(dict([score_row(f"mean reversion (k={k:.2f})", y_cls, y_chg,
                                -k * test.speed_vs_own_mean.values)]))

    reg = RandomForestRegressor(n_estimators=trees, min_samples_leaf=5,
                                random_state=seed, n_jobs=-1)
    reg.fit(train[feats], train.target_change)
    rows.update(dict([score_row("random forest", y_cls, y_chg,
                                reg.predict(test[feats]))]))

    out = pd.DataFrame(rows).T
    print(out.round(3).to_string())

    mr = out.loc[[i for i in out.index if i.startswith("mean reversion")][0]]
    rf = out.loc["random forest"]
    print(f"\nforest vs mean reversion: MAE {rf.MAE - mr.MAE:+.3f}, "
          f"average precision {rf.average_precision - mr.average_precision:+.3f}, "
          f"recall@15% {rf['recall@15%'] - mr['recall@15%']:+.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/panel.csv")
    p.add_argument("--split", type=int, default=2024)
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.panel, a.split, a.trees, a.seed)
