"""
Where does the forest beat the one-parameter mean-reversion rule?

If the gain is concentrated in a recognisable kind of region, the extra
complexity has a place to point at. If it is spread evenly, the forest may
just be fitting noise.

Usage:
    python src/where_rf_wins.py --panel data/panel.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from model import (FEATURES, LAG_FEATURES, add_area_type, add_history,
                   add_lags, add_target)
from regress import add_regression_target


def main(panel_path, split_year, trees, seed):
    df = pd.read_csv(panel_path)
    df = add_regression_target(add_area_type(add_lags(add_history(add_target(df)))))
    feats = FEATURES + LAG_FEATURES
    df = df.dropna(subset=feats + ["target", "target_change"])
    train, test = df[df.year < split_year].copy(), df[df.year >= split_year].copy()

    k = -np.linalg.lstsq(train.speed_vs_own_mean.values.reshape(-1, 1),
                         train.target_change.values, rcond=None)[0][0]
    reg = RandomForestRegressor(n_estimators=trees, min_samples_leaf=5,
                                random_state=seed, n_jobs=-1).fit(train[feats],
                                                                  train.target_change)
    test["err_mr"] = (test.target_change + k * test.speed_vs_own_mean).abs()
    test["err_rf"] = (test.target_change - reg.predict(test[feats])).abs()
    test["gain"] = test.err_mr - test.err_rf          # positive: forest is closer

    print(f"forest wins on {100 * (test.gain > 0).mean():.1f}% of rows, "
          f"mean gain in MAE {test.gain.mean():+.3f}\n")

    def table(by, label):
        g = test.groupby(by, observed=True).agg(
            rows=("gain", "size"),
            mae_mean_reversion=("err_mr", "mean"),
            mae_forest=("err_rf", "mean"),
            forest_better_pct=("gain", lambda s: 100 * (s > 0).mean()))
        print(label)
        print(g.round(3).to_string(), "\n")

    table("area_type", "by area type")

    test["tests_band"] = pd.qcut(test.tests, 4,
                                 labels=["fewest tests", "few", "many", "most tests"])
    table("tests_band", "by how much the region is used")

    test["volatility"] = pd.qcut(test.region_std, 4,
                                 labels=["steadiest", "steady", "variable", "most variable"])
    table("volatility", "by how much the region's speed normally moves")

    test["size_of_move"] = pd.qcut(test.target_change.abs(), 4,
                                   labels=["smallest moves", "small", "large", "largest moves"])
    table("size_of_move", "by how big the actual change turned out to be")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/panel.csv")
    p.add_argument("--split", type=int, default=2024)
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.panel, a.split, a.trees, a.seed)
