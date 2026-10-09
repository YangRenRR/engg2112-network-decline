"""
Score every region for the next quarter and write the files the dashboard reads.

The model itself is the regressor from regress.py: it predicts the size of next
quarter's change rather than a yes/no label, because the ranking it produces is
what a maintenance team actually needs and it does not depend on where the 20%
threshold was drawn.

Two files come out:

  predictions.csv   one row per region, ranked worst first
  history.csv       the download speed of every region in every quarter

Usage:
    python src/export_predictions.py --panel data/panel.csv --out ui
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from model import (FEATURES, LAG_FEATURES, add_area_type, add_history,
                   add_lags, add_target)
from regress import add_regression_target


def main(panel_path, out_dir, trees, seed):
    df = pd.read_csv(panel_path)
    df = add_area_type(add_lags(add_history(add_target(df))))
    df = add_regression_target(df)

    feats = FEATURES + LAG_FEATURES
    ready = df.dropna(subset=feats)

    # train on every region-quarter whose outcome is already known
    train = ready.dropna(subset=["target_change"])
    model = RandomForestRegressor(n_estimators=trees, min_samples_leaf=5,
                                  random_state=seed, n_jobs=-1)
    model.fit(train[feats], train.target_change)

    # score the latest quarter, the one whose outcome is still unknown
    latest = ready.t.max()
    now = ready[ready.t == latest].copy()
    now["predicted_change"] = model.predict(now[feats])
    now = now.sort_values("predicted_change")       # most negative first
    now["rank"] = range(1, len(now) + 1)

    cols = ["rank", "region", "lon", "lat", "year", "quarter", "down_mbps",
            "lat_ms", "tests", "n_tiles", "area_type", "pct_change",
            "predicted_change"]
    os.makedirs(out_dir, exist_ok=True)
    now[cols].round(4).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    hist = df[["region", "year", "quarter", "t", "down_mbps", "tests"]]
    hist.round(3).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    q = f"{int(now.year.iloc[0])} Q{int(now.quarter.iloc[0])}"
    print(f"trained on {len(train):,} region-quarters")
    print(f"scored {len(now):,} regions for the quarter after {q}")
    print(f"predicted change: median {now.predicted_change.median():+.3f}, "
          f"worst {now.predicted_change.min():+.3f}")
    print(f"\nwrote {out_dir}/predictions.csv and {out_dir}/history.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/panel.csv")
    p.add_argument("--out", default="ui")
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.panel, a.out, a.trees, a.seed)
