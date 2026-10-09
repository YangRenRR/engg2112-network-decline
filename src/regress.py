"""
Predict the size of next quarter's change, not just whether it crosses 20%.

The classifier throws away how far a region falls: a 21% drop and a 60% drop
are the same label, and a 19% drop counts as no decline at all. A regressor
keeps the magnitude, and the ranking it produces does not depend on where the
threshold was set.

Both are judged the same way, on the ranking they produce, so they can be
compared directly:

  average precision   against the 20% decline label
  recall at budget    share of declining regions inside the top K% ranked
  MAE                 how far the predicted change is from the real one

Usage:
    python src/regress.py --panel data/panel.csv --weather data/weather.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error

from compare import BUDGETS, recall_at_budget
from model import (FEATURES, LAG_FEATURES, WEATHER_FEATURES, add_area_type,
                   add_history, add_lags, add_target, add_weather)


def add_regression_target(df):
    """Next quarter's proportional change in median download speed."""
    df = df.sort_values(["region", "t"]).copy()
    nxt = df.groupby("region")[["pct_change", "t"]].shift(-1)
    df["target_change"] = nxt["pct_change"].where(nxt["t"] - df["t"] == 1)
    return df


def main(panel_path, weather_path, split_year, trees, seed, threshold):
    df = pd.read_csv(panel_path)
    df = add_area_type(add_lags(add_history(add_target(df))))
    df = add_weather(df, weather_path)
    df = add_regression_target(df)

    feats = FEATURES + LAG_FEATURES
    df = df.dropna(subset=feats + ["target", "target_change"])
    train, test = df[df.year < split_year], df[df.year >= split_year]
    y_cls = test.target.astype(int)
    y_chg = test.target_change

    print(f"{len(train):,} train rows, {len(test):,} test rows, "
          f"base rate {y_cls.mean():.3f}\n")

    rows = {}

    clf = RandomForestClassifier(n_estimators=trees, class_weight="balanced",
                                 min_samples_leaf=5, random_state=seed, n_jobs=-1)
    clf.fit(train[feats], train.target.astype(int))
    p_cls = clf.predict_proba(test[feats])[:, 1]
    rows["classifier (p of >20% drop)"] = {
        "average_precision": average_precision_score(y_cls, p_cls),
        **{f"recall@{int(b*100)}%": recall_at_budget(y_cls, p_cls, b) for b in BUDGETS},
        "MAE on change": np.nan,
    }

    reg = RandomForestRegressor(n_estimators=trees, min_samples_leaf=5,
                                random_state=seed, n_jobs=-1)
    reg.fit(train[feats], train.target_change)
    pred_chg = reg.predict(test[feats])
    score_reg = -pred_chg                      # a bigger drop ranks higher
    rows["regressor (size of change)"] = {
        "average_precision": average_precision_score(y_cls, score_reg),
        **{f"recall@{int(b*100)}%": recall_at_budget(y_cls, score_reg, b) for b in BUDGETS},
        "MAE on change": mean_absolute_error(y_chg, pred_chg),
    }

    print(pd.DataFrame(rows).T.round(3).to_string())

    print(f"\nactual change next quarter: median {y_chg.median():+.3f}, "
          f"mean {y_chg.mean():+.3f}, sd {y_chg.std():.3f}")
    print(f"predicting no change at all would give MAE "
          f"{mean_absolute_error(y_chg, np.zeros(len(y_chg))):.3f}")
    print(f"predicting the training mean would give MAE "
          f"{mean_absolute_error(y_chg, np.full(len(y_chg), train.target_change.mean())):.3f}")

    imp = pd.Series(reg.feature_importances_, index=feats)
    print("\nten most useful features for the regressor")
    print(imp.sort_values(ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/panel.csv")
    p.add_argument("--weather", default="data/weather.csv")
    p.add_argument("--split", type=int, default=2024)
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.20)
    a = p.parse_args()
    main(a.panel, a.weather, a.split, a.trees, a.seed, a.threshold)
