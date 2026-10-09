"""
Compare feature sets on equal terms.

Two changes from model.py's reporting. Every configuration is trained and
tested on the same rows, so a feature set is not rewarded or punished for
how much history it requires. And the metrics suit what we actually produce,
which is a ranking rather than a yes/no call:

  average precision   quality of the whole ranking, no threshold involved
  recall at budget    of the regions that decline, how many are inside the
                      top K% we could afford to inspect

Usage:
    python src/compare.py --panel data/panel.csv --weather data/weather.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from model import (FEATURES, LAG_FEATURES, WEATHER_FEATURES, add_area_type,
                   add_history, add_lags, add_target, add_weather)

BUDGETS = (0.10, 0.15, 0.20)


def recall_at_budget(y_true, score, budget):
    k = max(1, int(round(budget * len(score))))
    top = np.argsort(-score)[:k]
    return y_true.values[top].sum() / max(1, y_true.sum())


def run(df, features, split_year, trees, seed):
    train, test = df[df.year < split_year], df[df.year >= split_year]
    y_train, y_test = train.target.astype(int), test.target.astype(int)
    clf = RandomForestClassifier(n_estimators=trees, class_weight="balanced",
                                 min_samples_leaf=5, random_state=seed, n_jobs=-1)
    clf.fit(train[features], y_train)
    score = clf.predict_proba(test[features])[:, 1]
    row = {"average_precision": average_precision_score(y_test, score)}
    for b in BUDGETS:
        row[f"recall@{int(b*100)}%"] = recall_at_budget(y_test, score, b)
    return row, clf


def main(panel_path, weather_path, split_year, trees, seed):
    df = pd.read_csv(panel_path)
    df = add_area_type(add_lags(add_history(add_target(df))))
    df = add_weather(df, weather_path)

    sets = {
        "current quarter only": FEATURES,
        "+ multi-quarter history": FEATURES + LAG_FEATURES,
        "+ weather": FEATURES + WEATHER_FEATURES,
        "+ both": FEATURES + LAG_FEATURES + WEATHER_FEATURES,
    }
    everything = sorted({f for fs in sets.values() for f in fs})
    df = df.dropna(subset=everything + ["target"])      # same rows for all

    test_rows = (df.year >= split_year).sum()
    base = df[df.year >= split_year].target.mean()
    print(f"{len(df):,} rows, {test_rows:,} in test, base rate {base:.3f}")
    print(f"a random ranking scores {base:.3f} average precision\n")

    out = {}
    for name, feats in sets.items():
        out[name], clf = run(df, feats, split_year, trees, seed)
        if name == "+ both":
            imp = pd.Series(clf.feature_importances_, index=feats)
    print(pd.DataFrame(out).T.round(3).to_string())

    print("\nten most useful features, all features in play")
    print(imp.sort_values(ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/panel.csv")
    p.add_argument("--weather", default="data/weather.csv")
    p.add_argument("--split", type=int, default=2024)
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.panel, a.weather, a.split, a.trees, a.seed)
