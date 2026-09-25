"""
Train and evaluate a classifier for significant quarterly speed decline.

The task is to predict NEXT quarter's decline from what is known at the end
of THIS quarter. The panel's own `declined` column describes the move into
the current quarter, so it is shifted forward by one quarter to become the
target. Using it unshifted would hand the model the answer, since the label
is derived from `pct_change`, which is one of the features.

The split is by time, not at random: the model learns from earlier quarters
and is tested on later ones. A random split would let a region's own future
leak into training through its neighbouring quarters.

Two baselines are reported alongside the model, because a classifier is only
worth using if it beats the obvious guesses:

  never    always predict no decline, i.e. assume next quarter looks like this
  always   always predict decline, which finds everything at the cost of
           sending a crew to every region

Usage:
    python src/model.py --panel data/panel.csv --split 2024
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score)

FEATURES = [
    "down_mbps",        # speed this quarter
    "up_mbps",
    "lat_ms",
    "prev_down_mbps",   # speed last quarter
    "pct_change",       # how it moved into this quarter
    "tests",            # how much evidence we have
    "prev_tests",
    "n_tiles",          # how built-up the region is, roughly
    "lon", "lat",
    "quarter",          # seasonality
    "region_mean",      # the region's own long-run level
    "region_std",
]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target is whether this region declines in the FOLLOWING quarter.

    Only set where the next quarter is actually the next one: a gap means
    nobody ran a test there, which is not the same as no decline.
    """
    df = df.sort_values(["region", "t"]).copy()
    nxt = df.groupby("region")[["declined", "t"]].shift(-1)
    df["target"] = nxt.declined.where(nxt.t - df.t == 1)
    return df


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add each region's long-run level and volatility, computed only from
    quarters before the row in question so nothing leaks backwards."""
    df = df.sort_values(["region", "t"]).copy()
    g = df.groupby("region")["down_mbps"]
    df["region_mean"] = g.transform(lambda s: s.shift(1).expanding().mean())
    df["region_std"] = g.transform(lambda s: s.shift(1).expanding().std())
    return df


def evaluate(name, y_true, y_pred):
    return {
        "model": name,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "flagged": int(y_pred.sum()),
        "flagged_pct": 100 * y_pred.mean(),
    }


def main(panel_path, split_year, n_estimators, threshold, seed):
    df = pd.read_csv(panel_path)
    df = add_history(add_target(df))
    df = df.dropna(subset=FEATURES + ["target"])

    train = df[df.year < split_year]
    test = df[df.year >= split_year]

    X_train, y_train = train[FEATURES], train.target.astype(int)
    X_test, y_test = test[FEATURES], test.target.astype(int)

    print(f"train: {len(train):,} rows, {train.year.min()}-{train.year.max()}, "
          f"{100 * y_train.mean():.1f}% decline next quarter")
    print(f"test:  {len(test):,} rows, {test.year.min()}-{test.year.max()}, "
          f"{100 * y_test.mean():.1f}% declined\n")

    rows = [
        evaluate("baseline: never", y_test, np.zeros(len(y_test), dtype=int)),
        evaluate("baseline: always", y_test, np.ones(len(y_test), dtype=int)),
    ]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",      # declines are the minority class
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    rows.append(evaluate(f"random forest (p>{threshold})",
                         y_test, (proba >= threshold).astype(int)))

    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nconfusion matrix, random forest (rows: actual, cols: predicted)")
    print(pd.DataFrame(
        confusion_matrix(y_test, (proba >= threshold).astype(int)),
        index=["no decline", "declined"],
        columns=["predicted no", "predicted yes"]).to_string())

    print("\n" + classification_report(y_test, (proba >= threshold).astype(int),
                                       target_names=["no decline", "declined"],
                                       zero_division=0))

    imp = (pd.Series(clf.feature_importances_, index=FEATURES)
           .sort_values(ascending=False).head(8))
    print("most useful features")
    print(imp.round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/panel.csv")
    p.add_argument("--split", type=int, default=2024,
                   help="first year of the test set")
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="probability above which a region is flagged")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.panel, a.split, a.trees, a.threshold, a.seed)
