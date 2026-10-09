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

WEATHER_FEATURES = [
    "max_temp", "mean_max_temp", "mean_min_temp", "hot_days",
    "total_rain_mm", "wet_days", "max_daily_rain_mm",
]

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


def add_weather(df: pd.DataFrame, weather_path) -> pd.DataFrame:
    """Join quarterly weather on the coarse weather cell each region sits in."""
    if weather_path is None:
        return df
    w = pd.read_csv(weather_path)
    df = df.copy()
    df["wlat"] = (df.lat / 2.0).round() * 2.0
    df["wlon"] = (df.lon / 2.0).round() * 2.0
    return df.merge(w, on=["wlat", "wlon", "year", "quarter"], how="left")


def add_area_type(df: pd.DataFrame) -> pd.DataFrame:
    """Split regions into three bands by how much they are used.

    The proposal promised results broken down by urban, outer suburban and
    regional areas. Until ABS population is joined, a region's median number
    of speed tests stands in for how built-up it is: tests track both
    population and how much of the area is covered.
    """
    df = df.copy()
    usage = df.groupby("region").tests.median()
    bands = pd.qcut(usage, 3, labels=["regional", "outer", "urban"])
    df["area_type"] = df.region.map(bands)
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


def main(panel_path, split_year, n_estimators, threshold, seed, weather_path):
    df = pd.read_csv(panel_path)
    df = add_area_type(add_history(add_target(df)))
    features = list(FEATURES)
    if weather_path:
        df = add_weather(df, weather_path)
        features += WEATHER_FEATURES
    df = df.dropna(subset=features + ["target"])

    train = df[df.year < split_year]
    test = df[df.year >= split_year]

    X_train, y_train = train[features], train.target.astype(int)
    X_test, y_test = test[features], test.target.astype(int)

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

    test = test.assign(pred=(proba >= threshold).astype(int),
                       truth=y_test.values)
    by_area = (test.groupby("area_type", observed=True)
               .apply(lambda g: pd.Series({
                   "regions": g.region.nunique(),
                   "rows": len(g),
                   "base_rate": g.truth.mean(),
                   "recall": recall_score(g.truth, g.pred, zero_division=0),
                   "precision": precision_score(g.truth, g.pred, zero_division=0),
                   "flagged_pct": 100 * g.pred.mean(),
               }), include_groups=False)
               .reindex(["urban", "outer", "regional"]))
    print("\nby area type (bands set by median tests per region)")
    print(by_area.round(3).to_string())

    imp = (pd.Series(clf.feature_importances_, index=features)
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
    p.add_argument("--weather", default=None,
                   help="path to weather.csv; omit to train without weather")
    a = p.parse_args()
    main(a.panel, a.split, a.trees, a.threshold, a.seed, a.weather)
