"""
Repeat two of the quarterly findings at daily resolution, on a different source.

A quarter is a long time to average over. If weather affects mobile speeds at
all, a quarterly mean could easily hide it, and the same goes for any short
disruption. Measurement Lab publishes individual speed tests with timestamps,
so the same two questions can be asked a day at a time:

  weather     does rainfall, heat or wind explain day-to-day speed?
  forecast    can tomorrow's change be predicted, and by what?

The series is Australia-wide rather than per region, so it cannot produce a
maintenance ranking. It is a check on whether the quarterly conclusions hold
at a different granularity, not a replacement for the panel.

Usage:
    python src/daily.py --daily data/mlab_daily.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

SYDNEY = (-33.87, 151.21)      # most Australian NDT tests land on the syd0x servers
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
WEATHER = ["temperature_2m_max", "temperature_2m_min",
           "precipitation_sum", "wind_speed_10m_max"]
FEATURES = ["pct_change", "lag1", "lag2", "lag3",
            "vs_own_mean", "roll7", "log_tests", "dow"]


def fetch_weather(start, end):
    lat, lon = SYDNEY
    r = requests.get(ARCHIVE, timeout=180, params={
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ",".join(WEATHER), "timezone": "Australia/Sydney"})
    r.raise_for_status()
    w = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
    w["date"] = pd.to_datetime(w["date"])
    return w


def weather_question(d):
    """Does weather explain speed once sample size and weekday are allowed for?"""
    start, end = d.date.min().date(), d.date.max().date()
    m = d.merge(fetch_weather(str(start), str(end)), on="date")
    print(f"=== weather, {len(m)} days ===\n")

    print("correlation with median speed:")
    for c in WEATHER + ["log_tests", "dow"]:
        print(f"  {c:22s} {m.median_mbps.corr(m[c]):+.3f}")

    # the honest comparison: weather has to beat sample size and weekday alone
    y = m.median_mbps
    for name, cols in [("sample size + weekday", ["log_tests", "dow"]),
                       ("    the same + weather", ["log_tests", "dow"] + WEATHER)]:
        mae = -cross_val_score(LinearRegression(), m[cols], y, cv=5,
                               scoring="neg_mean_absolute_error").mean()
        print(f"\n  {name}  MAE {mae:.2f} Mbit/s")
    print(f"  {'    predicting the mean':22s}  MAE {abs(y - y.mean()).mean():.2f} Mbit/s")


def forecast_question(d, trees, seed):
    """Can tomorrow's change be predicted, and does a forest beat mean reversion?"""
    m = d.dropna(subset=FEATURES + ["target_change"])
    cut = int(len(m) * 0.6)
    tr, te = m.iloc[:cut], m.iloc[cut:]
    y = te.target_change
    print(f"\n=== forecast, {len(tr)} train days, {len(te)} test days ===\n")

    preds = {"predict no change": np.zeros(len(te)),
             "repeat today": te["pct_change"].values}
    k = np.linalg.lstsq(tr[["vs_own_mean"]].values,
                        tr.target_change.values, rcond=None)[0][0]
    preds[f"mean reversion (k={k:+.3f})"] = te.vs_own_mean.values * k

    rf = RandomForestRegressor(n_estimators=trees, min_samples_leaf=5,
                               random_state=seed, n_jobs=-1)
    rf.fit(tr[FEATURES], tr.target_change)
    preds["random forest"] = rf.predict(te[FEATURES])

    for name, p in preds.items():
        print(f"  {name:28s} MAE {mean_absolute_error(y, p):.4f}")

    print("\n  importances:")
    for f, i in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"    {f:14s} {i:.3f}")


def add_features(d):
    d = d.sort_values("date").copy()
    s = d.median_mbps
    d["pct_change"] = s.pct_change()
    d["target_change"] = d["pct_change"].shift(-1)
    for k in (1, 2, 3):
        d[f"lag{k}"] = d["pct_change"].shift(k)
    d["vs_own_mean"] = s / s.expanding().mean() - 1   # the strongest quarterly feature
    d["roll7"] = s.rolling(7).mean() / s - 1
    d["log_tests"] = np.log(d.n_tests)
    d["dow"] = d.date.dt.dayofweek
    return d


def main(daily_path, trees, seed):
    d = add_features(pd.read_csv(daily_path, parse_dates=["date"]))
    weather_question(d)
    forecast_question(d, trees, seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--daily", default="data/mlab_daily.csv")
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.daily, a.trees, a.seed)
