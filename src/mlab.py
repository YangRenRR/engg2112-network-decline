"""
Measurement Lab NDT as a second, finer-grained view of network performance.

Ookla publishes quarterly aggregates, which is why the weather features in
our model carry no signal: a storm that takes a tower down for two days is
averaged away long before the model sees it. M-Lab publishes every individual
speed test with a timestamp, so the same question can be asked at daily
resolution.

The archives sit in a public Google Cloud Storage bucket and need no
credentials:

    gs://archive-measurement-lab/ndt/ndt7/YYYY/MM/DD/

Each file is half an hour of tests from one measurement server. Australian
servers are named syd0x and mel0x, so filtering on the filename keeps the
download to Australian traffic without needing the separate geolocation
dataset.

Throughput is not stored directly; it is computed the way NDT does it, from
the bytes acknowledged and the elapsed time of the last server measurement.

This module is written to be reusable: `daily_stats` produces a daily series
that can serve as a feature source or as an independent baseline for the
Ookla-based model, not only for the weather experiment it was built for.

Usage:
    python src/mlab.py --start 2026-06-01 --end 2026-08-31 --out data/mlab_daily.csv
"""
import argparse
import datetime as dt
import gzip
import io
import json
import tarfile
import urllib.parse
import urllib.request

import pandas as pd

BUCKET = "archive-measurement-lab"
LIST_API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"
AU_SITES = ("syd", "mel")


def list_archives(day: dt.date, sites=AU_SITES, timeout=60):
    """Names of the half-hour archives published for one day, Australian sites only."""
    prefix = f"ndt/ndt7/{day:%Y/%m/%d}/"
    names, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": 1000, "fields": "items/name,nextPageToken"}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(f"{LIST_API}?{urllib.parse.urlencode(q)}",
                                    timeout=timeout) as r:
            page = json.load(r)
        names += [i["name"] for i in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            break
    return [n for n in names if any(s in n.rsplit("/", 1)[-1] for s in sites)]


def throughput_mbps(measurement) -> float | None:
    """Download rate from the last server measurement, as NDT computes it."""
    tcp = measurement.get("TCPInfo") or {}
    acked, elapsed = tcp.get("BytesAcked"), tcp.get("ElapsedTime")
    if not acked or not elapsed:
        return None
    return acked * 8 / elapsed          # bytes/µs * 8 == Mbit/s


def read_archive(name: str, timeout=180):
    """Every download test in one archive, as (start time, Mbit/s) pairs."""
    with urllib.request.urlopen(MEDIA + name, timeout=timeout) as r:
        blob = r.read()
    out = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile() or "download" not in member.name:
                continue
            try:
                test = json.load(gzip.open(tar.extractfile(member)))
                sm = test["Download"]["ServerMeasurements"]
                mbps = throughput_mbps(sm[-1])
            except Exception:
                continue
            if mbps:
                out.append((test["StartTime"], mbps))
    return out


def daily_stats(start: dt.date, end: dt.date, sites=AU_SITES,
                archives_per_day=2, verbose=True):
    """Daily download statistics, sampling a few archives per day.

    A sample is enough for a daily median and keeps the download to a few
    hundred megabytes instead of several gigabytes per quarter.
    """
    rows = []
    day = start
    while day <= end:
        try:
            names = list_archives(day, sites)
        except Exception as exc:
            if verbose:
                print(f"  {day}: listing failed ({type(exc).__name__})")
            day += dt.timedelta(days=1)
            continue
        if names:
            step = max(1, len(names) // archives_per_day)
            chosen = names[::step][:archives_per_day]
            tests = [t for n in chosen for t in read_archive(n)]
            if tests:
                s = pd.Series([m for _, m in tests])
                rows.append({"date": day, "n_tests": len(s),
                             "median_mbps": s.median(),
                             "p25_mbps": s.quantile(.25),
                             "p75_mbps": s.quantile(.75),
                             "mean_mbps": s.mean(),
                             "archives_sampled": len(chosen)})
                if verbose:
                    print(f"  {day}: {len(s):>5} tests, "
                          f"median {s.median():6.1f} Mbit/s")
        day += dt.timedelta(days=1)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--per-day", type=int, default=2)
    p.add_argument("--out", default="data/mlab_daily.csv")
    a = p.parse_args()

    df = daily_stats(dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end),
                     archives_per_day=a.per_day)
    df.to_csv(a.out, index=False)
    print(f"\n{len(df)} days, {df.n_tests.sum():,} tests")
    print("written:", a.out)
