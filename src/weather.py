"""
Quarterly weather features for each region in the panel.

The proposal named the Bureau of Meteorology as the weather source, but BoM's
Climate Data Online blocks automated requests and serves data per station
rather than per location, which would need a nearest-station join for every
region. We use the Open-Meteo historical archive instead: it is free, needs no
key, and serves daily values for any coordinate from the ERA5 reanalysis,
which is itself assimilated from station observations.

Weather varies far more slowly across space than mobile coverage does, so
values are fetched once per coarse weather cell (two degrees, about 200 km)
and shared by every region inside it, rather than once for each of the 768
regions. That keeps the job inside the free tier's request budget.

Usage:
    python src/weather.py --panel data/panel.csv --out data/weather.csv
"""
import argparse
import time

import pandas as pd
import requests

API = "https://archive-api.open-meteo.com/v1/archive"
HOT_DAY_C = 35.0        # days at or above this count as hot
WET_DAY_MM = 25.0       # days at or above this count as heavy rain

CELL = 2.0              # degrees; one weather cell is about 200 km


BATCH = 1                # one coordinate per request; batching several
                         # coordinates over seven years exceeds the free tier
CHUNK_YEARS = 2          # years per request; the API times out streaming
                         # eight years of daily data in one response


def fetch_cells(coords, start, end, retries=4):
    """Fetch daily weather for several coordinates in one request.

    Returns a list of dataframes in the same order as `coords`.
    """
    params = {
        "latitude": ",".join(f"{lat:.2f}" for lat, _ in coords),
        "longitude": ",".join(f"{lon:.2f}" for _, lon in coords),
        "start_date": start, "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Australia/Sydney",
    }
    for attempt in range(retries):
        r = requests.get(API, params=params, timeout=120)
        if r.ok and r.text.lstrip().startswith(("{", "[")):
            payload = r.json()
            if isinstance(payload, dict):      # a single coordinate
                payload = [payload]
            return [pd.DataFrame(p["daily"]) for p in payload]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"open-meteo failed: {r.status_code} {r.text[:120]}")


def fetch_cell_series(lat, lon, start, end):
    """Fetch one coordinate in multi-year chunks and stitch the days together."""
    y0, y1 = int(start[:4]), int(end[:4])
    parts = []
    for ys in range(y0, y1 + 1, CHUNK_YEARS):
        ye = min(ys + CHUNK_YEARS - 1, y1)
        s = max(start, f"{ys}-01-01")
        e = min(end, f"{ye}-12-31")
        parts.append(fetch_cells([(lat, lon)], s, e)[0])
        time.sleep(1.5)
    return pd.concat(parts, ignore_index=True)


def to_quarters(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["date"] = pd.to_datetime(d.time)
    d["year"] = d.date.dt.year
    d["quarter"] = d.date.dt.quarter
    out = d.groupby(["year", "quarter"]).agg(
        max_temp=("temperature_2m_max", "max"),
        mean_max_temp=("temperature_2m_max", "mean"),
        mean_min_temp=("temperature_2m_min", "mean"),
        hot_days=("temperature_2m_max", lambda s: int((s >= HOT_DAY_C).sum())),
        total_rain_mm=("precipitation_sum", "sum"),
        wet_days=("precipitation_sum", lambda s: int((s >= WET_DAY_MM).sum())),
        max_daily_rain_mm=("precipitation_sum", "max"),
    ).reset_index()
    return out


def build(panel_path, verbose=True):
    panel = pd.read_csv(panel_path)
    panel["wlat"] = (panel.lat / CELL).round() * CELL
    panel["wlon"] = (panel.lon / CELL).round() * CELL
    cells = panel[["wlat", "wlon"]].drop_duplicates().sort_values(["wlat", "wlon"])

    start = f"{panel.year.min()}-01-01"
    end = f"{panel.year.max()}-12-31"
    end = min(end, (pd.Timestamp.today() - pd.Timedelta(days=6)).strftime("%Y-%m-%d"))

    coords = list(cells.itertuples(index=False, name=None))
    frames = []
    for i in range(0, len(coords), BATCH):
        batch = coords[i:i + BATCH]
        for (lat, lon), daily in zip(batch, fetch_cells(batch, start, end)):
            q = to_quarters(daily)
            q["wlat"], q["wlon"] = lat, lon
            frames.append(q)
        if verbose:
            print(f"  {min(i + BATCH, len(coords))}/{len(coords)} cells")
        time.sleep(3)            # be polite to a free service
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/panel.csv")
    p.add_argument("--out", default="data/weather.csv")
    a = p.parse_args()

    w = build(a.panel)
    w.to_csv(a.out, index=False)
    print(f"\n{len(w):,} rows, {w.groupby(['wlat','wlon']).ngroups} weather cells, "
          f"{w.year.min()}-{w.year.max()}")
    print("written:", a.out)
