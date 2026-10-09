"""
Build the modelling table: one row per region per quarter.

Each quarter of Ookla data is aggregated into regions, then quarters are
stacked and joined to the quarter before to produce the change in speed
and the label we are trying to predict.

Regions are quadkey prefixes (zoom 12, roughly 8 km across) as a stand-in
for ABS SA2 areas until the boundary files are wired in.

Usage:
    python src/build_panel.py --start 2019 1 --end 2026 1 --out data/panel.csv
"""
import argparse

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import duckdb
import pandas as pd

from fetch_ookla import NSW, url_for
from quadkey import bbox_prefixes, quadkey_to_lonlat

REGION_ZOOM = 12      # ~8 km cells
FILTER_ZOOM = 8       # coarse prefixes used to cut the file down to NSW


def quarters(start, end):
    y, q = start
    while (y, q) <= end:
        yield y, q
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)


def load_quarter(con, year, quarter, prefixes, min_tests, region_zoom=REGION_ZOOM):
    """Aggregate one quarter of tiles into regions."""
    where = " OR ".join(f"quadkey LIKE '{p}%'" for p in prefixes)
    sql = f"""
        SELECT substr(quadkey, 1, {region_zoom})              AS region,
               sum(avg_d_kbps * tests) / sum(tests) / 1000.0  AS down_mbps,
               sum(avg_u_kbps * tests) / sum(tests) / 1000.0  AS up_mbps,
               sum(avg_lat_ms * tests) / sum(tests)           AS lat_ms,
               sum(tests)                                     AS tests,
               sum(devices)                                   AS devices,
               count(*)                                       AS n_tiles
        FROM read_parquet('{url_for(year, quarter)}')
        WHERE {where}
        GROUP BY 1
        HAVING sum(tests) >= {min_tests}
    """
    df = con.execute(sql).df()
    df["year"], df["quarter"] = year, quarter
    return df


def build(start=(2019, 1), end=(2026, 1), bbox=NSW, min_tests=50,
          threshold=0.20, verbose=True, region_zoom=REGION_ZOOM):
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    prefixes = bbox_prefixes(bbox["xmin"], bbox["ymin"],
                             bbox["xmax"], bbox["ymax"], FILTER_ZOOM)

    frames = []
    for year, quarter in quarters(start, end):
        try:
            df = load_quarter(con, year, quarter, prefixes, min_tests,
                              region_zoom)
        except Exception as exc:                      # quarter not published yet
            if verbose:
                print(f"  {year} Q{quarter}: skipped ({type(exc).__name__})")
            continue
        if verbose:
            print(f"  {year} Q{quarter}: {len(df):>4} regions, "
                  f"{df.tests.sum():>9,.0f} tests")
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["t"] = panel.year * 4 + panel.quarter          # running quarter index
    panel = panel.sort_values(["region", "t"])

    # Compare each region with itself one quarter earlier.
    prev = panel.groupby("region")[["down_mbps", "tests", "t"]].shift(1)
    panel["prev_down_mbps"] = prev.down_mbps
    panel["prev_tests"] = prev.tests
    consecutive = panel.t - prev.t == 1
    panel.loc[~consecutive, ["prev_down_mbps", "prev_tests"]] = None

    panel["pct_change"] = (panel.down_mbps - panel.prev_down_mbps) / panel.prev_down_mbps
    panel["declined"] = (panel["pct_change"] <= -threshold).astype("Int64")
    panel.loc[panel["pct_change"].isna(), "declined"] = None

    centres = panel.region.map(quadkey_to_lonlat)
    panel["lon"] = [c[0] for c in centres]
    panel["lat"] = [c[1] for c in centres]

    cols = ["region", "lon", "lat", "year", "quarter", "t", "down_mbps",
            "up_mbps", "lat_ms", "tests", "devices", "n_tiles",
            "prev_down_mbps", "prev_tests", "pct_change", "declined"]
    return panel[cols].reset_index(drop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", nargs=2, type=int, default=[2019, 1])
    p.add_argument("--end", nargs=2, type=int, default=[2026, 1])
    p.add_argument("--min-tests", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.20)
    p.add_argument("--zoom", type=int, default=REGION_ZOOM,
                   help="quadkey prefix length; lower means larger regions")
    p.add_argument("--out", default="data/panel.csv")
    a = p.parse_args()

    panel = build(tuple(a.start), tuple(a.end),
                  min_tests=a.min_tests, threshold=a.threshold,
                  region_zoom=a.zoom)
    panel.to_csv(a.out, index=False)

    labelled = panel["declined"].notna().sum()
    print(f"\n{len(panel):,} rows, {panel.region.nunique()} regions, "
          f"{panel.t.nunique()} quarters")
    print(f"{labelled:,} rows have a previous quarter to compare with; "
          f"{panel['declined'].sum():,.0f} of those declined "
          f"({100 * panel['declined'].mean():.1f}%)")
    print("written:", a.out)
