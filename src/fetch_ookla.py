"""
Fetch Ookla Open Data mobile performance tiles for a bounding box,
aggregated into a coarse grid (stand-in for SA2 until ABS boundaries are added).

Data source: https://github.com/teamookla/ookla-open-data
Licence: CC BY-NC-SA 4.0

Usage:
    python src/fetch_ookla.py --year 2026 --quarter 1 --out data/nsw_2026Q1.csv
"""
import argparse

import duckdb

BASE = ("https://ookla-open-data.s3.amazonaws.com/parquet/performance"
        "/type=mobile/year={year}/quarter={q}/{year}-{month}-01_performance_mobile_tiles.parquet")
QUARTER_MONTH = {1: "01", 2: "04", 3: "07", 4: "10"}

# Rough bounding box for New South Wales
NSW = dict(xmin=141.0, xmax=154.0, ymin=-37.6, ymax=-28.1)


def url_for(year: int, quarter: int) -> str:
    return BASE.format(year=year, q=quarter, month=QUARTER_MONTH[quarter])


def fetch(year: int, quarter: int, bbox=NSW, grid=1, min_tests=50):
    """Return a dataframe of weighted mean speeds per grid cell.

    grid: number of decimal places to round coordinates to.
          1 -> ~11 km cells. Replace with a spatial join on ABS SA2
          boundaries once those are wired in.
    """
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    sql = f"""
        SELECT round(tile_x, {grid})                          AS gx,
               round(tile_y, {grid})                          AS gy,
               sum(avg_d_kbps * tests) / sum(tests) / 1000.0  AS down_mbps,
               sum(avg_u_kbps * tests) / sum(tests) / 1000.0  AS up_mbps,
               sum(avg_lat_ms * tests) / sum(tests)           AS lat_ms,
               sum(tests)                                     AS tests,
               count(*)                                       AS n_tiles
        FROM read_parquet('{url_for(year, quarter)}')
        WHERE tile_x BETWEEN {bbox['xmin']} AND {bbox['xmax']}
          AND tile_y BETWEEN {bbox['ymin']} AND {bbox['ymax']}
        GROUP BY 1, 2
        HAVING sum(tests) >= {min_tests}
    """
    df = con.execute(sql).df()
    df["year"], df["quarter"] = year, quarter
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--quarter", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--grid", type=int, default=1)
    p.add_argument("--min-tests", type=int, default=50)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    df = fetch(a.year, a.quarter, grid=a.grid, min_tests=a.min_tests)
    print(f"{a.year} Q{a.quarter}: {len(df)} regions, {df.tests.sum():,.0f} tests")
    if a.out:
        df.to_csv(a.out, index=False)
        print("written:", a.out)
