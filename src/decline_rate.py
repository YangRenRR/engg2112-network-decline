"""
How often does a region's median download speed fall by more than X%
from one quarter to the next?

This is the check that tells us whether the 20% threshold in the proposal
makes the classification task too easy or too hard.

Usage:
    python src/decline_rate.py --from 2025 4 --to 2026 1
"""
import argparse

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import duckdb

from fetch_ookla import NSW, url_for


def decline_rates(y0, q0, y1, q1, bbox=NSW, grid=1, min_tests=50,
                  thresholds=(0.10, 0.20, 0.30)):
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    box = (f"tile_x BETWEEN {bbox['xmin']} AND {bbox['xmax']} "
           f"AND tile_y BETWEEN {bbox['ymin']} AND {bbox['ymax']}")
    agg = lambda url: f"""
        SELECT round(tile_x,{grid}) gx, round(tile_y,{grid}) gy,
               sum(avg_d_kbps*tests)/sum(tests) d, sum(tests) t
        FROM read_parquet('{url}') WHERE {box} GROUP BY 1,2
    """
    cases = ",\n".join(
        f"round(100.0*sum(CASE WHEN chg <= -{t} THEN 1 ELSE 0 END)/count(*),1) "
        f'AS "drop_over_{int(t*100)}pct"' for t in thresholds)
    sql = f"""
        WITH a AS ({agg(url_for(y0, q0))}),
             b AS ({agg(url_for(y1, q1))}),
             j AS (SELECT a.gx, a.gy, (b.d - a.d) / a.d AS chg
                   FROM a JOIN b USING (gx, gy)
                   WHERE a.t >= {min_tests} AND b.t >= {min_tests})
        SELECT count(*) AS regions,
               {cases},
               round(100*median(chg),1) AS median_pct_change
        FROM j
    """
    return con.execute(sql).df()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="frm", nargs=2, type=int, default=[2025, 4],
                   metavar=("YEAR", "QUARTER"))
    p.add_argument("--to", nargs=2, type=int, default=[2026, 1],
                   metavar=("YEAR", "QUARTER"))
    p.add_argument("--grid", type=int, default=1)
    p.add_argument("--min-tests", type=int, default=50)
    a = p.parse_args()

    df = decline_rates(*a.frm, *a.to, grid=a.grid, min_tests=a.min_tests)
    print(f"{a.frm[0]} Q{a.frm[1]}  ->  {a.to[0]} Q{a.to[1]}   (NSW)")
    print(df.to_string(index=False))
