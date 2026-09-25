"""
Helpers for Bing-style quadkeys, which is how Ookla identifies its tiles.

A quadkey is a base-4 string: each extra character halves the tile size, so
a prefix of length z is the tile containing it at zoom level z. That makes
prefixes a cheap way to both filter a region and aggregate tiles, and unlike
the tile_x / tile_y columns it works for every quarter Ookla has published.
"""
import math


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    sin_lat = math.sin(math.radians(lat))
    y = int((0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_to_quadkey(x: int, y: int, zoom: int) -> str:
    out = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        out.append(str(digit))
    return "".join(out)


def bbox_prefixes(xmin: float, ymin: float, xmax: float, ymax: float,
                  zoom: int = 8) -> list[str]:
    """Quadkey prefixes covering a bounding box.

    Used to filter the global parquet file down to one region before any
    rows are read back.
    """
    x0, y0 = lonlat_to_tile(xmin, ymax, zoom)   # top-left
    x1, y1 = lonlat_to_tile(xmax, ymin, zoom)   # bottom-right
    return [tile_to_quadkey(x, y, zoom)
            for x in range(x0, x1 + 1)
            for y in range(y0, y1 + 1)]


def quadkey_to_lonlat(qk: str) -> tuple[float, float]:
    """Centre of the tile a quadkey refers to."""
    x = y = 0
    zoom = len(qk)
    for i, c in enumerate(qk):
        mask = 1 << (zoom - i - 1)
        if c in "13":
            x |= mask
        if c in "23":
            y |= mask
    n = 2 ** zoom
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n))))
    return lon, lat
