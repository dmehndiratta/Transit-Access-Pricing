"""Step 3 - Engineer transit-accessibility features per listing.

Bullet two (the clever bit): turn GTFS into 'how good is transit here?'.
Uses sklearn BallTree with the haversine metric, so there is NO geopandas /
GDAL dependency - much smoother to install on Windows.

Features per listing:
  - dist_nearest_stop_m
  - dist_nearest_rail_m
  - n_stops_within_buffer
  - freq_weighted_access     (nearby stops, weighted by AM-peak trips)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from _common import load_config, raw_dir, processed_dir

EARTH_M = 6_371_000  # metres


def load_stops(gtfs_dir) -> pd.DataFrame:
    """Read stops.txt -> DataFrame[stop_id, lat, lon, is_rail].

    is_rail comes from routes.txt route_type, joined through trips.txt and
    stop_times.txt (rail_route_types in config).
    """
    raise NotImplementedError  # TODO


def stop_frequencies(gtfs_dir, am_peak) -> pd.Series:
    """AM-peak trips per stop_id, from stop_times.txt filtered to the window."""
    raise NotImplementedError  # TODO


def radians(df, lat="lat", lon="lon") -> np.ndarray:
    return np.radians(df[[lat, lon]].to_numpy())


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    buffer_rad = cfg["transit_features"]["walk_buffer_m"] / EARTH_M

    listings = pd.read_parquet(processed_dir(cfg) / "listings_clean.parquet")

    # TODO:
    # 1. Load + concat stops from each GTFS feed in raw_dir; attach frequency.
    # 2. tree = BallTree(radians(stops), metric='haversine')
    # 3. nearest stop:  d, _ = tree.query(radians(listings), k=1)
    #                   dist_nearest_stop_m = d[:, 0] * EARTH_M
    # 4. within buffer:  idx = tree.query_radius(radians(listings), buffer_rad)
    #                   n_stops_within_buffer = [len(i) for i in idx]
    #                   freq_weighted_access  = [freq[i].sum() for i in idx]
    # 5. rail tree on stops[is_rail] -> dist_nearest_rail_m
    # 6. keep ['id', <features>] and write parquet.

    out = processed_dir(cfg) / "transit_features.parquet"
    print(f"Step 3 complete -> {out.name}")


if __name__ == "__main__":
    main()
