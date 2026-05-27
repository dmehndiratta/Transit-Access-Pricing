"""Step 3 - Engineer transit-accessibility features per listing.

Bullet two (the clever bit): turn raw GTFS into 'how good is transit here?'.
Uses sklearn BallTree with the haversine metric, so there is NO geopandas /
GDAL dependency - much smoother to install on Windows.

stops.txt has no route_type, so 'is this a rail stop?' and 'how frequent is
service here?' are recovered by joining:
    stop_times.txt -> trips.txt (route_id) -> routes.txt (route_type)

Features written per listing id:
  - dist_nearest_stop_m    nearest serviced stop of any kind
  - dist_nearest_rail_m    nearest metro/tram/rail stop (route_type in config)
  - n_stops_within_buffer  serviced stops within the walk buffer
  - freq_weighted_access   sum of AM-peak departures at stops within the buffer
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from _common import load_config, raw_dir, processed_dir

EARTH_M = 6_371_000  # mean Earth radius in metres


def _to_seconds(t: str) -> int:
    """'HH:MM:SS' -> seconds since midnight. Handles GTFS hours >= 24 and
    values that aren't zero-padded ('7:00:00')."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _radians(df: pd.DataFrame, lat: str, lon: str) -> np.ndarray:
    return np.radians(df[[lat, lon]].to_numpy(dtype=float))


def build_stop_table(gtfs_dir: Path, am_peak, rail_types) -> pd.DataFrame:
    """Return serviced stops with coords, AM-peak departure counts, is_rail."""
    # route_id -> route_type
    routes = pd.read_csv(gtfs_dir / "routes.txt", dtype={"route_id": str})
    route_type = dict(zip(routes["route_id"], routes["route_type"]))

    # trip_id -> route_type  (composed through trips.txt)
    trips = pd.read_csv(gtfs_dir / "trips.txt",
                        dtype={"route_id": str, "trip_id": str})
    trip_type = {t: route_type.get(r) for t, r in
                 zip(trips["trip_id"], trips["route_id"])}

    # stop_times can be large - read only what we need, as strings.
    st_path = gtfs_dir / "stop_times.txt"
    header = pd.read_csv(st_path, nrows=0).columns
    timecol = "departure_time" if "departure_time" in header else "arrival_time"
    st = pd.read_csv(st_path, usecols=["trip_id", "stop_id", timecol],
                     dtype={"trip_id": str, "stop_id": str, timecol: str})
    st = st.dropna(subset=[timecol])

    st["route_type"] = st["trip_id"].map(trip_type)

    secs = st[timecol].map(_to_seconds)
    am0, am1 = _to_seconds(am_peak[0]), _to_seconds(am_peak[1])
    am_mask = (secs >= am0) & (secs < am1)

    am_peak_trips = (st.loc[am_mask].groupby("stop_id").size()
                     .rename("am_peak_trips"))
    rail_stop_ids = set(st.loc[st["route_type"].isin(rail_types), "stop_id"].unique())
    serviced = set(st["stop_id"].unique())

    stops = pd.read_csv(gtfs_dir / "stops.txt", dtype={"stop_id": str},
                        usecols=["stop_id", "stop_lat", "stop_lon"])
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    stops = stops[stops["stop_id"].isin(serviced)].copy()
    stops = stops.merge(am_peak_trips, on="stop_id", how="left")
    stops["am_peak_trips"] = stops["am_peak_trips"].fillna(0).astype(int)
    stops["is_rail"] = stops["stop_id"].isin(rail_stop_ids)
    return stops


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    tf = cfg["transit_features"]
    buffer_rad = tf["walk_buffer_m"] / EARTH_M
    rail_types = set(tf["rail_route_types"])

    listings = pd.read_parquet(processed_dir(cfg) / "listings_clean.parquet")
    listings = listings.dropna(subset=["latitude", "longitude"]).copy()

    stops = build_stop_table(raw_dir(cfg) / "gtfs_stm", tf["am_peak"], rail_types)
    print(f"  stops: {len(stops):,} serviced "
          f"({int(stops['is_rail'].sum()):,} rail)")

    L = _radians(listings, "latitude", "longitude")
    freq = stops["am_peak_trips"].to_numpy()

    tree = BallTree(_radians(stops, "stop_lat", "stop_lon"), metric="haversine")
    d_near, _ = tree.query(L, k=1)
    within = tree.query_radius(L, r=buffer_rad)

    listings["dist_nearest_stop_m"] = d_near[:, 0] * EARTH_M
    listings["n_stops_within_buffer"] = [len(ix) for ix in within]
    listings["freq_weighted_access"] = [int(freq[ix].sum()) for ix in within]

    rail = stops[stops["is_rail"]]
    if len(rail):
        rail_tree = BallTree(_radians(rail, "stop_lat", "stop_lon"), metric="haversine")
        dr, _ = rail_tree.query(L, k=1)
        listings["dist_nearest_rail_m"] = dr[:, 0] * EARTH_M
    else:
        listings["dist_nearest_rail_m"] = np.nan

    cols = ["id", "dist_nearest_stop_m", "dist_nearest_rail_m",
            "n_stops_within_buffer", "freq_weighted_access"]
    out = processed_dir(cfg) / "transit_features.parquet"
    listings[cols].to_parquet(out, index=False)

    print(f"  median dist to nearest stop: "
          f"{listings['dist_nearest_stop_m'].median():.0f} m")
    print(f"  median dist to nearest rail: "
          f"{listings['dist_nearest_rail_m'].median():.0f} m")
    print(f"Step 3 complete -> {out.name} ({len(listings):,} rows)")


if __name__ == "__main__":
    main()