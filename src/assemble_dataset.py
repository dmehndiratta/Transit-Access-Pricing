"""Step 4 - Assemble the analysis-ready dataset.

Bullet one: writes 'listings_features.parquet' - the analysis-ready dataset
others can build on without re-running the messy steps.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from _common import load_config, processed_dir

EARTH_M = 6_371_000


def haversine_m(lat, lon, ref_lat, ref_lon) -> np.ndarray:
    """Great-circle distance (m) from each (lat, lon) to one reference point."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat, lon, ref_lat, ref_lon))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_M * 2 * np.arcsin(np.sqrt(a))


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    pdir = processed_dir(cfg)

    listings = pd.read_parquet(pdir / "listings_clean.parquet")
    transit = pd.read_parquet(pdir / "transit_features.parquet")
    df = listings.merge(transit, on="id", how="inner")

    ref = cfg["downtown_ref"]
    df["dist_to_core_m"] = haversine_m(df["latitude"], df["longitude"], ref["lat"], ref["lon"])
    df["log_price"] = np.log(df["price"])

    out = pdir / "listings_features.parquet"
    df.to_parquet(out, index=False)
    print(f"Step 4 complete -> {out.name} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
