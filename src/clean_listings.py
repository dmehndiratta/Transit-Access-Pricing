"""Step 2 - Clean the raw Airbnb listings into a tidy table.

Bullet two: 'automate ... processing tasks.' This is the publishable twin of
the salary-parsing data-integrity work - same judgment, open data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from _common import load_config, raw_dir, processed_dir

# Columns we keep from Inside Airbnb's detailed listings file.
KEEP = [
    "id", "latitude", "longitude", "price", "room_type",
    "accommodates", "bedrooms", "bathrooms_text",
    "number_of_reviews", "review_scores_rating", "neighbourhood_cleansed",
]


def parse_price(series: pd.Series) -> pd.Series:
    """'$1,234.00' -> 1234.0 ; blanks/junk -> NaN."""
    return (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan})
        .astype(float)
    )


def parse_bathrooms(series: pd.Series) -> pd.Series:
    """'1.5 baths' / 'Half-bath' -> float count."""
    s = series.astype(str).str.lower()
    half = s.str.contains("half")
    num = pd.to_numeric(s.str.extract(r"([\d.]+)")[0], errors="coerce")
    return num.where(~half, 0.5)


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    df = pd.read_csv(raw_dir(cfg) / "listings.csv.gz", compression="gzip", low_memory=False)

    df = df[[c for c in KEEP if c in df.columns]].copy()
    df["price"] = parse_price(df["price"])
    if "bathrooms_text" in df:
        df["bathrooms"] = parse_bathrooms(df["bathrooms_text"])

    # --- TODO: integrity calls (document each in the README) ---
    # 1. Drop / winsorize implausible prices (e.g. keep 1st-99th percentile).
    # 2. Missing bedrooms / bathrooms: impute vs drop - decide and note why.
    # 3. Drop rows with no coordinates.
    # 4. De-duplicate repeat listings if present.

    out = processed_dir(cfg) / "listings_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"Step 2 complete -> {out.name} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
