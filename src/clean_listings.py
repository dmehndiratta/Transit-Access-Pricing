"""Step 2 - Clean the raw Airbnb listings into a tidy table.

Bullet two: 'automate ... processing tasks.' The publishable twin of the
salary-parsing data-integrity work - same judgment, open data.

Handles the 'blank-price snapshot' quirk: some Inside Airbnb quarters ship
listings.csv with an EMPTY price column. When that happens we derive a
per-listing price from calendar.csv.gz (median quoted nightly rate).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from _common import load_config, raw_dir, processed_dir

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
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
    )


def parse_bathrooms(series: pd.Series) -> pd.Series:
    """'1.5 baths' / 'Half-bath' -> float count."""
    s = series.astype(str).str.lower()
    half = s.str.contains("half")
    num = pd.to_numeric(s.str.extract(r"([\d.]+)")[0], errors="coerce")
    return num.where(~half, 0.5)


def price_from_calendar(cal_path) -> pd.Series:
    """Median quoted nightly price per listing_id from calendar.csv.gz."""
    cal = pd.read_csv(cal_path, compression="gzip",
                      usecols=["listing_id", "price"],
                      dtype={"listing_id": "int64", "price": str})
    cal["price"] = parse_price(cal["price"])
    cal = cal.dropna(subset=["price"])
    return cal.groupby("listing_id")["price"].median()


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    rd = raw_dir(cfg)
    df = pd.read_csv(rd / "listings.csv.gz", compression="gzip", low_memory=False)

    df = df[[c for c in KEEP if c in df.columns]].copy()
    df["price"] = parse_price(df["price"])
    if "bathrooms_text" in df:
        df["bathrooms"] = parse_bathrooms(df["bathrooms_text"])

    # --- blank-price snapshot fallback ---
    if df["price"].isna().mean() > 0.5:
        cal_path = rd / "calendar.csv.gz"
        if not cal_path.exists():
            raise FileNotFoundError(
                "price is blank in listings.csv.gz and calendar.csv.gz is "
                "missing. Re-run Step 1 so the calendar is downloaded.")
        med = price_from_calendar(cal_path)
        df["price"] = df["id"].map(med)
        print(f"  price blank in listings -> derived from calendar "
              f"for {df['price'].notna().sum():,} listings")

    # --- light imputation so one sparse column can't wipe the sample ---
    if "bedrooms" in df:
        est = np.ceil(df["accommodates"] / 2).clip(lower=1)
        df["bedrooms"] = df["bedrooms"].fillna(est)
    if "bathrooms" in df:
        df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    # --- integrity ---
    df = df.dropna(subset=["latitude", "longitude", "price"]).copy()
    df = df[df["price"] > 0]
    lo, hi = df["price"].quantile([0.01, 0.99])   # winsorize extreme prices
    df["price"] = df["price"].clip(lo, hi)

    out = processed_dir(cfg) / "listings_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"Step 2 complete -> {out.name} ({len(df):,} rows)")


if __name__ == "__main__":
    main()