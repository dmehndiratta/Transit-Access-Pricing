"""Quick snapshot probe - does this Inside Airbnb snapshot still have prices?

Usage:
    python check_snapshot.py <listings.csv.gz URL>

Downloads only the listings file, reports how many rows have a real price,
and shows a few samples. Use it to vet a snapshot BEFORE wiring it into
config.yaml and re-running the whole pipeline. Nothing is saved to data/.
"""
import sys
import tempfile
import pandas as pd
import requests


def main(url: str) -> None:
    print(f"probing: {url}")
    with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False) as tmp:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                tmp.write(chunk)
        path = tmp.name

    df = pd.read_csv(path, compression="gzip", usecols=["price"], low_memory=False)
    price = (df["price"].astype(str)
             .str.replace(r"[\$,]", "", regex=True)
             .str.strip()
             .replace({"": None, "nan": None, "None": None}))
    n = len(price)
    good = price.notna().sum()
    print(f"rows: {n:,}")
    print(f"non-blank price: {good:,}  ({100*good/n:.1f}%)")
    print(f"samples: {price.dropna().head().tolist()}")
    if good == 0:
        print(">> price-stripped snapshot - try an earlier date.")
    else:
        print(">> usable. Put this URL in config.yaml: airbnb_listings.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python check_snapshot.py <listings.csv.gz URL>")
        sys.exit(1)
    main(sys.argv[1])