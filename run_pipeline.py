"""Run the whole pipeline end-to-end:  python run_pipeline.py

This single command is the 'workflow' half of 'scripts and workflows' - it
rebuilds the analysis-ready dataset from raw open data in one shot.

Note: the step files use plain names (no leading digits) because Python module
names can't start with a number - 'import 01_fetch_data' is a syntax error.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from _common import load_config
import fetch_data
import clean_listings
import build_transit_features
import assemble_dataset
import train_model

STEPS = [
    ("Fetch raw data", fetch_data.main),
    ("Clean listings", clean_listings.main),
    ("Build transit features", build_transit_features.main),
    ("Assemble dataset", assemble_dataset.main),
    ("Train models", train_model.main),
]


def main() -> None:
    cfg = load_config()
    for name, fn in STEPS:
        print(f"\n=== {name} ===")
        t0 = time.time()
        fn(cfg)
        print(f"    ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
