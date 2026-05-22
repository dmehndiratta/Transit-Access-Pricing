"""Step 1 - Download raw open data (Airbnb listings + GTFS feeds).

Bullet two: 'develop scripts and workflows to automate data collection.'
Idempotent - files already present are skipped, so re-running is cheap and
safe. Point a scheduler (cron / GitHub Actions) at this to make it recurring.
"""
from __future__ import annotations
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from _common import load_config, ensure_dirs, raw_dir


def download(url: str, dest: Path) -> None:
    if not url:
        print(f"  [skip] no URL configured for {dest.name}")
        return
    if dest.exists():
        print(f"  [skip] {dest.name} already present")
        return
    print(f"  [get ] {dest.name}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


def unzip(path: Path, out_dir: Path) -> None:
    if not path.exists() or path.suffix != ".zip":
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(out_dir)
    print(f"  [unzip] {path.name} -> {out_dir.name}/")


def main(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    rd = raw_dir(cfg)
    urls = cfg["data_urls"]

    download(urls.get("airbnb_listings", ""), rd / "listings.csv.gz")
    for feed in ("gtfs_stm", "gtfs_rem", "gtfs_exo"):
        zip_path = rd / f"{feed}.zip"
        download(urls.get(feed, ""), zip_path)
        unzip(zip_path, rd / feed)

    print("Step 1 complete.")


if __name__ == "__main__":
    main()
