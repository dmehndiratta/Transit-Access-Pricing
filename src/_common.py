"""Shared helpers: config loading and path setup."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict) -> None:
    for key in ("raw_dir", "processed_dir"):
        (ROOT / cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)


def raw_dir(cfg: dict) -> Path:
    return ROOT / cfg["paths"]["raw_dir"]


def processed_dir(cfg: dict) -> Path:
    return ROOT / cfg["paths"]["processed_dir"]
