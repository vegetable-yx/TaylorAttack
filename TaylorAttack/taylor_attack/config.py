from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def find_model_config(config: dict[str, Any], model: str) -> dict[str, Any]:
    for item in config.get("models", []):
        if item.get("id") == model or item.get("short_name") == model:
            return item
    raise KeyError(f"Model {model!r} not found in config")


def find_benchmark_config(config: dict[str, Any], benchmark: str) -> dict[str, Any]:
    for item in config.get("datasets", []):
        if item.get("name") == benchmark:
            return item
    raise KeyError(f"Benchmark {benchmark!r} not found in config")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
