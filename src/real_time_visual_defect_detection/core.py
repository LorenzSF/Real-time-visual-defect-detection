from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON or YAML config file into a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix == ".json":
            cfg = json.load(f)
        else:
            cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dictionary-like object: {path}")

    return cfg
