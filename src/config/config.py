from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ProjectConfig:
    region_bbox: List[float]
    buoys: List[str]
    patch_size: int
    time_steps: int
    horizon: int
    thresholds: Dict[str, float]
    webhook_url: Optional[str] = None
    training: Optional[Dict[str, Any]] = None


def load_config(path: str | Path) -> ProjectConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ProjectConfig(
        region_bbox=raw.get("region_bbox", [-130, 20, -110, 40]),
        buoys=raw.get("buoys", []),
        patch_size=int(raw.get("patch_size", 9)),
        time_steps=int(raw.get("time_steps", 12)),
        horizon=int(raw.get("horizon", 6)),
        thresholds=raw.get("thresholds", {"swh": 4.0}),
        webhook_url=raw.get("webhook_url"),
        training=raw.get("training", {}),
    )
