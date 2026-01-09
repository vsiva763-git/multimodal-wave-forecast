import os
import time
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import requests


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_file(url: str, dst_path: str | Path, retries: int = 3, timeout: int = 60) -> Path:
    dst = Path(dst_path)
    ensure_dir(dst.parent)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return dst
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(2 * attempt)
            else:
                raise
    if last_err:
        raise last_err
    return dst


def daterange(start_ts: int, end_ts: int, step_seconds: int) -> Iterator[int]:
    t = start_ts
    while t <= end_ts:
        yield t
        t += step_seconds


def nearest_idx(values: Iterable[float], target: float) -> int:
    values_list = list(values)
    best_idx = 0
    best_diff = float("inf")
    for i, v in enumerate(values_list):
        d = abs(v - target)
        if d < best_diff:
            best_diff = d
            best_idx = i
    return best_idx
