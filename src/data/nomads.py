from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Tuple

import requests


def _list_dirs(url: str) -> List[str]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    # Simple href dir listing parser
    hrefs = re.findall(r'href=["\']([^"\']+/)["\']', r.text)
    # Filter out parent dirs and non-date-like
    return hrefs


def latest_ww3_product(base: str = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/wave/prod/") -> Optional[str]:
    dirs = _list_dirs(base)
    # Expect multi_1.YYYYMMDD/
    candidates: List[Tuple[datetime, str]] = []
    for d in dirs:
        m = re.match(r"multi_1\.(\d{8})/", d)
        if m:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
            candidates.append((dt, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1].rstrip("/")


def latest_gfs_cycle(base: str = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/") -> Optional[str]:
    day_dirs = _list_dirs(base)
    # Expect gfs.YYYYMMDD/
    day_candidates: List[Tuple[datetime, str]] = []
    for d in day_dirs:
        m = re.match(r"gfs\.(\d{8})/", d)
        if m:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
            day_candidates.append((dt, d))
    if not day_candidates:
        return None
    day_candidates.sort(key=lambda x: x[0])
    latest_day = day_candidates[-1][1]
    # list cycles (00,06,12,18)
    cycle_dirs = _list_dirs(base + latest_day)
    cycles = [c for c in cycle_dirs if re.match(r"\d{2}/", c)]
    if not cycles:
        return None
    cycles.sort()
    return (latest_day + cycles[-1]).rstrip("/")  # e.g., gfs.YYYYMMDD/HH


def ww3_glo30m_filenames(fhrs: List[int]) -> List[str]:
    return [f"multi_1.glo_30m.f{fh:03d}.grib2" for fh in fhrs]


def gfs_0p25_filenames(cycle_hh: str, fhrs: List[int]) -> List[str]:
    # files like gfs.tHHz.pgrb2.0p25.f000
    return [f"gfs.t{cycle_hh}z.pgrb2.0p25.f{fh:03d}" for fh in fhrs]
