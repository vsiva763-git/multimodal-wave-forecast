from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import xarray as xr

from .utils import download_file, ensure_dir


@dataclass
class WW3Source:
    """
    Helper describing a WW3 source.

    Notes:
    - NOAA NOMADS WW3 directory example (global multi-grid):
      https://nomads.ncep.noaa.gov/pub/data/nccf/com/wave/prod/
      Inside: multi_1.YYYYMMDD/ files for each cycle and forecast hour.

    This module provides utilities but expects you to supply concrete URLs or
    patterns since WW3 file naming varies by product (glo_30m, ak, at_10m, etc.).
    """

    base_url: str  # e.g., https://nomads.ncep.noaa.gov/pub/data/nccf/com/wave/prod
    product_dir: str  # e.g., multi_1.20260109
    filenames: list[str]  # e.g., ["multi_1.glo_30m.f000.grib2", ...]


def fetch_ww3_files(dst_dir: str | Path, source: WW3Source) -> list[Path]:
    ensure_dir(dst_dir)
    saved: list[Path] = []
    for fname in source.filenames:
        url = f"{source.base_url.rstrip('/')}/{source.product_dir}/{fname}"
        dst = Path(dst_dir) / fname
        saved.append(download_file(url, dst))
    return saved


def open_ww3_grib(file_path: str | Path, variables: Iterable[str] | None = None) -> xr.Dataset:
    """
    Open a WW3 GRIB2 file via cfgrib -> xarray.Dataset.

    Common shortNames:
    - swh: significant wave height (m)
    - mwp: mean wave period (s)
    - mwd: mean wave direction (degree)

    Requires eccodes system library. On Ubuntu:
      sudo apt update && sudo apt install -y eccodes libeccodes0 libeccodes-dev
    """
    try:
        if variables:
            # cfgrib supports filtering by shortName
            ds_all = []
            for var in variables:
                ds_var = xr.open_dataset(
                    file_path, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": var}}
                )
                ds_all.append(ds_var)
            return xr.merge(ds_all)
        return xr.open_dataset(file_path, engine="cfgrib")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Failed to open WW3 GRIB. Ensure eccodes is installed and cfgrib is available."
        ) from e
