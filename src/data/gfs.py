from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import xarray as xr

from .utils import download_file, ensure_dir


@dataclass
class GFSSource:
    """
    Helper for GFS downloads from NOMADS.

    Example base:
      https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/

    Note: GFS GRIB shortNames for 10m winds and pressure can include: 10u, 10v, prmsl.
    Exact availability depends on product/resolution.
    """

    base_url: str
    product_dir: str
    filenames: list[str]


def fetch_gfs_files(dst_dir: str | Path, source: GFSSource) -> list[Path]:
    ensure_dir(dst_dir)
    saved: list[Path] = []
    for fname in source.filenames:
        url = f"{source.base_url.rstrip('/')}/{source.product_dir}/{fname}"
        dst = Path(dst_dir) / fname
        saved.append(download_file(url, dst))
    return saved


def open_gfs_grib(file_path: str | Path, variables: Iterable[str] | None = None) -> xr.Dataset:
    try:
        if variables:
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
            "Failed to open GFS GRIB. Ensure eccodes is installed and cfgrib is available."
        ) from e
