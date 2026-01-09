from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import xarray as xr


def crop_bbox(ds: xr.Dataset | xr.DataArray, bbox: Tuple[float, float, float, float]) -> xr.Dataset | xr.DataArray:
    min_lon, min_lat, max_lon, max_lat = bbox
    # Attempt to find standard coordinate names
    lon_name = next((n for n in ["lon", "longitude", "x"] if n in ds.coords), None)
    lat_name = next((n for n in ["lat", "latitude", "y"] if n in ds.coords), None)
    if lon_name is None or lat_name is None:
        return ds
    return ds.sel({lon_name: slice(min_lon, max_lon), lat_name: slice(min_lat, max_lat)})


def sample_points(ds: xr.Dataset, lats: Iterable[float], lons: Iterable[float]) -> xr.Dataset:
    lat_name = next((n for n in ["lat", "latitude", "y"] if n in ds.coords), None)
    lon_name = next((n for n in ["lon", "longitude", "x"] if n in ds.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError("Dataset missing lat/lon coordinates for sampling")
    pts = ds.interp({lat_name: ("point", np.array(list(lats))), lon_name: ("point", np.array(list(lons)))}).transpose(
        ...
    )
    return pts
