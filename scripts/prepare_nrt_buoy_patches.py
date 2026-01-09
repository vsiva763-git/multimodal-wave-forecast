#!/usr/bin/env python3
from __future__ import annotations

"""
Prepare NRT buoy-centered patches from WW3 (waves) and GFS (winds/pressure).

Steps:
 - Discover latest WW3 and GFS cycles on NOMADS
 - Download a few forecast hours (default: 0,3,6)
 - Open GRIB2 via cfgrib + eccodes
 - Extract N x N spatial patch around buoy
 - Align hourly time and build rolling sequences: past T -> future H SWH targets
 - Save arrays to data/processed/nrt_{station}.npz

Requirements:
  sudo apt install -y eccodes libeccodes0 libeccodes-dev
  pip install -r requirements.txt
"""

import argparse
from pathlib import Path
from typing import List
import sys

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.nomads import (
    latest_ww3_product,
    latest_gfs_cycle,
    ww3_glo30m_filenames,
    gfs_0p25_filenames,
)
from src.data.ww3 import fetch_ww3_files, open_ww3_grib, WW3Source
from src.data.gfs import fetch_gfs_files, open_gfs_grib, GFSSource
from src.data.ndbc_meta import get_station_latlon, get_stations_in_bbox
from src.data.ocean_regions import get_ocean_region, list_ocean_regions
from src.preprocess.temporal import align_time
from src.data.utils import ensure_dir


def _coord_names(ds: xr.Dataset):
    lat = next((n for n in ["lat", "latitude", "y"] if n in ds.coords), None)
    lon = next((n for n in ["lon", "longitude", "x"] if n in ds.coords), None)
    if lat is None or lon is None:
        raise ValueError("Dataset missing lat/lon coordinates")
    return lat, lon


def extract_patch(ds: xr.Dataset, lat: float, lon: float, patch: int) -> xr.Dataset:
    lat_name, lon_name = _coord_names(ds)
    # Find nearest grid point index
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values
    # Handle 0..360 lon if present
    if lon_vals.min() >= 0 and lon < 0:
        lon = (lon + 360) % 360
    lat_idx = int(np.argmin(np.abs(lat_vals - lat)))
    lon_idx = int(np.argmin(np.abs(lon_vals - lon)))
    k = patch // 2
    lat_start = max(0, min(lat_idx - k, len(lat_vals) - patch))
    lon_start = max(0, min(lon_idx - k, len(lon_vals) - patch))
    sel = ds.isel({lat_name: slice(lat_start, lat_start + patch), lon_name: slice(lon_start, lon_start + patch)})
    return sel


def stack_vars(ds: xr.Dataset, var_order: List[str]) -> xr.DataArray:
    # Stack selected variables into channel dim [time, channel, lat, lon]
    arrays = []
    for v in var_order:
        if v not in ds:
            raise KeyError(f"Variable {v} not in dataset")
        arrays.append(ds[v])
    stacked = xr.concat(arrays, dim="channel").transpose("time", "channel", ...)
    return stacked


def build_sequences(ww3_patch: np.ndarray, gfs_patch: np.ndarray, T: int, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ww3_patch: [time, Cw, H, W] includes swh at channel 0
    # gfs_patch: [time, Cg, H, W]
    # Output: Xw [N,T,Cw,H,W], Xg [N,T,Cg,H,W], y [N,H]
    times = ww3_patch.shape[0]
    N = times - (T + H) + 1
    if N <= 0:
        return np.empty((0, T, *ww3_patch.shape[1:])), np.empty((0, T, *gfs_patch.shape[1:])), np.empty((0, H))
    Xw = []
    Xg = []
    Y = []
    center = ww3_patch.shape[-1] // 2
    for i in range(N):
        Xw.append(ww3_patch[i : i + T])
        Xg.append(gfs_patch[i : i + T])
        # target from SWH (channel 0) at central pixel
        swh_future = ww3_patch[i + T : i + T + H, 0, center, center]
        Y.append(swh_future)
    return np.stack(Xw), np.stack(Xg), np.stack(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", help="NDBC station id, e.g., 46042 (or use --ocean)")
    ap.add_argument("--ocean", help=f"Ocean region name (alternative to --station): {', '.join(list_ocean_regions()[:5])}...")
    ap.add_argument("--list-oceans", action="store_true", help="List available ocean regions and exit")
    ap.add_argument("--fhrs", type=int, nargs="*", default=[0, 3, 6], help="Forecast hours to fetch")
    ap.add_argument("--time_steps", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--out", default=None, help="Output npz path (default: data/processed/nrt_{station}.npz)")
    ap.add_argument("--ww3_base", default="https://nomads.ncep.noaa.gov/pub/data/nccf/com/wave/prod/", help="WW3 base URL")
    ap.add_argument("--ww3_dir", default=None, help="Override WW3 product dir, e.g., multi_1.YYYYMMDD")
    ap.add_argument("--gfs_base", default="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/", help="GFS base URL")
    ap.add_argument("--gfs_dir", default=None, help="Override GFS product dir, e.g., gfs.YYYYMMDD/HH")
    args = ap.parse_args()

    if args.list_oceans:
        print("Available ocean regions:")
        for region_key in list_ocean_regions():
            print(f"  {region_key}")
        return

    if not args.station and not args.ocean:
        raise SystemExit("Must specify either --station or --ocean")

    if args.ocean:
        region = get_ocean_region(args.ocean)
        if not region:
            raise SystemExit(f"Unknown ocean region: {args.ocean}. Use --list-oceans to see available regions.")
        print(f"Using ocean region: {region.name} {region.bbox}")
        # Find stations in this region
        stations = get_stations_in_bbox(region.bbox)
        if not stations:
            raise SystemExit(f"No NDBC stations found in {region.name}")
        # Use the first station as representative (or you could loop over all)
        args.station = stations[0][0]
        lat, lon = stations[0][1], stations[0][2]
        print(f"Selected station {args.station} at ({lat:.2f}, {lon:.2f})")
        print(f"Found {len(stations)} total stations in {region.name}: {', '.join([s[0] for s in stations[:10]])}{'...' if len(stations) > 10 else ''}")
    else:
        latlon = get_station_latlon(args.station)
        if not latlon:
            raise SystemExit(f"Could not find station {args.station} in NDBC active stations")
        lat, lon = latlon

    ww3_base = args.ww3_base
    gfs_base = args.gfs_base
    ww3_dir = args.ww3_dir or latest_ww3_product(ww3_base)
    gfs_dir = args.gfs_dir or latest_gfs_cycle(gfs_base)
    if not ww3_dir or not gfs_dir:
        raise SystemExit("Could not locate latest WW3/GFS directories on NOMADS. Pass --ww3_dir and --gfs_dir overrides.")

    # Determine cycle hour for GFS filenames
    gfs_hh = gfs_dir.split("/")[-1]

    ww3_files = ww3_glo30m_filenames(args.fhrs)
    gfs_files = gfs_0p25_filenames(gfs_hh, args.fhrs)

    raw_ww3 = ensure_dir("data/raw/ww3")
    raw_gfs = ensure_dir("data/raw/gfs")

    ww3_paths = fetch_ww3_files(raw_ww3, WW3Source(ww3_base, ww3_dir, ww3_files))
    gfs_paths = fetch_gfs_files(raw_gfs, GFSSource(gfs_base, gfs_dir, gfs_files))

    # Open and concat over time
    ww3_ds_list = [open_ww3_grib(p, variables=["swh", "mwp", "mwd"]) for p in ww3_paths]
    ww3_ds = xr.concat(ww3_ds_list, dim="time")
    gfs_ds_list = [open_gfs_grib(p, variables=["10u", "10v", "prmsl"]) for p in gfs_paths]
    gfs_ds = xr.concat(gfs_ds_list, dim="time")

    # Align time to hourly
    ww3_ds, gfs_ds = align_time([ww3_ds, gfs_ds], freq="1H")

    # Extract patches and stack channels
    ww3_patch_ds = extract_patch(ww3_ds, lat, lon, args.patch)
    gfs_patch_ds = extract_patch(gfs_ds, lat, lon, args.patch)
    ww3_patch = stack_vars(ww3_patch_ds, ["swh", "mwp", "mwd"]).to_numpy().astype(np.float32)
    gfs_patch = stack_vars(gfs_patch_ds, ["10u", "10v", "prmsl"]).to_numpy().astype(np.float32)

    Xw, Xg, Y = build_sequences(ww3_patch, gfs_patch, args.time_steps, args.horizon)
    if Xw.shape[0] == 0:
        raise SystemExit("Not enough time steps to build sequences with the requested T and H.")

    out_path = args.out or f"data/processed/nrt_{args.station}.npz"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ww3=Xw, gfs=Xg, target=Y)
    print(f"Wrote {out} with {Xw.shape[0]} samples: Xw {Xw.shape}, Xg {Xg.shape}, Y {Y.shape}")


if __name__ == "__main__":
    main()
