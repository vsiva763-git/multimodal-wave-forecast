#!/usr/bin/env python3
from __future__ import annotations

"""
Lightweight download helper.

This script provides examples and a starting point for fetching WW3 and GFS GRIB2
files from NOMADS. Exact filenames vary by cycle and grid. Adjust the 'examples'
section to your target product.

Usage:
  python scripts/download_data.py --dst data/raw/ww3 --type ww3 \
      --base https://nomads.ncep.noaa.gov/pub/data/nccf/com/wave/prod \
      --dir multi_1.20260109 --file multi_1.glo_30m.f000.grib2
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.ww3 import WW3Source, fetch_ww3_files
from src.data.gfs import GFSSource, fetch_gfs_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True, help="Destination folder")
    ap.add_argument("--type", choices=["ww3", "gfs"], required=True)
    ap.add_argument("--base", required=True, help="Base URL to NOMADS product root")
    ap.add_argument("--dir", required=True, help="Product directory, e.g., multi_1.YYYYMMDD or gfs.YYYYMMDD/HH")
    ap.add_argument("--file", action="append", dest="files", help="Filename to download (repeatable)")
    args = ap.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    if args.type == "ww3":
        src = WW3Source(base_url=args.base, product_dir=args.dir, filenames=args.files or [])
        saved = fetch_ww3_files(dst, src)
    else:
        src = GFSSource(base_url=args.base, product_dir=args.dir, filenames=args.files or [])
        saved = fetch_gfs_files(dst, src)
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
