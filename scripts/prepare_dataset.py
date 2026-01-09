#!/usr/bin/env python3
from __future__ import annotations

"""
Prepare training samples for selected buoys.

For this initial scaffold, this script creates a synthetic dataset compatible with
the model pipeline. Extend this to:
 - Open WW3 and GFS GRIB via cfgrib/xarray
 - Spatially crop patches around buoy locations
 - Temporally align sequences (past T -> future H)
 - Save tensors to npz or similar in data/processed
"""

from pathlib import Path
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/synthetic.npz")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--time_steps", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--ww3_channels", type=int, default=3)
    ap.add_argument("--gfs_channels", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    ww3 = rng.normal(size=(args.n, args.time_steps, args.ww3_channels, args.patch, args.patch)).astype(np.float32)
    gfs = rng.normal(size=(args.n, args.time_steps, args.gfs_channels, args.patch, args.patch)).astype(np.float32)
    ww3_sig = ww3[..., 0, :, :].mean(axis=(-1, -2))
    gfs_mag = np.linalg.norm(gfs[..., :2, :, :].mean(axis=(-1, -2)), axis=-1)
    base = 0.7 * ww3_sig[:, -1] + 0.3 * gfs_mag[:, -1]
    y = np.stack([base + 0.1 * (h + 1) for h in range(args.horizon)], axis=-1).astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ww3=ww3, gfs=gfs, target=y)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
