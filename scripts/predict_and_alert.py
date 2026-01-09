#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import List
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.inference.alerting import evaluate_and_alert
from src.model.multimodal_model import MultiModalWaveForecaster, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=False, help="Path to checkpoint (optional)")
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--webhook", type=str, default=None)
    ap.add_argument("--station", type=str, default="demo")
    ap.add_argument("--time_steps", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--patch", type=int, default=9)
    args = ap.parse_args()

    cfg = ModelConfig(time_steps=args.time_steps, horizon=args.horizon, patch_size=args.patch)
    model = MultiModalWaveForecaster(cfg)
    if args.ckpt:
        model = model.load_from_checkpoint(args.ckpt)
    model.eval()

    # Create a single synthetic sample for demo
    ww3 = torch.randn(1, cfg.time_steps, cfg.ww3_channels, cfg.patch_size, cfg.patch_size)
    gfs = torch.randn(1, cfg.time_steps, cfg.gfs_channels, cfg.patch_size, cfg.patch_size)
    with torch.no_grad():
        y_hat = model({"ww3_patch": ww3, "gfs_patch": gfs}).squeeze(0).cpu().numpy().tolist()  # [H]

    event = evaluate_and_alert(
        {"station_id": args.station, "lead_hours": list(range(1, cfg.horizon + 1)), "swh": y_hat},
        threshold_m=args.threshold,
        webhook_url=args.webhook,
    )
    print(event)


if __name__ == "__main__":
    main()
