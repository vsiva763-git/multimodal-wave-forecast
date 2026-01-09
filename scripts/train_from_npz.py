#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytorch_lightning as pl

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model.multimodal_model import MultiModalWaveForecaster, ModelConfig
from src.model.npz_data_module import NPZDataModule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to npz produced by prepare_nrt_buoy_patches.py")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--time_steps", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--patch", type=int, default=9)
    args = ap.parse_args()

    dm = NPZDataModule(args.npz, batch_size=args.batch)
    model_cfg = ModelConfig(time_steps=args.time_steps, horizon=args.horizon, patch_size=args.patch)
    model = MultiModalWaveForecaster(model_cfg)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
