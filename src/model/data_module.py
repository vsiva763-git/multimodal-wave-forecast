from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


@dataclass
class DMConfig:
    time_steps: int = 12
    horizon: int = 6
    ww3_channels: int = 3
    gfs_channels: int = 3
    patch_size: int = 9
    batch_size: int = 16
    num_workers: int = 2
    train_size: int = 512
    val_size: int = 128


class SyntheticWaveDataset(Dataset):
    def __init__(self, n_samples: int, cfg: DMConfig, seed: int = 42):
        self.n = n_samples
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.ww3 = rng.normal(size=(n_samples, cfg.time_steps, cfg.ww3_channels, cfg.patch_size, cfg.patch_size)).astype(np.float32)
        self.gfs = rng.normal(size=(n_samples, cfg.time_steps, cfg.gfs_channels, cfg.patch_size, cfg.patch_size)).astype(np.float32)
        # Create a target that depends on sums/means over recent inputs to simulate predictability
        ww3_sig = self.ww3[..., 0, :, :].mean(axis=(-1, -2))  # use channel 0 as proxy for swh
        gfs_mag = np.linalg.norm(self.gfs[..., :2, :, :].mean(axis=(-1, -2)), axis=-1)  # wind magnitude
        base = 0.7 * ww3_sig[:, -1] + 0.3 * gfs_mag[:, -1]
        targets = np.stack([base + 0.1 * (h + 1) for h in range(cfg.horizon)], axis=-1).astype(np.float32)
        self.targets = targets

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {
            "ww3_patch": torch.from_numpy(self.ww3[idx]),
            "gfs_patch": torch.from_numpy(self.gfs[idx]),
            "target_swh": torch.from_numpy(self.targets[idx]),
        }


class WaveDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DMConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = SyntheticWaveDataset(self.cfg.train_size, self.cfg)
        self.val_ds = SyntheticWaveDataset(self.cfg.val_size, self.cfg, seed=123)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
