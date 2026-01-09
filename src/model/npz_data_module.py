from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class NPZWaveDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.ww3 = data["ww3"].astype(np.float32)
        self.gfs = data["gfs"].astype(np.float32)
        self.targets = data["target"].astype(np.float32)
        assert self.ww3.shape[0] == self.gfs.shape[0] == self.targets.shape[0]

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        return {
            "ww3_patch": torch.from_numpy(self.ww3[idx]),
            "gfs_patch": torch.from_numpy(self.gfs[idx]),
            "target_swh": torch.from_numpy(self.targets[idx]),
        }


class NPZDataModule(pl.LightningDataModule):
    def __init__(self, npz_path: str, batch_size: int = 16, num_workers: int = 2, val_fraction: float = 0.1, seed: int = 42):
        super().__init__()
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed

    def setup(self, stage: Optional[str] = None) -> None:
        full = NPZWaveDataset(self.npz_path)
        n_val = max(1, int(len(full) * self.val_fraction))
        n_train = len(full) - n_val
        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds = random_split(full, [n_train, n_val], generator=generator)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
