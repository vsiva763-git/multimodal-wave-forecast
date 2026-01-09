from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .components import SpatialCNN, TransformerFusion, TemporalLSTM


@dataclass
class ModelConfig:
    ww3_channels: int = 3  # e.g., swh, mwp, mwd (encoded suitably)
    gfs_channels: int = 3  # e.g., 10u, 10v, prmsl (normalized)
    patch_size: int = 9
    time_steps: int = 12
    horizon: int = 6
    cnn_dim: int = 128
    fusion_dim: int = 256
    lstm_hidden: int = 256
    lr: float = 1e-3


class MultiModalWaveForecaster(pl.LightningModule):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.save_hyperparameters(cfg.__dict__)
        self.cfg = cfg
        # Two CNNs for spatial encoding per modality
        self.cnn_ww3 = SpatialCNN(cfg.ww3_channels, cfg.cnn_dim)
        self.cnn_gfs = SpatialCNN(cfg.gfs_channels, cfg.cnn_dim)
        # Project concatenated embeddings to fusion dim
        self.proj = nn.Linear(cfg.cnn_dim * 2, cfg.fusion_dim)
        self.fuser = TransformerFusion(d_model=cfg.fusion_dim)
        self.temporal = TemporalLSTM(cfg.fusion_dim, cfg.lstm_hidden)
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden, cfg.lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.lstm_hidden, cfg.horizon),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Inputs: ww3_patch [B,T,Cw,H,W], gfs_patch [B,T,Cg,H,W]
        ww3 = batch["ww3_patch"]
        gfs = batch["gfs_patch"]
        B, T, Cw, H, W = ww3.shape
        _, _, Cg, _, _ = gfs.shape
        ww3_flat = ww3.reshape(B * T, Cw, H, W)
        gfs_flat = gfs.reshape(B * T, Cg, H, W)
        e_ww3 = self.cnn_ww3(ww3_flat)  # [B*T, E]
        e_gfs = self.cnn_gfs(gfs_flat)  # [B*T, E]
        e = torch.cat([e_ww3, e_gfs], dim=-1).reshape(B, T, -1)
        e = self.proj(e)
        f = self.fuser(e)
        h = self.temporal(f)  # [B, hidden]
        y = self.head(h)  # [B, horizon]
        return y

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        y_hat = self.forward(batch)
        y = batch["target_swh"]  # [B, horizon]
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        y_hat = self.forward(batch)
        y = batch["target_swh"]
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
