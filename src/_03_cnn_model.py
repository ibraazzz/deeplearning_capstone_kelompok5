"""
src/_03_cnn_model.py
====================
Definisi arsitektur CNN1D yang dapat diimport oleh modul lain.
Diekstrak dari 03_train_cnn.py agar dapat digunakan di 05_evaluate_cnn.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_KELAS = 5


class Block1DCNN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, pool: int = 2, dropout: float = 0.3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


class CNN1D(nn.Module):
    """
    1D-CNN untuk klasifikasi postur duduk (opsional kepala regresi RULA).
    Input: (Batch, 132, 70)  → Channels=132, Time=70
    """
    def __init__(self, n_kelas: int = N_KELAS, in_channels: int = 132,
                 ada_rula: bool = False, dropout: float = 0.3):
        super().__init__()
        self.ada_rula = ada_rula
        self.backbone = nn.Sequential(
            Block1DCNN(in_channels, 128, kernel=5, pool=2, dropout=dropout),
            Block1DCNN(128, 256, kernel=5, pool=2, dropout=dropout),
            Block1DCNN(256, 256, kernel=3, pool=1, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.kepala_klasifikasi = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_kelas),
        )
        if ada_rula:
            self.kepala_rula = nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1),
            )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.backbone(x).squeeze(-1)
        logits = self.kepala_klasifikasi(feat)
        if self.ada_rula:
            return logits, self.kepala_rula(feat).squeeze(-1)
        return logits
