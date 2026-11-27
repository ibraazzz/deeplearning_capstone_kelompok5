"""
src/template_transformer.py
============================
Tim 05 – Penilaian Postur Duduk
TEMPLATE untuk anggota tim jalur Transformer.

Input format: (Batch, 70, 132) → (B, Time, Features)
Gunakan: get_dataloaders(format_cnn=False)

Arsitektur dasar: Positional Encoding + TransformerEncoder + ClassificationHead
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

AKAR = Path(__file__).resolve().parents[1]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))
RESULTS_DIR = AKAR / "results" / "transformer"
KELAS = ["tegak", "membungkuk", "bersandar", "condong ke kiri", "condong ke kanan"]
N_KELAS = 5


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ModelTransformer(nn.Module):
    """
    Template arsitektur Transformer untuk klasifikasi postur duduk.
    Input: (B, 70, 132) → (Batch, Time, Features)
    Output: (B, 5) logits
    """

    def __init__(
        self,
        input_size: int = 132,
        d_model: int = 128,
        n_head: int = 8,
        n_layer: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        n_kelas: int = N_KELAS,
    ):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.kepala = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_kelas)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → proj → (B, T, d_model) → encoder → mean pooling → head
        x = self.pos_enc(self.proj(x))
        x = self.encoder(x)
        feat = x.mean(dim=1)  # Global Average Pooling temporal
        return self.kepala(feat)


def latih_transformer(seed: int = 1, epoch: int = 50, lr: float = 1e-4, batch: int = 32) -> None:
    """Template training loop Transformer. Implementasikan sesuai kebutuhan."""
    torch.manual_seed(seed)
    from src.dataset_loader import get_dataloaders

    dl_tr, dl_va, dl_te, info = get_dataloaders(batch_size=batch, format_cnn=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelTransformer().to(device)

    # TODO: Implementasikan training loop
    # Referensi: src/03_train_cnn.py

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hasil_json = {
        "metrik": {"val_acc": 0.0, "val_macro_f1": 0.0, "test_acc": 0.0, "test_macro_f1": 0.0},
        "seed": seed, "perangkat": str(device), "model": "transformer",
        "n_val": info.n_val, "n_test": info.n_test, "durasi_detik": 0.0,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(hasil_json, indent=2) + "\n")
    tambahan = {
        "jalur": "transformer", "seed": seed,
        "hiper": {"epoch": epoch, "lr": lr, "batch": batch},
        "val_f1_makro": 0.0, "val_f1_rinci": {k: 0.0 for k in KELAS},
        "test_f1_makro": 0.0, "test_f1_rinci": {k: 0.0 for k in KELAS}, "riwayat": [],
    }
    (RESULTS_DIR / "tambahan.json").write_text(json.dumps(tambahan, indent=2) + "\n")
    print("[template_transformer] Implementasikan latih_transformer() secara lengkap.")


if __name__ == "__main__":
    latih_transformer()
