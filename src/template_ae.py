"""
src/template_ae.py
==================
Tim 05 – Penilaian Postur Duduk
TEMPLATE untuk anggota tim jalur Autoencoder (AE).

Pendekatan: 1D Convolutional Autoencoder
  - Encoder mengompresi urutan pose menjadi representasi laten
  - Kepala klasifikasi dipasang pada representasi laten
  - Dapat dilatih dengan loss rekonstruksi + classification loss

Input format: (Batch, 132, 70) → (B, Channels, Time)
Gunakan: get_dataloaders(format_cnn=True)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

AKAR = Path(__file__).resolve().parents[1]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))
RESULTS_DIR = AKAR / "results" / "ae"
KELAS = ["tegak", "membungkuk", "bersandar", "condong ke kiri", "condong ke kanan"]
N_KELAS = 5


class ModelAE(nn.Module):
    """
    Template arsitektur 1D Convolutional Autoencoder untuk klasifikasi postur duduk.
    Input: (B, 132, 70) → Encoder → Laten → Decoder + KepalaKlasifikasi
    """

    def __init__(self, in_channels: int = 132, laten_dim: int = 64,
                 dropout: float = 0.3, n_kelas: int = N_KELAS):
        super().__init__()
        self.laten_dim = laten_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),  # T=35
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8),   # T=8
        )
        self.fc_laten = nn.Linear(64 * 8, laten_dim)

        # Decoder
        self.fc_decode = nn.Linear(laten_dim, 64 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # T=16
            nn.ReLU(),
            nn.ConvTranspose1d(128, in_channels, kernel_size=5, stride=4, padding=2, output_padding=1),  # T≈70
            nn.AdaptiveAvgPool1d(70),  # Pastikan output T=70
        )

        # Kepala klasifikasi
        self.kepala = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(laten_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_kelas),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                    # (B, 64, 8)
        return self.fc_laten(h.flatten(1))     # (B, laten_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_decode(z)).view(-1, 64, 8)  # (B, 64, 8)
        return self.decoder(h)                           # (B, 132, 70)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits_klasifikasi, rekonstruksi)."""
        z = self.encode(x)
        logits = self.kepala(z)
        rekon = self.decode(z)
        return logits, rekon


def latih_ae(seed: int = 1, epoch: int = 50, lr: float = 1e-3,
             batch: int = 32, lambda_rekon: float = 0.5) -> None:
    """
    Template training loop AE. Implementasikan sesuai kebutuhan.
    Loss = CrossEntropy(klasifikasi) + lambda_rekon * MSE(rekonstruksi)
    """
    torch.manual_seed(seed)
    from src.dataset_loader import get_dataloaders

    dl_tr, dl_va, dl_te, info = get_dataloaders(batch_size=batch, format_cnn=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelAE().to(device)

    # TODO: Implementasikan training loop
    # Contoh loss:
    #   logits, rekon = model(X)
    #   loss = F.cross_entropy(logits, y) + lambda_rekon * F.mse_loss(rekon, X)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hasil_json = {
        "metrik": {"val_acc": 0.0, "val_macro_f1": 0.0, "test_acc": 0.0, "test_macro_f1": 0.0},
        "seed": seed, "perangkat": str(device), "model": "1d-cnn-ae",
        "n_val": info.n_val, "n_test": info.n_test, "durasi_detik": 0.0,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(hasil_json, indent=2) + "\n")
    tambahan = {
        "jalur": "ae", "seed": seed,
        "hiper": {"epoch": epoch, "lr": lr, "batch": batch, "lambda_rekon": lambda_rekon},
        "val_f1_makro": 0.0, "val_f1_rinci": {k: 0.0 for k in KELAS},
        "test_f1_makro": 0.0, "test_f1_rinci": {k: 0.0 for k in KELAS}, "riwayat": [],
    }
    (RESULTS_DIR / "tambahan.json").write_text(json.dumps(tambahan, indent=2) + "\n")
    print("[template_ae] Implementasikan latih_ae() secara lengkap.")


if __name__ == "__main__":
    latih_ae()
