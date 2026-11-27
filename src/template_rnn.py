"""
src/template_rnn.py
===================
Tim 05 – Penilaian Postur Duduk
TEMPLATE untuk anggota tim jalur RNN (LSTM/GRU).

Salin file ini sebagai titik awal, lalu implementasikan model RNN Anda.
Lihat dataset_loader.py untuk cara memuat data.

Input format untuk RNN: (Batch, 70, 132)  → (B, Time, Features)
Gunakan: get_dataloaders(format_cnn=False)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AKAR = Path(__file__).resolve().parents[1]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))
RESULTS_DIR = AKAR / "results" / "rnn"
KELAS = ["tegak", "membungkuk", "bersandar", "condong ke kiri", "condong ke kanan"]
N_KELAS = 5


class ModelRNN(nn.Module):
    """
    Template arsitektur RNN (LSTM bidirectional).
    Ganti sesuai kebutuhan eksperimen Anda.
    
    Input: (B, 70, 132)  → (Batch, Time, Features)
    Output: (B, 5)        logits klasifikasi
    """

    def __init__(self, input_size: int = 132, hidden: int = 128,
                 n_layer: int = 2, dropout: float = 0.3, n_kelas: int = N_KELAS):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=n_layer,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layer > 1 else 0.0,
        )
        self.kepala = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 64),  # *2 karena bidirectional
            nn.ReLU(),
            nn.Linear(64, n_kelas),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.rnn(x)
        # Ambil output frame terakhir
        feat = out[:, -1, :]  # (B, hidden*2)
        return self.kepala(feat)


def latih_rnn(seed: int = 1, epoch: int = 50, lr: float = 1e-3, batch: int = 32) -> None:
    """Latih dan simpan model RNN. Implementasikan sesuai kebutuhan."""
    torch.manual_seed(seed)
    from src.dataset_loader import get_dataloaders

    # format_cnn=False → input (B, 70, 132) untuk RNN
    dl_tr, dl_va, dl_te, info = get_dataloaders(batch_size=batch, format_cnn=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelRNN().to(device)

    # TODO: Implementasikan training loop
    # Lihat src/03_train_cnn.py sebagai referensi

    # Format output yang WAJIB diikuti:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hasil_json = {
        "metrik": {
            "val_acc": 0.0,         # ganti dengan nilai aktual
            "val_macro_f1": 0.0,    # ganti dengan nilai aktual
            "test_acc": 0.0,
            "test_macro_f1": 0.0,
        },
        "seed": seed,
        "perangkat": str(device),
        "model": "rnn-lstm",
        "n_val": info.n_val,
        "n_test": info.n_test,
        "durasi_detik": 0.0,
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(hasil_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tambahan = {
        "jalur": "rnn",
        "seed": seed,
        "hiper": {"epoch": epoch, "lr": lr, "batch": batch},
        "val_f1_makro": 0.0,       # ganti
        "val_f1_rinci": {k: 0.0 for k in KELAS},  # ganti
        "test_f1_makro": 0.0,
        "test_f1_rinci": {k: 0.0 for k in KELAS},
        "riwayat": [],             # daftar {"epoch":e, "loss":L, "val_f1":F}
    }
    (RESULTS_DIR / "tambahan.json").write_text(
        json.dumps(tambahan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[template_rnn] Template selesai. Implementasikan latih_rnn() secara lengkap.")


if __name__ == "__main__":
    latih_rnn()
