"""
src/03_train_cnn.py
===================
Tim 05 – Penilaian Postur Duduk
Tahap 3: Pelatihan model 1D-CNN (jalur CNN – anggota saya).

Arsitektur 1D-CNN:
  Input: (Batch, 132, 70)  → Channels=132 (33 landmark × 4 nilai), Time=70 frame
  
  Block 1: Conv1d(132→128, k=5, pad=2) → BN → ReLU → MaxPool(2) → Dropout
  Block 2: Conv1d(128→256, k=5, pad=2) → BN → ReLU → MaxPool(2) → Dropout
  Block 3: Conv1d(256→256, k=3, pad=1) → BN → ReLU → AdaptiveAvgPool(1)
  
  Multi-task head:
    - Klasifikasi: Linear(256→128) → ReLU → Dropout → Linear(128→5)  [5 postur]
    - Regresi RULA: Linear(256→64) → ReLU → Linear(64→1)  [opsional, jika rula.csv ada]

Metrik utama:
  - Klasifikasi: F1-Score Macro (val_f1_makro)
  - Regresi: MAE skor RULA (jika tersedia)

Output:
  - results/cnn/best_cnn.pt          bobot model terbaik (max val_f1)
  - results/cnn/training_history.json riwayat pelatihan
  - results/cnn/metrics.json          metrik evaluasi (kompatibel format sprint-1)
  - results/cnn/tambahan.json         info detail (hiper, arsitektur, riwayat)

Penggunaan:
  python src/03_train_cnn.py
  python src/03_train_cnn.py --epoch 50 --lr 0.001 --batch 32 --seed 1
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

AKAR = Path(__file__).resolve().parents[1]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))
DATASET_NPZ = AKAR / "data" / "processed" / "dataset_kondisi.npz"
RESULTS_DIR = AKAR / "results" / "cnn"

KELAS = ["tegak", "membungkuk", "bersandar", "condong ke kiri", "condong ke kanan"]
N_KELAS = 5


# ─────────────────────────── Arsitektur ────────────────────────────────────

class Block1DCNN(nn.Module):
    """Blok konvolusi 1D dengan BN + ReLU + MaxPool + Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, pool: int = 2, dropout: float = 0.3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


class CNN1D(nn.Module):
    """
    1D-CNN untuk klasifikasi postur duduk (dan opsional regresi RULA).

    Args:
        n_kelas    : jumlah kelas postur (default=5)
        in_channels: jumlah channel input = 33 landmark × 4 nilai = 132
        ada_rula   : jika True, tambahkan kepala regresi RULA
        dropout    : tingkat dropout
    """

    def __init__(
        self,
        n_kelas: int = N_KELAS,
        in_channels: int = 132,
        ada_rula: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.ada_rula = ada_rula

        # Backbone
        self.backbone = nn.Sequential(
            Block1DCNN(in_channels, 128, kernel=5, pool=2, dropout=dropout),
            Block1DCNN(128, 256, kernel=5, pool=2, dropout=dropout),
            Block1DCNN(256, 256, kernel=3, pool=1, dropout=dropout),  # pool=1: tanpa pooling
            nn.AdaptiveAvgPool1d(1),  # → [B, 256, 1]
        )

        # Kepala klasifikasi (5 postur)
        self.kepala_klasifikasi = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_kelas),
        )

        # Kepala regresi RULA (opsional)
        if ada_rula:
            self.kepala_rula = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor (B, 132, 70) – Channels, Time
        Returns:
            logits (B, 5) atau (logits, rula_pred) jika ada_rula
        """
        feat = self.backbone(x).squeeze(-1)  # [B, 256]
        logits = self.kepala_klasifikasi(feat)  # [B, 5]

        if self.ada_rula:
            rula_pred = self.kepala_rula(feat).squeeze(-1)  # [B]
            return logits, rula_pred

        return logits


# ─────────────────────────── Metrik ────────────────────────────────────────

def hitung_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1_per_kelas = []
    for c in range(N_KELAS):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            f1_per_kelas.append(0.0)
        else:
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            f1_per_kelas.append(f1)
    return float(np.mean(f1_per_kelas)), f1_per_kelas


def hitung_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true.astype(float) - y_pred.astype(float)).mean())


# ─────────────────────────── Training Loop ─────────────────────────────────

def latih_satu_epoch(
    model: CNN1D,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bobot_kelas: torch.Tensor | None,
    ada_rula: bool,
    lambda_rula: float = 0.1,
) -> float:
    model.train()
    total_loss = 0.0
    n_total = 0

    for batch in loader:
        if ada_rula:
            X, y_postur, y_rula = [b.to(device) for b in batch]
        else:
            X, y_postur = [b.to(device) for b in batch]

        optimizer.zero_grad()

        if ada_rula:
            logits, rula_pred = model(X)
            loss_cls = F.cross_entropy(logits, y_postur, weight=bobot_kelas)
            loss_reg = F.l1_loss(rula_pred, y_rula.float())
            loss = loss_cls + lambda_rula * loss_reg
        else:
            logits = model(X)
            loss = F.cross_entropy(logits, y_postur, weight=bobot_kelas)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(X)
        n_total += len(X)

    return total_loss / n_total


@torch.no_grad()
def evaluasi(
    model: CNN1D,
    loader,
    device: torch.device,
    ada_rula: bool,
) -> dict:
    model.eval()
    all_pred = []
    all_true = []
    all_pred_rula = []
    all_true_rula = []

    for batch in loader:
        if ada_rula:
            X, y_postur, y_rula = [b.to(device) for b in batch]
        else:
            X, y_postur = [b.to(device) for b in batch]

        if ada_rula:
            logits, rula_pred = model(X)
            all_pred_rula.extend(rula_pred.cpu().numpy().tolist())
            all_true_rula.extend(y_rula.cpu().numpy().tolist())
        else:
            logits = model(X)

        pred = logits.argmax(dim=1)
        all_pred.extend(pred.cpu().numpy().tolist())
        all_true.extend(y_postur.cpu().numpy().tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    f1_macro, f1_per_kelas = hitung_macro_f1(y_true, y_pred)
    acc = float((y_true == y_pred).mean())

    hasil = {
        "val_f1_makro": round(f1_macro, 4),
        "val_acc": round(acc, 4),
        "f1_per_kelas": {k: round(f, 4) for k, f in zip(KELAS, f1_per_kelas)},
    }

    if ada_rula and all_pred_rula:
        mae = hitung_mae(np.array(all_true_rula), np.array(all_pred_rula))
        hasil["val_mae_rula"] = round(mae, 4)

    return hasil


# ─────────────────────────── Dataloader ────────────────────────────────────

def buat_loader(
    X: np.ndarray,
    y_postur: np.ndarray,
    y_rula: np.ndarray | None = None,
    batch_size: int = 32,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    tensors = [
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y_postur.astype(np.int64)),
    ]
    if y_rula is not None:
        tensors.append(torch.from_numpy(y_rula.astype(np.float32)))

    ds = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ─────────────────────────── Main Training ─────────────────────────────────

def latih(
    seed: int = 1,
    epoch: int = 50,
    lr: float = 1e-3,
    batch: int = 32,
    wd: float = 1e-4,
    dropout: float = 0.3,
    patience: int = 15,  # early stopping patience
    lambda_rula: float = 0.1,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[03_train_cnn] Device: {device} | Seed: {seed}")

    # ── Muat data ──
    from src.dataset_loader import get_numpy_splits
    import pandas as pd

    X_tr, y_tr, X_va, y_va, X_te, y_te = get_numpy_splits(mode="pose", format_cnn=True)
    print(f"  Shape Train: {X_tr.shape} | Val: {X_va.shape} | Test: {X_te.shape}")

    # Cek RULA
    with np.load(DATASET_NPZ, allow_pickle=True) as z:
        ada_rula = "y_rula" in z.files
        if ada_rula:
            y_rula_all = z["y_rula"]
            # Ambil split yang sesuai
            from src.dataset_loader import split_per_partisipan
            info = split_per_partisipan()
            y_rula_tr = y_rula_all[info.idx_train]
            y_rula_va = y_rula_all[info.idx_val]
            y_rula_te = y_rula_all[info.idx_test]
            # Ganti NaN dengan rata-rata
            mean_rula = np.nanmean(y_rula_tr)
            y_rula_tr = np.where(np.isnan(y_rula_tr), mean_rula, y_rula_tr)
            y_rula_va = np.where(np.isnan(y_rula_va), mean_rula, y_rula_va)
            y_rula_te = np.where(np.isnan(y_rula_te), mean_rula, y_rula_te)
            print(f"  RULA tersedia: rata-rata train = {mean_rula:.2f}")
        else:
            y_rula_tr = y_rula_va = y_rula_te = None
            print("  RULA tidak tersedia → hanya klasifikasi 5 postur")

    # ── Bobot kelas (menangani imbalance) ──
    n_kelas = np.bincount(y_tr, minlength=N_KELAS).astype(np.float32)
    bobot_kelas = torch.tensor(
        n_kelas.sum() / (N_KELAS * np.maximum(n_kelas, 1)),
        dtype=torch.float32,
    ).to(device)
    print(f"  Bobot kelas: {[f'{w:.3f}' for w in bobot_kelas.cpu().numpy()]}")

    # ── DataLoaders ──
    loader_tr = buat_loader(X_tr, y_tr, y_rula_tr, batch_size=batch, shuffle=True)
    loader_va = buat_loader(X_va, y_va, y_rula_va, batch_size=batch, shuffle=False)
    loader_te = buat_loader(X_te, y_te, y_rula_te, batch_size=batch, shuffle=False)

    # ── Model ──
    model = CNN1D(n_kelas=N_KELAS, in_channels=132, ada_rula=ada_rula, dropout=dropout)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameter model: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch)

    # ── Training loop dengan early stopping ──
    riwayat = []
    terbaik = {"val_f1": -1.0, "epoch": 0, "bobot": None}
    t_mulai = time.perf_counter()

    for e in range(1, epoch + 1):
        loss = latih_satu_epoch(model, loader_tr, optimizer, device,
                                bobot_kelas, ada_rula, lambda_rula)
        scheduler.step()
        metrik_val = evaluasi(model, loader_va, device, ada_rula)
        f1_val = metrik_val["val_f1_makro"]
        acc_val = metrik_val["val_acc"]

        riwayat.append({
            "epoch": e,
            "loss": round(loss, 4),
            "val_f1": f1_val,
            "val_acc": acc_val,
        })

        if f1_val > terbaik["val_f1"]:
            terbaik["val_f1"] = f1_val
            terbaik["epoch"] = e
            terbaik["bobot"] = copy.deepcopy(model.state_dict())

        print(f"  [Epoch {e:3d}/{epoch}] loss={loss:.4f} | val_f1={f1_val:.4f} | "
              f"val_acc={acc_val:.4f} {'◀ TERBAIK' if e == terbaik['epoch'] else ''}")

        # Early stopping
        if e - terbaik["epoch"] >= patience:
            print(f"  Early stopping di epoch {e} (tidak ada perbaikan {patience} epoch)")
            break

    durasi = time.perf_counter() - t_mulai
    print(f"\n[03_train_cnn] Pelatihan selesai dalam {durasi:.1f} detik")
    print(f"  Epoch terbaik  : {terbaik['epoch']} (val_f1={terbaik['val_f1']:.4f})")

    # ── Simpan bobot terbaik ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path_bobot = RESULTS_DIR / "best_cnn.pt"
    model.load_state_dict(terbaik["bobot"])
    torch.save({
        "state_dict": terbaik["bobot"],
        "hyperparams": {"epoch": epoch, "lr": lr, "batch": batch, "wd": wd, "dropout": dropout},
        "n_params": n_params,
        "ada_rula": ada_rula,
    }, path_bobot)
    print(f"  Bobot tersimpan: {path_bobot}")

    # ── Evaluasi Test Set ──
    metrik_test = evaluasi(model, loader_te, device, ada_rula)
    metrik_val_final = evaluasi(model, loader_va, device, ada_rula)

    print(f"\n[Evaluasi Test]")
    print(f"  Test F1-Macro  : {metrik_test.get('val_f1_makro', '—')}")
    print(f"  Test Accuracy  : {metrik_test.get('val_acc', '—')}")
    if ada_rula:
        print(f"  Test MAE RULA  : {metrik_test.get('val_mae_rula', '—')}")

    # ── Simpan metrics.json ──
    metrik_out = {
        "val_acc": metrik_val_final["val_acc"],
        "val_macro_f1": metrik_val_final["val_f1_makro"],
        "test_acc": metrik_test["val_acc"],
        "test_macro_f1": metrik_test["val_f1_makro"],
    }
    if ada_rula:
        metrik_out["val_mae_rula"] = metrik_val_final.get("val_mae_rula", None)
        metrik_out["test_mae_rula"] = metrik_test.get("val_mae_rula", None)

    hasil_json = {
        "metrik": {k: round(float(v), 6) for k, v in metrik_out.items() if v is not None},
        "seed": seed,
        "perangkat": str(device),
        "model": "1d-cnn",
        "n_val": len(y_va),
        "n_test": len(y_te),
        "durasi_detik": round(durasi, 1),
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(hasil_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # ── Simpan tambahan.json ──
    tambahan = {
        "jalur": "cnn",
        "seed": seed,
        "hiper": {"epoch": epoch, "lr": lr, "batch": batch, "wd": wd, "dropout": dropout},
        "parameter": n_params,
        "ada_rula": ada_rula,
        "epoch_terbaik": terbaik["epoch"],
        "val_f1_makro": metrik_val_final["val_f1_makro"],
        "val_f1_rinci": metrik_val_final["f1_per_kelas"],
        "test_f1_makro": metrik_test["val_f1_makro"],
        "test_f1_rinci": metrik_test["f1_per_kelas"],
        "riwayat": riwayat,
        "arsitektur": str(model),
    }
    (RESULTS_DIR / "tambahan.json").write_text(
        json.dumps(tambahan, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    # ── Simpan riwayat pelatihan ──
    (RESULTS_DIR / "training_history.json").write_text(
        json.dumps(riwayat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\n[03_train_cnn] Semua hasil tersimpan di: {RESULTS_DIR}")
    print(f"  metrics.json | tambahan.json | training_history.json | best_cnn.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lambda-rula", type=float, default=0.1)
    args = ap.parse_args()

    latih(
        seed=args.seed,
        epoch=args.epoch,
        lr=args.lr,
        batch=args.batch,
        wd=args.wd,
        dropout=args.dropout,
        patience=args.patience,
        lambda_rula=args.lambda_rula,
    )


if __name__ == "__main__":
    main()
