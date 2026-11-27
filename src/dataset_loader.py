"""
src/dataset_loader.py
=====================
Tim 05 – Penilaian Postur Duduk
Modul bersama untuk semua jalur model (CNN, RNN, Transformer, AE).

Menyediakan:
  - split_per_partisipan()  : pembagian data train/val/test per partisipan
  - get_dataloaders()       : DataLoader PyTorch siap pakai (normalized)
  - get_numpy_splits()      : array numpy mentah setelah normalisasi
  - SplitInfo               : dataclass berisi metadata split

Aturan Subject-Independent Split (seed=2026, tetap):
  - Train : P04, P05, P06, P07  (4 partisipan)
  - Val   : P02, P03             (2 partisipan)
  - Test  : P01, P08             (2 partisipan)
  Urutan ini TIDAK BOLEH DIUBAH agar hasil 4 jalur sebanding.

Skema normalisasi:
  - Fit scaler HANYA pada data Train (mencegah kebocoran).
  - Normalisasi: (X - mean_train) / std_train per fitur.

Penggunaan (anggota RNN, Transformer, AE):
  from src.dataset_loader import get_dataloaders, get_numpy_splits
  
  # Untuk model PyTorch (otomatis normalisasi):
  dl_train, dl_val, dl_test, info = get_dataloaders(batch_size=32)
  
  # Untuk baseline / sklearn:
  X_train, y_train, X_val, y_val, X_test, y_test = get_numpy_splits()
  
  # Input shape: (Batch, 132, 70)  → format (B, Channels, Time) untuk 1D-CNN
  # Input shape: (Batch, 70, 132)  → format (B, Time, Features) untuk RNN/Transformer
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

AKAR = Path(__file__).resolve().parents[1]
DATASET_NPZ = AKAR / "data" / "processed" / "dataset_kondisi.npz"
SPLIT_NPZ = AKAR / "data" / "processed" / "split_indices.npz"
LABEL_CSV = AKAR / "data" / "processed" / "label_bersih.csv"

KELAS = ["tegak", "membungkuk", "bersandar", "condong ke kiri", "condong ke kanan"]
N_KELAS = len(KELAS)

# Split partisipan tetap (subject-independent, seed=2026)
PARTISIPAN_TRAIN = ["P04", "P05", "P06", "P07"]
PARTISIPAN_VAL   = ["P02", "P03"]
PARTISIPAN_TEST  = ["P01", "P08"]

SEED = 2026


@dataclass
class SplitInfo:
    """Metadata pembagian data."""
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    partisipan_train: list = field(default_factory=lambda: PARTISIPAN_TRAIN)
    partisipan_val: list   = field(default_factory=lambda: PARTISIPAN_VAL)
    partisipan_test: list  = field(default_factory=lambda: PARTISIPAN_TEST)
    n_train: int = 0
    n_val: int   = 0
    n_test: int  = 0
    ada_rula: bool = False
    fitur_shape: tuple = ()


def split_per_partisipan(dataset_npz: Path = DATASET_NPZ) -> SplitInfo:
    """
    Buat atau muat indeks split berdasarkan partisipan.
    Partisipan TIDAK boleh terbagi antara train dan test.
    """
    import pandas as pd

    if not dataset_npz.exists():
        raise FileNotFoundError(
            f"dataset_kondisi.npz tidak ditemukan di {dataset_npz}. "
            "Jalankan dulu: python src/02_prepare_dataset.py"
        )

    # Muat label_bersih.csv untuk info partisipan
    label_csv = LABEL_CSV
    if not label_csv.exists():
        raise FileNotFoundError(f"label_bersih.csv tidak ditemukan: {label_csv}")

    df = pd.read_csv(label_csv)
    partisipan_arr = df["partisipan"].values  # ['P01', 'P01', ..., 'P08']

    # Buat indeks berdasarkan set partisipan
    idx_train = np.where(np.isin(partisipan_arr, PARTISIPAN_TRAIN))[0]
    idx_val   = np.where(np.isin(partisipan_arr, PARTISIPAN_VAL))[0]
    idx_test  = np.where(np.isin(partisipan_arr, PARTISIPAN_TEST))[0]

    # Validasi: tidak ada kebocoran antar split
    _validasi_split(partisipan_arr, idx_train, idx_val, idx_test)

    # Cek apakah ada y_rula
    with np.load(dataset_npz) as z:
        ada_rula = "y_rula" in z.files
        fitur_shape = z["X_fitur"].shape

    # Simpan split_indices.npz agar reproducible
    split_path = SPLIT_NPZ
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(split_path,
             idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
             partisipan_train=np.array(PARTISIPAN_TRAIN),
             partisipan_val=np.array(PARTISIPAN_VAL),
             partisipan_test=np.array(PARTISIPAN_TEST))

    info = SplitInfo(
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        n_train=len(idx_train), n_val=len(idx_val), n_test=len(idx_test),
        ada_rula=ada_rula, fitur_shape=fitur_shape,
    )
    return info


def _validasi_split(
    partisipan_arr: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
) -> None:
    """Pastikan tidak ada partisipan yang bocor antar split."""
    p_train = set(partisipan_arr[idx_train])
    p_val   = set(partisipan_arr[idx_val])
    p_test  = set(partisipan_arr[idx_test])

    bocor_tv = p_train & p_val
    bocor_tt = p_train & p_test
    bocor_vt = p_val & p_test

    if bocor_tv or bocor_tt or bocor_vt:
        raise AssertionError(
            f"KEBOCORAN SPLIT TERDETEKSI! "
            f"Train∩Val={bocor_tv}, Train∩Test={bocor_tt}, Val∩Test={bocor_vt}"
        )

    # Pastikan semua partisipan tercakup
    semua_p = set(PARTISIPAN_TRAIN) | set(PARTISIPAN_VAL) | set(PARTISIPAN_TEST)
    p_diketahui = p_train | p_val | p_test
    tidak_tercakup = p_diketahui - semua_p
    if tidak_tercakup:
        print(f"  [PERINGATAN] Partisipan tidak dikenal di split: {tidak_tercakup}")

    total = len(idx_train) + len(idx_val) + len(idx_test)
    print(f"  [Split Validasi OK] Train={len(idx_train)} | Val={len(idx_val)} | Test={len(idx_test)} | Total={total}")
    print(f"    Partisipan Train: {sorted(p_train)}")
    print(f"    Partisipan Val  : {sorted(p_val)}")
    print(f"    Partisipan Test : {sorted(p_test)}")


def _muat_data(dataset_npz: Path = DATASET_NPZ) -> dict:
    """Muat dataset_kondisi.npz → dict array numpy."""
    if not dataset_npz.exists():
        raise FileNotFoundError(
            f"dataset_kondisi.npz tidak ada: {dataset_npz}\n"
            "Jalankan: python src/02_prepare_dataset.py"
        )
    with np.load(dataset_npz, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    return data


def _normalisasi(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalisasi Z-score. Fit hanya pada train.
    Returns: X_train_n, X_val_n, X_test_n, mean_train, std_train
    """
    # Reshape ke 2D untuk hitung statistik
    orig_shape = X_train.shape
    X_flat = X_train.reshape(len(X_train), -1).astype(np.float32)
    mu = X_flat.mean(axis=0, keepdims=True)
    sd = X_flat.std(axis=0, keepdims=True) + 1e-7

    def norm(X):
        return ((X.reshape(len(X), -1).astype(np.float32) - mu) / sd).reshape(X.shape)

    return norm(X_train), norm(X_val), norm(X_test), mu, sd


def get_numpy_splits(
    dataset_npz: Path = DATASET_NPZ,
    mode: str = "pose",  # "pose" atau "fitur"
    format_cnn: bool = True,  # True → (B, C, T) | False → (B, T, C)
) -> tuple[np.ndarray, ...]:
    """
    Kembalikan (X_train, y_train, X_val, y_val, X_test, y_test) sebagai numpy.

    Args:
        mode: "pose" → X_pose [N, 70, 132], "fitur" → X_fitur [N, N_fitur]
        format_cnn: Jika mode="pose" dan True, transpose ke [N, 132, 70] untuk 1D-CNN.
    """
    data = _muat_data(dataset_npz)
    info = split_per_partisipan(dataset_npz)

    if mode == "pose":
        X_all = data["X_pose"]  # [N, 70, 132]
    elif mode == "fitur":
        X_all = data["X_fitur"]  # [N, N_fitur]
    else:
        raise ValueError(f"mode harus 'pose' atau 'fitur', bukan '{mode}'")

    y_all = data["y_postur"]  # [N]

    X_train = X_all[info.idx_train]
    X_val   = X_all[info.idx_val]
    X_test  = X_all[info.idx_test]

    X_train_n, X_val_n, X_test_n, _, _ = _normalisasi(X_train, X_val, X_test)

    if mode == "pose" and format_cnn:
        # Transpose: (N, 70, 132) → (N, 132, 70)  = (Batch, Channels, Time)
        X_train_n = X_train_n.transpose(0, 2, 1)
        X_val_n   = X_val_n.transpose(0, 2, 1)
        X_test_n  = X_test_n.transpose(0, 2, 1)

    y_train = y_all[info.idx_train].astype(np.int64)
    y_val   = y_all[info.idx_val].astype(np.int64)
    y_test  = y_all[info.idx_test].astype(np.int64)

    return X_train_n, y_train, X_val_n, y_val, X_test_n, y_test


def get_dataloaders(
    dataset_npz: Path = DATASET_NPZ,
    batch_size: int = 32,
    mode: str = "pose",
    format_cnn: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, SplitInfo]:
    """
    Kembalikan (dl_train, dl_val, dl_test, info) PyTorch DataLoader.
    
    Format input ke model:
      - mode="pose", format_cnn=True  → Tensor (B, 132, 70)  ← 1D-CNN (default)
      - mode="pose", format_cnn=False → Tensor (B, 70, 132)  ← RNN/Transformer
      - mode="fitur"                  → Tensor (B, N_fitur)  ← MLP/RF
    """
    X_tr, y_tr, X_va, y_va, X_te, y_te = get_numpy_splits(
        dataset_npz=dataset_npz, mode=mode, format_cnn=format_cnn
    )
    info = split_per_partisipan(dataset_npz)

    def to_dl(X, y, shuffle=False):
        ds = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.int64)),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)

    dl_train = to_dl(X_tr, y_tr, shuffle=True)
    dl_val   = to_dl(X_va, y_va, shuffle=False)
    dl_test  = to_dl(X_te, y_te, shuffle=False)

    print(f"[dataset_loader] DataLoader siap:")
    print(f"  Train : {len(X_tr)} sampel | Batch={batch_size} | Shape={X_tr.shape}")
    print(f"  Val   : {len(X_va)} sampel | Shape={X_va.shape}")
    print(f"  Test  : {len(X_te)} sampel | Shape={X_te.shape}")
    print(f"  Kelas : {KELAS}")

    return dl_train, dl_val, dl_test, info


def verifikasi_split(dataset_npz: Path = DATASET_NPZ) -> None:
    """
    Verifikasi otomatis: jalankan semua pengecekan validasi split.
    Dipanggil oleh tahap verifikasi atau test.
    """
    import pandas as pd
    print("\n[VERIFIKASI SPLIT]")

    data = _muat_data(dataset_npz)
    info = split_per_partisipan(dataset_npz)

    df = pd.read_csv(LABEL_CSV)
    partisipan_arr = df["partisipan"].values

    # 1. Tidak ada kebocoran
    _validasi_split(partisipan_arr, info.idx_train, info.idx_val, info.idx_test)

    # 2. Jumlah kondisi total (64 setelah penyelarasan k+2, atau 80 tanpa geser)
    total = info.n_train + info.n_val + info.n_test
    assert total in (64, 80), f"Total kondisi {total} tidak valid (harus 64 atau 80)"
    print(f"  [OK] Total kondisi: {total}")

    # 3. 70 frame per kondisi
    X = data["X_pose"]  # [80, 70, 132]
    assert X.shape[1] == 70, f"Frame per kondisi {X.shape[1]} ≠ 70"
    print(f"  [OK] Frame per kondisi: {X.shape[1]}")

    # 4. Tidak ada NaN/Inf
    assert not np.isnan(X).any(), "NaN di X_pose!"
    assert not np.isinf(X).any(), "Inf di X_pose!"
    print("  [OK] Tidak ada NaN/Inf")

    # 5. Distribusi kelas di train
    y = data["y_postur"]
    y_train = y[info.idx_train]
    print("  Distribusi kelas Train:")
    for c, nama in enumerate(KELAS):
        n = int((y_train == c).sum())
        print(f"    {c} ({nama:20s}): {n}")

    print("[VERIFIKASI] Semua pengecekan LOLOS ✓\n")


if __name__ == "__main__":
    # Jika dijalankan langsung: tampilkan info split
    verifikasi_split()
