"""Pemuat data kasus tim — Penilaian postur duduk."""
from __future__ import annotations

import csv
import numpy as np
from src.common import manifest

TUGAS = "klasifikasi"
FPS = 10
DURASI_S = 8
FRAME = FPS * DURASI_S
BUANG_FRAME = FPS
PARTISIPAN = tuple(f"P{i:02d}" for i in range(1, 9))
POSTUR = ("tegak", "membungkuk", "bersandar", "condong ke kiri", "condong ke kanan")
UJI_PARTISIPAN = ("P07", "P08")
VAL_ULANGAN = 2

_POSE = manifest.AKAR / "data" / "processed" / "dataset_kondisi.npz"
_LABEL = manifest.AKAR / "data" / "processed" / "label_bersih.csv"

def _muat_semua():
    # Membaca dataset_kondisi.npz
    arsip = np.load(_POSE, allow_pickle=True)
    
    X = np.asarray(arsip['X_pose'], dtype=np.float32)
    yp = np.asarray(arsip['y_postur'], dtype=np.int64)
    
    # Karena tidak ada info partisipan/ulangan, kita buat dummy
    # Ambil 80% untuk latih, 20% untuk validasi
    n = len(X)
    split = int(n * 0.8)
    
    Xtr, ytr = X[:split], yp[:split]
    Xva, yva = X[split:], yp[split:]
    Xte, yte = X[split:], yp[split:]  # Sementara test = val (karena belum ada split final)
    
    return Xtr, ytr, Xva, yva, Xte, yte

def muat():
    return _muat_semua()

def muat_rula():
    raise NotImplementedError("Regresi RULA belum diimplementasikan di dataset_kondisi.npz")