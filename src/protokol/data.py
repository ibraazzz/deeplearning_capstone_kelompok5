"""Pemuat data kasus tim — Penilaian postur duduk. MILIK TIM, dibekukan setelah Sprint 0.

Kasus: klasifikasi 5 postur + regresi skor RULA dari urutan pose 10 fps, 8 detik,
direkam tim sendiri di meja uji (kamera tiap 0.1 s).

Format data (disepakati Sprint 0; operator rekaman 6 Nov menghasilkan persis ini):

- ``data/pose.npz`` — satu kunci per kondisi: ``<partisipan>__<postur>__<ulangan>``,
  contoh ``P01__tegak__1``. Isinya array NumPy ``[80, J, 3]`` float32: 80 frame
  (10 fps × 8 s), J titik pose, koordinat (x, y, z). Frame DISIMPAN lengkap;
  pembuangan 1 detik pertama dilakukan di sini, bukan saat merekam.
- ``data/label.csv`` — kolom: ``kondisi, partisipan, postur, ulangan, y_postur, rula_1, rula_2``.
  ``kondisi`` = kunci di pose.npz. ``postur`` = slug dari POSTUR di bawah.
  ``y_postur`` bulat 0–4. ``rula_1``, ``rula_2`` = skor RULA penilai 1 dan 2 (A dan D);
  skor efektif = rerata keduanya.

Split beku berbasis grup (pengganti ``split.indeks``; setara GroupKFold per sesi/blok/
partisipan dari kartu kasus; disetujui pembimbing pada PR Sprint 0):

- Uji (hold-out partisipan): seluruh kondisi P07 dan P08.
- Validasi (blok): ulangan 2 dari P01–P06.
- Latih: ulangan 1 dari P01–P06.

Tanpa pengacakan → hasil split identik di semua laptop; seed 2026 tetap masuk
``data_hash`` lewat penilai starter. ``src.common.split`` tidak dipakai karena kasus ini
mewajibkan pemisahan per partisipan/blok, bukan pengacakan jendela.

Aturan praproses: normalisasi TIDAK dilakukan di sini. Tiap model mem-fitting parameter
normalisasi dari data latih saja (lihat PROTOKOL.md).

Kontrak dua tugas: ``TUGAS = "klasifikasi"`` (utama) — ``muat()`` mengembalikan label
postur dan dinilai penilai starter (acc, macro-F1). Regresi RULA sekunder —
``muat_rula()`` mengembalikan split yang sama dengan y = skor RULA; prediksinya
dinilai ke ``results/<nim>/metrics_rula.json`` lewat ``evaluate --out``.
"""
from __future__ import annotations

import csv

import numpy as np

from src.common import manifest

TUGAS = "klasifikasi"

FPS = 10
DURASI_S = 8
FRAME = FPS * DURASI_S      # 80 frame per kondisi
BUANG_FRAME = FPS           # 1 detik pertama tiap kondisi dibuang → 70 frame dipakai
PARTISIPAN = tuple(f"P{i:02d}" for i in range(1, 9))
POSTUR = ("tegak", "membungkuk", "bersandar", "condong-kiri", "condong-kanan")
UJI_PARTISIPAN = ("P07", "P08")   # data uji dipisah per partisipan
VAL_ULANGAN = 2                   # validasi dipisah per blok ulangan
_POSE = manifest.AKAR / "data" / "pose.npz"
_LABEL = manifest.AKAR / "data" / "label.csv"


def _muat_semua():
    """Semua kondisi, urut sesuai label.csv: (X, y_postur, y_rula, partisipan, ulangan)."""
    label = list(csv.DictReader(_LABEL.read_text("utf-8").splitlines()))
    if not label:
        raise SystemExit("data/label.csv kosong; isi sesuai format di docstring modul ini.")
    arsip = np.load(_POSE)
    xs, ys_postur, ys_rula, part, ul = [], [], [], [], []
    for baris in label:
        kunci = baris["kondisi"].strip()
        if kunci not in arsip:
            raise SystemExit(f"data/pose.npz tidak punya kondisi {kunci}")
        X = np.asarray(arsip[kunci], dtype=np.float32)
        if X.ndim != 3 or X.shape[0] != FRAME or X.shape[2] != 3:
            raise SystemExit(f"{kunci}: bentuk {X.shape}, seharusnya ({FRAME}, J, 3)")
        p, u = baris["partisipan"].strip(), int(baris["ulangan"])
        slug = baris["postur"].strip()
        if p not in PARTISIPAN:
            raise SystemExit(f"{kunci}: partisipan {p} di luar {PARTISIPAN}")
        if slug not in POSTUR:
            raise SystemExit(f"{kunci}: postur {slug} di luar {POSTUR}")
        yp = int(baris["y_postur"])
        if yp != POSTUR.index(slug):
            raise SystemExit(f"{kunci}: y_postur {yp} tidak cocok dengan postur {slug}")
        if u not in (1, 2):
            raise SystemExit(f"{kunci}: ulangan {u} di luar 1..2")
        r1, r2 = float(baris["rula_1"]), float(baris["rula_2"])
        xs.append(X[BUANG_FRAME:])               # buang 1 detik pertama
        ys_postur.append(yp)
        ys_rula.append((r1 + r2) / 2.0)
        part.append(p)
        ul.append(u)
    return (np.stack(xs), np.array(ys_postur, dtype=np.int64),
            np.array(ys_rula, dtype=np.float32), part, ul)


def _indeks_grup(part, ul):
    """Mask boolean (latih, validasi, uji) menurut aturan grup beku."""
    part = np.asarray(part)
    ul = np.asarray(ul)
    uji = np.isin(part, UJI_PARTISIPAN)
    va = (~uji) & (ul == VAL_ULANGAN)
    tr = ~(uji | va)
    if not (tr.any() and va.any() and uji.any()):
        raise SystemExit("split grup menghasilkan bagian kosong; periksa label.csv.")
    return tr, va, uji


def muat():
    """Split klasifikasi postur: (X_tr, y_tr, X_va, y_va, X_uji, y_uji)."""
    X, yp, _, part, ul = _muat_semua()
    tr, va, uji = _indeks_grup(part, ul)
    return X[tr], yp[tr], X[va], yp[va], X[uji], yp[uji]


def muat_rula():
    """Split regresi RULA (bagian yang sama, label berbeda): seperti muat(), y = skor RULA."""
    X, _, yr, part, ul = _muat_semua()
    tr, va, uji = _indeks_grup(part, ul)
    return X[tr], yr[tr], X[va], yr[va], X[uji], yr[uji]
