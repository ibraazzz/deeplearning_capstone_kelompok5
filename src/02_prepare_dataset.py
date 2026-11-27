"""
src/02_prepare_dataset.py
=========================
Tim 05 – Penilaian Postur Duduk
Tahap 2: Pemotongan kondisi, pembuangan 1 detik pertama, rekayasa fitur.

Input:
  - data/processed/pose.npz   (dari 01_extract_pose.py)
  - data_label/sesi-20261117-143626/label.csv
  - data_label/sesi-20261117-143626/rula.csv  (opsional)

Output:
  - data/processed/dataset_kondisi.npz   (X_pose, X_fitur, y_postur, [y_rula], meta)
  - data/processed/label_bersih.csv      (80 kondisi, termasuk kolom frame_valid)

Spesifikasi:
  - 10 FPS (interval ~100ms), 8 detik/kondisi → 80 frame/kondisi teoritis
  - Buang 1 detik pertama (10 frame awal) → 70 frame valid per kondisi
  - X_pose  : float32 [80_kondisi, 70, 33*4] = [80, 70, 132]  (urutan frame × landmark*4)
  - X_fitur : float32 [80_kondisi, N_fitur]   (fitur statistik untuk baseline RF)
  - y_postur: int64   [80_kondisi]            (label kelas 0-4)
  - y_rula  : float32 [80_kondisi]            (rata-rata 2 penilai; hanya jika rula.csv ada)

Rekayasa fitur (untuk baseline Random Forest & analisis):
  Sudut sendi utama (6 sudut):
    - Siku kiri/kanan (shoulder-elbow-wrist)
    - Bahu kiri/kanan (hip-shoulder-elbow)
    - Pinggul kiri/kanan (shoulder-hip-knee)
  Kecepatan titik pose (kecepatan delta frame untuk 33 landmark x,y):
    - mean, std, max kecepatan per landmark
  Statistik posisi (mean, std per landmark x,y per kondisi):
  Total fitur ≈ 6 sudut × 4 stat + 33*2 kecepatan × 3 stat + 33*2 posisi × 2 stat =
              24 + 198 + 132 = 354 fitur
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AKAR = Path(__file__).resolve().parents[1]
POSE_NPZ_DEFAULT = AKAR / "data" / "processed" / "pose.npz"
LABEL_CSV_DEFAULT = AKAR / "data_label" / "sesi-20261117-143626" / "label.csv"
RULA_CSV_DEFAULT = AKAR / "data_label" / "sesi-20261117-143626" / "rula.csv"
OUTPUT_NPZ_DEFAULT = AKAR / "data" / "processed" / "dataset_kondisi.npz"
OUTPUT_LABEL_CSV_DEFAULT = AKAR / "data" / "processed" / "label_bersih.csv"

N_FRAME_KONDISI = 80   # 8 detik × 10 FPS (teoritis)
N_BUANG = 10           # 1 detik pertama (fase transisi)
N_VALID = 70           # frame valid yang tersisa
FPS = 10.0
GESER_KONDISI = 2      # Penyelarasan label isyarat (PROTOKOL.md): postur kondisi k terlihat di kondisi k+2

KELAS = ["tegak", "membungkuk", "bersandar", "condong ke kiri", "condong ke kanan"]
KELAS_IDX = {k: i for i, k in enumerate(KELAS)}

# Pasangan titik untuk perhitungan sudut sendi (MediaPipe Pose landmark index)
# Format: (a, vertex, b) → sudut di titik vertex
SUDUT_SENDI = {
    "siku_kiri":    (11, 13, 15),   # shoulder_L - elbow_L - wrist_L
    "siku_kanan":   (12, 14, 16),   # shoulder_R - elbow_R - wrist_R
    "bahu_kiri":    (23, 11, 13),   # hip_L - shoulder_L - elbow_L
    "bahu_kanan":   (24, 12, 14),   # hip_R - shoulder_R - elbow_R
    "pinggul_kiri": (11, 23, 25),   # shoulder_L - hip_L - knee_L
    "pinggul_kanan":(12, 24, 26),   # shoulder_R - hip_R - knee_R
}


def hitung_sudut(a: np.ndarray, v: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Hitung sudut (derajat) di titik v antara vektor v→a dan v→b.
    Input shape: (..., 2) atau (..., 3) untuk koordinat x,y[,z].
    """
    va = a - v
    vb = b - v
    # Normalisasi
    norma = np.linalg.norm(va, axis=-1, keepdims=True)
    normb = np.linalg.norm(vb, axis=-1, keepdims=True)
    eps = 1e-8
    va_n = va / (norma + eps)
    vb_n = vb / (norma + eps)
    dot = np.clip((va_n * vb_n).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def rekayasa_fitur(kp_kondisi: np.ndarray) -> np.ndarray:
    """
    Hitung fitur statistik dari sekuens keypoints satu kondisi.
    Input: kp_kondisi shape [T, 33, 4]  (T=70 frame valid)
    Output: vektor fitur 1-D (float32)
    """
    T = kp_kondisi.shape[0]
    xy = kp_kondisi[:, :, :2]  # [T, 33, 2]

    fitur_bag = []

    # 1. Sudut sendi per frame, lalu stat (mean, std, min, max)
    for nama, (ia, iv, ib) in SUDUT_SENDI.items():
        a_seq = xy[:, ia, :]   # [T, 2]
        v_seq = xy[:, iv, :]
        b_seq = xy[:, ib, :]
        sudut = hitung_sudut(a_seq, v_seq, b_seq)  # [T]
        fitur_bag.extend([sudut.mean(), sudut.std(), sudut.min(), sudut.max()])

    # 2. Kecepatan (delta frame) per landmark x,y
    delta = np.diff(xy, axis=0)  # [T-1, 33, 2]
    kecepatan = np.linalg.norm(delta, axis=-1)  # [T-1, 33]
    # stat per landmark: mean, std, max
    fitur_bag.extend(kecepatan.mean(axis=0).tolist())   # 33
    fitur_bag.extend(kecepatan.std(axis=0).tolist())    # 33
    fitur_bag.extend(kecepatan.max(axis=0).tolist())    # 33

    # 3. Statistik posisi (mean, std per landmark x,y)
    fitur_bag.extend(xy.mean(axis=0).ravel().tolist())  # 33*2=66
    fitur_bag.extend(xy.std(axis=0).ravel().tolist())   # 33*2=66

    return np.array(fitur_bag, dtype=np.float32)


def potong_frame_kondisi(
    t_host_all: np.ndarray,
    keypoints_all: np.ndarray,
    deteksi_ok_all: np.ndarray,
    t_mulai: float,
    t_selesai: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Ambil frame dalam rentang [t_mulai, t_selesai], buang N_BUANG frame pertama.
    Kembalikan (kp_valid, ok_valid, n_frame_asli).
    Frame yang gagal (deteksi_ok=False) diganti interpolasi linier atau 0 jika tidak bisa.
    """
    mask = (t_host_all >= t_mulai) & (t_host_all < t_selesai)
    idx = np.where(mask)[0]
    n_asli = len(idx)

    if n_asli <= N_BUANG:
        # Kondisi tidak cukup panjang
        return np.zeros((N_VALID, 33, 4), dtype=np.float32), np.zeros(N_VALID, dtype=bool), n_asli

    # Ambil frame setelah membuang N_BUANG pertama
    idx_valid = idx[N_BUANG:]
    kp = keypoints_all[idx_valid].copy()          # [n_sisa, 33, 4]
    ok = deteksi_ok_all[idx_valid].copy()          # [n_sisa]

    # Isi frame gagal dengan nilai frame sebelumnya (forward fill) atau nol
    for t in range(len(kp)):
        if not ok[t]:
            if t > 0:
                kp[t] = kp[t - 1]
                ok[t] = ok[t - 1]  # propagasi status
            # jika frame pertama gagal, biarkan nol

    # Resampling ke tepat N_VALID frame dengan interpolasi linier
    n_sisa = len(kp)
    if n_sisa >= N_VALID:
        # Ambil N_VALID frame pertama (atau sampel merata)
        idx_resample = np.round(np.linspace(0, n_sisa - 1, N_VALID)).astype(int)
        kp_out = kp[idx_resample]
        ok_out = ok[idx_resample]
    else:
        # Kurang dari N_VALID: pad dengan frame terakhir
        pad = N_VALID - n_sisa
        kp_out = np.concatenate([kp, np.tile(kp[-1:], (pad, 1, 1))], axis=0)
        ok_out = np.concatenate([ok, np.zeros(pad, dtype=bool)], axis=0)

    return kp_out.astype(np.float32), ok_out, n_asli


def prepare_dataset(
    pose_npz: Path,
    label_csv: Path,
    rula_csv: Path | None,
    output_npz: Path,
    output_label_csv: Path,
) -> None:
    print("[02_prepare_dataset] Memuat pose.npz...")
    with np.load(pose_npz) as z:
        keypoints_all = z["keypoints"]    # [N_frame, 33, 4]
        berkas_all = z["berkas"]           # [N_frame]
        t_host_all = z["t_host"]          # [N_frame] float64
        deteksi_ok_all = z["deteksi_ok"]  # [N_frame] bool
    N_total = len(t_host_all)
    print(f"  Total frame: {N_total} | Deteksi OK: {deteksi_ok_all.sum()} "
          f"({deteksi_ok_all.mean()*100:.1f}%)")

    print("[02_prepare_dataset] Memuat label.csv...")
    df_label = pd.read_csv(label_csv)
    df_label = df_label.sort_values("t_mulai_host").reset_index(drop=True)
    N_total_label = len(df_label)
    print(f"  Jumlah kondisi awal di label.csv: {N_total_label}")

    # Penyelarasan label isyarat (k -> k+GESER_KONDISI di blok/partisipan yang sama)
    pasangan = []
    dibuang = []
    for i in range(N_total_label):
        j = i + GESER_KONDISI
        row_k = df_label.iloc[i]
        if (
            j < N_total_label
            and df_label.iloc[j]["partisipan"] == row_k["partisipan"]
            and df_label.iloc[j]["blok"] == row_k["blok"]
        ):
            pasangan.append((row_k, df_label.iloc[j]))
        else:
            dibuang.append(int(row_k["no"]))

    N_kondisi = len(pasangan)
    print(f"  Penyelarasan k+{GESER_KONDISI}: {N_kondisi} kondisi dipakai, {len(dibuang)} kondisi akhir blok dibuang")

    # Cek rula.csv
    ada_rula = False
    df_rula = None
    if rula_csv is not None and rula_csv.exists():
        df_rula = pd.read_csv(rula_csv)
        ada_rula = True
        print(f"  rula.csv ditemukan: {len(df_rula)} baris")
        # Hitung kappa dan rata-rata jika ada
        _proses_rula(df_rula)
    else:
        print("  rula.csv TIDAK ditemukan → mode klasifikasi saja (kepala regresi RULA disiapkan tapi kosong)")

    # Array output utama
    X_pose = np.zeros((N_kondisi, N_VALID, 33, 4), dtype=np.float32)
    X_fitur = []
    y_postur = np.zeros(N_kondisi, dtype=np.int64)
    y_rula = np.full(N_kondisi, np.nan, dtype=np.float32)
    meta_rows = []  # untuk label_bersih.csv

    print(f"[02_prepare_dataset] Memotong frame kondisi k+{GESER_KONDISI} untuk label kondisi k...")
    for i, (row, row_target) in enumerate(pasangan):
        t_mulai = float(row_target["t_mulai_host"])
        t_selesai = float(row_target["t_selesai_host"])
        partisipan = row["partisipan"]
        postur = row["postur"]
        no_kondisi = int(row["no"])
        no_target = int(row_target["no"])

        kp_valid, ok_valid, n_asli = potong_frame_kondisi(
            t_host_all, keypoints_all, deteksi_ok_all, t_mulai, t_selesai
        )

        X_pose[i] = kp_valid
        fitur = rekayasa_fitur(kp_valid)
        X_fitur.append(fitur)

        label_idx = KELAS_IDX.get(postur, -1)
        if label_idx == -1:
            print(f"  [PERINGATAN] Postur tidak dikenal: '{postur}' di kondisi {no_kondisi}", file=sys.stderr)
        y_postur[i] = label_idx

        # RULA
        if ada_rula and df_rula is not None:
            baris_rula = df_rula[df_rula["no"] == no_kondisi]
            if len(baris_rula) == 1:
                # Ambil rata-rata dua penilai (kolom penilai_1, penilai_2 atau nama serupa)
                cols_rula = [c for c in baris_rula.columns if c.startswith("penilai") or c.startswith("rula")]
                if cols_rula:
                    y_rula[i] = float(baris_rula[cols_rula].values[0].astype(float).mean())

        meta_rows.append({
            "no": no_kondisi,
            "no_target_rekaman": no_target,
            "partisipan": partisipan,
            "postur": postur,
            "label_idx": label_idx,
            "t_mulai_host": t_mulai,
            "t_selesai_host": t_selesai,
            "n_frame_asli": n_asli,
            "n_frame_valid": int(ok_valid.sum()),
            "deteksi_persen": float(ok_valid.mean() * 100),
        })

        if (i + 1) % 20 == 0 or (i + 1) == N_kondisi:
            print(f"  [{i+1:3d}/{N_kondisi}] {partisipan} | {postur} | "
                  f"frame={n_asli} → valid={ok_valid.sum()}")

    X_fitur = np.array(X_fitur, dtype=np.float32)  # [N_kondisi, N_fitur]

    # Reshape X_pose untuk model: [N_kondisi, N_VALID, 33*4] = [80, 70, 132]
    X_pose_flat = X_pose.reshape(N_kondisi, N_VALID, 33 * 4)

    # Simpan dataset_kondisi.npz
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    savekwargs = dict(
        X_pose=X_pose_flat,
        X_fitur=X_fitur,
        y_postur=y_postur,
        kelas=np.array(KELAS),
    )
    if ada_rula:
        savekwargs["y_rula"] = y_rula
    np.savez_compressed(output_npz, **savekwargs)

    # Simpan label_bersih.csv
    df_meta = pd.DataFrame(meta_rows)
    df_meta.to_csv(output_label_csv, index=False)

    print(f"\n[02_prepare_dataset] Selesai!")
    print(f"  X_pose shape    : {X_pose_flat.shape}  (kondisi, frame_valid, landmark*4)")
    print(f"  X_fitur shape   : {X_fitur.shape}  (kondisi, fitur statistik)")
    print(f"  y_postur shape  : {y_postur.shape}")
    if ada_rula:
        print(f"  y_rula shape    : {y_rula.shape} | non-NaN: {(~np.isnan(y_rula)).sum()}")
    print(f"  Output NPZ      : {output_npz}")
    print(f"  Output label CSV: {output_label_csv}")

    # ── Validasi otomatis ──────────────────────────────────────────────────
    print("\n[VALIDASI]")
    assert X_pose_flat.shape == (N_kondisi, N_VALID, 132), \
        f"Shape X_pose salah: {X_pose_flat.shape}"
    assert len(y_postur) == N_kondisi, "Jumlah label tidak sesuai kondisi"
    assert not np.isnan(X_pose_flat).any(), "GAGAL: X_pose mengandung NaN!"
    assert not np.isinf(X_pose_flat).any(), "GAGAL: X_pose mengandung Inf!"
    assert not np.isnan(X_fitur).any(), "GAGAL: X_fitur mengandung NaN!"
    # Pastikan jumlah kondisi pasca penyelarasan k+2 (64 kondisi)
    assert N_kondisi == 64, f"Jumlah kondisi setelah geser k+{GESER_KONDISI} seharusnya 64, dapat {N_kondisi}"
    # Distribusi kelas
    for c, nama in enumerate(KELAS):
        n_c = int((y_postur == c).sum())
        print(f"  Kelas {c} ({nama:20s}): {n_c} kondisi")
    print("  [OK] Semua validasi lolos.")


def _proses_rula(df_rula: pd.DataFrame) -> None:
    """Hitung Cohen's Kappa dan statistik RULA jika ada 2 kolom penilai."""
    from sklearn.metrics import cohen_kappa_score
    cols_penilai = [c for c in df_rula.columns if c.startswith("penilai") or c == "rula_1" or c == "rula_2"]
    if len(cols_penilai) >= 2:
        p1 = df_rula[cols_penilai[0]].dropna().astype(int)
        p2 = df_rula[cols_penilai[1]].dropna().astype(int)
        idx_valid = p1.index.intersection(p2.index)
        kappa = cohen_kappa_score(p1[idx_valid], p2[idx_valid])
        print(f"  Cohen's Kappa antar penilai RULA: {kappa:.3f}")
        rata2 = (df_rula[cols_penilai[0]] + df_rula[cols_penilai[1]]) / 2
        print(f"  Rata-rata skor RULA: {rata2.mean():.2f} ± {rata2.std():.2f} "
              f"(min={rata2.min():.0f}, max={rata2.max():.0f})")
    else:
        print("  [INFO] Hanya 1 kolom penilai RULA ditemukan; kappa tidak dihitung.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose", default=str(POSE_NPZ_DEFAULT))
    ap.add_argument("--label", default=str(LABEL_CSV_DEFAULT))
    ap.add_argument("--rula", default=str(RULA_CSV_DEFAULT), help="Path rula.csv (opsional)")
    ap.add_argument("--output", default=str(OUTPUT_NPZ_DEFAULT))
    ap.add_argument("--output-label", default=str(OUTPUT_LABEL_CSV_DEFAULT))
    args = ap.parse_args()

    rula_path = Path(args.rula)
    prepare_dataset(
        pose_npz=Path(args.pose),
        label_csv=Path(args.label),
        rula_csv=rula_path if rula_path.exists() else None,
        output_npz=Path(args.output),
        output_label_csv=Path(args.output_label),
    )


if __name__ == "__main__":
    main()
