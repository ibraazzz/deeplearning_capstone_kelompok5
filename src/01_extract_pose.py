"""
src/01_extract_pose.py
======================
Tim 05 – Penilaian Postur Duduk
Tahap 1: Ekstraksi 33 Landmark MediaPipe Pose dari seluruh frame kamera.

Mendukung:
  - MediaPipe Tasks API (mediapipe >= 0.10.x / 1.x: PoseLandmarker)
  - MediaPipe Solutions API (mediapipe < 0.10.30: mp.solutions.pose) sebagai fallback

Output: data/processed/pose.npz
  - keypoints  : float32 [N_frame, 33, 4]  (x, y, z, visibility) ternormalisasi [0,1]
  - berkas      : str array [N_frame]       nama file gambar (mis. f000001.jpg)
  - t_host      : float64 [N_frame]         timestamp jam komputer meja (detik epoch)
  - deteksi_ok  : bool    [N_frame]         True jika pose terdeteksi

Strategi frame gagal:
  - Jika pose tidak terdeteksi, keypoints diisi 0.0 dan deteksi_ok=False.
  - Frame ini tidak dihapus; penanganan dilakukan di tahap 02_prepare_dataset.py.

Penggunaan:
  python src/01_extract_pose.py
  python src/01_extract_pose.py --kamera dataset_gambar/kamera --output data/processed/pose.npz
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

AKAR = Path(__file__).resolve().parents[1]
KAMERA_DIR_DEFAULT = AKAR / "dataset_gambar" / "kamera"
KAMERA_INDEX_DEFAULT = AKAR / "dataset_gambar" / "kamera_index.csv"
OUTPUT_DEFAULT = AKAR / "data" / "processed" / "pose.npz"
MODEL_DIR_DEFAULT = AKAR / "data" / "models"

N_LANDMARK = 33

MODEL_URLS = {
    0: ("pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
    1: ("pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"),
    2: ("pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"),
}


def baca_kamera_index(path: Path) -> list[dict]:
    """Baca kamera_index.csv → list dict {berkas, t_host}."""
    with path.open(encoding="utf-8") as f:
        baris = list(csv.DictReader(f))
    for b in baris:
        b["t_host"] = float(b["t_host"])
    return sorted(baris, key=lambda x: x["berkas"])


def pastikan_model_task(complexity: int, model_dir: Path) -> Path:
    """Pastikan file model .task tersedia di model_dir; unduh jika belum ada."""
    nama_file, url = MODEL_URLS.get(complexity, MODEL_URLS[1])
    path_model = model_dir / nama_file
    if not path_model.exists() or path_model.stat().st_size == 0:
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[01_extract_pose] Mengunduh model MediaPipe ({nama_file})...")
        urllib.request.urlretrieve(url, path_model)
        print(f"  Model tersimpan di: {path_model}")
    return path_model


def ekstrak_pose(
    kamera_dir: Path,
    kamera_index: Path,
    output: Path,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> None:
    """Proses semua frame di kamera_dir menggunakan MediaPipe Pose."""
    indeks = baca_kamera_index(kamera_index)
    N = len(indeks)
    print(f"[01_extract_pose] Total frame di index: {N}")

    # Siapkan array output
    keypoints = np.zeros((N, N_LANDMARK, 4), dtype=np.float32)
    berkas_arr = np.array([b["berkas"] for b in indeks])
    t_host_arr = np.array([b["t_host"] for b in indeks], dtype=np.float64)
    deteksi_ok = np.zeros(N, dtype=bool)

    gunakan_tasks = not hasattr(mp, "solutions")
    if gunakan_tasks:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        model_path = pastikan_model_task(model_complexity, MODEL_DIR_DEFAULT)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
        print(f"[01_extract_pose] Menggunakan MediaPipe Tasks PoseLandmarker ({model_path.name})")
    else:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            smooth_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        print("[01_extract_pose] Menggunakan MediaPipe Solutions Pose")

    t_mulai = time.perf_counter()
    gagal = 0
    for i, baris in enumerate(indeks):
        nama = baris["berkas"]
        path_img = kamera_dir / nama
        if not path_img.exists():
            print(f"  [PERINGATAN] File tidak ada: {nama}", file=sys.stderr)
            gagal += 1
            continue

        img_bgr = cv2.imread(str(path_img))
        if img_bgr is None:
            print(f"  [PERINGATAN] Gagal baca gambar: {nama}", file=sys.stderr)
            gagal += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if gunakan_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            hasil = landmarker.detect(mp_image)
            if hasil.pose_landmarks and len(hasil.pose_landmarks) > 0:
                lms = hasil.pose_landmarks[0]
                for j, lm in enumerate(lms[:N_LANDMARK]):
                    vis = getattr(lm, "visibility", 1.0)
                    keypoints[i, j] = [lm.x, lm.y, lm.z, float(vis if vis is not None else 1.0)]
                deteksi_ok[i] = True
            else:
                gagal += 1
        else:
            hasil = pose.process(img_rgb)
            if hasil.pose_landmarks:
                for j, lm in enumerate(hasil.pose_landmarks.landmark):
                    keypoints[i, j] = [lm.x, lm.y, lm.z, lm.visibility]
                deteksi_ok[i] = True
            else:
                gagal += 1

        if (i + 1) % 500 == 0 or (i + 1) == N:
            elapsed = time.perf_counter() - t_mulai
            fps_efektif = (i + 1) / elapsed
            persen_ok = deteksi_ok[: i + 1].mean() * 100
            print(
                f"  [{i+1:5d}/{N}] {fps_efektif:.1f} fps | "
                f"deteksi OK: {persen_ok:.1f}% | gagal kumulatif: {gagal}",
                flush=True,
            )

    if gunakan_tasks:
        landmarker.close()
    else:
        pose.close()

    # Simpan output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        keypoints=keypoints,
        berkas=berkas_arr,
        t_host=t_host_arr,
        deteksi_ok=deteksi_ok,
    )

    elapsed_total = time.perf_counter() - t_mulai
    persen_ok_total = deteksi_ok.mean() * 100
    print(f"\n[01_extract_pose] Selesai dalam {elapsed_total:.1f} detik")
    print(f"  Total frame    : {N}")
    print(f"  Deteksi OK     : {deteksi_ok.sum()} ({persen_ok_total:.1f}%)")
    print(f"  Frame gagal    : {gagal}")
    print(f"  Output disimpan: {output}")

    # Validasi cepat: tidak boleh ada NaN atau Inf
    assert not np.isnan(keypoints).any(), "GAGAL: keypoints mengandung NaN!"
    assert not np.isinf(keypoints).any(), "GAGAL: keypoints mengandung Inf!"
    print("  [OK] Tidak ada NaN/Inf di keypoints.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kamera", default=str(KAMERA_DIR_DEFAULT), help="Folder gambar kamera")
    ap.add_argument("--index", default=str(KAMERA_INDEX_DEFAULT), help="kamera_index.csv")
    ap.add_argument("--output", default=str(OUTPUT_DEFAULT), help="Jalur output .npz")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                    help="MediaPipe model_complexity (0=cepat/lite, 1=default/full, 2=akurat/heavy)")
    ap.add_argument("--conf-deteksi", type=float, default=0.5)
    ap.add_argument("--conf-tracking", type=float, default=0.5)
    args = ap.parse_args()

    ekstrak_pose(
        kamera_dir=Path(args.kamera),
        kamera_index=Path(args.index),
        output=Path(args.output),
        model_complexity=args.complexity,
        min_detection_confidence=args.conf_deteksi,
        min_tracking_confidence=args.conf_tracking,
    )


if __name__ == "__main__":
    main()
