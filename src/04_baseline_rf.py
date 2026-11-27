"""
src/04_baseline_rf.py
=====================
Tim 05 – Penilaian Postur Duduk
Tahap 4: Baseline non-Deep Learning – Random Forest dari fitur statistik.

Fitur yang digunakan (sesuai tim-05.pdf):
  - Sudut sendi per frame (statistik: mean, std, min, max) → 6 sudut × 4 = 24
  - Kecepatan titik pose (mean, std, max per landmark) → 33 × 3 = 99
  - Statistik posisi landmark (mean, std) → 33 × 2 × 2 = 132
  Total: ~354 fitur statistik agregat per kondisi

Model yang dibandingkan:
  - Random Forest (utama)
  - SVM (pembanding sekunder)
  - LDA (Linear Discriminant Analysis)

Output:
  - results/random_forest/metrics.json
  - results/random_forest/tambahan.json
  - results/random_forest/classification_report.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

AKAR = Path(__file__).resolve().parents[1]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))
DATASET_NPZ = AKAR / "data" / "processed" / "dataset_kondisi.npz"
RESULTS_DIR = AKAR / "results" / "random_forest"

KELAS = ["tegak", "membungkuk", "bersandar", "condong ke kiri", "condong ke kanan"]
N_KELAS = 5


def latih_dan_evaluasi_rf(
    seed: int = 1,
    n_estimators: int = 300,
) -> None:
    t_mulai_global = time.perf_counter()
    print(f"[04_baseline_rf] Seed={seed} | n_estimators={n_estimators}")

    # ── Muat fitur statistik ──
    from src.dataset_loader import get_numpy_splits, split_per_partisipan

    X_tr, y_tr, X_va, y_va, X_te, y_te = get_numpy_splits(
        mode="fitur", format_cnn=False
    )
    print(f"  Shape Train : {X_tr.shape} | Val: {X_va.shape} | Test: {X_te.shape}")
    print(f"  Distribusi Train: {dict(zip(KELAS, np.bincount(y_tr, minlength=N_KELAS).tolist()))}")

    # ── Definisi model ──
    kandidat = {
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
        "lda": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis()),
        ]),
    }

    # ── Evaluasi semua kandidat pada Val ──
    hasil_val = {}
    for nama, pipe in kandidat.items():
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        t1 = time.perf_counter()
        pred_val = pipe.predict(X_va)
        f1_val = f1_score(y_va, pred_val, average="macro", zero_division=0)
        acc_val = accuracy_score(y_va, pred_val)
        f1_rinci = f1_score(y_va, pred_val, average=None, zero_division=0, labels=range(N_KELAS))
        print(f"  [{nama:4s}] val_f1_macro={f1_val:.4f} | val_acc={acc_val:.4f} | waktu={t1-t0:.2f}s")
        hasil_val[nama] = {
            "val_f1_makro": round(f1_val, 4),
            "val_acc": round(acc_val, 4),
            "val_f1_rinci": {k: round(float(v), 4) for k, v in zip(KELAS, f1_rinci)},
            "waktu_latih_s": round(t1 - t0, 2),
        }

    # ── Pilih model terbaik ──
    terbaik_nama = max(hasil_val, key=lambda n: hasil_val[n]["val_f1_makro"])
    print(f"\n  → Model terpilih: {terbaik_nama} (val_f1={hasil_val[terbaik_nama]['val_f1_makro']:.4f})")

    model_terbaik = kandidat[terbaik_nama]

    # ── Evaluasi Test ──
    pred_test = model_terbaik.predict(X_te)
    pred_val_final = model_terbaik.predict(X_va)

    f1_test = f1_score(y_te, pred_test, average="macro", zero_division=0)
    acc_test = accuracy_score(y_te, pred_test)
    f1_test_rinci = f1_score(y_te, pred_test, average=None, zero_division=0, labels=range(N_KELAS))

    print(f"\n[Evaluasi Test – {terbaik_nama}]")
    print(f"  Test F1-Macro : {f1_test:.4f}")
    print(f"  Test Accuracy : {acc_test:.4f}")

    # Classification report
    report = classification_report(
        y_te, pred_test,
        target_names=KELAS,
        labels=list(range(N_KELAS)),
        zero_division=0,
    )
    print("\n" + report)

    # Confusion matrix
    cm = confusion_matrix(y_te, pred_test, labels=list(range(N_KELAS)))
    print("Confusion Matrix (Test):")
    print(cm)

    durasi = time.perf_counter() - t_mulai_global

    # ── Simpan output ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # metrics.json (format kompatibel sprint-1)
    f1_val_final = hasil_val[terbaik_nama]["val_f1_makro"]
    acc_val_final = hasil_val[terbaik_nama]["val_acc"]
    hasil_json = {
        "metrik": {
            "val_acc": float(acc_val_final),
            "val_macro_f1": float(f1_val_final),
            "test_acc": float(acc_test),
            "test_macro_f1": float(f1_test),
        },
        "seed": seed,
        "perangkat": "cpu laptop",
        "model": f"non-dl-{terbaik_nama}",
        "n_val": len(y_va),
        "n_test": len(y_te),
        "durasi_detik": round(durasi, 1),
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(hasil_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # tambahan.json
    tambahan = {
        "jalur": "random_forest",
        "seed": seed,
        "dipilih": terbaik_nama,
        "calon": hasil_val,
        "fitur": f"sudut sendi (6×4=24) + kecepatan titik pose (33×3=99) + statistik posisi (33×2×2=132) = {X_tr.shape[1]} total fitur",
        "test_f1_makro": round(f1_test, 4),
        "test_f1_rinci": {k: round(float(v), 4) for k, v in zip(KELAS, f1_test_rinci)},
        "confusion_matrix_test": cm.tolist(),
    }
    (RESULTS_DIR / "tambahan.json").write_text(
        json.dumps(tambahan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # classification_report.txt
    report_full = (
        f"Model: {terbaik_nama} (baseline non-DL)\n"
        f"Seed: {seed}\n"
        f"Val F1-Macro: {f1_val_final:.4f} | Val Acc: {acc_val_final:.4f}\n"
        f"Test F1-Macro: {f1_test:.4f} | Test Acc: {acc_test:.4f}\n\n"
        f"=== Classification Report (Test) ===\n{report}\n"
        f"=== Confusion Matrix (Test) ===\n{cm}\n"
    )
    (RESULTS_DIR / "classification_report.txt").write_text(report_full, encoding="utf-8")

    print(f"\n[04_baseline_rf] Semua hasil tersimpan di: {RESULTS_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-estimators", type=int, default=300)
    args = ap.parse_args()
    latih_dan_evaluasi_rf(seed=args.seed, n_estimators=args.n_estimators)


if __name__ == "__main__":
    main()
