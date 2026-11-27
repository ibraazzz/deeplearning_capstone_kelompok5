"""
src/05_evaluate_cnn.py
======================
Tim 05 – Penilaian Postur Duduk
Tahap 5: Evaluasi lengkap model 1D-CNN dengan visualisasi.

Menghasilkan:
  - results/cnn/classification_report.txt
  - results/cnn/confusion_matrix.png       (resolusi tinggi, 300 DPI)
  - results/cnn/training_curves.png        (loss & F1 per epoch)
  - results/cnn/metrics.json               (diperbarui dengan nilai test)

Penggunaan:
  python src/05_evaluate_cnn.py
  python src/05_evaluate_cnn.py --bobot results/cnn/best_cnn.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Tanpa GUI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

AKAR = Path(__file__).resolve().parents[1]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))
RESULTS_DIR = AKAR / "results" / "cnn"
KELAS = ["tegak", "membungkuk", "bersandar", "condong ke kiri", "condong ke kanan"]
N_KELAS = 5


def plot_training_curves(riwayat: list[dict], save_path: Path) -> None:
    """Plot loss dan val_f1 per epoch."""
    epochs = [r["epoch"] for r in riwayat]
    losses = [r["loss"] for r in riwayat]
    f1s = [r["val_f1"] for r in riwayat]
    accs = [r["val_acc"] for r in riwayat]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, losses, "b-o", markersize=4, linewidth=1.5, label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 & Accuracy
    axes[1].plot(epochs, f1s, "g-o", markersize=4, linewidth=1.5, label="Val F1-Macro")
    axes[1].plot(epochs, accs, "r-s", markersize=4, linewidth=1.5, label="Val Accuracy")
    # Tandai epoch terbaik
    best_ep = max(riwayat, key=lambda x: x["val_f1"])["epoch"]
    best_f1 = max(f1s)
    axes[1].axvline(x=best_ep, color="gray", linestyle="--", alpha=0.7, label=f"Best (ep {best_ep})")
    axes[1].scatter([best_ep], [best_f1], color="green", s=100, zorder=5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.suptitle("Tim 05 – 1D-CNN Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Tersimpan] {save_path.name}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, judul: str = "Test Set") -> None:
    """Plot confusion matrix ternormalisasi dan absolut."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_KELAS)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Absolut
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=KELAS, yticklabels=KELAS, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix – Absolut ({judul})")
    axes[0].set_xlabel("Prediksi")
    axes[0].set_ylabel("Aktual")
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=25, ha="right")

    # Ternormalisasi
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=KELAS, yticklabels=KELAS, ax=axes[1],
                vmin=0, vmax=1)
    axes[1].set_title(f"Confusion Matrix – Ternormalisasi ({judul})")
    axes[1].set_xlabel("Prediksi")
    axes[1].set_ylabel("Aktual")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=25, ha="right")

    plt.suptitle("Tim 05 – 1D-CNN Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Tersimpan] {save_path.name}")


def evaluasi_lengkap(bobot_path: Path | None = None) -> None:
    from src.dataset_loader import get_numpy_splits
    from src._03_cnn_model import CNN1D

    if bobot_path is None:
        bobot_path = RESULTS_DIR / "best_cnn.pt"

    if not bobot_path.exists():
        raise FileNotFoundError(
            f"Bobot model tidak ditemukan: {bobot_path}\n"
            "Jalankan dulu: python src/03_train_cnn.py"
        )

    # Muat model
    checkpoint = torch.load(bobot_path, map_location="cpu")
    ada_rula = checkpoint.get("ada_rula", False)

    model = CNN1D(n_kelas=N_KELAS, in_channels=132, ada_rula=ada_rula)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"[05_evaluate_cnn] Model dimuat dari: {bobot_path}")
    print(f"  ada_rula={ada_rula} | Parameter={sum(p.numel() for p in model.parameters()):,}")

    # Muat data
    X_tr, y_tr, X_va, y_va, X_te, y_te = get_numpy_splits(mode="pose", format_cnn=True)

    @torch.no_grad()
    def prediksi(X):
        X_t = torch.from_numpy(X.astype(np.float32))
        out = model(X_t)
        if isinstance(out, tuple):
            return out[0].argmax(dim=1).numpy()
        return out.argmax(dim=1).numpy()

    pred_val = prediksi(X_va)
    pred_test = prediksi(X_te)

    # Metrik Val
    f1_val = f1_score(y_va, pred_val, average="macro", zero_division=0)
    acc_val = accuracy_score(y_va, pred_val)

    # Metrik Test
    f1_test = f1_score(y_te, pred_test, average="macro", zero_division=0)
    acc_test = accuracy_score(y_te, pred_test)

    print(f"\n  Val  F1-Macro: {f1_val:.4f} | Acc: {acc_val:.4f}")
    print(f"  Test F1-Macro: {f1_test:.4f} | Acc: {acc_test:.4f}")

    report_test = classification_report(
        y_te, pred_test, target_names=KELAS, labels=list(range(N_KELAS)), zero_division=0
    )
    print("\n=== Classification Report (Test) ===")
    print(report_test)

    # Simpan report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_full = (
        f"=== Tim 05 – 1D-CNN Evaluation ===\n"
        f"Val  F1-Macro: {f1_val:.4f} | Val  Acc: {acc_val:.4f}\n"
        f"Test F1-Macro: {f1_test:.4f} | Test Acc: {acc_test:.4f}\n\n"
        f"=== Classification Report (Test) ===\n{report_test}\n"
    )
    (RESULTS_DIR / "classification_report.txt").write_text(report_full, encoding="utf-8")
    print(f"  [Tersimpan] classification_report.txt")

    # Perbarui metrics.json
    if (RESULTS_DIR / "metrics.json").exists():
        with open(RESULTS_DIR / "metrics.json", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    existing["metrik"] = existing.get("metrik", {})
    existing["metrik"].update({
        "val_acc": round(float(acc_val), 6),
        "val_macro_f1": round(float(f1_val), 6),
        "test_acc": round(float(acc_test), 6),
        "test_macro_f1": round(float(f1_test), 6),
    })
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  [Tersimpan] metrics.json (diperbarui)")

    # Plot Confusion Matrix
    plot_confusion_matrix(y_te, pred_test, RESULTS_DIR / "confusion_matrix.png")

    # Plot Training Curves (dari training_history.json)
    history_path = RESULTS_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            riwayat = json.load(f)
        plot_training_curves(riwayat, RESULTS_DIR / "training_curves.png")
    else:
        print("  [INFO] training_history.json tidak ditemukan; kurva pelatihan tidak dibuat")

    print(f"\n[05_evaluate_cnn] Evaluasi selesai. Hasil di: {RESULTS_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bobot", default=None, help="Path best_cnn.pt")
    args = ap.parse_args()
    evaluasi_lengkap(Path(args.bobot) if args.bobot else None)


if __name__ == "__main__":
    main()
