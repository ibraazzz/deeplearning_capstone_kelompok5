"""
src/06_compare_models.py
========================
Tim 05 – Penilaian Postur Duduk
Tahap 6: Perbandingan akhir semua jalur model tim.

Membaca metrics.json dari masing-masing folder jalur dan membuat tabel perbandingan.
Jalankan setelah semua anggota tim selesai melatih model masing-masing.

Jalur yang dibandingkan:
  - results/cnn/           (CNN – anggota saya)
  - results/rnn/           (RNN – anggota RNN, dipisah LSTM & GRU)
  - results/transformer/   (Transformer – anggota Transformer)
  - results/ae/            (Autoencoder – anggota AE)
  - results/random_forest/ (Baseline non-DL)

Output:
  - results/perbandingan_model.md   Tabel markdown
  - results/perbandingan_model.csv  CSV untuk analisis lanjut
  - results/perbandingan_model.png  Grafik batang F1-Macro
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AKAR = Path(__file__).resolve().parents[1]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))
RESULTS_DIR = AKAR / "results"

# PERBAIKAN: Menambahkan entri 'lstm' dan 'gru' dengan suffix masing-masing
JALUR_MODEL = {
    "cnn":           {"nama": "1D-CNN",       "folder": "cnn"},
    "lstm":          {"nama": "RNN (LSTM)",    "folder": "rnn", "suffix": "_lstm"},
    "gru":           {"nama": "RNN (GRU)",     "folder": "rnn", "suffix": "_gru"},
    "transformer":   {"nama": "Transformer",   "folder": "transformer"},
    "ae":            {"nama": "Autoencoder",   "folder": "ae"},
    "random_forest": {"nama": "Random Forest", "folder": "random_forest"},
}

KELAS = ["tegak", "membungkuk", "bersandar", "condong ke kiri", "condong ke kanan"]


def baca_metrics(folder: Path, suffix: str = "") -> dict | None:
    """Baca metrics.json dari folder jalur (dengan atau tanpa suffix)."""
    p = folder / f"metrics{suffix}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def baca_tambahan(folder: Path, suffix: str = "") -> dict | None:
    """Baca tambahan.json dari folder jalur."""
    p = folder / f"tambahan{suffix}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def bandingkan_model() -> None:
    print("[06_compare_models] Mengumpulkan hasil semua jalur...")

    baris = []
    for key, info in JALUR_MODEL.items():
        folder = RESULTS_DIR / info["folder"]
        suffix = info.get("suffix", "")
        
        metrics = baca_metrics(folder, suffix)
        tambahan = baca_tambahan(folder, suffix)

        if metrics is None:
            print(f"  [{key:14s}] ⚠ metrics{suffix}.json belum ada (model belum selesai dilatih)")
            baris.append({
                "jalur": key,
                "nama_model": info["nama"],
                "val_f1_macro": None,
                "val_acc": None,
                "test_f1_macro": None,
                "test_acc": None,
                "n_params": None,
                "waktu_latih_s": None,
                "status": "belum ada",
            })
            continue

        # Cek dulu apakah ada key "metrik" (format lama) atau langsung (format baru)
        m = metrics.get("metrik", metrics)
        val_f1 = m.get("val_macro_f1", m.get("val_f1_macro", m.get("val_f1", None)))
        val_acc = m.get("val_acc", None)
        test_f1 = m.get("test_macro_f1", m.get("test_f1_macro", m.get("test_f1", None)))
        test_acc = m.get("test_acc", None)

        # Dari tambahan.json
        n_params = tambahan.get("parameter", None) if tambahan else None
        waktu = metrics.get("durasi_detik", None)

        val_f1_str = f"{val_f1:.4f}" if val_f1 is not None else "—"
        test_f1_str = f"{test_f1:.4f}" if test_f1 is not None else "—"
        print(f"  [{key:14s}] val_f1={val_f1_str} | test_f1={test_f1_str}")

        baris.append({
            "jalur": key,
            "nama_model": info["nama"],
            "val_f1_macro": val_f1,
            "val_acc": val_acc,
            "test_f1_macro": test_f1,
            "test_acc": test_acc,
            "n_params": n_params,
            "waktu_latih_s": waktu,
            "status": "selesai",
        })

    df = pd.DataFrame(baris)

    # Urutkan berdasarkan val_f1_macro
    df_selesai = df[df["status"] == "selesai"].copy()
    df_belum = df[df["status"] != "selesai"].copy()
    if len(df_selesai) > 0:
        df_selesai = df_selesai.sort_values("val_f1_macro", ascending=False)
    df = pd.concat([df_selesai, df_belum], ignore_index=True)

    # ── Simpan CSV ──
    csv_path = RESULTS_DIR / "perbandingan_model.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  [Tersimpan] {csv_path.name}")

    # ── Buat tabel Markdown ──
    baris_md = ["# Perbandingan Model – Tim 05 Penilaian Postur Duduk\n",
                "| Model | Val F1-Macro | Val Acc | Test F1-Macro | Test Acc | #Params | Waktu (s) |",
                "|-------|:-----------:|:-------:|:-------------:|:--------:|:-------:|:---------:|"]

    for _, row in df.iterrows():
        def fmt(v, d=4):
            return f"{v:.{d}f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"

        n_params_str = str(int(row["n_params"])) if pd.notna(row["n_params"]) else "—"
        waktu_str = fmt(row["waktu_latih_s"], 1) if pd.notna(row["waktu_latih_s"]) else "—"
        baris_md.append(
            f"| {row['nama_model']} | {fmt(row['val_f1_macro'])} | {fmt(row['val_acc'])} | "
            f"{fmt(row['test_f1_macro'])} | {fmt(row['test_acc'])} | "
            f"{n_params_str} | {waktu_str} |"
        )

    baris_md.append("\n> *Metrik utama: F1-Score Macro (5 kelas postur duduk). "
                    "Split per partisipan (subject-independent, seed=2026).*\n")

    # F1 per kelas (dari tambahan.json jika tersedia)
    baris_md.append("## F1 per Kelas Postur\n")
    baris_md.append("| Model | " + " | ".join(KELAS) + " |")
    baris_md.append("|-------|" + "|".join(["------"] * len(KELAS)) + "|")

    for key, info in JALUR_MODEL.items():
        folder = RESULTS_DIR / info["folder"]
        suffix = info.get("suffix", "")
        tambahan = baca_tambahan(folder, suffix)
        if tambahan and "test_f1_rinci" in tambahan:
            rinci = tambahan["test_f1_rinci"]
            vals = [f"{rinci.get(k, 0):.3f}" for k in KELAS]
            baris_md.append(f"| {info['nama']} | " + " | ".join(vals) + " |")
        elif tambahan and "val_f1_rinci" in tambahan:
            rinci = tambahan["val_f1_rinci"]
            vals = [f"{rinci.get(k, 0):.3f}" for k in KELAS]
            baris_md.append(f"| {info['nama']} (val) | " + " | ".join(vals) + " |")

    md_path = RESULTS_DIR / "perbandingan_model.md"
    md_path.write_text("\n".join(baris_md), encoding="utf-8")
    print(f"  [Tersimpan] {md_path.name}")

    # ── Plot grafik batang ──
    df_plot = df[df["status"] == "selesai"].copy()
    if len(df_plot) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df_plot))
        width = 0.35
        ax.bar(x - width / 2, df_plot["val_f1_macro"].fillna(0), width, label="Val F1-Macro", color="steelblue")
        ax.bar(x + width / 2, df_plot["test_f1_macro"].fillna(0), width, label="Test F1-Macro", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot["nama_model"], rotation=15, ha="right")
        ax.set_ylabel("F1-Score Macro")
        ax.set_title("Tim 05 – Perbandingan F1-Score Macro Antar Model")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, label="Target 0.80")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "perbandingan_model.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [Tersimpan] perbandingan_model.png")

    print(f"\n[06_compare_models] Selesai. Lihat {RESULTS_DIR}")
    print("\n=== RINGKASAN ===")
    print(df[["nama_model", "val_f1_macro", "test_f1_macro", "status"]].to_string(index=False))


if __name__ == "__main__":
    bandingkan_model()