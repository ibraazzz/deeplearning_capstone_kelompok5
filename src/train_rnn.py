"""
src/train_rnn.py
================
Tim 05 – Penilaian Postur Duduk
Model RNN (Bidirectional LSTM/GRU) untuk klasifikasi postur.

Jalankan: python -m src.train_rnn --seed 1 --model lstm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset_loader import get_dataloaders, KELAS, N_KELAS

# Path output
AKAR = Path(__file__).resolve().parents[1]
RESULTS_DIR = AKAR / "results" / "rnn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class ModelRNN(nn.Module):
    def __init__(self, input_size: int = 132, hidden: int = 128, n_layer: int = 2,
                 dropout: float = 0.3, n_kelas: int = N_KELAS, rnn_type: str = "lstm"):
        super().__init__()
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden, n_layer, batch_first=True,
                               bidirectional=True, dropout=dropout if n_layer > 1 else 0.0)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size, hidden, n_layer, batch_first=True,
                              bidirectional=True, dropout=dropout if n_layer > 1 else 0.0)
        else:
            raise ValueError("rnn_type harus 'lstm' atau 'gru'")

        self.layer_norm = nn.LayerNorm(hidden * 2)
        self.kepala = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, n_kelas)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        feat = out[:, -1, :]
        feat = self.layer_norm(feat)
        return self.kepala(feat)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, return_preds=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    if return_preds:
        return acc, f1, all_labels, all_preds
    return acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Muat DataLoader (format_cnn=False → (B, 70, 132))
    dl_train, dl_val, dl_test, info = get_dataloaders(batch_size=args.batch, format_cnn=False)

    # 2. Tangani Class Imbalance (WeightedRandomSampler)
    # Hitung bobot per kelas dari data latih
    y_train = np.concatenate([yb.numpy() for _, yb in dl_train])
    class_counts = np.bincount(y_train, minlength=N_KELAS)
    class_weights = 1.0 / (class_counts + 1e-7)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    dl_train_balanced = DataLoader(dl_train.dataset, batch_size=args.batch, sampler=sampler)

    # 3. Inisialisasi Model, Loss, Optimizer
    model = ModelRNN(rnn_type=args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 4. Training Loop dengan Early Stopping
    history = {"loss": [], "val_acc": [], "val_f1": []}
    best_f1 = 0.0
    patience = 10
    no_improve = 0
    best_model_path = RESULTS_DIR / f"best_{args.model}.pt"

    print(f"[{args.model.upper()}] Mulai training...")
    t0 = time.time()
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, dl_train_balanced, optimizer, criterion, device)
        val_acc, val_f1 = evaluate(model, dl_val, device)

        history["loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1:02d}/{args.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        # Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Model terbaik disimpan (F1: {best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  → Early stopping di epoch {epoch+1}")
                break

    durasi = time.time() - t0
    print(f"Training selesai dalam {durasi:.1f} detik.")

    # 5. Evaluasi Final (Test) dengan Model Terbaik
    model.load_state_dict(torch.load(best_model_path))
    test_acc, test_f1, y_true, y_pred = evaluate(model, dl_test, device, return_preds=True)
    print(f"\n[TEST] Akurasi: {test_acc:.4f} | F1-Macro: {test_f1:.4f}")

    # 6. Simpan Artefak
    # 6a. metrics.json
    metrics = {
        "val_acc": float(history["val_acc"][-1]),
        "val_f1_macro": float(history["val_f1"][-1]),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1),
        "test_precision": float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        "seed": args.seed,
        "model": args.model,
        "durasi_detik": durasi
    }
    with open(RESULTS_DIR / f"metrics_{args.model}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 6b. classification_report.txt
    report = classification_report(y_true, y_pred, target_names=KELAS, zero_division=0)
    with open(RESULTS_DIR / f"classification_report_{args.model}.txt", "w") as f:
        f.write(report)

    # 6c. confusion_matrix.png
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=KELAS)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f'Confusion Matrix - {args.model.upper()}')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"confusion_matrix_{args.model}.png")
    plt.close()

    # 6d. training_curves.png
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(history["loss"], color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('F1-Macro / Acc', color='tab:blue')
    ax2.plot(history["val_f1"], color='tab:blue', label='Val F1-Macro')
    ax2.plot(history["val_acc"], color='tab:cyan', linestyle='--', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f'Training Curves - {args.model.upper()}')
    fig.tight_layout()
    plt.savefig(RESULTS_DIR / f"training_curves_{args.model}.png")
    plt.close()

    print(f"\nSemua artefak disimpan di {RESULTS_DIR}/")


if __name__ == "__main__":
    main()