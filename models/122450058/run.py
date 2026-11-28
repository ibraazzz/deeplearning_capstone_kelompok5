"""Kerangka run.py anggota untuk model RNN (LSTM/GRU)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "rnn"

class ModelRNN(nn.Module):
    def __init__(self, input_size: int = 132, hidden: int = 128, n_layer: int = 2, 
                 dropout: float = 0.3, n_kelas: int = 5, rnn_type: str = "lstm"):
        super().__init__()
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size, 
                hidden_size=hidden, 
                num_layers=n_layer, 
                batch_first=True, 
                bidirectional=True, 
                dropout=dropout if n_layer > 1 else 0.0
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size, 
                hidden_size=hidden, 
                num_layers=n_layer, 
                batch_first=True, 
                bidirectional=True, 
                dropout=dropout if n_layer > 1 else 0.0
            )
        else:
            raise ValueError("rnn_type harus 'lstm' atau 'gru'")
            
        self.kepala = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(hidden * 2, 64), 
            nn.ReLU(), 
            nn.Linear(64, n_kelas)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        feat = out[:, -1, :]
        return self.kepala(feat)

def latih_dan_prediksi(seed: int, final: bool, model_type: str = "lstm"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Muat data via data.py
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelRNN(rnn_type=model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Data loader sederhana
    batch_size = 32
    X_tensor = torch.tensor(Xtr, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(ytr, dtype=torch.long).to(device)
    
    # Untuk menyimpan riwayat training
    history = {"loss": [], "val_acc": []}
    
    model.train()
    for epoch in range(30):
        epoch_loss = 0
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i:i+batch_size]
            yb = y_tensor[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Evaluasi validasi tiap epoch
        model.eval()
        with torch.no_grad():
            val_out = model(torch.tensor(Xva, dtype=torch.float32).to(device))
            val_pred = val_out.argmax(1).cpu().numpy()
            val_acc = accuracy_score(yva, val_pred)
            
        history["loss"].append(epoch_loss / (len(X_tensor) // batch_size))
        history["val_acc"].append(val_acc)
        model.train()
            
    model.eval()
    with torch.no_grad():
        pred_val = model(torch.tensor(Xva, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
        pred_test = model(torch.tensor(Xte, dtype=torch.float32).to(device)).argmax(1).cpu().numpy() if final and Xte is not None else None
        
    return pred_val, pred_test, yva, yte, history

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru"], help="Pilih LSTM atau GRU")
    a = ap.parse_args()
    
    t0 = time.time()
    pred_val, pred_test, yva, yte, history = latih_dan_prediksi(a.seed, a.final, a.model)
    durasi = time.time() - t0
    
    # Simpan hasil di folder results/NIM/model_type/
    out = AKAR / "results" / NIM / a.model
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    
    perintah = [
        sys.executable, "-m", "src.common.evaluate", 
        "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"), 
        "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)

    # ==========================================
    # --- MEMBUAT GRAFIK OTOMATIS ---
    # ==========================================
    # 1. Grafik Training Curves
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(history["loss"], color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='tab:blue')
    ax2.plot(history["val_acc"], color='tab:blue', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title(f'Model {a.model.upper()} - Training Curves')
    fig.tight_layout()
    plt.savefig(out / f"training_curves_{a.model}.png")
    plt.close()

    # 2. Grafik Confusion Matrix
    cm = confusion_matrix(yva, pred_val)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.POSTUR)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f'Model {a.model.upper()} - Confusion Matrix (Validasi)')
    plt.tight_layout()
    plt.savefig(out / f"confusion_matrix_{a.model}.png")
    plt.close()

    print(f"Grafik training curves & confusion matrix disimpan di {out}/")

if __name__ == "__main__":
    main()