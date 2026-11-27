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
from sklearn.metrics import accuracy_score, f1_score

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

def latih_dan_prediksi(seed: int, final: bool, model_type: str = "lstm") -> tuple[np.ndarray, np.ndarray | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Muat data via data.py
    Xtr, ytr, Xva, _, Xte, _ = data.muat()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelRNN(rnn_type=model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Data loader sederhana
    batch_size = 32
    X_tensor = torch.tensor(Xtr, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(ytr, dtype=torch.long).to(device)
    
    model.train()
    for epoch in range(30):
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i:i+batch_size]
            yb = y_tensor[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        pred_val = model(torch.tensor(Xva, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
        pred_test = model(torch.tensor(Xte, dtype=torch.float32).to(device)).argmax(1).cpu().numpy() if final and Xte is not None else None
        
    return pred_val, pred_test

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru"], help="Pilih LSTM atau GRU")
    a = ap.parse_args()
    
    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final, a.model)
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

if __name__ == "__main__":
    main()