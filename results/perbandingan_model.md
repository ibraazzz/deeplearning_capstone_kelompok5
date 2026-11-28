# Perbandingan Model – Tim 05 Penilaian Postur Duduk

| Model | Val F1-Macro | Val Acc | Test F1-Macro | Test Acc | #Params | Waktu (s) |
|-------|:-----------:|:-------:|:-------------:|:--------:|:-------:|:---------:|
| Random Forest | 0.6133 | 0.6250 | 0.6788 | 0.6875 | — | 2.2 |
| 1D-CNN | 0.5676 | 0.6250 | 0.2467 | 0.3750 | 480389 | 0.4 |
| RNN (GRU) | 0.1600 | 0.2500 | 0.2633 | 0.2500 | — | 4.1 |
| RNN (LSTM) | 0.1511 | 0.1875 | 0.1667 | 0.1875 | — | 1.8 |
| Transformer | — | — | — | — | — | — |
| Autoencoder | — | — | — | — | — | — |

> *Metrik utama: F1-Score Macro (5 kelas postur duduk). Split per partisipan (subject-independent, seed=2026).*

## F1 per Kelas Postur

| Model | tegak | membungkuk | bersandar | condong ke kiri | condong ke kanan |
|-------|------|------|------|------|------|
| 1D-CNN | 0.000 | 0.500 | 0.000 | 0.400 | 0.333 |
| Random Forest | 0.667 | 0.727 | 0.667 | 0.667 | 0.667 |