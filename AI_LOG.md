# AI_LOG — Tim 05 Capstone Deep Learning

Satu baris per pemakaan AI yang bermakna (ChatGPT, Claude, Copilot, Bravais, dan sejenisnya).
Ringkas dan jujur; transkrip penuh tidak diminta. Yang dinilai bukan "pakai AI atau tidak",
melainkan "mengerti atau tidak" — isi kolom terakhir dengan apa yang Anda ubah dari keluarannya.

| Tanggal | Siapa (NIM) | Alat | Untuk apa | Apa yang diubah dari keluarannya |
|---|---|---|---|---|
| 2026-11-27 | 122450100 | Gemini | Kerangka preprocessing & training 1D-CNN | Perbaikan offset `k+2`, split *subject-independent*, dan kriteria *early stopping* F1-Macro. |
| 2026-11-27 | 122450058 | DeepSeek | Membantu debugging error path `data.py`, penyesuaian arsitektur RNN, pembuatan skrip `train_rnn.py`, dan perbaikan `06_compare_models.py`. | Mengganti path dari `pose.npz` ke `dataset_kondisi.npz`, menambahkan argumen `--model` (LSTM/GRU), membuat grafik otomatis (training curves & confusion matrix), dan memperbaiki format pembacaan `metrics.json` agar LSTM/GRU muncul di grafik perbandingan. |