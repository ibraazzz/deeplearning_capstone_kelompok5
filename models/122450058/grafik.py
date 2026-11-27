import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Lokasi folder hasil LSTM
base_dir = Path("results/122450058/lstm")

# Pastikan file metrics.json ada
if not (base_dir / "metrics.json").exists():
    print("Error: File metrics.json tidak ditemukan.")
    print("Jalankan dulu: python models/122450058/run.py --seed 1 --model lstm --final")
    exit(1)

with open(base_dir / "metrics.json") as f:
    lstm_metrics = json.load(f)

# Ambil nilai metrik
val_f1 = lstm_metrics["metrik"]["val_macro_f1"]
test_f1 = lstm_metrics["metrik"]["test_macro_f1"]

# Buat grafik batang
labels = ['Val F1-Macro', 'Test F1-Macro']
values = [val_f1, test_f1]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 5))
rects = ax.bar(x, values, width, color=['tab:blue', 'tab:orange'])

ax.set_ylabel('F1-Score Macro')
ax.set_title('Model LSTM - F1-Score Macro (Validasi vs Uji)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0)

# Tampilkan nilai di atas bar
for rect in rects:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
# Simpan gambar di folder LSTM
plt.savefig(base_dir / "grafik_lstm.png")
print(f"Grafik LSTM berhasil dibuat di {base_dir}/grafik_lstm.png")