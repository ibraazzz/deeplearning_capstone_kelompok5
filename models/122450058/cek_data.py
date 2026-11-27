import numpy as np
arsip = np.load('data/processed/pose.npz')
print('Jumlah kunci:', len(arsip))
print('Contoh kunci:', list(arsip.keys())[:10])