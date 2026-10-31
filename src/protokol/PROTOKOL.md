# Protokol evaluasi tim — Penilaian postur duduk

Diisi Penjaga Data & Evaluasi pada Sprint 0, digabung ke `main` lewat *pull request* yang
disetujui pembimbing, lalu **dibekukan**. Perubahan setelah itu hanya lewat *pull request*
baru yang disetujui pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Penilaian postur duduk: klasifikasi 5 postur + regresi skor RULA dari urutan pose 10 fps, 8 detik; pembanding non-DL; batasan ukuran/latensi untuk pemasangan kembali di laptop meja |
| Sumber dataset | Rekaman tim sendiri di meja uji (kamera tiap 0,1 s), sesi operator Jum 6 Nov 15.00–16.00; arsip & kartu kasus: https://github.com/sains-data/capstone-dl-2026-starter |
| Jenis tugas (`TUGAS`) | klasifikasi (utama) + regresi (sekunder, skor RULA). Kontrak penilai starter memakai `TUGAS = "klasifikasi"`; RULA dinilai lewat `muat_rula()` + `evaluate --out results/<nim>/metrics_rula.json` |
| Target / label | `y_postur` bulat 0–4: tegak, membungkuk, bersandar, condong-kiri, condong-kanan. `y_rula` float = rerata skor RULA dua penilai (penilai 1: A, penilai 2: D); skor keduanya disimpan di `label.csv` untuk cek kesepakatan antar-penilai |
| Ukuran data latih / validasi / uji | 30 / 30 / 20 kondisi (total 80 = 8 partisipan × 5 postur × 2 ulangan). Tiap kondisi = 1 sampel urutan 70 frame (8 s @ 10 fps, 10 frame pertama dibuang). X = `[N, 70, J, 3]` float32 |
| Cara split | Berbasis grup, tanpa pengacakan (pengganti `split.indeks`, setara GroupKFold kartu kasus): **uji** = hold-out partisipan P07–P08; **validasi** = blok ulangan 2 dari P01–P06; **latih** = ulangan 1 dari P01–P06. Alasan: mencegah bocoran identitas orang/blok. Seed 2026 tetap masuk `data_hash` lewat penilai. Format data & kode: `src/protokol/data.py` |
| Metrik utama untuk komparasi | macro-F1 (klasifikasi postur) dan MAE (skor RULA); dilengkapi acc dan RMSE/R². Rerata ± simpangan baku 3 seed. Ditambah ukuran ONNX (KB) dan latensi (ms) di laptop meja |
| Praproses yang sama untuk semua model | 10 frame pertama tiap kondisi dibuang di loader (1 detik pertama). Normalisasi: parameter (mean/std) di-*fit* **dari data latih saja**, diterapkan ke validasi/uji. Tidak ada augmentasi yang memakai info validasi/uji. Data tim lain hanya boleh jadi uji tambahan (tidak masuk metrik utama); data bersama = rekaman program harian |
| Anggaran komputasi per run | ≤ 20 menit pelatihan CPU laptop; catat `--durasi` dan `--perangkat`. Batasan pemasangan: model diekspor ONNX, diukur ukurannya (KB) dan latensi inferensinya (ms) di laptop meja oleh anggota B |
| Evaluasi data uji | sekali, oleh Penjaga Data (A), pada sprint terakhir (`--final`). Termasuk prediksi RULA (`metrics_rula.json`) |
| `data_hash` saat dibekukan | diisi setelah data rekaman final masuk `data/` dan baseline jalan: ambil dari `results/baseline/metrics.json` |

## Catatan Sprint 0

1. Format berkas data didefinisikan di docstring `src/protokol/data.py` — operator rekaman
   wajib menghasilkan `data/pose.npz` + `data/label.csv` persis seperti itu.
2. `data/` tidak di-commit; yang di-commit hanya `MANIFEST.json`, protokol, dan kode.
3. Baseline starter jalan setelah data ada: `python -m src.common.baseline --seed 2026`.
4. Pembanding non-DL (statistik sudut sendi + kecepatan titik pose → Random Forest, anggota D)
   dijalankan sebagai model biasa di `models/<nim>/` — bukan bagian protokol.

Disetujui pembimbing pada: … (tautan *pull request*)
