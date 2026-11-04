# Analisis Clustering - ## Heart Disease

### Konfigurasi DBSCAN

- Fitur: seluruh fitur terstandar + one-hot
- Parameter: eps=2.6, min_samples=4
- Kriteria pemilihan: >= 2 cluster, noise <= 15%, tiap cluster >= 5% data; dipilih Silhouette tertinggi.
- Status: Tidak memenuhi kriteria ukuran cluster (cadangan)

### Ruang fitur: Seluruh fitur terstandar + one-hot (ruang K-Means/FCM)

**Cakupan evaluasi masing-masing metode**

| Algoritma | Valid | Jumlah cluster | Ukuran cluster | Noise (%) | Data dievaluasi | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---:|---|---:|---:|---:|---:|---:|
| K-Means | Ya | 2 | 469,449 | 0.00% | 918 | 0.175169 | 2.009263 | 210.728750 |
| FCM | Ya | 2 | 460,458 | 0.00% | 918 | 0.175512 | 2.012030 | 210.520018 |
| DBSCAN | Tidak | 2 | 906,4 | 0.87% | 910 | 0.357515 | 0.827725 | 13.811272 |

K-Means dan FCM dievaluasi pada seluruh data; metrik DBSCAN hanya pada baris non-noise.

**Perbandingan pada subset non-noise DBSCAN yang sama**

| Algoritma | Valid | Jumlah cluster | Ukuran cluster | Noise (%) | Data dievaluasi | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---:|---|---:|---:|---:|---:|---:|
| K-Means | Ya | 2 | 462,448 | 0.00% | 910 | 0.176893 | 1.996857 | 213.601014 |
| FCM | Ya | 2 | 453,457 | 0.00% | 910 | 0.177109 | 1.999153 | 213.348000 |
| DBSCAN | Tidak | 2 | 906,4 | 0.87% | 910 | 0.357515 | 0.827725 | 13.811272 |

Pemenang metrik (hanya di antara metode yang valid), cakupan masing-masing:
- Silhouette: FCM (0.175512; lebih tinggi lebih baik).
- Davies_Bouldin: K-Means (2.009263; lebih rendah lebih baik).
- Calinski_Harabasz: K-Means (210.728750; lebih tinggi lebih baik).

Pemenang metrik pada subset yang sama:
- Silhouette: FCM (0.177109; lebih tinggi lebih baik).
- Davies_Bouldin: K-Means (1.996857; lebih rendah lebih baik).
- Calinski_Harabasz: K-Means (213.601014; lebih tinggi lebih baik).

Catatan: DBSCAN ditandai TIDAK VALID (cluster terkecil 4 baris (< 5% data)), sehingga nilai metriknya tidak dihitung sebagai pemenang dan tidak boleh ditafsirkan sebagai kualitas yang baik.

### Jumlah cluster dan noise

| Algoritma | Jumlah cluster | Ukuran cluster | Noise | Noise (%) |
|---|---:|---|---:|---:|
| K-Means | 2 | 469,449 | 0 | 0.00% |
| FCM | 2 | 460,458 | 0 | 0.00% |
| DBSCAN | 2 | 906,4 | 8 | 0.87% |

### Kemiripan label antar metode (Adjusted Rand Index)

- K-Means vs FCM: 0.953
- K-Means vs DBSCAN: -0.000
- FCM vs DBSCAN: 0.000

Catatan: Silhouette, Davies-Bouldin, dan Calinski-Harabasz adalah metrik internal, bukan accuracy; nilainya bergantung pada ruang fitur dan data yang dievaluasi. ARI untuk DBSCAN menganggap label noise (-1) sebagai satu kelompok.
