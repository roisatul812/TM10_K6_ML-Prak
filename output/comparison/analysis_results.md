# Perbandingan Hasil Clustering

## Mall Customers

### Konfigurasi DBSCAN

- Fitur: Age, Annual Income, Spending Score (terstandar; tanpa Gender)
- Parameter: eps=0.6, min_samples=5
- Kriteria pemilihan: >= 2 cluster, noise <= 15%, tiap cluster >= 5% data; dipilih Silhouette tertinggi.
- Status: Memenuhi kriteria

### Ruang fitur: Seluruh fitur terstandar + one-hot (ruang K-Means/FCM)

**Cakupan evaluasi masing-masing metode**

| Algoritma | Valid | Jumlah cluster | Ukuran cluster | Noise (%) | Data dievaluasi | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---:|---|---:|---:|---:|---:|---:|
| K-Means | Ya | 4 | 65,57,38,40 | 0.00% | 200 | 0.319470 | 1.179273 | 85.148484 |
| FCM | Ya | 4 | 35,63,62,40 | 0.00% | 200 | 0.319119 | 1.165554 | 84.652464 |
| DBSCAN | Ya | 2 | 137,35 | 14.00% | 172 | 0.220601 | 1.288921 | 43.459616 |

K-Means dan FCM dievaluasi pada seluruh data; metrik DBSCAN hanya pada baris non-noise.

**Perbandingan pada subset non-noise DBSCAN yang sama**

| Algoritma | Valid | Jumlah cluster | Ukuran cluster | Noise (%) | Data dievaluasi | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---:|---|---:|---:|---:|---:|---:|
| K-Means | Ya | 4 | 62,52,23,35 | 0.00% | 172 | 0.324114 | 1.205888 | 77.154635 |
| FCM | Ya | 4 | 20,61,56,35 | 0.00% | 172 | 0.323260 | 1.177870 | 77.377946 |
| DBSCAN | Ya | 2 | 137,35 | 14.00% | 172 | 0.220601 | 1.288921 | 43.459616 |

Pemenang metrik (hanya di antara metode yang valid), cakupan masing-masing:
- Silhouette: K-Means (0.319470; lebih tinggi lebih baik).
- Davies_Bouldin: FCM (1.165554; lebih rendah lebih baik).
- Calinski_Harabasz: K-Means (85.148484; lebih tinggi lebih baik).

Pemenang metrik pada subset yang sama:
- Silhouette: K-Means (0.324114; lebih tinggi lebih baik).
- Davies_Bouldin: FCM (1.177870; lebih rendah lebih baik).
- Calinski_Harabasz: FCM (77.377946; lebih tinggi lebih baik).


### Ruang fitur: Fitur numerik saja (ruang DBSCAN)

**Cakupan evaluasi masing-masing metode**

| Algoritma | Valid | Jumlah cluster | Ukuran cluster | Noise (%) | Data dievaluasi | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---:|---|---:|---:|---:|---:|---:|
| K-Means | Ya | 4 | 65,57,38,40 | 0.00% | 200 | 0.404669 | 0.933460 | 125.390948 |
| FCM | Ya | 4 | 35,63,62,40 | 0.00% | 200 | 0.402359 | 0.921777 | 123.532254 |
| DBSCAN | Ya | 2 | 137,35 | 14.00% | 172 | 0.273047 | 1.020402 | 54.421388 |

K-Means dan FCM dievaluasi pada seluruh data; metrik DBSCAN hanya pada baris non-noise.

**Perbandingan pada subset non-noise DBSCAN yang sama**

| Algoritma | Valid | Jumlah cluster | Ukuran cluster | Noise (%) | Data dievaluasi | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---:|---|---:|---:|---:|---:|---:|
| K-Means | Ya | 4 | 62,52,23,35 | 0.00% | 172 | 0.422569 | 0.898133 | 123.580661 |
| FCM | Ya | 4 | 20,61,56,35 | 0.00% | 172 | 0.419199 | 0.875212 | 122.505270 |
| DBSCAN | Ya | 2 | 137,35 | 14.00% | 172 | 0.273047 | 1.020402 | 54.421388 |

Pemenang metrik (hanya di antara metode yang valid), cakupan masing-masing:
- Silhouette: K-Means (0.404669; lebih tinggi lebih baik).
- Davies_Bouldin: FCM (0.921777; lebih rendah lebih baik).
- Calinski_Harabasz: K-Means (125.390948; lebih tinggi lebih baik).

Pemenang metrik pada subset yang sama:
- Silhouette: K-Means (0.422569; lebih tinggi lebih baik).
- Davies_Bouldin: FCM (0.875212; lebih rendah lebih baik).
- Calinski_Harabasz: K-Means (123.580661; lebih tinggi lebih baik).


### Jumlah cluster dan noise

| Algoritma | Jumlah cluster | Ukuran cluster | Noise | Noise (%) |
|---|---:|---|---:|---:|
| K-Means | 4 | 65,57,38,40 | 0 | 0.00% |
| FCM | 4 | 35,63,62,40 | 0 | 0.00% |
| DBSCAN | 2 | 137,35 | 28 | 14.00% |

### Kemiripan label antar metode (Adjusted Rand Index)

- K-Means vs FCM: 0.932
- K-Means vs DBSCAN: 0.302
- FCM vs DBSCAN: 0.314

Catatan: Silhouette, Davies-Bouldin, dan Calinski-Harabasz adalah metrik internal, bukan accuracy; nilainya bergantung pada ruang fitur dan data yang dievaluasi. ARI untuk DBSCAN menganggap label noise (-1) sebagai satu kelompok.

## Heart Disease

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

