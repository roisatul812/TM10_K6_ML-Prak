# Analisis Clustering - ## Heart Disease

### Perbandingan utama (cakupan evaluasi masing-masing eksperimen)

| Algoritma | Jumlah cluster | Ukuran cluster | Noise | Noise (%) | Data dievaluasi | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| K-Means | 2 | 469,449 | 0 | 0.00% | 918 | 0.175169 | 2.009263 | 210.728750 |
| FCM | 2 | 460,458 | 0 | 0.00% | 918 | 0.175512 | 2.012030 | 210.520018 |
| DBSCAN | 2 | 778,1 | 139 | 15.14% | 779 | 0.095689 | 0.809253 | 1.469490 |

K-Means dan FCM dievaluasi pada seluruh data. Metrik DBSCAN dihitung hanya pada baris non-noise (label -1 dikeluarkan), sehingga tabel utama tidak memiliki cakupan observasi yang sama dan metrik DBSCAN tidak boleh dibandingkan langsung tanpa konteks.

### Perbandingan tambahan yang adil pada subset non-noise DBSCAN

| Algoritma | Jumlah cluster | Ukuran cluster | Noise | Noise (%) | Data dievaluasi | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| K-Means | 2 | 375,404 | 0 | 0.00% | 779 | 0.196626 | 1.852795 | 216.135622 |
| FCM | 2 | 367,412 | 0 | 0.00% | 779 | 0.196808 | 1.851084 | 215.958432 |
| DBSCAN | 2 | 778,1 | 139 | 15.14% | 779 | 0.095689 | 0.809253 | 1.469490 |

Semua algoritma pada tabel ini dievaluasi pada 779 baris observasi yang sama, yaitu baris non-noise DBSCAN. K-Means dan FCM tidak dilatih ulang; label yang telah tersimpan hanya digunakan pada subset tersebut. Tabel tambahan ini tidak menggantikan evaluasi utama.

### Jumlah cluster dan noise

| Algoritma | Jumlah cluster | Ukuran cluster | Noise | Noise (%) |
|---|---:|---|---:|---:|
| K-Means | 2 | 469,449 | 0 | 0.00% |
| FCM | 2 | 460,458 | 0 | 0.00% |
| DBSCAN | 2 | 778,1 | 139 | 15.14% |

### Hasil dan interpretasi metrik

- Silhouette: semakin tinggi semakin baik.
- Davies-Bouldin: semakin rendah semakin baik.
- Calinski-Harabasz: semakin tinggi semakin baik.

Pemenang numerik pada tabel utama (cakupan DBSCAN berbeda):
- Silhouette: FCM (0.175512; lebih tinggi lebih baik).
- Davies_Bouldin: DBSCAN (0.809253; lebih rendah lebih baik).
- Calinski_Harabasz: K-Means (210.728750; lebih tinggi lebih baik).

Pemenang pada perbandingan tambahan dengan baris observasi yang sama:
- Silhouette: FCM (0.196808; lebih tinggi lebih baik).
- Davies_Bouldin: DBSCAN (0.809253; lebih rendah lebih baik).
- Calinski_Harabasz: K-Means (216.135622; lebih tinggi lebih baik).

Pemenang pada tabel utama harus ditafsirkan hati-hati karena metrik DBSCAN mengecualikan noise sedangkan K-Means/FCM memakai seluruh data. Nilai metrik internal tidak mengukur accuracy dan hasilnya juga bergantung pada representasi fitur serta jumlah observasi yang dievaluasi.

### Konfigurasi DBSCAN Heart

- DBSCAN menggunakan matriks fitur preprocessing yang tersedia, tanpa preprocessing ulang.
- Parameter: eps=2.0, min_samples=5, metric=Euclidean.
- Hasil: 2 cluster, ukuran 778,1; 139 noise dari 918 baris (15.14%).
- Metrik DBSCAN dihitung pada 779 baris non-noise saja.
- Satu cluster hanya beranggotakan satu baris, sehingga struktur cluster DBSCAN perlu ditafsirkan dengan hati-hati. Banyaknya noise yang lebih sedikit tidak dengan sendirinya menunjukkan kualitas clustering yang lebih baik.
- PCA digunakan hanya untuk visualisasi DBSCAN, bukan sebagai input clustering.
