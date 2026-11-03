# TM 10 - Person 1: Preprocessing dan K-Means

Project ini menjalankan preprocessing dan K-Means pada **dua dataset secara terpisah**:
1. `heart.csv` (Heart Disease)
2. `Mall_Customers.csv` (Mall Customers)

## Struktur folder

```text
TM10_Person1_SourceCode/
├── dataset/
│   ├── heart.csv
│   └── Mall_Customers.csv
├── code/
│   └── run_person1.py
├── output/
└── requirements.txt
```

## Instalasi

Disarankan menggunakan Python 3.10 atau lebih baru.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Install library:
```bash
pip install -r requirements.txt
```

## Menjalankan program

Jalankan dari direktori utama `TM10_Person1_SourceCode`:

```bash
python code/run_person1.py
```

## Alur program

1. Membaca dataset dan menampilkan ukuran, tipe data, missing value, duplikasi, serta statistik deskriptif.
2. Menghapus baris duplikat.
3. Pada Heart Disease, nilai 0 di `RestingBP` dan `Cholesterol` diperlakukan sebagai nilai tidak valid dan diganti median dari nilai positif, mengikuti pendekatan pada laporan TM 9.
4. Tidak menggunakan `HeartDisease` sebagai fitur clustering. Kolom tersebut tetap disimpan di CSV hasil agar dapat dipakai untuk interpretasi terpisah.
5. Tidak menggunakan `CustomerID` sebagai fitur Mall Customers.
6. Melakukan imputasi median untuk fitur numerik dan modus untuk fitur kategorikal jika ditemukan missing value.
7. Melakukan `StandardScaler` pada fitur numerik dan `OneHotEncoder` pada fitur kategorikal.
8. Menguji K-Means untuk K=2 sampai K=10.
9. Menghitung Inertia, Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Index.
10. Memilih K final secara otomatis berdasarkan Silhouette Score tertinggi, kemudian menyimpan hasil clustering dan visualisasi.

## Fitur yang digunakan

### Heart Disease
Numerik:
- Age
- RestingBP
- Cholesterol
- MaxHR
- Oldpeak

Kategorikal:
- Sex
- ChestPainType
- FastingBS
- RestingECG
- ExerciseAngina
- ST_Slope

Tidak digunakan sebagai fitur: `HeartDisease` (label).

### Mall Customers
Numerik:
- Age
- Annual Income (k$)
- Spending Score (1-100)

Kategorikal:
- Gender

Tidak digunakan sebagai fitur: `CustomerID` (identitas).

## Output

Program membuat folder `output/heart/` dan `output/mall/`. Masing-masing berisi:
- `*_cleaned.csv`: dataset setelah pemeriksaan dan pembersihan.
- `*_features_scaled_encoded.csv`: matriks fitur setelah scaling dan encoding.
- `*_kmeans_clustered.csv`: data asli dengan kolom `Cluster`.
- `kmeans_evaluation.csv`: evaluasi untuk setiap nilai K.
- `kmeans_final_metrics.csv`: metrik model final.
- `kmeans_cluster_counts.csv`: jumlah anggota tiap cluster.
- `kmeans_cluster_profile.csv`: rata-rata fitur numerik asli per cluster.
- `kmeans_elbow.png`: grafik Elbow.
- `kmeans_silhouette.png`: grafik Silhouette.
- `kmeans_cluster_distribution.png`: grafik jumlah anggota per cluster.
- `kmeans_pca_visualization.png`: visualisasi 2D dengan PCA.

## Catatan metodologis penting untuk kelompok

- Untuk perbandingan K-Means, Fuzzy C-Means, dan DBSCAN, Person 2 dan Person 3 harus menggunakan **baris data, fitur, encoding, dan scaling yang sama** untuk dataset yang sama.
- Gunakan file `*_features_scaled_encoded.csv` sebagai matriks input bersama, dengan urutan baris yang sama seperti `*_cleaned.csv`.
- `HeartDisease` dan `CustomerID` tidak boleh dimasukkan ke matriks fitur.
- `kmeans_pca_visualization.png` hanya untuk visualisasi. Model K-Means tetap dilatih pada seluruh fitur hasil preprocessing, bukan hanya dua komponen PCA.
- Nilai K final dipilih otomatis berdasarkan Silhouette tertinggi sebagai baseline program. Tetap periksa grafik Elbow, distribusi anggota, dan interpretasi cluster sebelum menetapkan K pada laporan.
- Clustering tidak memiliki label kelas acuan, sehingga Silhouette, Davies-Bouldin, dan Calinski-Harabasz adalah **metrik evaluasi internal**, bukan accuracy klasifikasi.
- DBSCAN dapat menghasilkan label noise (-1). Saat menghitung metrik DBSCAN, anggota noise perlu ditangani dan dijelaskan secara konsisten; jangan menganggap noise sebagai cluster biasa tanpa penjelasan.
- Karena One-Hot Encoding memperluas dimensi fitur, hasil K-Means pada data campuran ini merupakan pendekatan numerik atas fitur kategorikal. Jelaskan keputusan ini di laporan.

## Pembagian tanggung jawab

File ini mencakup bagian Person 1: preprocessing bersama dan K-Means baseline. Person 2 mengimplementasikan Fuzzy C-Means, sedangkan Person 3 mengimplementasikan DBSCAN dan mengintegrasikan perbandingan akhir.
