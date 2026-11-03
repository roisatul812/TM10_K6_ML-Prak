
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# KONFIGURASI DIREKTORI

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

INPUT_DIR = ROOT_DIR / "output" / "mall" / "preprocessing"
OUTPUT_DIR = ROOT_DIR / "output" / "mall" / "kmeans"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = INPUT_DIR / "mall_features_scaled_encoded.csv"
CLEANED_PATH = INPUT_DIR / "mall_cleaned.csv"

RANDOM_STATE = 42
K_RANGE = range(2, 7)


def run():
    print("=== K-MEANS MALL CUSTOMERS ===")

    # 1. Membaca hasil preprocessing
    X = pd.read_csv(DATA_PATH)
    df = pd.read_csv(CLEANED_PATH)

    if len(X) != len(df):
        raise ValueError("Jumlah baris data preprocessing dan data asli berbeda.")

    print("Jumlah data:", X.shape[0])
    print("Jumlah fitur:", X.shape[1])

    # 2. Menguji beberapa nilai K
    records = []

    for k in K_RANGE:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=RANDOM_STATE
        )

        labels = model.fit_predict(X)

        records.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette": silhouette_score(X, labels),
            "Davies_Bouldin": davies_bouldin_score(X, labels),
            "Calinski_Harabasz": calinski_harabasz_score(X, labels)
        })

    evaluation = pd.DataFrame(records)

    print("\n=== HASIL EVALUASI K-MEANS ===")
    print(evaluation.round(4).to_string(index=False))

    evaluation.to_csv(
        OUTPUT_DIR / "mall_kmeans_evaluation.csv",
        index=False
    )

    # 3. Menentukan K berdasarkan Silhouette tertinggi
    best_index = evaluation["Silhouette"].idxmax()
    best_k = int(evaluation.loc[best_index, "K"])

    print("\nK terbaik berdasarkan Silhouette:", best_k)

    # 4. Visualisasi Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(
        evaluation["K"],
        evaluation["Inertia"],
        marker="o"
    )
    plt.xlabel("Jumlah Cluster (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method - K-Means Mall Customers")
    plt.xticks(list(K_RANGE))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "mall_kmeans_elbow.png",
        dpi=300
    )
    plt.close()

    # 5. Visualisasi Silhouette Score
    plt.figure(figsize=(8, 5))
    plt.plot(
        evaluation["K"],
        evaluation["Silhouette"],
        marker="o"
    )
    plt.xlabel("Jumlah Cluster (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score - K-Means Mall Customers")
    plt.xticks(list(K_RANGE))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "mall_kmeans_silhouette.png",
        dpi=300
    )
    plt.close()

    # 6. Membuat model final
    final_model = KMeans(
        n_clusters=best_k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE
    )

    labels = final_model.fit_predict(X)

    # 7. Menambahkan label cluster
    result = df.copy()
    result["Cluster_KMeans"] = labels

    result.to_csv(
        OUTPUT_DIR / "mall_kmeans_clustered.csv",
        index=False
    )

    # 8. Menghitung distribusi cluster
    counts = (
        result["Cluster_KMeans"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index()
    )

    counts.columns = ["Cluster", "Jumlah_Anggota"]

    counts["Persentase(%)"] = (
        counts["Jumlah_Anggota"] / len(result) * 100
    ).round(2)

    counts.to_csv(
        OUTPUT_DIR / "mall_kmeans_cluster_counts.csv",
        index=False
    )

    print("\n=== DISTRIBUSI CLUSTER ===")
    print(counts.to_string(index=False))

    # 9. Profil cluster berdasarkan fitur numerik asli
    numeric_features = [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    profile = result.groupby("Cluster_KMeans")[
        numeric_features
    ].mean().round(2)

    profile["Jumlah_Anggota"] = (
        result.groupby("Cluster_KMeans").size()
    )

    profile.to_csv(
        OUTPUT_DIR / "mall_kmeans_cluster_profile.csv"
    )

    # 10. Visualisasi distribusi cluster
    plt.figure(figsize=(8, 5))
    plt.bar(
        counts["Cluster"].astype(str),
        counts["Jumlah_Anggota"]
    )
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah Anggota")
    plt.title("Distribusi Cluster K-Means - Mall Customers")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "mall_kmeans_distribution.png",
        dpi=300
    )
    plt.close()

    # 11. Menyimpan metrik final
    final_metrics = {
        "Dataset": "Mall Customers",
        "Metode": "K-Means",
        "Jumlah_Cluster": best_k,
        "Inertia": final_model.inertia_,
        "Silhouette": silhouette_score(X, labels),
        "Davies_Bouldin": davies_bouldin_score(X, labels),
        "Calinski_Harabasz": calinski_harabasz_score(X, labels)
    }

    pd.DataFrame([final_metrics]).to_csv(
        OUTPUT_DIR / "mall_kmeans_final_metrics.csv",
        index=False
    )

    print("\n=== METRIK MODEL FINAL ===")
    print(
        pd.DataFrame([final_metrics])
        .round(4)
        .to_string(index=False)
    )

    print("\nK-Means Mall Customers selesai!")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
