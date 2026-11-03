from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

# KONFIGURASI DIREKTORI
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

INPUT_DIR = ROOT_DIR / "output" / "mall" / "preprocessing"
OUTPUT_DIR = ROOT_DIR / "output" / "mall" / "dbscan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = INPUT_DIR / "mall_features_scaled_encoded.csv"
CLEANED_PATH = INPUT_DIR / "mall_cleaned.csv"
NUMERIC_FEATURES = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]


def compute_k_distance_plot(X: np.ndarray, k: int, eps: float, output_path: Path):
    """Membuat k-distance graph untuk menentukan epsilon."""
    k = min(k, X.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distance = np.sort(distances[:, -1])

    plt.figure(figsize=(8, 5))
    plt.plot(k_distance, marker="o", linestyle="-", alpha=0.8)
    plt.axhline(eps, color="red", linestyle="--", label=f"Epsilon terpilih = {eps:g}")
    plt.xlabel("Indeks Data")
    plt.ylabel(f"Jarak ke-{k} tetangga terdekat")
    plt.title("K-Distance Graph - Mall Customers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_candidate_table(X: np.ndarray):
    """Menguji epsilon di sekitar kurva jarak dan beberapa min_samples."""
    table = []
    eps_candidates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
    min_samples_candidates = [4, 5, 6, 8, 10]

    for eps in eps_candidates:
        for min_samples in min_samples_candidates:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            valid_mask = labels != -1
            valid_labels = labels[valid_mask]
            n_clusters = len(np.unique(valid_labels)) if valid_mask.any() else 0
            cluster_sizes = np.unique(valid_labels, return_counts=True)[1]
            noise_count = int(np.sum(labels == -1))
            noise_pct = noise_count / len(X)

            if n_clusters >= 2:
                sil = silhouette_score(X[valid_mask], valid_labels)
                dbi = davies_bouldin_score(X[valid_mask], valid_labels)
                ch = calinski_harabasz_score(X[valid_mask], valid_labels)
            else:
                sil = np.nan
                dbi = np.nan
                ch = np.nan

            table.append({
                "Epsilon": eps,
                "Min_Samples": min_samples,
                "Jumlah_Cluster": n_clusters,
                "Smallest_Cluster_Size": int(cluster_sizes.min()) if len(cluster_sizes) else 0,
                "Largest_Cluster_Size": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
                "Noise_Count": noise_count,
                "Noise_Percentage": noise_pct,
                "Silhouette": sil,
                "Davies_Bouldin": dbi,
                "Calinski_Harabasz": ch,
            })

    return pd.DataFrame(table)


def select_best_candidate(candidates: pd.DataFrame, min_samples: int):
    """Ambil epsilon terbesar yang masih memisahkan minimal dua cluster."""
    valid_candidates = candidates[
        (candidates["Min_Samples"] == min_samples)
        & (candidates["Jumlah_Cluster"] >= 2)
        & (candidates["Smallest_Cluster_Size"] >= min_samples)
    ].copy()

    if valid_candidates.empty:
        return None

    return valid_candidates.sort_values("Epsilon", ascending=False).iloc[0]


def compute_metrics(X: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -1
    valid_X = X[valid_mask]
    valid_labels = labels[valid_mask]

    if len(np.unique(valid_labels)) < 2:
        return {
            "Silhouette": np.nan,
            "Davies_Bouldin": np.nan,
            "Calinski_Harabasz": np.nan,
            "Jumlah_Data_Evaluasi": int(valid_X.shape[0]),
        }

    return {
        "Silhouette": float(silhouette_score(valid_X, valid_labels)),
        "Davies_Bouldin": float(davies_bouldin_score(valid_X, valid_labels)),
        "Calinski_Harabasz": float(calinski_harabasz_score(valid_X, valid_labels)),
        "Jumlah_Data_Evaluasi": int(valid_X.shape[0]),
    }


def save_cluster_counts(result: pd.DataFrame, output_path: Path):
    counts = result["Cluster_DBSCAN"].value_counts().sort_index().rename_axis("Cluster").reset_index()
    counts.columns = ["Cluster", "Jumlah_Anggota"]
    counts["Persentase(%)"] = (counts["Jumlah_Anggota"] / len(result) * 100).round(2)
    counts["Tipe"] = np.where(counts["Cluster"] == -1, "Noise", "Cluster")
    counts.to_csv(output_path, index=False)
    return counts


def save_cluster_profile(result: pd.DataFrame, output_path: Path):
    valid = result[result["Cluster_DBSCAN"] != -1].copy()
    profile = valid.groupby("Cluster_DBSCAN")[NUMERIC_FEATURES].mean().round(2)
    profile["Jumlah_Anggota"] = valid.groupby("Cluster_DBSCAN").size()
    profile.to_csv(output_path)
    return profile


def save_distribution(counts: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(8, 5))
    colors = ["black" if cluster == -1 else "C0" for cluster in counts["Cluster"]]
    labels = ["Noise (-1)" if cluster == -1 else str(cluster) for cluster in counts["Cluster"]]
    plt.bar(labels, counts["Jumlah_Anggota"], color=colors)
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah Anggota")
    plt.title("Distribusi Cluster DBSCAN - Mall Customers")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_visualization(result: pd.DataFrame, output_path: Path):
    valid = result[result["Cluster_DBSCAN"] != -1].copy()
    plt.figure(figsize=(8, 6))
    noise = result[result["Cluster_DBSCAN"] == -1]
    if not noise.empty:
        plt.scatter(
            noise["Annual Income (k$)"],
            noise["Spending Score (1-100)"],
            c="black",
            marker="x",
            s=45,
            linewidths=1.2,
            label="Noise (-1)",
        )

    for cluster in sorted(valid["Cluster_DBSCAN"].unique()):
        mask = valid["Cluster_DBSCAN"] == cluster
        plt.scatter(
            valid.loc[mask, "Annual Income (k$)"],
            valid.loc[mask, "Spending Score (1-100)"],
            s=50,
            alpha=0.8,
            label=f"Cluster {cluster}",
        )

    plt.title("DBSCAN - Mall Customers")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run():
    print("=== DBSCAN MALL CUSTOMERS ===")
    X = pd.read_csv(DATA_PATH).to_numpy(dtype=float)
    df = pd.read_csv(CLEANED_PATH)

    if X.shape[0] != len(df):
        raise ValueError("Jumlah baris data preprocessing dan data asli berbeda.")

    candidate_table = build_candidate_table(X)

    min_samples = X.shape[1] + 1
    selected = select_best_candidate(candidate_table, min_samples)
    if selected is None:
        raise ValueError("Tidak ada kombinasi DBSCAN yang menghasilkan setidaknya dua cluster valid.")

    eps = float(selected["Epsilon"])
    candidate_table["Selected"] = (
        (candidate_table["Epsilon"] == eps)
        & (candidate_table["Min_Samples"] == min_samples)
    )
    candidate_table.to_csv(OUTPUT_DIR / "mall_dbscan_evaluation.csv", index=False)
    compute_k_distance_plot(
        X,
        min_samples,
        eps,
        OUTPUT_DIR / "mall_dbscan_k_distance.png",
    )
    print("Parameter terpilih: eps=", eps, "min_samples=", min_samples)
    print("Alasan: epsilon terbesar pada grid yang masih mempertahankan >=2 cluster dengan min_samples=jumlah fitur+1.")
    print("Kandidat terbaik:\n", selected.to_frame().T.to_string(index=False))

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    result = df.copy()
    result["Cluster_DBSCAN"] = labels
    result["Is_Noise"] = labels == -1
    result.to_csv(OUTPUT_DIR / "mall_dbscan_clustered.csv", index=False)

    noise_count = int(np.sum(labels == -1))
    noise_pct = noise_count / len(df)
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    n_clusters = len(np.unique(valid_labels)) if valid_mask.any() else 0
    counts = save_cluster_counts(result, OUTPUT_DIR / "mall_dbscan_cluster_counts.csv")
    save_cluster_profile(result, OUTPUT_DIR / "mall_dbscan_cluster_profile.csv")
    save_distribution(counts, OUTPUT_DIR / "mall_dbscan_distribution.png")
    save_visualization(result, OUTPUT_DIR / "mall_dbscan_visualization.png")

    metrics = compute_metrics(X, labels)
    final_metrics = {
        "Dataset": "Mall Customers",
        "Metode": "DBSCAN",
        "Jumlah_Cluster": n_clusters,
        "Epsilon": eps,
        "Min_Samples": min_samples,
        "Smallest_Cluster_Size": int(selected["Smallest_Cluster_Size"]),
        "Largest_Cluster_Size": int(selected["Largest_Cluster_Size"]),
        "Noise_Count": noise_count,
        "Noise_Percentage": noise_pct,
        "Selection_Criteria": "Epsilon terbesar pada grid yang mempertahankan >=2 cluster; min_samples=jumlah fitur+1.",
        "Silhouette": metrics["Silhouette"],
        "Davies_Bouldin": metrics["Davies_Bouldin"],
        "Calinski_Harabasz": metrics["Calinski_Harabasz"],
        "Jumlah_Data_Evaluasi": metrics["Jumlah_Data_Evaluasi"],
    }
    pd.DataFrame([final_metrics]).to_csv(
        OUTPUT_DIR / "mall_dbscan_final_metrics.csv",
        index=False,
    )

    print("\n=== DISTRIBUSI CLUSTER DBSCAN ===")
    print(counts.to_string(index=False))
    print("\nJumlah noise:", noise_count)
    print("Persentase noise:", round(noise_pct * 100, 2), "%")
    print("\n=== METRIK MODEL FINAL ===")
    print(pd.DataFrame([final_metrics]).round(4).to_string(index=False))
    print("\nDBSCAN Mall Customers selesai!")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
