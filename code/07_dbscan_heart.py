from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
INPUT_DIR = ROOT_DIR / "output" / "heart" / "preprocessing"
OUTPUT_DIR = ROOT_DIR / "output" / "heart" / "dbscan"
FEATURES_PATH = INPUT_DIR / "heart_features_scaled_encoded.csv"
CLEANED_PATH = INPUT_DIR / "heart_cleaned.csv"

EPS = 2.0
MIN_SAMPLES = 5
METRIC = "euclidean"
NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    non_noise = labels != -1
    valid_X = X[non_noise]
    valid_labels = labels[non_noise]
    cluster_count = len(np.unique(valid_labels))

    metrics = {
        "Silhouette": np.nan,
        "Davies_Bouldin": np.nan,
        "Calinski_Harabasz": np.nan,
        "Jumlah_Data_Evaluasi": int(non_noise.sum()),
        "Catatan_Evaluasi": (
            "Metrik dihitung hanya pada data non-noise (label -1 dikeluarkan)."
        ),
    }
    if cluster_count < 2 or len(valid_labels) <= cluster_count:
        metrics["Catatan_Evaluasi"] += (
            " Metrik tidak terdefinisi karena diperlukan minimal dua cluster "
            "dan jumlah observasi lebih besar dari jumlah cluster."
        )
        return metrics

    metrics.update(
        {
            "Silhouette": float(silhouette_score(valid_X, valid_labels)),
            "Davies_Bouldin": float(davies_bouldin_score(valid_X, valid_labels)),
            "Calinski_Harabasz": float(calinski_harabasz_score(valid_X, valid_labels)),
        }
    )
    return metrics


def save_cluster_counts(labels: np.ndarray, output_path: Path) -> pd.DataFrame:
    counts = (
        pd.Series(labels, name="Cluster_DBSCAN")
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Jumlah_Anggota")
    )
    counts["Persentase(%)"] = counts["Jumlah_Anggota"] / len(labels) * 100
    counts["Tipe"] = np.where(counts["Cluster"] == -1, "Noise", "Cluster")
    counts.to_csv(output_path, index=False)
    return counts


def save_cluster_profile(
    result: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    non_noise = result[result["Cluster_DBSCAN"] != -1]
    profile = non_noise.groupby("Cluster_DBSCAN")[NUMERIC_FEATURES].mean().round(2)
    profile["Jumlah_Anggota"] = non_noise.groupby("Cluster_DBSCAN").size()
    profile.to_csv(output_path)
    return profile


def save_distribution(counts: pd.DataFrame, output_path: Path) -> None:
    labels = [
        "Noise (-1)" if cluster == -1 else f"Cluster {cluster}"
        for cluster in counts["Cluster"]
    ]
    colors = ["black" if cluster == -1 else "C0" for cluster in counts["Cluster"]]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts["Jumlah_Anggota"], color=colors)
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah Anggota")
    plt.title("Distribusi Cluster DBSCAN - Heart Disease")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_pca_visualization(
    X: np.ndarray, labels: np.ndarray, output_path: Path
) -> None:
    projection = PCA(n_components=2, svd_solver="full").fit_transform(X)
    plt.figure(figsize=(8, 6))
    for cluster in sorted(value for value in np.unique(labels) if value != -1):
        mask = labels == cluster
        plt.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=35,
            alpha=0.65,
            label=f"Cluster {cluster}",
        )
    noise = labels == -1
    if noise.any():
        plt.scatter(
            projection[noise, 0],
            projection[noise, 1],
            c="black",
            marker="x",
            s=38,
            linewidths=1.2,
            label="Noise (-1)",
        )
    plt.title("DBSCAN Heart Disease (PCA untuk visualisasi saja)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_k_distance_plot(X: np.ndarray, output_path: Path) -> None:
    neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES, metric=METRIC).fit(X)
    distances, _ = neighbors.kneighbors(X)
    k_distance = np.sort(distances[:, -1])

    plt.figure(figsize=(8, 5))
    plt.plot(k_distance, alpha=0.8)
    plt.axhline(
        EPS,
        color="red",
        linestyle="--",
        label=f"eps = {EPS:g}, min_samples = {MIN_SAMPLES}",
    )
    plt.xlabel("Observasi (jarak diurutkan)")
    plt.ylabel(f"Jarak ke-{MIN_SAMPLES} tetangga terdekat")
    plt.title("K-Distance Graph - Heart Disease")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run() -> None:
    print("=== DBSCAN HEART DISEASE ===")
    X = pd.read_csv(FEATURES_PATH).to_numpy(dtype=float)
    original = pd.read_csv(CLEANED_PATH)
    if X.shape[0] != len(original):
        raise ValueError("Jumlah baris fitur preprocessing dan data Heart berbeda.")
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Fitur Heart harus berupa matriks dua dimensi dengan nilai finite.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric=METRIC).fit_predict(X)
    result = original.copy()
    result["Cluster_DBSCAN"] = labels
    result["Is_Noise"] = labels == -1
    result.to_csv(OUTPUT_DIR / "heart_dbscan_clustered.csv", index=False)

    counts = save_cluster_counts(
        labels, OUTPUT_DIR / "heart_dbscan_cluster_counts.csv"
    )
    profile = save_cluster_profile(
        result, OUTPUT_DIR / "heart_dbscan_cluster_profile.csv"
    )
    save_distribution(counts, OUTPUT_DIR / "heart_dbscan_distribution.png")
    save_pca_visualization(X, labels, OUTPUT_DIR / "heart_dbscan_pca.png")
    save_k_distance_plot(X, OUTPUT_DIR / "heart_dbscan_k_distance.png")

    cluster_sizes = np.unique(labels[labels != -1], return_counts=True)[1]
    noise_count = int(np.sum(labels == -1))
    metrics = compute_metrics(X, labels)
    final_metrics = {
        "Dataset": "Heart Disease",
        "Metode": "DBSCAN",
        "Metric": METRIC,
        "Epsilon": EPS,
        "Min_Samples": MIN_SAMPLES,
        "Jumlah_Cluster": len(cluster_sizes),
        "Cluster_Sizes": ",".join(map(str, sorted(cluster_sizes.tolist()))),
        "Noise_Count": noise_count,
        "Noise_Percentage": noise_count / len(labels),
        "Jumlah_Data": len(labels),
        "PCA_Digunakan_Untuk_Clustering": False,
        **metrics,
    }
    pd.DataFrame([final_metrics]).to_csv(
        OUTPUT_DIR / "heart_dbscan_final_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {
                **final_metrics,
                "Parameter": f"eps={EPS}; min_samples={MIN_SAMPLES}; metric={METRIC}",
                "Catatan": metrics["Catatan_Evaluasi"],
            }
        ]
    ).to_csv(OUTPUT_DIR / "heart_dbscan_evaluation.csv", index=False)

    print("Fitur:", X.shape[1], "| baris:", len(X))
    print("Cluster counts:\n", counts.to_string(index=False))
    print("Profil cluster:\n", profile.to_string())
    print("Metrik final:\n", pd.DataFrame([final_metrics]).to_string(index=False))
    print("Catatan:", metrics["Catatan_Evaluasi"])


if __name__ == "__main__":
    run()
