from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# KONFIGURASI DIREKTORI
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

INPUT_DIR = ROOT_DIR / "output" / "heart" / "preprocessing"
KMEANS_DIR = ROOT_DIR / "output" / "heart" / "kmeans"
OUTPUT_DIR = ROOT_DIR / "output" / "heart" / "fcm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = INPUT_DIR / "heart_features_scaled_encoded.csv"
CLEANED_PATH = INPUT_DIR / "heart_cleaned.csv"
RANDOM_STATE = 42
NUMERIC_FEATURES = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak",
]


def load_best_k():
    """Membaca jumlah cluster terbaik dari baseline K-Means."""
    metrics = pd.read_csv(KMEANS_DIR / "heart_kmeans_final_metrics.csv")
    return int(metrics.iloc[0]["Jumlah_Cluster"])


def compute_metrics(X: np.ndarray, labels: np.ndarray):
    """Menghitung metrik internal untuk label cluster."""
    valid_mask = labels >= 0
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
    counts = (
        result["Cluster_FCM"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index()
    )
    counts.columns = ["Cluster", "Jumlah_Anggota"]
    counts["Persentase(%)"] = (counts["Jumlah_Anggota"] / len(result) * 100).round(2)
    counts.to_csv(output_path, index=False)
    return counts


def save_cluster_profile(result: pd.DataFrame, output_path: Path):
    profile = result.groupby("Cluster_FCM")[NUMERIC_FEATURES].mean().round(2)
    profile["Jumlah_Anggota"] = result.groupby("Cluster_FCM").size()
    profile.to_csv(output_path)
    return profile


def save_visualization(result: pd.DataFrame, X: np.ndarray, output_path: Path):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for cluster in sorted(result["Cluster_FCM"].unique()):
        mask = result["Cluster_FCM"] == cluster
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=40,
            alpha=0.8,
            label=f"Cluster {cluster}",
        )

    plt.title("Fuzzy C-Means - Heart Disease (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_distribution(counts: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(8, 5))
    plt.bar(counts["Cluster"].astype(str), counts["Jumlah_Anggota"])
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah Anggota")
    plt.title("Distribusi Cluster Fuzzy C-Means - Heart Disease")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run():
    print("=== FUZZY C-MEANS HEART DISEASE ===")

    X = pd.read_csv(DATA_PATH)
    df = pd.read_csv(CLEANED_PATH)

    if len(X) != len(df):
        raise ValueError("Jumlah baris data preprocessing dan data asli berbeda.")

    best_k = load_best_k()
    print("Jumlah data:", X.shape[0])
    print("Jumlah fitur:", X.shape[1])
    print("Jumlah cluster baseline K-Means:", best_k)

    data = X.to_numpy(dtype=float)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data.T,
        c=best_k,
        m=2,
        error=1e-5,
        maxiter=1000,
        seed=RANDOM_STATE,
    )

    if u.shape != (best_k, len(X)):
        raise ValueError("Dimensi membership FCM tidak sesuai dengan jumlah cluster dan data.")
    if not np.isfinite(u).all() or not np.allclose(u.sum(axis=0), 1.0, atol=1e-5):
        raise ValueError("Membership FCM tidak valid atau tidak berjumlah 1 per data.")

    labels = np.argmax(u, axis=0).astype(int)

    membership = pd.DataFrame(
        u.T,
        columns=[f"Membership_Cluster_{i}" for i in range(best_k)],
    )
    membership.to_csv(OUTPUT_DIR / "heart_fcm_membership.csv", index=False)

    result = df.copy()
    result["Cluster_FCM"] = labels
    if len(result) != len(X) or not np.array_equal(
        result["Cluster_FCM"].to_numpy(),
        membership.to_numpy().argmax(axis=1),
    ):
        raise ValueError("Label atau jumlah baris hasil FCM tidak cocok dengan membership.")
    result.to_csv(OUTPUT_DIR / "heart_fcm_clustered.csv", index=False)

    counts = save_cluster_counts(result, OUTPUT_DIR / "heart_fcm_cluster_counts.csv")
    save_cluster_profile(result, OUTPUT_DIR / "heart_fcm_cluster_profile.csv")
    save_distribution(counts, OUTPUT_DIR / "heart_fcm_distribution.png")
    save_visualization(result, data, OUTPUT_DIR / "heart_fcm_visualization.png")

    metrics = compute_metrics(data, labels)
    evaluation = pd.DataFrame([
        {
            "Dataset": "Heart Disease",
            "Metode": "FCM",
            "Jumlah_Cluster": best_k,
            "Silhouette": metrics["Silhouette"],
            "Davies_Bouldin": metrics["Davies_Bouldin"],
            "Calinski_Harabasz": metrics["Calinski_Harabasz"],
            "Jumlah_Data_Evaluasi": metrics["Jumlah_Data_Evaluasi"],
        }
    ])
    evaluation.to_csv(OUTPUT_DIR / "heart_fcm_evaluation.csv", index=False)

    final_metrics = {
        "Dataset": "Heart Disease",
        "Metode": "FCM",
        "Jumlah_Cluster": best_k,
        "Fuzziness": 2,
        "Error_Tolerance": 1e-5,
        "Max_Iter": 1000,
        "Objective_Function": float(jm[-1]),
        "Iterations": int(len(jm)),
        "Silhouette": metrics["Silhouette"],
        "Davies_Bouldin": metrics["Davies_Bouldin"],
        "Calinski_Harabasz": metrics["Calinski_Harabasz"],
        "Jumlah_Data_Evaluasi": metrics["Jumlah_Data_Evaluasi"],
    }
    pd.DataFrame([final_metrics]).to_csv(
        OUTPUT_DIR / "heart_fcm_final_metrics.csv",
        index=False,
    )

    print("\n=== DISTRIBUSI CLUSTER FCM ===")
    print(counts.to_string(index=False))
    print("\n=== METRIK MODEL FINAL ===")
    print(pd.DataFrame([final_metrics]).round(4).to_string(index=False))
    print("\nFCM Heart Disease selesai!")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
