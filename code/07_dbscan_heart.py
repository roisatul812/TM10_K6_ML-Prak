from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
INPUT_DIR = ROOT_DIR / "output" / "heart" / "preprocessing"
OUTPUT_DIR = ROOT_DIR / "output" / "heart" / "dbscan"
FEATURES_PATH = INPUT_DIR / "heart_features_scaled_encoded.csv"
CLEANED_PATH = INPUT_DIR / "heart_cleaned.csv"

METRIC = "euclidean"
NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

MAX_NOISE_PCT = 15.0
MIN_CLUSTER_PCT = 5.0
MIN_CLUSTERS = 2
EPS_GRID = np.round(np.arange(1.0, 5.001, 0.1), 3)
MIN_SAMPLES_GRID = [4, 5, 6, 8, 10, 15, 20, 30]
CRITERIA_TEXT = (
    f">= {MIN_CLUSTERS} cluster, noise <= {MAX_NOISE_PCT:g}%, tiap cluster >= {MIN_CLUSTER_PCT:g}% data; "
    "dipilih Silhouette tertinggi."
)


def compute_metrics(X, labels):
    mask = labels != -1
    out = {"Silhouette": np.nan, "Davies_Bouldin": np.nan, "Calinski_Harabasz": np.nan,
           "Jumlah_Data_Evaluasi": int(mask.sum()),
           "Catatan_Evaluasi": "Metrik dihitung hanya pada data non-noise (label -1 dikeluarkan)."}
    if len(np.unique(labels[mask])) < 2:
        out["Catatan_Evaluasi"] += " Metrik tidak terdefinisi: kurang dari dua cluster."
        return out
    out.update({
        "Silhouette": float(silhouette_score(X[mask], labels[mask])),
        "Davies_Bouldin": float(davies_bouldin_score(X[mask], labels[mask])),
        "Calinski_Harabasz": float(calinski_harabasz_score(X[mask], labels[mask])),
    })
    return out


def build_grid(X):
    rows, n = [], len(X)
    for ms in MIN_SAMPLES_GRID:
        for eps in EPS_GRID:
            labels = DBSCAN(eps=float(eps), min_samples=ms, metric=METRIC).fit_predict(X)
            mask = labels != -1
            k = len(np.unique(labels[mask])) if mask.any() else 0
            sizes = np.bincount(labels[mask]) if mask.any() else np.array([0])
            m = compute_metrics(X, labels) if k >= 2 else {
                "Silhouette": np.nan, "Davies_Bouldin": np.nan, "Calinski_Harabasz": np.nan}
            noise_pct = float((~mask).mean() * 100)
            rows.append({
                "Epsilon": float(eps), "Min_Samples": ms, "Jumlah_Cluster": k,
                "Smallest_Cluster_Size": int(sizes.min()), "Largest_Cluster_Size": int(sizes.max()),
                "Noise_Count": int((~mask).sum()), "Noise_Percentage": noise_pct / 100,
                "Silhouette": m["Silhouette"], "Davies_Bouldin": m["Davies_Bouldin"],
                "Calinski_Harabasz": m["Calinski_Harabasz"],
                "Memenuhi_Kriteria": bool(k >= MIN_CLUSTERS and noise_pct <= MAX_NOISE_PCT
                                          and sizes.min() >= n * MIN_CLUSTER_PCT / 100),
            })
    return pd.DataFrame(rows)


def select_parameters(grid):
    ok = grid[grid["Memenuhi_Kriteria"]]
    if not ok.empty:
        return ok.sort_values(["Silhouette", "Noise_Count"], ascending=[False, True]).iloc[0], "Memenuhi kriteria"
    relaxed = grid[(grid["Jumlah_Cluster"] >= MIN_CLUSTERS) & (grid["Noise_Percentage"] * 100 <= MAX_NOISE_PCT)]
    if relaxed.empty:
        raise ValueError("Tidak ada kombinasi DBSCAN dengan >= 2 cluster dan noise yang wajar.")
    return (relaxed.sort_values("Silhouette", ascending=False).iloc[0],
            "Tidak memenuhi kriteria ukuran cluster (cadangan)")


def save_cluster_counts(labels, path):
    c = (pd.Series(labels, name="Cluster_DBSCAN").value_counts().sort_index()
         .rename_axis("Cluster").reset_index(name="Jumlah_Anggota"))
    c["Persentase(%)"] = c["Jumlah_Anggota"] / len(labels) * 100
    c["Tipe"] = np.where(c["Cluster"] == -1, "Noise", "Cluster")
    c.to_csv(path, index=False)
    return c


def save_cluster_profile(result, path):
    v = result[result["Cluster_DBSCAN"] != -1]
    p = v.groupby("Cluster_DBSCAN")[NUMERIC_FEATURES].mean().round(2)
    p["Jumlah_Anggota"] = v.groupby("Cluster_DBSCAN").size()
    p.to_csv(path)
    return p


def save_distribution(counts, path):
    names = ["Noise (-1)" if c == -1 else f"Cluster {c}" for c in counts["Cluster"]]
    colors = ["black" if c == -1 else "C0" for c in counts["Cluster"]]
    plt.figure(figsize=(8, 5))
    plt.bar(names, counts["Jumlah_Anggota"], color=colors)
    plt.ylabel("Jumlah Anggota"); plt.title("Distribusi Cluster DBSCAN - Heart Disease")
    plt.grid(axis="y", alpha=0.25); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def save_pca_visualization(X, labels, eps, ms, path):
    proj = PCA(n_components=2, svd_solver="full").fit_transform(X)
    plt.figure(figsize=(8, 6))
    for c in sorted(v for v in np.unique(labels) if v != -1):
        m = labels == c
        plt.scatter(proj[m, 0], proj[m, 1], s=35, alpha=0.65, label=f"Cluster {c} (n={m.sum()})")
    noise = labels == -1
    if noise.any():
        plt.scatter(proj[noise, 0], proj[noise, 1], c="black", marker="x", s=38, linewidths=1.2,
                    label=f"Noise (-1) (n={noise.sum()})")
    plt.title(f"DBSCAN Heart Disease (PCA hanya untuk visualisasi)\neps={eps:g}, min_samples={ms}")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()


def save_k_distance_plot(X, ms, eps, path):
    d, _ = NearestNeighbors(n_neighbors=ms, metric=METRIC).fit(X).kneighbors(X)
    kd = np.sort(d[:, -1])
    plt.figure(figsize=(8, 5))
    plt.plot(kd, alpha=0.8)
    plt.axhline(eps, color="red", ls="--", label=f"eps = {eps:g}, min_samples = {ms}")
    plt.xlabel("Observasi (jarak diurutkan)"); plt.ylabel(f"Jarak ke-{ms} tetangga terdekat")
    plt.title("K-Distance Graph - Heart Disease")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def save_noise_vs_eps(grid, ms, path):
    g = grid[grid["Min_Samples"] == ms]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(g["Epsilon"], g["Noise_Percentage"] * 100, "o-", color="black")
    ax1.set_xlabel("Epsilon"); ax1.set_ylabel("Noise (%)")
    ax2 = ax1.twinx(); ax2.plot(g["Epsilon"], g["Jumlah_Cluster"], "s--", color="C1")
    ax2.set_ylabel("Jumlah cluster")
    ax1.set_title(f"Heart Disease: Noise dan Jumlah Cluster vs Epsilon (min_samples={ms})")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def run():
    print("=== DBSCAN HEART DISEASE (REVISI) ===")
    X = pd.read_csv(FEATURES_PATH).to_numpy(dtype=float)
    original = pd.read_csv(CLEANED_PATH)
    if X.shape[0] != len(original):
        raise ValueError("Jumlah baris fitur preprocessing dan data Heart berbeda.")
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Fitur Heart harus berupa matriks dua dimensi dengan nilai finite.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = build_grid(X)
    selected, status = select_parameters(grid)
    eps, ms = float(selected["Epsilon"]), int(selected["Min_Samples"])
    grid["Selected"] = (grid["Epsilon"] == eps) & (grid["Min_Samples"] == ms)
    grid.to_csv(OUTPUT_DIR / "heart_dbscan_grid.csv", index=False)
    print(f"Kombinasi memenuhi kriteria: {int(grid['Memenuhi_Kriteria'].sum())} dari {len(grid)}")
    print(f"Parameter terpilih: eps={eps}, min_samples={ms} | {status}")

    labels = DBSCAN(eps=eps, min_samples=ms, metric=METRIC).fit_predict(X)
    result = original.copy()
    result["Cluster_DBSCAN"] = labels
    result["Is_Noise"] = labels == -1
    result.to_csv(OUTPUT_DIR / "heart_dbscan_clustered.csv", index=False)

    counts = save_cluster_counts(labels, OUTPUT_DIR / "heart_dbscan_cluster_counts.csv")
    profile = save_cluster_profile(result, OUTPUT_DIR / "heart_dbscan_cluster_profile.csv")
    save_distribution(counts, OUTPUT_DIR / "heart_dbscan_distribution.png")
    save_pca_visualization(X, labels, eps, ms, OUTPUT_DIR / "heart_dbscan_pca.png")
    save_k_distance_plot(X, ms, eps, OUTPUT_DIR / "heart_dbscan_k_distance.png")
    save_noise_vs_eps(grid, ms, OUTPUT_DIR / "heart_dbscan_noise_vs_eps.png")

    sizes = np.bincount(labels[labels != -1])
    noise_count = int((labels == -1).sum())
    metrics = compute_metrics(X, labels)
    final = {
        "Dataset": "Heart Disease", "Metode": "DBSCAN", "Metric": METRIC,
        "Epsilon": eps, "Min_Samples": ms, "Jumlah_Cluster": len(sizes),
        "Cluster_Sizes": ",".join(map(str, sorted(sizes.tolist()))),
        "Smallest_Cluster_Size": int(sizes.min()), "Largest_Cluster_Size": int(sizes.max()),
        "Noise_Count": noise_count, "Noise_Percentage": noise_count / len(labels),
        "Jumlah_Data": len(labels), "PCA_Digunakan_Untuk_Clustering": False,
        "Selection_Criteria": CRITERIA_TEXT, "Status_Kriteria": status, **metrics,
    }
    pd.DataFrame([final]).to_csv(OUTPUT_DIR / "heart_dbscan_final_metrics.csv", index=False)
    pd.DataFrame([{**final, "Parameter": f"eps={eps}; min_samples={ms}; metric={METRIC}",
                   "Catatan": metrics["Catatan_Evaluasi"]}]).to_csv(
        OUTPUT_DIR / "heart_dbscan_evaluation.csv", index=False)

    print("Fitur:", X.shape[1], "| baris:", len(X))
    print("Cluster counts:\n", counts.to_string(index=False))
    print("Profil cluster:\n", profile.to_string())
    print("Metrik final:\n", pd.DataFrame([final]).drop(columns=["Selection_Criteria"]).round(4).to_string(index=False))
    if status != "Memenuhi kriteria":
        print("PERINGATAN: tidak ada konfigurasi DBSCAN yang memenuhi kriteria; hasil ini tidak "
              "boleh diperlakukan sebagai clustering yang valid.")


if __name__ == "__main__":
    run()
