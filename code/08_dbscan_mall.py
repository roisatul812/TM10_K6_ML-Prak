from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
INPUT_DIR = ROOT_DIR / "output" / "mall" / "preprocessing"
OUTPUT_DIR = ROOT_DIR / "output" / "mall" / "dbscan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = INPUT_DIR / "mall_features_scaled_encoded.csv"
CLEANED_PATH = INPUT_DIR / "mall_cleaned.csv"
NUMERIC_FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

# Kriteria pemilihan parameter
MAX_NOISE_PCT = 15.0
MIN_CLUSTER_PCT = 5.0
MIN_CLUSTERS = 2
EPS_GRID = np.round(np.arange(0.2, 1.501, 0.05), 3)
MIN_SAMPLES_GRID = [3, 4, 5, 6, 8, 10]
CRITERIA_TEXT = (
    f">= {MIN_CLUSTERS} cluster, noise <= {MAX_NOISE_PCT:g}%, tiap cluster >= {MIN_CLUSTER_PCT:g}% data; "
    "dipilih Silhouette tertinggi."
)


def compute_metrics(X, labels):
    mask = labels != -1
    if len(np.unique(labels[mask])) < 2:
        return {"Silhouette": np.nan, "Davies_Bouldin": np.nan, "Calinski_Harabasz": np.nan,
                "Jumlah_Data_Evaluasi": int(mask.sum())}
    return {
        "Silhouette": float(silhouette_score(X[mask], labels[mask])),
        "Davies_Bouldin": float(davies_bouldin_score(X[mask], labels[mask])),
        "Calinski_Harabasz": float(calinski_harabasz_score(X[mask], labels[mask])),
        "Jumlah_Data_Evaluasi": int(mask.sum()),
    }


def build_grid(X):
    rows, n = [], len(X)
    for ms in MIN_SAMPLES_GRID:
        for eps in EPS_GRID:
            labels = DBSCAN(eps=float(eps), min_samples=ms).fit_predict(X)
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
    # Cadangan: relaksasi ukuran cluster minimum
    relaxed = grid[(grid["Jumlah_Cluster"] >= MIN_CLUSTERS) & (grid["Noise_Percentage"] * 100 <= MAX_NOISE_PCT)]
    if relaxed.empty:
        raise ValueError("Tidak ada kombinasi DBSCAN dengan >= 2 cluster dan noise yang wajar.")
    return relaxed.sort_values("Silhouette", ascending=False).iloc[0], "Tidak memenuhi kriteria ukuran cluster (cadangan)"


def save_k_distance(X, ms, eps, path):
    d, _ = NearestNeighbors(n_neighbors=ms).fit(X).kneighbors(X)
    kd = np.sort(d[:, -1])
    plt.figure(figsize=(8, 5))
    plt.plot(kd, lw=2)
    plt.axhline(eps, color="red", ls="--", label=f"Epsilon terpilih = {eps:g}")
    plt.xlabel("Indeks Data (diurutkan)")
    plt.ylabel(f"Jarak ke-{ms} tetangga terdekat")
    plt.title("K-Distance Graph - Mall Customers (fitur numerik)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()


def save_counts(result, path):
    c = result["Cluster_DBSCAN"].value_counts().sort_index().rename_axis("Cluster").reset_index()
    c.columns = ["Cluster", "Jumlah_Anggota"]
    c["Persentase(%)"] = (c["Jumlah_Anggota"] / len(result) * 100).round(2)
    c["Tipe"] = np.where(c["Cluster"] == -1, "Noise", "Cluster")
    c.to_csv(path, index=False)
    return c


def save_profile(result, path):
    v = result[result["Cluster_DBSCAN"] != -1]
    p = v.groupby("Cluster_DBSCAN")[NUMERIC_FEATURES].mean().round(2)
    p["Jumlah_Anggota"] = v.groupby("Cluster_DBSCAN").size()
    p.to_csv(path)
    return p


def save_distribution(counts, path):
    plt.figure(figsize=(8, 5))
    colors = ["black" if c == -1 else "C0" for c in counts["Cluster"]]
    names = ["Noise (-1)" if c == -1 else f"Cluster {c}" for c in counts["Cluster"]]
    plt.bar(names, counts["Jumlah_Anggota"], color=colors)
    plt.ylabel("Jumlah Anggota"); plt.title("Distribusi Cluster DBSCAN - Mall Customers")
    plt.grid(axis="y", alpha=0.25); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def save_visualization(result, eps, ms, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (x, y) in zip(axes, [("Annual Income (k$)", "Spending Score (1-100)"),
                                 ("Age", "Spending Score (1-100)")]):
        noise = result[result["Cluster_DBSCAN"] == -1]
        if not noise.empty:
            ax.scatter(noise[x], noise[y], c="black", marker="x", s=45, linewidths=1.2, label="Noise (-1)")
        for c in sorted(result.loc[result["Cluster_DBSCAN"] != -1, "Cluster_DBSCAN"].unique()):
            m = result["Cluster_DBSCAN"] == c
            ax.scatter(result.loc[m, x], result.loc[m, y], s=50, alpha=0.8, label=f"Cluster {c}")
        ax.set_xlabel(x); ax.set_ylabel(y); ax.legend()
    fig.suptitle(f"DBSCAN - Mall Customers (eps={eps:g}, min_samples={ms})")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def run():
    print("=== DBSCAN MALL CUSTOMERS (REVISI) ===")
    Xall = pd.read_csv(DATA_PATH)
    df = pd.read_csv(CLEANED_PATH)
    if len(Xall) != len(df):
        raise ValueError("Jumlah baris data preprocessing dan data asli berbeda.")
    X = Xall[NUMERIC_FEATURES].to_numpy(dtype=float)

    grid = build_grid(X)
    selected, status = select_parameters(grid)
    eps, ms = float(selected["Epsilon"]), int(selected["Min_Samples"])
    grid["Selected"] = (grid["Epsilon"] == eps) & (grid["Min_Samples"] == ms)
    grid.to_csv(OUTPUT_DIR / "mall_dbscan_evaluation.csv", index=False)
    save_k_distance(X, ms, eps, OUTPUT_DIR / "mall_dbscan_k_distance.png")
    print(f"Parameter terpilih: eps={eps}, min_samples={ms} | {status}")
    print("Kriteria:", CRITERIA_TEXT)

    labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
    result = df.copy()
    result["Cluster_DBSCAN"] = labels
    result["Is_Noise"] = labels == -1
    result.to_csv(OUTPUT_DIR / "mall_dbscan_clustered.csv", index=False)

    counts = save_counts(result, OUTPUT_DIR / "mall_dbscan_cluster_counts.csv")
    profile = save_profile(result, OUTPUT_DIR / "mall_dbscan_cluster_profile.csv")
    gender = pd.crosstab(result["Cluster_DBSCAN"], result["Gender"])
    gender.to_csv(OUTPUT_DIR / "mall_dbscan_gender_per_cluster.csv")
    save_distribution(counts, OUTPUT_DIR / "mall_dbscan_distribution.png")
    save_visualization(result, eps, ms, OUTPUT_DIR / "mall_dbscan_visualization.png")

    noise_count = int((labels == -1).sum())
    sizes = np.bincount(labels[labels != -1])
    metrics = compute_metrics(X, labels)
    final = {
        "Dataset": "Mall Customers", "Metode": "DBSCAN",
        "Fitur_Digunakan": "Age, Annual Income, Spending Score (terstandar; tanpa Gender)",
        "Jumlah_Cluster": len(sizes), "Epsilon": eps, "Min_Samples": ms,
        "Smallest_Cluster_Size": int(sizes.min()), "Largest_Cluster_Size": int(sizes.max()),
        "Noise_Count": noise_count, "Noise_Percentage": noise_count / len(labels),
        "Selection_Criteria": CRITERIA_TEXT, "Status_Kriteria": status, **metrics,
    }
    pd.DataFrame([final]).to_csv(OUTPUT_DIR / "mall_dbscan_final_metrics.csv", index=False)

    print("\n=== DISTRIBUSI CLUSTER ===\n", counts.to_string(index=False))
    print("\n=== PROFIL CLUSTER ===\n", profile.to_string())
    print("\n=== GENDER PER CLUSTER ===\n", gender.to_string())
    print("\n=== METRIK FINAL ===\n", pd.DataFrame([final]).drop(columns=["Selection_Criteria"]).round(4).to_string(index=False))
    print("\nDBSCAN Mall Customers selesai. Output:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
