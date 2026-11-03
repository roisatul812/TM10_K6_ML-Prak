from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
OUTPUT_DIR = ROOT_DIR / "output" / "comparison"
ALGORITHMS = ["K-Means", "FCM", "DBSCAN"]
METRICS = ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz"]


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(labels) <= len(unique_labels):
        return {
            "Silhouette": np.nan,
            "Davies_Bouldin": np.nan,
            "Calinski_Harabasz": np.nan,
        }
    return {
        "Silhouette": float(silhouette_score(X, labels)),
        "Davies_Bouldin": float(davies_bouldin_score(X, labels)),
        "Calinski_Harabasz": float(calinski_harabasz_score(X, labels)),
    }


def load_experiment(dataset_name: str) -> tuple[Path, np.ndarray, dict]:
    dataset_dir = ROOT_DIR / "output" / dataset_name
    X = pd.read_csv(
        dataset_dir
        / "preprocessing"
        / f"{dataset_name}_features_scaled_encoded.csv"
    ).to_numpy(dtype=float)
    label_files = {
        "K-Means": dataset_dir / "kmeans" / f"{dataset_name}_kmeans_clustered.csv",
        "FCM": dataset_dir / "fcm" / f"{dataset_name}_fcm_clustered.csv",
        "DBSCAN": dataset_dir / "dbscan" / f"{dataset_name}_dbscan_clustered.csv",
    }
    label_columns = {
        "K-Means": "Cluster_KMeans",
        "FCM": "Cluster_FCM",
        "DBSCAN": "Cluster_DBSCAN",
    }
    labels = {
        algorithm: pd.read_csv(path)[label_columns[algorithm]].to_numpy()
        for algorithm, path in label_files.items()
    }
    if any(len(cluster_labels) != len(X) for cluster_labels in labels.values()):
        raise ValueError(
            f"Jumlah baris label clustering {dataset_name} berbeda dari matriks fitur."
        )
    return dataset_dir, X, labels


def metric_row(
    dataset_name: str,
    algorithm: str,
    labels: np.ndarray,
    metrics: dict,
    basis: str,
    evaluated_count: int,
    total_count: int,
    dbscan_noise_count: int,
) -> dict:
    non_noise_labels = labels[labels != -1]
    cluster_ids, cluster_sizes = np.unique(non_noise_labels, return_counts=True)
    noise_count = dbscan_noise_count if algorithm == "DBSCAN" else 0
    return {
        "Dataset": "Mall Customers" if dataset_name == "mall" else "Heart Disease",
        "Algorithm": algorithm,
        "Basis_Evaluasi": basis,
        "Jumlah_Data_Evaluasi": evaluated_count,
        "Jumlah_Cluster": len(cluster_ids),
        "Cluster_Sizes": ",".join(map(str, cluster_sizes.tolist())),
        "Noise_Count": noise_count,
        "Noise_Percentage": noise_count / total_count if algorithm == "DBSCAN" else 0.0,
        **metrics,
    }


def build_comparison_tables(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir, X, labels = load_experiment(dataset_name)
    dbscan_labels = labels["DBSCAN"]
    non_noise = dbscan_labels != -1
    if not non_noise.any():
        raise ValueError(f"DBSCAN menandai seluruh data {dataset_name} sebagai noise.")
    if dataset_name == "heart" and int(non_noise.sum()) != 779:
        raise ValueError(
            "DBSCAN Heart harus menghasilkan tepat 779 baris non-noise "
            f"untuk konfigurasi awal; ditemukan {int(non_noise.sum())}."
        )

    main_rows = []
    fair_rows = []
    for algorithm in ALGORITHMS:
        algorithm_labels = labels[algorithm]
        if algorithm == "DBSCAN":
            main_mask = non_noise
            main_basis = "Data non-noise DBSCAN"
        else:
            main_mask = np.ones(len(X), dtype=bool)
            main_basis = "Seluruh data"

        main_rows.append(
            metric_row(
                dataset_name,
                algorithm,
                algorithm_labels[main_mask],
                compute_metrics(X[main_mask], algorithm_labels[main_mask]),
                main_basis,
                int(main_mask.sum()),
                len(X),
                int(np.sum(dbscan_labels == -1)),
            )
        )
        fair_rows.append(
            metric_row(
                dataset_name,
                algorithm,
                algorithm_labels[non_noise],
                compute_metrics(X[non_noise], algorithm_labels[non_noise]),
                "Subset baris non-noise DBSCAN yang sama",
                int(non_noise.sum()),
                len(X),
                int(np.sum(dbscan_labels == -1)),
            )
        )

    main = pd.DataFrame(main_rows)
    fair = pd.DataFrame(fair_rows)

    for algorithm, folder in [("K-Means", "kmeans"), ("FCM", "fcm")]:
        final_path = dataset_dir / folder / f"{dataset_name}_{folder}_final_metrics.csv"
        stored = pd.read_csv(final_path).iloc[0]
        calculated = main.loc[main["Algorithm"] == algorithm].iloc[0]
        for metric in METRICS:
            if not np.isclose(
                float(calculated[metric]), float(stored[metric]), atol=1e-10
            ):
                raise ValueError(
                    f"Metrik {algorithm} {dataset_name} pada comparison tidak sama "
                    "dengan hasil eksperimen tersimpan."
                )
    return main, fair


def build_cluster_counts(dataset_name: str) -> pd.DataFrame:
    dataset_dir, _, labels = load_experiment(dataset_name)
    rows = []
    for algorithm in ALGORITHMS:
        values = labels[algorithm]
        cluster_ids, sizes = np.unique(values[values != -1], return_counts=True)
        noise_count = int(np.sum(values == -1)) if algorithm == "DBSCAN" else 0
        rows.append(
            {
                "Dataset": "Mall Customers"
                if dataset_name == "mall"
                else "Heart Disease",
                "Algorithm": algorithm,
                "Jumlah_Cluster": len(cluster_ids),
                "Cluster_Sizes": ",".join(map(str, sizes.tolist())),
                "Jumlah_Noise": noise_count,
                "Persentase_Noise": noise_count / len(values),
                "Jumlah_Data": len(values),
            }
        )
    return pd.DataFrame(rows)


def save_metric_plot(comparison: pd.DataFrame, output_path: Path, title: str) -> None:
    positions = np.arange(len(comparison))
    labels = [
        f"{row['Algorithm']}\n(n={row['Jumlah_Data_Evaluasi']})"
        for _, row in comparison.iterrows()
    ]
    metric_titles = {
        "Silhouette": "Silhouette (lebih tinggi lebih baik)",
        "Davies_Bouldin": "Davies-Bouldin (lebih rendah lebih baik)",
        "Calinski_Harabasz": "Calinski-Harabasz (lebih tinggi lebih baik)",
    }
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, metric in zip(axes, METRICS):
        axis.bar(positions, comparison[metric], color=colors)
        axis.set_xticks(positions, labels)
        axis.set_title(metric_titles[metric])
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_cluster_count_plot(counts: pd.DataFrame, output_path: Path, title: str) -> None:
    positions = np.arange(len(counts))
    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        positions,
        counts["Jumlah_Cluster"],
        color=["#4C72B0", "#55A868", "#C44E52"],
        label="Cluster non-noise",
    )
    axis.set_xticks(positions, counts["Algorithm"])
    axis.set_ylabel("Jumlah cluster")
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.25)
    for bar, row in zip(bars, counts.itertuples()):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"noise={row.Jumlah_Noise} ({row.Persentase_Noise * 100:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def fmt(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.6f}"


def metric_markdown_table(comparison: pd.DataFrame) -> list[str]:
    lines = [
        "| Algoritma | Jumlah cluster | Ukuran cluster | Noise | Noise (%) | Data dievaluasi | Silhouette | Davies-Bouldin | Calinski-Harabasz |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['Algorithm']} | {row['Jumlah_Cluster']} | {row['Cluster_Sizes']} | "
            f"{row['Noise_Count']} | {row['Noise_Percentage'] * 100:.2f}% | "
            f"{row['Jumlah_Data_Evaluasi']} | {fmt(row['Silhouette'])} | "
            f"{fmt(row['Davies_Bouldin'])} | {fmt(row['Calinski_Harabasz'])} |"
        )
    return lines


def metric_winners(comparison: pd.DataFrame) -> list[str]:
    directions = {
        "Silhouette": "max",
        "Davies_Bouldin": "min",
        "Calinski_Harabasz": "max",
    }
    lines = []
    for metric, direction in directions.items():
        values = comparison[metric]
        best = values.max() if direction == "max" else values.min()
        winners = comparison.loc[np.isclose(values, best), "Algorithm"].tolist()
        lines.append(
            f"- {metric}: {' / '.join(winners)} ({fmt(best)}; "
            f"{'lebih tinggi' if direction == 'max' else 'lebih rendah'} lebih baik)."
        )
    return lines


def build_analysis(
    dataset_name: str, main: pd.DataFrame, fair: pd.DataFrame, counts: pd.DataFrame
) -> str:
    title = "Mall Customers" if dataset_name == "mall" else "Heart Disease"
    main_index = main.set_index("Algorithm")
    fair_index = fair.set_index("Algorithm")
    dbscan = main_index.loc["DBSCAN"]
    lines = [
        f"## {title}",
        "",
        "### Perbandingan utama (cakupan evaluasi masing-masing eksperimen)",
        "",
        *metric_markdown_table(main),
        "",
        "K-Means dan FCM dievaluasi pada seluruh data. Metrik DBSCAN dihitung hanya pada "
        "baris non-noise (label -1 dikeluarkan), sehingga tabel utama tidak memiliki cakupan "
        "observasi yang sama dan metrik DBSCAN tidak boleh dibandingkan langsung tanpa konteks.",
        "",
        "### Perbandingan tambahan yang adil pada subset non-noise DBSCAN",
        "",
        *metric_markdown_table(fair),
        "",
        f"Semua algoritma pada tabel ini dievaluasi pada {int(fair['Jumlah_Data_Evaluasi'].iloc[0])} "
        "baris observasi yang sama, yaitu baris non-noise DBSCAN. K-Means dan FCM tidak dilatih "
        "ulang; label yang telah tersimpan hanya digunakan pada subset tersebut. Tabel tambahan "
        "ini tidak menggantikan evaluasi utama.",
        "",
        "### Jumlah cluster dan noise",
        "",
        "| Algoritma | Jumlah cluster | Ukuran cluster | Noise | Noise (%) |",
        "|---|---:|---|---:|---:|",
    ]
    for _, row in counts.iterrows():
        lines.append(
            f"| {row['Algorithm']} | {row['Jumlah_Cluster']} | {row['Cluster_Sizes']} | "
            f"{row['Jumlah_Noise']} | {row['Persentase_Noise'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "### Hasil dan interpretasi metrik",
            "",
            "- Silhouette: semakin tinggi semakin baik.",
            "- Davies-Bouldin: semakin rendah semakin baik.",
            "- Calinski-Harabasz: semakin tinggi semakin baik.",
            "",
            "Pemenang numerik pada tabel utama (cakupan DBSCAN berbeda):",
            *metric_winners(main),
            "",
            "Pemenang pada perbandingan tambahan dengan baris observasi yang sama:",
            *metric_winners(fair),
            "",
            "Pemenang pada tabel utama harus ditafsirkan hati-hati karena metrik DBSCAN mengecualikan "
            "noise sedangkan K-Means/FCM memakai seluruh data. Nilai metrik internal tidak mengukur "
            "accuracy dan hasilnya juga bergantung pada representasi fitur serta jumlah observasi "
            "yang dievaluasi.",
            "",
        ]
    )
    if dataset_name == "heart":
        lines.extend(
            [
                "### Konfigurasi DBSCAN Heart",
                "",
                "- DBSCAN menggunakan matriks fitur preprocessing yang tersedia, tanpa preprocessing ulang.",
                "- Parameter: eps=2.0, min_samples=5, metric=Euclidean.",
                f"- Hasil: {int(dbscan['Jumlah_Cluster'])} cluster, ukuran {dbscan['Cluster_Sizes']}; "
                f"{int(dbscan['Noise_Count'])} noise dari 918 baris "
                f"({dbscan['Noise_Percentage'] * 100:.2f}%).",
                "- Metrik DBSCAN dihitung pada 779 baris non-noise saja.",
                "- Satu cluster hanya beranggotakan satu baris, sehingga struktur cluster DBSCAN "
                "perlu ditafsirkan dengan hati-hati. Banyaknya noise yang lebih sedikit tidak "
                "dengan sendirinya menunjukkan kualitas clustering yang lebih baik.",
                "- PCA digunakan hanya untuk visualisasi DBSCAN, bukan sebagai input clustering.",
                "",
            ]
        )
    return "\n".join(lines)


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analyses = []
    for dataset_name in ["mall", "heart"]:
        main, fair = build_comparison_tables(dataset_name)
        counts = build_cluster_counts(dataset_name)

        main.to_csv(
            OUTPUT_DIR / f"{dataset_name}_clustering_comparison.csv", index=False
        )
        fair.to_csv(
            OUTPUT_DIR / f"{dataset_name}_clustering_comparison_non_noise.csv",
            index=False,
        )
        counts.to_csv(
            OUTPUT_DIR / f"{dataset_name}_cluster_noise_comparison.csv", index=False
        )
        save_metric_plot(
            main,
            OUTPUT_DIR / f"{dataset_name}_clustering_comparison.png",
            f"{dataset_name.title()} - evaluasi utama",
        )
        save_metric_plot(
            fair,
            OUTPUT_DIR / f"{dataset_name}_clustering_comparison_non_noise.png",
            f"{dataset_name.title()} - subset non-noise DBSCAN yang sama",
        )
        save_cluster_count_plot(
            counts,
            OUTPUT_DIR / f"{dataset_name}_cluster_noise_comparison.png",
            f"{dataset_name.title()} - jumlah cluster dan noise",
        )

        analysis = build_analysis(dataset_name, main, fair, counts)
        (OUTPUT_DIR / f"{dataset_name}_analysis_results.md").write_text(
            f"# Analisis Clustering - {analysis}", encoding="utf-8"
        )
        analyses.append(analysis)

    (OUTPUT_DIR / "analysis_results.md").write_text(
        "# Perbandingan Hasil Clustering\n\n"
        + "\n".join(analyses)
        + "\n",
        encoding="utf-8",
    )
    print("Comparison K-Means, FCM, dan DBSCAN selesai.")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
