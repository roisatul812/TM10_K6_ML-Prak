from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score,
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
OUTPUT_DIR = ROOT_DIR / "output" / "comparison"
ALGORITHMS = ["K-Means", "FCM", "DBSCAN"]
METRICS = ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz"]
DIRECTIONS = {"Silhouette": "max", "Davies_Bouldin": "min", "Calinski_Harabasz": "max"}
MALL_NUMERIC = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

MAX_NOISE_PCT = 15.0
MIN_CLUSTER_PCT = 5.0
MIN_CLUSTERS = 2
COLORS = {"K-Means": "#4C72B0", "FCM": "#55A868", "DBSCAN": "#C44E52"}


def title_of(ds):
    return "Mall Customers" if ds == "mall" else "Heart Disease"


def compute_metrics(X, labels):
    if len(np.unique(labels)) < 2 or len(labels) <= len(np.unique(labels)):
        return {m: np.nan for m in METRICS}
    return {
        "Silhouette": float(silhouette_score(X, labels)),
        "Davies_Bouldin": float(davies_bouldin_score(X, labels)),
        "Calinski_Harabasz": float(calinski_harabasz_score(X, labels)),
    }


def load_experiment(ds):
    d = ROOT_DIR / "output" / ds
    Xdf = pd.read_csv(d / "preprocessing" / f"{ds}_features_scaled_encoded.csv")
    spaces = {"encoded": ("Seluruh fitur terstandar + one-hot (ruang K-Means/FCM)", Xdf.to_numpy(float))}
    if ds == "mall":
        spaces["numerik"] = ("Fitur numerik saja (ruang DBSCAN)", Xdf[MALL_NUMERIC].to_numpy(float))
    files = {"K-Means": ("kmeans", "Cluster_KMeans"), "FCM": ("fcm", "Cluster_FCM"),
             "DBSCAN": ("dbscan", "Cluster_DBSCAN")}
    labels = {a: pd.read_csv(d / f / f"{ds}_{f}_clustered.csv")[c].to_numpy() for a, (f, c) in files.items()}
    n = len(Xdf)
    if any(len(v) != n for v in labels.values()):
        raise ValueError(f"Jumlah baris label {ds} berbeda dari matriks fitur.")
    return d, spaces, labels


def validity(labels):
    n = len(labels)
    nn = labels[labels != -1]
    sizes = np.unique(nn, return_counts=True)[1] if len(nn) else np.array([0])
    noise_pct = float((labels == -1).mean() * 100)
    k = len(sizes) if len(nn) else 0
    ok = k >= MIN_CLUSTERS and noise_pct <= MAX_NOISE_PCT and sizes.min() >= n * MIN_CLUSTER_PCT / 100
    reasons = []
    if k < MIN_CLUSTERS:
        reasons.append("kurang dari 2 cluster")
    if noise_pct > MAX_NOISE_PCT:
        reasons.append(f"noise {noise_pct:.1f}% > {MAX_NOISE_PCT:g}%")
    if k and sizes.min() < n * MIN_CLUSTER_PCT / 100:
        reasons.append(f"cluster terkecil {sizes.min()} baris (< {MIN_CLUSTER_PCT:g}% data)")
    return bool(ok), "; ".join(reasons) if reasons else "-"


def metric_row(ds, algo, labels_full, mask, X, basis, space_label):
    labels = labels_full[mask]
    valid, reason = validity(labels_full)
    ids, sizes = np.unique(labels[labels != -1], return_counts=True)
    noise = int((labels_full == -1).sum())
    return {
        "Dataset": title_of(ds), "Algorithm": algo, "Ruang_Fitur": space_label, "Basis_Evaluasi": basis,
        "Jumlah_Data_Evaluasi": int(mask.sum()), "Jumlah_Cluster": len(ids),
        "Cluster_Sizes": ",".join(map(str, sizes.tolist())), "Noise_Count": noise,
        "Noise_Percentage": noise / len(labels_full), "Valid": valid, "Catatan_Validitas": reason,
        **compute_metrics(X[mask][labels != -1], labels[labels != -1]),
    }


def build_tables(ds, X, space_label, labels):
    db = labels["DBSCAN"]
    non_noise = db != -1
    if not non_noise.any():
        raise ValueError(f"DBSCAN menandai seluruh data {ds} sebagai noise.")
    main_rows, fair_rows = [], []
    for algo in ALGORITHMS:
        lab = labels[algo]
        # metrik dihitung pada data non-noise (label -1 dikeluarkan) untuk DBSCAN
        mask_main = non_noise if algo == "DBSCAN" else np.ones(len(X), bool)
        basis_main = "Data non-noise DBSCAN" if algo == "DBSCAN" else "Seluruh data"
        main_rows.append(metric_row(ds, algo, lab, mask_main, X, basis_main, space_label))
        fair_rows.append(metric_row(ds, algo, lab, non_noise, X, "Subset baris non-noise DBSCAN yang sama", space_label))
    return pd.DataFrame(main_rows), pd.DataFrame(fair_rows)


def verify_stored(ds, dataset_dir, main):
    for algo, folder in [("K-Means", "kmeans"), ("FCM", "fcm")]:
        stored = pd.read_csv(dataset_dir / folder / f"{ds}_{folder}_final_metrics.csv").iloc[0]
        calc = main.loc[main["Algorithm"] == algo].iloc[0]
        for m in METRICS:
            if not np.isclose(float(calc[m]), float(stored[m]), atol=1e-8):
                raise ValueError(f"Metrik {algo} {ds} berbeda dari hasil eksperimen tersimpan.")


def build_counts(ds, labels):
    rows = []
    for algo in ALGORITHMS:
        v = labels[algo]
        ids, sizes = np.unique(v[v != -1], return_counts=True)
        noise = int((v == -1).sum())
        rows.append({"Dataset": title_of(ds), "Algorithm": algo, "Jumlah_Cluster": len(ids),
                     "Cluster_Sizes": ",".join(map(str, sizes.tolist())), "Jumlah_Noise": noise,
                     "Persentase_Noise": noise / len(v), "Jumlah_Data": len(v)})
    return pd.DataFrame(rows)


def save_metric_plot(df, path, title):
    pos = np.arange(len(df))
    names = [f"{r['Algorithm']}{'' if r['Valid'] else ' (tidak valid)'}\n(n={r['Jumlah_Data_Evaluasi']})"
             for _, r in df.iterrows()]
    titles = {"Silhouette": "Silhouette (lebih tinggi lebih baik)",
              "Davies_Bouldin": "Davies-Bouldin (lebih rendah lebih baik)",
              "Calinski_Harabasz": "Calinski-Harabasz (lebih tinggi lebih baik)"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, m in zip(axes, METRICS):
        bars = ax.bar(pos, df[m], color=[COLORS[a] for a in df["Algorithm"]])
        for b, (_, r) in zip(bars, df.iterrows()):
            if not r["Valid"]:
                b.set_hatch("//"); b.set_alpha(0.6)
        ax.set_xticks(pos, names, fontsize=9); ax.set_title(titles[m]); ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def save_count_plot(counts, path, title):
    pos = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(pos, counts["Jumlah_Cluster"], color=[COLORS[a] for a in counts["Algorithm"]])
    ax.set_xticks(pos, counts["Algorithm"]); ax.set_ylabel("Jumlah cluster"); ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for b, r in zip(bars, counts.itertuples()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"noise={r.Jumlah_Noise} ({r.Persentase_Noise * 100:.2f}%)", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fmt(v):
    return "N/A" if pd.isna(v) else f"{float(v):.6f}"


def md_table(df):
    lines = ["| Algoritma | Valid | Jumlah cluster | Ukuran cluster | Noise (%) | Data dievaluasi | Silhouette | Davies-Bouldin | Calinski-Harabasz |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['Algorithm']} | {'Ya' if r['Valid'] else 'Tidak'} | {r['Jumlah_Cluster']} | {r['Cluster_Sizes']} | "
                     f"{r['Noise_Percentage'] * 100:.2f}% | {r['Jumlah_Data_Evaluasi']} | {fmt(r['Silhouette'])} | "
                     f"{fmt(r['Davies_Bouldin'])} | {fmt(r['Calinski_Harabasz'])} |")
    return lines


def winners(df):
    """Pemenang hanya dihitung di antara metode yang valid."""
    pool = df[df["Valid"]]
    out, win = [], {}
    for m, d in DIRECTIONS.items():
        if pool.empty:
            continue
        best = pool[m].max() if d == "max" else pool[m].min()
        names = pool.loc[np.isclose(pool[m], best), "Algorithm"].tolist()
        for a in names:
            win[a] = win.get(a, 0) + 1
        out.append(f"- {m}: {' / '.join(names)} ({fmt(best)}; {'lebih tinggi' if d == 'max' else 'lebih rendah'} lebih baik).")
    return out, win


def build_analysis(ds, dataset_dir, space_results, counts, labels):
    title = title_of(ds)
    dbm = pd.read_csv(dataset_dir / "dbscan" / f"{ds}_dbscan_final_metrics.csv").iloc[0]
    lines = [f"## {title}", "", "### Konfigurasi DBSCAN", "",
             f"- Fitur: {dbm['Fitur_Digunakan'] if 'Fitur_Digunakan' in dbm else 'seluruh fitur terstandar + one-hot'}",
             f"- Parameter: eps={dbm['Epsilon']:g}, min_samples={int(dbm['Min_Samples'])}",
             f"- Kriteria pemilihan: {dbm['Selection_Criteria']}",
             f"- Status: {dbm['Status_Kriteria']}", ""]
    for key, (space_label, main, fair) in space_results.items():
        lines += [f"### Ruang fitur: {space_label}", "", "**Cakupan evaluasi masing-masing metode**", "", *md_table(main), "",
                  "K-Means dan FCM dievaluasi pada seluruh data; metrik DBSCAN hanya pada baris non-noise.", "",
                  "**Perbandingan pada subset non-noise DBSCAN yang sama**", "", *md_table(fair), ""]
        w_main, win_main = winners(main)
        w_fair, win_fair = winners(fair)
        lines += ["Pemenang metrik (hanya di antara metode yang valid), cakupan masing-masing:", *w_main, "",
                  "Pemenang metrik pada subset yang sama:", *w_fair, ""]
        invalid = main[~main["Valid"]]
        for _, r in invalid.iterrows():
            lines.append(f"Catatan: {r['Algorithm']} ditandai TIDAK VALID ({r['Catatan_Validitas']}), "
                         "sehingga nilai metriknya tidak dihitung sebagai pemenang dan tidak boleh ditafsirkan sebagai kualitas yang baik.")
        lines.append("")
    # jumlah noise/cluster
    lines += ["### Jumlah cluster dan noise", "", "| Algoritma | Jumlah cluster | Ukuran cluster | Noise | Noise (%) |", "|---|---:|---|---:|---:|"]
    for _, r in counts.iterrows():
        lines.append(f"| {r['Algorithm']} | {r['Jumlah_Cluster']} | {r['Cluster_Sizes']} | {r['Jumlah_Noise']} | {r['Persentase_Noise'] * 100:.2f}% |")
    # kesamaan hasil antar metode
    ari = {f"{a} vs {b}": adjusted_rand_score(labels[a], labels[b])
           for a, b in [("K-Means", "FCM"), ("K-Means", "DBSCAN"), ("FCM", "DBSCAN")]}
    lines += ["", "### Kemiripan label antar metode (Adjusted Rand Index)", ""]
    lines += [f"- {k}: {v:.3f}" for k, v in ari.items()]
    lines += ["", "Catatan: Silhouette, Davies-Bouldin, dan Calinski-Harabasz adalah metrik internal, bukan accuracy; "
              "nilainya bergantung pada ruang fitur dan data yang dievaluasi. ARI untuk DBSCAN menganggap label noise (-1) sebagai satu kelompok.", ""]
    return "\n".join(lines)


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analyses, summary_rows = [], []
    for ds in ["mall", "heart"]:
        dataset_dir, spaces, labels = load_experiment(ds)
        counts = build_counts(ds, labels)
        counts.to_csv(OUTPUT_DIR / f"{ds}_cluster_noise_comparison.csv", index=False)
        save_count_plot(counts, OUTPUT_DIR / f"{ds}_cluster_noise_comparison.png", f"{ds.title()} - jumlah cluster dan noise")
        space_results = {}
        for key, (space_label, X) in spaces.items():
            main, fair = build_tables(ds, X, space_label, labels)
            if key == "encoded":
                verify_stored(ds, dataset_dir, main)
            suffix = "" if key == "encoded" else f"_{key}"
            main.to_csv(OUTPUT_DIR / f"{ds}_clustering_comparison{suffix}.csv", index=False)
            fair.to_csv(OUTPUT_DIR / f"{ds}_clustering_comparison_non_noise{suffix}.csv", index=False)
            save_metric_plot(main, OUTPUT_DIR / f"{ds}_clustering_comparison{suffix}.png", f"{ds.title()} - evaluasi utama ({key})")
            save_metric_plot(fair, OUTPUT_DIR / f"{ds}_clustering_comparison_non_noise{suffix}.png",
                             f"{ds.title()} - subset non-noise DBSCAN yang sama ({key})")
            space_results[key] = (space_label, main, fair)
            _, win = winners(main)
            for a in ALGORITHMS:
                summary_rows.append({"Dataset": title_of(ds), "Ruang_Fitur": space_label, "Algorithm": a,
                                     "Valid": bool(main.loc[main["Algorithm"] == a, "Valid"].iloc[0]),
                                     "Jumlah_Kemenangan_Metrik(dari_3)": win.get(a, 0)})
        analysis = build_analysis(ds, dataset_dir, space_results, counts, labels)
        (OUTPUT_DIR / f"{ds}_analysis_results.md").write_text(f"# Analisis Clustering - {analysis}", encoding="utf-8")
        analyses.append(analysis)
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(OUTPUT_DIR / "ringkasan_kemenangan_metrik.csv", index=False)
    (OUTPUT_DIR / "analysis_results.md").write_text("# Perbandingan Hasil Clustering\n\n" + "\n".join(analyses) + "\n", encoding="utf-8")
    print(summ.to_string(index=False))
    print("Comparison K-Means, FCM, dan DBSCAN selesai. Output:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
