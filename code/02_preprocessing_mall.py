
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# KONFIGURASI DIREKTORI

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DATA_PATH = ROOT_DIR / "dataset" / "Mall_Customers.csv"
OUTPUT_DIR = ROOT_DIR / "output" / "mall" / "preprocessing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run():
    print("=== PREPROCESSING MALL CUSTOMERS ===")

    # 1. Membaca dataset
    df = pd.read_csv(DATA_PATH)

    print("Ukuran dataset:", df.shape)
    print("\n5 data pertama:")
    print(df.head())

    print("\nTipe data:")
    print(df.dtypes)

    print("\nMissing value:")
    print(df.isnull().sum())

    print("\nJumlah duplikat:", df.duplicated().sum())

    # 2. Menghapus data duplikat
    df = df.drop_duplicates().copy()

    # 3. Menentukan fitur
    numeric_features = [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    categorical_features = ["Gender"]

    # CustomerID tidak digunakan karena hanya identitas.
    X = df[numeric_features + categorical_features].copy()

    # 4. Deteksi outlier menggunakan IQR
    outlier_records = []

    for col in numeric_features:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((df[col] < lower) | (df[col] > upper)).sum()
        )

        outlier_records.append({
            "Feature": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "Outlier_Count": count
        })

    outlier_df = pd.DataFrame(outlier_records)
    outlier_df.to_csv(
        OUTPUT_DIR / "mall_outlier_results.csv",
        index=False
    )

    print("\n=== HASIL DETEKSI OUTLIER ===")
    print(outlier_df)

    # 5. Pipeline numerik
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 6. Pipeline kategorikal
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        ))
    ])

    # 7. Menggabungkan preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    # 8. Transformasi data
    X_transformed = preprocessor.fit_transform(X)

    # Membuat nama fitur secara manual agar kompatibel
    # dengan scikit-learn versi lama.
    encoder = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["encoder"]
    )

    feature_names = list(numeric_features)

    for feature, categories in zip(
        categorical_features, encoder.categories_
    ):
        for category in categories:
            feature_names.append(
                "{}_{}".format(feature, category)
            )

    X_preprocessed = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=df.index
    )

    # 9. Menyimpan hasil preprocessing
    X_preprocessed.to_csv(
        OUTPUT_DIR / "mall_features_scaled_encoded.csv",
        index=False
    )

    df.to_csv(
        OUTPUT_DIR / "mall_cleaned.csv",
        index=False
    )

    # 10. Menyimpan ringkasan
    with open(
        OUTPUT_DIR / "mall_preprocessing_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("PREPROCESSING MALL CUSTOMERS\n")
        file.write("=" * 40 + "\n")
        file.write("Jumlah data: {}\n".format(len(df)))
        file.write("Fitur numerik: {}\n".format(numeric_features))
        file.write("Fitur kategorikal: {}\n".format(categorical_features))
        file.write(
            "Jumlah fitur setelah encoding: {}\n".format(
                len(feature_names)
            )
        )
        file.write("CustomerID tidak digunakan untuk clustering.\n")
        file.write(
            "Outlier dideteksi menggunakan IQR dan tidak dihapus.\n"
        )
        file.write("Numerik menggunakan StandardScaler.\n")
        file.write("Kategorikal menggunakan OneHotEncoder.\n")

    print("\nJumlah fitur setelah preprocessing:", len(feature_names))
    print("Ukuran hasil preprocessing:", X_preprocessed.shape)
    print("\nPreprocessing Mall Customers selesai!")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    run()
