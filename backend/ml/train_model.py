import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "fraudTrain.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
DEFAULT_SAMPLE_ROWS = 200_000

TARGET = "is_fraud"
FEATURES = [
    "amt",
    "category",
    "merchant",
    "gender",
    "city",
    "state",
    "job",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "city_pop",
    "unix_time",
]


def build_pipeline() -> Pipeline:
    numeric_features = [
        "amt",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "city_pop",
        "unix_time",
    ]
    categorical_features = ["category", "merchant", "gender", "city", "state", "job"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=80,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=60,
                    max_depth=18,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Download fraudTrain.csv from Kaggle "
            "and place it in backend/data/."
        )

    print(f"Loading dataset: {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH)
    missing = [column for column in [*FEATURES, TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    sample_rows = int(os.getenv("FRAUD_SAMPLE_ROWS", DEFAULT_SAMPLE_ROWS))
    if sample_rows > 0 and len(df) > sample_rows:
        print(f"Using stratified sample of {sample_rows:,} rows for faster training.", flush=True)
        sampled_groups = [
            group.sample(frac=sample_rows / len(df), random_state=42)
            for _, group in df.groupby(TARGET)
        ]
        df = pd.concat(sampled_groups).reset_index(drop=True)

    x = df[FEATURES]
    y = df[TARGET]

    print(f"Training rows: {len(x):,}", flush=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline()
    print("Training Random Forest model...", flush=True)
    model.fit(x_train, y_train)

    print("Evaluating model...", flush=True)
    predictions = model.predict(x_test)
    print("Classification Report")
    print(classification_report(y_test, predictions, digits=4))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
