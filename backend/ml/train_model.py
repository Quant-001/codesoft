import json
import os
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "fraudTrain.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"
DEFAULT_SAMPLE_ROWS = 200_000
MODEL_VERSION = "Random Forest v1.0"

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


def choose_threshold(y_true: pd.Series, probabilities: list[float]) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    best_threshold = 0.55
    best_score = -1.0

    for index, threshold in enumerate(thresholds):
        if recall[index] < 0.75:
            continue
        score = (2 * precision[index] * recall[index]) / (
            precision[index] + recall[index] + 1e-9
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return round(best_threshold, 4)


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
    probabilities = model.predict_proba(x_test)[:, 1]
    threshold = choose_threshold(y_test, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    print("Classification Report")
    print(classification_report(y_test, predictions, digits=4))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Kaggle fraudTrain.csv",
        "sample_rows": len(df),
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "feature_count": len(FEATURES),
        "positive_rate": round(float(y.mean()), 4),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
