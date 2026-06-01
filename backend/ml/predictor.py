import json
import os
from pathlib import Path
from typing import Any
from math import atan2, cos, radians, sin, sqrt

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "fraud_model.joblib"
METRICS_PATH = Path(__file__).resolve().parents[1] / "models" / "model_metrics.json"
DEFAULT_THRESHOLD = float(os.getenv("FRAUD_DECISION_THRESHOLD", "0.55"))
DEFAULT_MODEL_VERSION = "Random Forest v1.0"


class FraudPredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.metrics_path = METRICS_PATH
        self.model: Any | None = None
        self.metrics = self._load_metrics()
        self.threshold = float(self.metrics.get("threshold", DEFAULT_THRESHOLD))
        self.model_version = str(self.metrics.get("model_version", DEFAULT_MODEL_VERSION))
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, transaction: dict[str, Any]) -> dict[str, Any]:
        explanations = self._explanations(transaction)
        risk_factors = [item["signal"] for item in explanations]

        if self.model is None:
            fraud_probability = self._fallback_probability(transaction, risk_factors)
            prediction = int(fraud_probability >= self.threshold)
            model_source = "Demo rules engine"
        else:
            frame = pd.DataFrame([transaction])
            if hasattr(self.model, "predict_proba"):
                fraud_probability = float(self.model.predict_proba(frame)[0][1])
            else:
                fraud_probability = float(self.model.predict(frame)[0])
            prediction = int(fraud_probability >= self.threshold)
            model_source = "Random Forest classifier"

        return {
            "prediction": prediction,
            "label": "Fraudulent" if prediction == 1 else "Legitimate",
            "fraud_probability": round(fraud_probability, 4),
            "model_loaded": self.is_loaded,
            "risk_level": self._risk_level(fraud_probability),
            "model_source": model_source,
            "model_version": self.model_version,
            "threshold": self.threshold,
            "recommendation": self._recommendation(fraud_probability),
            "risk_factors": risk_factors
            or ["No strong manual risk triggers found for this transaction."],
            "explanations": explanations
            or [
                {
                    "signal": "No strong manual risk triggers found for this transaction.",
                    "impact": "Low",
                    "contribution": 0.0,
                }
            ],
        }

    def model_metrics(self) -> dict[str, Any]:
        if self.metrics:
            return {
                "model_version": self.model_version,
                "model_source": "Random Forest classifier"
                if self.is_loaded
                else "Demo rules engine",
                "threshold": self.threshold,
                **self.metrics,
            }

        return {
            "model_version": self.model_version,
            "model_source": "Random Forest classifier"
            if self.is_loaded
            else "Demo rules engine",
            "threshold": self.threshold,
            "trained_at": None,
            "dataset": "Kaggle fraudTrain.csv",
            "sample_rows": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "roc_auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "feature_count": 13,
            "positive_rate": None,
        }

    def _load_metrics(self) -> dict[str, Any]:
        if not self.metrics_path.exists():
            return {}
        try:
            return json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _explanations(self, transaction: dict[str, Any]) -> list[dict[str, Any]]:
        factors: list[dict[str, Any]] = []
        amount = float(transaction.get("amt", 0))
        category = str(transaction.get("category", "")).lower()
        city_pop = int(transaction.get("city_pop", 0))
        distance = self._distance_miles(transaction)

        if amount >= 750:
            factors.append(
                {
                    "signal": "Transaction amount is unusually high.",
                    "impact": "High",
                    "contribution": 0.18 if amount >= 1000 else 0.11,
                }
            )
        elif amount >= 300:
            factors.append(
                {
                    "signal": "Transaction amount is above the normal review threshold.",
                    "impact": "Medium",
                    "contribution": 0.11,
                }
            )

        if category in {"shopping_net", "misc_net", "grocery_net"}:
            factors.append(
                {
                    "signal": "Online category can carry higher card-not-present risk.",
                    "impact": "Medium",
                    "contribution": 0.11,
                }
            )

        if distance >= 250:
            factors.append(
                {
                    "signal": "Merchant location is far from the cardholder location.",
                    "impact": "High" if distance >= 500 else "Medium",
                    "contribution": 0.25 if distance >= 500 else 0.11,
                }
            )

        if 0 < city_pop < 5000:
            factors.append(
                {
                    "signal": "Small population area may need extra location validation.",
                    "impact": "Low",
                    "contribution": 0.07,
                }
            )

        return factors[:4]

    def _fallback_probability(
        self, transaction: dict[str, Any], risk_factors: list[str]
    ) -> float:
        amount = float(transaction.get("amt", 0))
        distance = self._distance_miles(transaction)
        score = 0.12 + (0.11 * len(risk_factors))

        if amount >= 1000:
            score += 0.18
        if distance >= 500:
            score += 0.14

        return min(score, 0.93)

    def _distance_miles(self, transaction: dict[str, Any]) -> float:
        try:
            lat1 = float(transaction["lat"])
            lon1 = float(transaction["long"])
            lat2 = float(transaction["merch_lat"])
            lon2 = float(transaction["merch_long"])
        except (KeyError, TypeError, ValueError):
            return 0.0

        radius_miles = 3958.8
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        )
        return radius_miles * 2 * atan2(sqrt(a), sqrt(1 - a))

    def _risk_level(self, probability: float) -> str:
        if probability >= 0.7:
            return "Critical"
        if probability >= 0.45:
            return "Elevated"
        if probability >= 0.2:
            return "Moderate"
        return "Low"

    def _recommendation(self, probability: float) -> str:
        if probability >= 0.7:
            return "Block or hold this transaction until a fraud analyst reviews it."
        if probability >= 0.45:
            return "Send this transaction to manual review before approval."
        if probability >= 0.2:
            return "Approve with monitoring and keep the transaction in review history."
        return "Approve normally. No immediate fraud action is recommended."
