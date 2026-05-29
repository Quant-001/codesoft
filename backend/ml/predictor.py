from pathlib import Path
from typing import Any
from math import atan2, cos, radians, sin, sqrt

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "fraud_model.joblib"


class FraudPredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model: Any | None = None
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, transaction: dict[str, Any]) -> dict[str, Any]:
        risk_factors = self._risk_factors(transaction)

        if self.model is None:
            fraud_probability = self._fallback_probability(transaction, risk_factors)
            prediction = int(fraud_probability >= 0.55)
            model_source = "Demo rules engine"
        else:
            frame = pd.DataFrame([transaction])
            prediction = int(self.model.predict(frame)[0])

            if hasattr(self.model, "predict_proba"):
                fraud_probability = float(self.model.predict_proba(frame)[0][1])
            else:
                fraud_probability = float(prediction)
            model_source = "Random Forest classifier"

        return {
            "prediction": prediction,
            "label": "Fraudulent" if prediction == 1 else "Legitimate",
            "fraud_probability": round(fraud_probability, 4),
            "model_loaded": self.is_loaded,
            "risk_level": self._risk_level(fraud_probability),
            "model_source": model_source,
            "recommendation": self._recommendation(fraud_probability),
            "risk_factors": risk_factors
            or ["No strong manual risk triggers found for this transaction."],
        }

    def _risk_factors(self, transaction: dict[str, Any]) -> list[str]:
        factors: list[str] = []
        amount = float(transaction.get("amt", 0))
        category = str(transaction.get("category", "")).lower()
        city_pop = int(transaction.get("city_pop", 0))
        distance = self._distance_miles(transaction)

        if amount >= 750:
            factors.append("Transaction amount is unusually high.")
        elif amount >= 300:
            factors.append("Transaction amount is above the normal review threshold.")

        if category in {"shopping_net", "misc_net", "grocery_net"}:
            factors.append("Online category can carry higher card-not-present risk.")

        if distance >= 250:
            factors.append("Merchant location is far from the cardholder location.")

        if 0 < city_pop < 5000:
            factors.append("Small population area may need extra location validation.")

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
