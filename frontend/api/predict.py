import json
from http.server import BaseHTTPRequestHandler
from math import atan2, cos, radians, sin, sqrt


REQUIRED_FIELDS = {
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
}


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self._send_json({}, status=204)

    def do_POST(self):
        try:
            content_length = int(self.headers.get("content-length", 0))
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            missing = sorted(REQUIRED_FIELDS.difference(payload))

            if missing:
                self._send_json(
                    {"detail": f"Missing required fields: {', '.join(missing)}"},
                    status=422,
                )
                return

            self._send_json(predict(payload))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"detail": str(exc)}, status=400)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        if status != 204:
            self.wfile.write(body)


def predict(transaction):
    risk_factors = get_risk_factors(transaction)
    explanations = get_explanations(transaction)
    fraud_probability = fallback_probability(transaction, risk_factors)
    prediction = int(fraud_probability >= 0.55)

    return {
        "transaction_id": None,
        "prediction": prediction,
        "label": "Fraudulent" if prediction == 1 else "Legitimate",
        "fraud_probability": round(fraud_probability, 4),
        "model_loaded": False,
        "risk_level": risk_level(fraud_probability),
        "model_source": "Demo rules engine",
        "model_version": "Random Forest v1.0",
        "threshold": 0.55,
        "recommendation": recommendation(fraud_probability),
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


def get_risk_factors(transaction):
    factors = []
    amount = float(transaction.get("amt", 0))
    category = str(transaction.get("category", "")).lower()
    city_pop = int(transaction.get("city_pop", 0))
    distance = distance_miles(transaction)

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


def get_explanations(transaction):
    explanations = []
    amount = float(transaction.get("amt", 0))
    category = str(transaction.get("category", "")).lower()
    city_pop = int(transaction.get("city_pop", 0))
    distance = distance_miles(transaction)

    if amount >= 750:
        explanations.append(
            {
                "signal": "Transaction amount is unusually high.",
                "impact": "High",
                "contribution": 0.18 if amount >= 1000 else 0.11,
            }
        )
    elif amount >= 300:
        explanations.append(
            {
                "signal": "Transaction amount is above the normal review threshold.",
                "impact": "Medium",
                "contribution": 0.11,
            }
        )

    if category in {"shopping_net", "misc_net", "grocery_net"}:
        explanations.append(
            {
                "signal": "Online category can carry higher card-not-present risk.",
                "impact": "Medium",
                "contribution": 0.11,
            }
        )

    if distance >= 250:
        explanations.append(
            {
                "signal": "Merchant location is far from the cardholder location.",
                "impact": "High" if distance >= 500 else "Medium",
                "contribution": 0.25 if distance >= 500 else 0.11,
            }
        )

    if 0 < city_pop < 5000:
        explanations.append(
            {
                "signal": "Small population area may need extra location validation.",
                "impact": "Low",
                "contribution": 0.07,
            }
        )

    return explanations[:4]


def fallback_probability(transaction, risk_factors):
    amount = float(transaction.get("amt", 0))
    distance = distance_miles(transaction)
    score = 0.12 + (0.11 * len(risk_factors))

    if amount >= 1000:
        score += 0.18
    if distance >= 500:
        score += 0.14

    return min(score, 0.93)


def distance_miles(transaction):
    lat1 = float(transaction["lat"])
    lon1 = float(transaction["long"])
    lat2 = float(transaction["merch_lat"])
    lon2 = float(transaction["merch_long"])

    radius_miles = 3958.8
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return radius_miles * 2 * atan2(sqrt(a), sqrt(1 - a))


def risk_level(probability):
    if probability >= 0.7:
        return "Critical"
    if probability >= 0.45:
        return "Elevated"
    if probability >= 0.2:
        return "Moderate"
    return "Low"


def recommendation(probability):
    if probability >= 0.7:
        return "Block or hold this transaction until a fraud analyst reviews it."
    if probability >= 0.45:
        return "Send this transaction to manual review before approval."
    if probability >= 0.2:
        return "Approve with monitoring and keep the transaction in review history."
    return "Approve normally. No immediate fraud action is recommended."
