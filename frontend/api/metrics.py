import json
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = {
            "model_version": "Random Forest v1.0",
            "model_source": "Demo rules engine",
            "threshold": 0.55,
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
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
