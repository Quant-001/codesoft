# FraudShield: Credit Card Fraud Detection

End-to-end fraud operations dashboard for scoring credit card transactions, explaining fraud risk, and recording analyst decisions.

## Tech Stack

- Frontend: React with Vite
- Backend: FastAPI
- ML: Pandas, Scikit-learn, Random Forest
- Storage: SQLite for transaction history and analyst decisions
- Deployment: Docker, Vercel-compatible frontend API fallback, GitHub Actions CI

## Features

- Fraudulent vs Legitimate transaction prediction
- Fraud probability, risk level, and recommendation
- Explainable risk signals with impact and score contribution
- Transaction history persisted in SQLite
- Analyst workflow: Approve, Reject, Manual Review
- Model evaluation metrics: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix
- Imbalanced-data handling with `class_weight="balanced"` and tuned decision threshold
- Model versioning with model source, trained dataset, feature count, and threshold
- Optional API-key roles for Admin and Fraud Analyst
- Health, metrics, transactions, decisions, and prediction APIs

## Project Structure

```text
CREDIT/
├── backend/
│   ├── app/
│   │   ├── auth.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── storage.py
│   ├── data/
│   │   └── fraudTrain.csv
│   ├── ml/
│   │   ├── predictor.py
│   │   └── train_model.py
│   ├── models/
│   │   ├── fraud_model.joblib
│   │   └── model_metrics.json
│   └── requirements.txt
├── frontend/
│   ├── api/
│   │   ├── metrics.py
│   │   ├── predict.py
│   │   └── transactions.py
│   └── src/
│       ├── main.jsx
│       └── styles.css
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Dataset Setup

Download the Kaggle Credit Card Transactions Fraud Detection dataset and place:

```text
backend/data/fraudTrain.csv
```

Required columns:

```text
amt, category, merchant, gender, city, state, job, lat, long,
merch_lat, merch_long, city_pop, unix_time, is_fraud
```

## Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Optional API-key auth is disabled by default. To enable it, set:

```text
FRAUD_ADMIN_KEY=your-admin-key
FRAUD_ANALYST_KEY=your-analyst-key
```

Then include this header in protected API requests:

```text
X-API-Key: your-analyst-key
```

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.train_model
uvicorn app.main:app --reload --port 8000
```

Useful endpoints:

- `GET /health`
- `POST /predict`
- `GET /transactions`
- `POST /decisions/{transaction_id}`
- `GET /metrics`

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at:

```text
http://localhost:5173
```

Vite proxies `/api` to `http://127.0.0.1:8765` by default. Set `VITE_PROXY_API_TARGET` if your backend runs somewhere else.

## Docker Setup

```bash
docker compose up --build
```

Open:

```text
http://localhost:5174
```

Backend API:

```text
http://localhost:8765
```

## How It Works

1. `backend/ml/train_model.py` loads and validates the Kaggle dataset.
2. Numeric and categorical features are processed in a Scikit-learn pipeline.
3. A balanced Random Forest model is trained.
4. The trainer selects a fraud decision threshold and saves evaluation metrics.
5. FastAPI loads the model and scores transactions through `/predict`.
6. Predictions are saved to SQLite with model metadata and explanations.
7. Analysts review transactions and save decisions through the dashboard.

## Deployment Notes

- Vercel frontend deployments can use the `frontend/api` serverless functions for demo scoring.
- Full deployments should run the FastAPI backend on Render, Railway, Fly.io, or Docker and set `VITE_API_URL` to that backend URL.
- SQLite is suitable for local/demo use. PostgreSQL is recommended for a multi-user production deployment.
