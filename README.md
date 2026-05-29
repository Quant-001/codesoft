# Credit Card Fraud Detection

Full-stack project for detecting fraudulent credit card transactions.

## Tech Stack

- Frontend: React with Vite
- Backend: FastAPI
- ML Model: Scikit-learn Random Forest pipeline
- Dataset: Kaggle Credit Card Transactions Fraud Detection Dataset
- Languages: Python and JavaScript

## Project Structure

```text
CREDIT/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/
│   │   └── fraudTrain.csv
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── train_model.py
│   ├── models/
│   │   └── fraud_model.joblib
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   └── package.json
├── .gitignore
└── README.md
```

## Dataset Setup

Download the Kaggle dataset shown in your screenshot:

`Credit Card Transactions Fraud Detection Dataset`

Place the training CSV here:

```text
backend/data/fraudTrain.csv
```

The training script expects the Kaggle columns:

```text
amt, category, merchant, gender, city, state, job, lat, long,
merch_lat, merch_long, city_pop, unix_time, is_fraud
```

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.train_model
uvicorn app.main:app --reload
```

The API runs at:

```text
http://localhost:8000
```

Useful endpoints:

- `GET /health`
- `POST /predict`

## Frontend Setup

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The React app runs at:

```text
http://localhost:5173
```

## Docker Setup

Run the full app with Docker Compose from the project root:

```bash
docker compose up --build
```

Open the frontend:

```text
http://localhost:5174
```

The backend API is exposed on:

```text
http://localhost:8765
```

Useful Docker commands:

```bash
docker compose ps
docker compose down
```

## How It Works

1. The Kaggle CSV is loaded by `backend/ml/train_model.py`.
2. Numeric and categorical features are cleaned with a scikit-learn preprocessing pipeline.
3. A balanced Random Forest classifier is trained.
4. The saved model is loaded by FastAPI.
5. The React dashboard sends transaction details to `/predict`.
6. FastAPI returns the fraud label, probability, risk level, recommendation, and risk signals.
