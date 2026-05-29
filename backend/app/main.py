from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictionResponse, TransactionInput
from ml.predictor import FraudPredictor


app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = FraudPredictor()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Credit Card Fraud Detection API"}


@app.get("/health")
def health() -> dict[str, bool | str]:
    return {
        "api_running": True,
        "model_loaded": predictor.is_loaded,
        "model_source": "Random Forest classifier"
        if predictor.is_loaded
        else "Demo rules engine",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput) -> PredictionResponse:
    try:
        result = predictor.predict(transaction.model_dump())
        return PredictionResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
