from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.auth import require_role
from app.schemas import (
    DecisionInput,
    MetricsResponse,
    PredictionResponse,
    TransactionInput,
    TransactionRecord,
)
from app.storage import init_db, list_transactions, save_prediction, update_decision
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
init_db()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Credit Card Fraud Detection API"}


@app.get("/health")
def health() -> dict[str, bool | str | float]:
    return {
        "api_running": True,
        "model_loaded": predictor.is_loaded,
        "model_source": "Random Forest classifier"
        if predictor.is_loaded
        else "Demo rules engine",
        "model_version": predictor.model_version,
        "threshold": predictor.threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    transaction: TransactionInput,
    role: str = Depends(require_role),
) -> PredictionResponse:
    try:
        result = predictor.predict(transaction.model_dump())
        record = save_prediction(transaction.model_dump(), result)
        result["transaction_id"] = record["id"]
        return PredictionResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/transactions", response_model=list[TransactionRecord])
def transactions(
    limit: int = 25,
    role: str = Depends(require_role),
) -> list[TransactionRecord]:
    return [TransactionRecord(**record) for record in list_transactions(limit=limit)]


@app.post("/decisions/{transaction_id}", response_model=TransactionRecord)
def decide(
    transaction_id: int,
    decision_input: DecisionInput,
    role: str = Depends(require_role),
) -> TransactionRecord:
    try:
        record = update_decision(
            transaction_id,
            decision_input.decision,
            decision_input.note,
        )
        return TransactionRecord(**record)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Transaction not found") from exc


@app.get("/metrics", response_model=MetricsResponse)
def metrics(role: str = Depends(require_role)) -> MetricsResponse:
    return MetricsResponse(**predictor.model_metrics())
