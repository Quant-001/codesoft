from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    amt: float = Field(..., gt=0, description="Transaction amount")
    category: str = Field(..., min_length=1)
    merchant: str = Field(..., min_length=1)
    gender: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    state: str = Field(..., min_length=1)
    job: str = Field(..., min_length=1)
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    city_pop: int = Field(..., ge=0)
    unix_time: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    transaction_id: int | None = None
    prediction: int
    label: str
    fraud_probability: float
    model_loaded: bool
    risk_level: str
    model_source: str
    model_version: str
    threshold: float
    recommendation: str
    risk_factors: list[str]
    explanations: list[dict[str, Any]]


class TransactionRecord(BaseModel):
    id: int
    created_at: datetime
    amount: float
    merchant: str
    category: str
    city: str
    state: str
    prediction: int
    label: str
    fraud_probability: float
    risk_level: str
    model_source: str
    model_version: str
    threshold: float
    analyst_decision: str
    decision_note: str | None = None
    decided_at: datetime | None = None
    request_payload: dict[str, Any]
    risk_factors: list[str]
    explanations: list[dict[str, Any]]


class DecisionInput(BaseModel):
    decision: Literal["Approved", "Rejected", "Manual Review"]
    note: str | None = Field(default=None, max_length=500)


class MetricsResponse(BaseModel):
    model_version: str
    model_source: str
    threshold: float
    trained_at: str | None = None
    dataset: str
    sample_rows: int | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    roc_auc: float | None = None
    confusion_matrix: list[list[int]]
    feature_count: int
    positive_rate: float | None = None
