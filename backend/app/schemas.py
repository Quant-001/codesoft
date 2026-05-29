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
    prediction: int
    label: str
    fraud_probability: float
    model_loaded: bool
    risk_level: str
    model_source: str
    recommendation: str
    risk_factors: list[str]
