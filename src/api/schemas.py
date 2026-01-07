from pydantic import BaseModel
from datetime import date

class PredictionRequest(BaseModel):
    symbol: str
    start_date: date
    end_date: date

class PredictionResponse(BaseModel):
    symbol: str
    prediction: float
