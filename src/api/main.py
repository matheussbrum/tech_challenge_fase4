from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date
import numpy as np
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from prometheus_fastapi_instrumentator import Instrumentator

from src.utils.config import MODEL_PATH, SCALER_PATH, WINDOW_SIZE

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(
    title="Tech Challenge Fase 4 - Stock Prediction API",
    version="1.0.0"
)

instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=[".*admin.*", "/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)

instrumentator.instrument(app).expose(app)

class PredictionRequest(BaseModel):
    symbol: str = Field(..., example="MSFT")
    start_date: date = Field(..., example="2020-01-01")
    end_date: date = Field(..., example="2025-01-01")
    window_size: int = Field(
        default=WINDOW_SIZE,
        gt=0,
        description="Must match training window size"
    )

class PredictionResponse(BaseModel):
    symbol: str
    window_size: int
    prediction: float

# ============================================================
# Endpoint
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")

    # Validação de consistência com o treino
    if req.window_size != WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid window_size. Model was trained with {WINDOW_SIZE}"
        )

    # Download dos dados
    df = yf.download(req.symbol, start=req.start_date, end=req.end_date)

    if df.empty or "Close" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="No data returned for given parameters"
        )

    closes = df["Close"].dropna().values

    if len(closes) < req.window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data. Need at least {req.window_size} records"
        )

    # Usa apenas a última janela
    closes = closes[-req.window_size:]

    # Pré-processamento (mesmo do treino)
    scaled = scaler.transform(closes.reshape(-1, 1))
    X = scaled.reshape(1, req.window_size, 1)

    # Inferência
    pred_scaled = model.predict(X, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)
    prediction = float(pred[0][0])

    # ============================================================
    # Validação de sanidade da previsão
    # ============================================================

    # Em vez de min/max da janela inteira:
    last_close = closes[-1]
    lower_bound = last_close * 0.70 # Permite queda de 30%
    upper_bound = last_close * 1.30 # Permite alta de 30%

    if not (lower_bound <= prediction <= upper_bound):
        raise HTTPException(
            status_code=422,
            detail=f"Predição irrealista. Último preço: {last_close:.2f}, Previsto: {prediction:.2f}"
        )

    return PredictionResponse(
        symbol=req.symbol,
        window_size=req.window_size,
        prediction=prediction
    )
