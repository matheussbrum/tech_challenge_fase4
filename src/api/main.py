from fastapi import FastAPI
from src.api.schemas import PredictionRequest, PredictionResponse
from src.inference.predictor import predict

app = FastAPI(title="Stock Price Prediction API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    result = predict(request.prices)
    return {"prediction": result}
