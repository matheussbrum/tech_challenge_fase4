import numpy as np
import joblib
from tensorflow.keras.models import load_model
from utils.config import MODELS_DIR, WINDOW_SIZE

model = load_model(MODELS_DIR / "lstm_close_price.keras")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")

def predict(prices: list[float]) -> float:
    prices = np.array(prices).reshape(-1, 1)
    scaled = scaler.transform(prices)

    X = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    pred_scaled = model.predict(X)

    pred = scaler.inverse_transform(pred_scaled)
    return float(pred[0][0])

predict([150.0, 152.5, 151.0, 153.0, 154.5])

