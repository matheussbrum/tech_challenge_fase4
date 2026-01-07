import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

from src.features.windowing import create_sequences
from src.utils.config import DATA_PROCESSED_DIR, WINDOW_SIZE, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate():
    logger.info("Starting model evaluation")

    data = pd.read_csv(DATA_PROCESSED_DIR / "scaled_data.csv").values
    X, y = create_sequences(data, WINDOW_SIZE)

    split = int(len(X) * 0.8)
    X_test, y_test = X[split:], y[split:]

    model = load_model(MODELS_DIR / "lstm_close_price.keras")
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    logger.info(f"MAE: {mae:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
