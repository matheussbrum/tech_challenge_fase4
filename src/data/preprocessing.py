import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, SYMBOL, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_data():
    logger.info("Starting preprocessing")

    df = pd.read_csv(DATA_RAW_DIR / f"{SYMBOL}.csv", index_col=0)

    if "Close" not in df.columns:
        raise ValueError("Column 'Close' not found in dataset")

    df = df[["Close"]]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Close"]].values)

    WINDOW_SIZE = 30

    def create_sequences(data, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, WINDOW_SIZE)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(DATA_PROCESSED_DIR / "X.npy", X)
    np.save(DATA_PROCESSED_DIR / "y.npy", y)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    logger.info("Preprocessing completed successfully")
