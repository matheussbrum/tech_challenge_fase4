import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from src.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, SYMBOL, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_data():
    logger.info("Starting preprocessing")

    df = pd.read_csv(DATA_RAW_DIR / f"{SYMBOL}.csv", index_col=0)
    df = df[["Close"]].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(scaled, columns=["Close"]).to_csv(
        DATA_PROCESSED_DIR / "scaled_data.csv", index=False
    )

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    logger.info("Preprocessing completed")