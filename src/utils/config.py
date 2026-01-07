from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "lstm_close_price.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

SYMBOL = "MSFT"
START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

WINDOW_SIZE = 1
TEST_SIZE = 0.2
RANDOM_STATE = 42