from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

SYMBOL = "MSFT"
START_DATE = "2020-10-01"
END_DATE = "2025-01-01"

WINDOW_SIZE = 1
TEST_SIZE = 0.2
RANDOM_STATE = 42