import yfinance as yf
from src.utils.config import SYMBOL, START_DATE, END_DATE, DATA_RAW_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def download_data():
    logger.info("Downloading data from Yahoo Finance")
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)
    
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW_DIR / f"{SYMBOL}.csv"
    df.to_csv(path)

    logger.info(f"Data saved at {path}")
    return df
