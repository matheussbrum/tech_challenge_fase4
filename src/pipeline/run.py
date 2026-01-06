import sys
from src.utils.logger import get_logger
from src.data.ingestion import download_data
from src.data.preprocessing import preprocess_data
from src.models.train import train
from src.models.evaluate import evaluate
from src.utils.config import MODELS_DIR

logger = get_logger("PIPELINE")

def run_pipeline():
    try:
        logger.info("=" * 60)
        logger.info("STARTING ML PIPELINE - TECH CHALLENGE FASE 4")
        logger.info("=" * 60)

        # 1. Data Ingestion
        logger.info("STEP 1: Data Ingestion")
        download_data()

        # 2. Data Preprocessing
        logger.info("STEP 2: Data Preprocessing")
        preprocess_data()

        # 3. Model Training
        logger.info("STEP 3: Model Training")
        train()

        # 4. Model Evaluation
        logger.info("STEP 4: Model Evaluation")
        evaluate()

        # 5. Final Validation
        logger.info("STEP 5: Artifact Validation")
        model_path = MODELS_DIR / "lstm_close_price.h5"
        scaler_path = MODELS_DIR / "scaler.pkl"

        if not model_path.exists():
            raise FileNotFoundError("Model file not found.")

        if not scaler_path.exists():
            raise FileNotFoundError("Scaler file not found.")

        logger.info("Pipeline executed successfully")
        logger.info(f"Model saved at: {model_path}")
        logger.info(f"Scaler saved at: {scaler_path}")

    except Exception as e:
        logger.error("Pipeline execution failed")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
