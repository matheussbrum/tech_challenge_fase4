import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from src.features.windowing import create_sequences
from src.models.lstm_model import build_model
from src.utils.config import DATA_PROCESSED_DIR, WINDOW_SIZE, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train():
    logger.info("Starting model training")

    data = pd.read_csv(DATA_PROCESSED_DIR / "scaled_data.csv").values
    X, y = create_sequences(data, WINDOW_SIZE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,shuffle=False
    )

    model = build_model((X_train.shape[1], 1))

    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop]
    )

    model.save(MODELS_DIR / "lstm_close_price.h5")
    logger.info("Model trained and saved")
