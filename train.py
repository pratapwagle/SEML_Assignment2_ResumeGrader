from config import METRICS_PATH, MODEL_PATH
from logging_config import configure_logging
from ml.data import build_training_data
from ml.trainer import train_model

configure_logging()
if __name__ == "__main__":
    metrics = train_model(build_training_data(), MODEL_PATH, METRICS_PATH)
    print(metrics)
