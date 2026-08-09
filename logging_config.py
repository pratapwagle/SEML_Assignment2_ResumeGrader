import logging
import sys
from logging.handlers import RotatingFileHandler

from config import LOG_PATH


def configure_logging() -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler,
        ],
        force=True,
    )
