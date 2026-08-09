"""Central logging setup for console and rotating project log files."""

import logging
import sys
from logging.handlers import RotatingFileHandler

from config import LOG_PATH


def configure_logging() -> None:
    """Configure root logging to stdout and a bounded rotating file.

    Creates the log directory if needed. Uses a 1 MB file size limit with
    three backups so local demos do not grow unbounded log directories.
    """
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
