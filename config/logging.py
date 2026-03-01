import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger() -> logging.Logger:
    # Create logs directory if it doesn't exist
    log_file_path = "logs/main.log"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Keep up to 5 logs of max size of 5 MB
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        mode="a+",
        maxBytes=5*1024**2,
        backupCount=5
    )

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[file_handler, logging.StreamHandler()]
    )

    log = logging.getLogger(__name__)
    return log