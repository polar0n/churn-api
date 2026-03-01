import os
import logging


def setup_logger() -> logging.Logger:
    # Keep up to 5 logs of max size of 5 MB
    file_handler = logging.handlers.RotatingFileHandler(
        filename="logs/main.log",
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