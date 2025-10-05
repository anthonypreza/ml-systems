"""Logger utilities."""

import os
import logging

PWD = os.path.dirname(__file__)
LOGS_PATH = f"{PWD}/logs"
LOG_LEVEL = logging.INFO


def get_logger(name, log_level=logging.INFO):
    """Get logger."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    log_file = os.path.join(f"{LOGS_PATH}/{name}.log")

    if not logger.handlers:
        os.makedirs(LOGS_PATH, exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Create structured formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
