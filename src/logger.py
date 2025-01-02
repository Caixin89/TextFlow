import logging
import os

from config import config


def setup_logger(log_file):
    log_level = config["logging"]["log_level"]

    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Shared handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the file handler to the root logger
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():  # Prevent duplicate handlers
        root_logger.setLevel(getattr(logging, log_level))
        root_logger.addHandler(file_handler)

    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
