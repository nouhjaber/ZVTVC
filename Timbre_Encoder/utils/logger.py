import logging
import os
from datetime import datetime

_DEFAULT_LOG_DIR = '/content/logs'

def setup_logger(name, log_file=None, level=logging.INFO):
    """Logger with INFO-level file handler on LOCAL disk (not Drive)."""
    log_dir = _DEFAULT_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'timbre_encoder_{timestamp}.log')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logger '{name}' initialized. Logging to: {log_file}")
    return logger

def get_logger(name):
    return setup_logger(name)
