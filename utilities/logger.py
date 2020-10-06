import os
import logging
from os.path import exists


def set_logger(log_path: str = None) -> None:
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
        logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s'))
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # Logging to file
    if log_path:
        if exists(log_path):
            os.remove(log_path)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s'))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)



if __name__ == '__main__':
    log_path = './logger_test.log'
    set_logger(log_path)
    logging.info("Logging one line")
    logging.info("Logging two lines")
    logging.info("Logging three lines")
    logging.warning("Warning one")
    logging.warning("Warning two")
    logging.error("Error one")
    logging.error("Error two")
