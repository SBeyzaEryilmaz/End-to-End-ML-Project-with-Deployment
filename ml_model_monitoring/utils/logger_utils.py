import logging


def get_logger(name: str) -> logging.Logger:
    """Returns a logger object.

    Args:
        name: Name

    Returns:
        Logger object
    """

    # Check if logger already exists
    if logging.getLogger(name).hasHandlers():
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set date format to YYYY-MM-DD-HH:MM
    formatter.datefmt = "%Y-%m-%d %H:%M"

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
