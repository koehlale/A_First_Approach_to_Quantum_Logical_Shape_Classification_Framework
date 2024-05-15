import logging
import logging.config


def create_logger(name: str) -> logging:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = 0

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    ch.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%d.%m.%Y %H:%M:%S",
        )
    )
    logger.addHandler(ch)

    return logger
