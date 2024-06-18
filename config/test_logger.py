from logger import create_logger


def main():
    # logging.config.fileConfig("config/logging.conf")
    # log = logging.getLogger("test2")
    log = create_logger("test_name")
    # log.level(logging.DEBUG)
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    log.critical("Critical message")


if __name__ == "__main__":
    main()
