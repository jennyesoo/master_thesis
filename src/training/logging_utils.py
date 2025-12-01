import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    fmt = "[%(asctime)s] %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logging.basicConfig(level=level, handlers=[handler])
