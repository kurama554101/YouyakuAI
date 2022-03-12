import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "log"))
from custom_log import LoggerFactory


def create_logger(logger_name):
    return LoggerFactory.get_logger(
        logger_type="print", logger_name=logger_name
    )
