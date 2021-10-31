import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "db"))
from db_wrapper import DBConfig, DBFactory
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "log"))
from log import LoggerFactory


def create_db_config():
    config = DBConfig(
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT"),
        username=os.environ.get("DB_USERNAME"),
        password=os.environ.get("DB_PASSWORD"),
        db_name=os.environ.get("DB_NAME"),
        db_type=os.environ.get("DB_TYPE")
    )
    return config


def create_logger(logger_name):
    return LoggerFactory.get_logger(logger_type="print", logger_name=logger_name)


def create_db_instance(config, logger):
    return DBFactory.get_db_instance(db_config=config, log_instance=logger)
