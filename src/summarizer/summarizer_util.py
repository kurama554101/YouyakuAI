import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "queue_api"))
from queue_client import QueueConfig
from queue_factory import QueueProducerCreator

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "db"))
from db_wrapper import DBConfig, DBFactory

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "log"))
from custom_log import LoggerFactory


def create_queue_config():
    config = QueueConfig(
        host=os.environ.get("QUEUE_HOST"),
        port=int(os.environ.get("QUEUE_PORT")),
        optional_param={
            "topic_name": os.environ.get("QUEUE_NAME"),
            "timeout": 5000,
        },
    )
    return config


def create_db_config():
    config = DBConfig(
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT"),
        username=os.environ.get("DB_USERNAME"),
        password=os.environ.get("DB_PASSWORD"),
        db_name=os.environ.get("DB_NAME"),
        db_type=os.environ.get("DB_TYPE"),
    )
    return config


def create_logger(logger_name):
    return LoggerFactory.get_logger(
        logger_type="print", logger_name=logger_name
    )


def create_db_instance(config, logger):
    return DBFactory.get_db_instance(db_config=config, log_instance=logger)


def create_queue_instance(config, logger):
    return QueueProducerCreator.create_producer(
        producer_type="kafka", config=config, logger=logger
    )
