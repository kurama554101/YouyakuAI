from typing import List
from kafka.admin import KafkaAdminClient, NewTopic
from queue_client import QueueConfig
from kafka.errors import TopicAlreadyExistsError

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "log"))
from log import AbstractLogger, LoggerFactory


def create_kafka_topics_if_needed(topics:List[str], client_id:str, config:QueueConfig):
    client = KafkaAdminClient(
        bootstrap_servers=config.get_url(),
        client_id=client_id
    )

    new_topics = []
    for topic in topics:
        new_topics.append(NewTopic(name=topic, num_partitions=1, replication_factor=1))

    try:
        client.create_topics(new_topics=new_topics, validate_only=False)
    except TopicAlreadyExistsError as e:
        logger = LoggerFactory.get_logger("print", logger_name="kafka_helper")
        logger.error("{} topic is already exists! detail is {}".format(topics, e))


def get_kafka_topics(client_id:str, config:QueueConfig):
    client = KafkaAdminClient(
        bootstrap_servers=config.get_url(),
        client_id=client_id
    )
    return client.list_topics()


def exist_kafka_topic(topic, client_id:str, config:QueueConfig) -> bool:
    client = KafkaAdminClient(
        bootstrap_servers=config.get_url(),
        client_id=client_id
    )
    topics = client.list_topics()
    return topic in topics


def delete_kafka_topics(topics:List[str], client_id:str, config:QueueConfig):
    client = KafkaAdminClient(
        bootstrap_servers=config.get_url(),
        client_id=client_id
    )

    client.delete_topics(topics=topics)
