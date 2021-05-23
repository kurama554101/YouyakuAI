from queue_client import QueueConfig, AbstractQueueInitializer, AbstractQueueConsumer, AbstractQueueProducer
from kafka_client import KafkaQueueInitializer, KafkaQueueConsumer, KafkaQueueProducer

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "log"))
from log import AbstractLogger



class QueueInitializerCreator:
    @staticmethod
    def create_initializer(initializer_type:str, config:QueueConfig, logger:AbstractLogger) -> AbstractQueueInitializer:
        if initializer_type == "kafka":
            return KafkaQueueInitializer(config=config, logger=logger)
        else:
            raise NotImplementedError("{} type is not implemented!".format(initializer_type))


class QueueProducerCreator:
    @staticmethod
    def create_producer(producer_type:str, config:QueueConfig, logger:AbstractLogger) -> AbstractQueueProducer:
        if producer_type == "kafka":
            return KafkaQueueProducer(config=config, logger=logger)
        else:
            raise NotImplementedError("{} type is not implemented!".format(producer_type))


class QueueConsumerCreator:
    @staticmethod
    def create_consumer(consumer_type:str, config:QueueConfig, logger:AbstractLogger) -> AbstractQueueConsumer:
        if consumer_type == "kafka":
            return KafkaQueueConsumer(config=config, logger=logger)
        else:
            raise NotImplementedError("{} type is not implemented!".format(consumer_type))
