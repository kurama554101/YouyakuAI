from queue_client import (
    AbstractQueueInitializer,
    AbstractQueueConsumer,
    AbstractQueueProducer,
    QueueConfig,
    QueueError,
    _QueueInternalResult,
)

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "log"))
from custom_log import AbstractLogger


class GcpPubSubQueueInitializer(AbstractQueueInitializer):
    def __init__(self, config: QueueConfig, logger: AbstractLogger) -> None:
        super().__init__(config, logger)

    def initialize(self):
        # nothing to do
        pass


class GcpPubSubQueueProducer(AbstractQueueProducer):
    def produce(self, messages: list):
        # TODO : imp
        return super().produce(messages)


class GcpPubSubQueueConsumer(AbstractQueueConsumer):
    def consume(self) -> list:
        # TODO : imp
        return super().consume()
