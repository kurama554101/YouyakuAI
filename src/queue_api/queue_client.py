from abc import ABCMeta, abstractmethod

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "log"))
from custom_log import AbstractLogger


class _QueueInternalResult:
    def __init__(self, result, e: Exception) -> None:
        self._result = result
        self._e = e


class QueueConfig:
    def __init__(self, host: str, port: int, optional_param: dict) -> None:
        self.host = host
        self.port = port
        self.optional_param = optional_param

    def get_url(self) -> str:
        return "{}:{}".format(self.host, self.port)


class AbstractQueueInitializer:
    def __init__(self, config: QueueConfig, logger: AbstractLogger) -> None:
        self._config = config
        self._logger = logger

    @abstractmethod
    def initialize(self):
        pass


class AbstractQueueProducer:
    __metadata__ = ABCMeta

    def __init__(self, config: QueueConfig, logger: AbstractLogger) -> None:
        self._config = config
        self._logger = logger

    @abstractmethod
    def produce(self, messages: list):
        pass

    @abstractmethod
    async def produce_task(self, loop, messages: list):
        pass


class AbstractQueueConsumer:
    __metadata__ = ABCMeta

    def __init__(self, config: QueueConfig, logger: AbstractLogger) -> None:
        self._config = config
        self._logger = logger

    @abstractmethod
    def consume(self) -> list:
        pass

    @abstractmethod
    async def consume_task(self, loop) -> list:
        pass


class QueueError(Exception):
    pass
