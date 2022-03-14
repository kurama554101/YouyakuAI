from google.cloud import pubsub_v1
from concurrent import futures
from typing import Callable
import json
from multiprocessing import Manager

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
        # TODO : create topic and subscription
        pass


class GcpPubSubQueueProducer(AbstractQueueProducer):
    def produce(self, messages: list):
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(
            project=self._config.optional_param["google_project_id"],
            topic=self._config.optional_param["topic_name"],
        )

        def seriarize(value: dict) -> str:
            return json.dumps(value).encode()

        def get_callback(
            data: str,
        ) -> Callable[[pubsub_v1.publisher.futures.Future], None]:
            def callback(
                publish_future: pubsub_v1.publisher.futures.Future,
            ) -> None:
                try:
                    print(publish_future.result(timeout=60))
                except futures.TimeoutError:
                    print(f"publishing {data} time out...")

            return callback

        publish_futures = []
        for message in messages:
            data = seriarize(message)
            publish_future = publisher.publish(topic_path, data)
            publish_future.add_done_callback(get_callback(data))
            publish_futures.append(publish_future)
        futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)


class GcpPubSubQueueConsumer(AbstractQueueConsumer):
    def __init__(self, config: QueueConfig, logger: AbstractLogger) -> None:
        super().__init__(config, logger)
        self.__subscription_name = "{}-sub".format(
            self._config.optional_param["topic_name"]
        )

    def consume(self) -> list:
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(
            self._config.optional_param["google_project_id"],
            self.__subscription_name,
        )

        def deseriarize(data: str) -> dict:
            return json.loads(data)

        # 少なくともマルチスレッドで、listに処理が入るため、datasをスレッドセーブにする必要がある
        manager = Manager()
        datas = manager.list()

        def callback(message: pubsub_v1.subscriber.message.Message):
            data = deseriarize(message.data)
            datas.append(data)
            message.ack()

        subscribe_future = subscriber.subscribe(subscription_path, callback)
        with subscriber:
            try:
                timeout_sec = int(
                    self._config.optional_param["timeout"] / 1000
                )
                subscribe_future.result(timeout=timeout_sec)
            except TimeoutError:
                subscribe_future.cancel()
                subscribe_future.result()
        return datas
