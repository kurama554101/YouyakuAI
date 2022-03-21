from google.cloud import pubsub_v1
from concurrent import futures
from typing import Callable
import json
from multiprocessing import Manager
import asyncio
from concurrent.futures import TimeoutError

from queue_client import (
    AbstractQueueInitializer,
    AbstractQueueConsumer,
    AbstractQueueProducer,
    QueueConfig,
    QueueError,
)

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "log"))
from custom_log import AbstractLogger


class GcpPubSubQueueInitializer(AbstractQueueInitializer):
    def __init__(self, config: QueueConfig, logger: AbstractLogger) -> None:
        super().__init__(config, logger)
        self.__subscription_name = "{}-sub".format(
            self._config.optional_param["topic_name"]
        )

    def initialize(self):
        # create topic if needed
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(
            project=self._config.optional_param["google_project_id"],
            topic=self._config.optional_param["topic_name"],
        )
        exist_topic_list = publisher.list_topics(
            request={
                "project": "projects/{}".format(
                    self._config.optional_param["google_project_id"]
                )
            }
        )
        exist_topic_list = [topic.name for topic in exist_topic_list]
        if topic_path in exist_topic_list:
            self._logger.info(f"{topic_path} is already exist.")
        else:
            topic = publisher.create_topic(name=topic_path)
            self._logger.info(f"create the {topic} topic")

        # create subscription if needed
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(
            self._config.optional_param["google_project_id"],
            self.__subscription_name,
        )
        with subscriber:
            exists_subscription_paths = publisher.list_topic_subscriptions(
                request={"topic": topic_path}
            )
            if subscription_path in exists_subscription_paths:
                self._logger.info(
                    f"{subscription_path} subscription is already exist."
                )
            else:
                subscription = subscriber.create_subscription(
                    name=subscription_path, topic=topic_path
                )
                self._logger.info(f"create the {subscription} subscription")


class GcpPubSubQueueProducer(AbstractQueueProducer):
    def produce(self, messages: list):
        self._logger.info("start pub/sub publish to queue..")
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
                    self._logger.info(publish_future.result(timeout=60))
                except futures.TimeoutError:
                    self._logger.error(f"publishing {data} time out...")
                except Exception as e:
                    self._logger.error(
                        f"publish error is occured! datail is {e}"
                    )

            return callback

        publish_futures = []
        with publisher:
            for message in messages:
                data = seriarize(message)
                publish_future = publisher.publish(topic_path, data)
                publish_future.add_done_callback(get_callback(data))
                publish_futures.append(publish_future)
            futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)
        self._logger.info("end pub/sub publish process.")

    async def produce_task(self, loop: asyncio.BaseEventLoop, messages: list):
        result = await loop.run_in_executor(None, self.produce, messages)
        return result


class GcpPubSubQueueConsumer(AbstractQueueConsumer):
    def __init__(self, config: QueueConfig, logger: AbstractLogger) -> None:
        super().__init__(config, logger)
        self.__subscription_name = "{}-sub".format(
            self._config.optional_param["topic_name"]
        )

    def consume(self) -> list:
        self._logger.info("start pub/sub consumer process...")
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

        with subscriber:
            try:
                subscribe_future = subscriber.subscribe(
                    subscription_path, callback
                )
                timeout_sec = int(
                    self._config.optional_param["timeout"] / 1000
                )
                subscribe_future.result(timeout=timeout_sec)
            except TimeoutError:
                subscribe_future.cancel()
                subscribe_future.result()
                raise QueueError(
                    "consume process is timeout! next consume process will be started..."
                )
            except Exception as e:
                subscribe_future.cancel()
                subscribe_future.result()
                raise QueueError(
                    "consume process occured error! detail is {}".format(e)
                )
        self._logger.info(
            "end pub/sub consume process. received datas is {}".format(datas)
        )
        return datas

    async def consume_task(self, loop: asyncio.BaseEventLoop) -> list:
        # TODO : implement
        return await super().consume_task(loop)
