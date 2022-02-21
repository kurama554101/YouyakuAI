import json
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import asyncio
from queue_client import (
    AbstractQueueInitializer,
    AbstractQueueConsumer,
    AbstractQueueProducer,
    QueueConfig,
    QueueError,
    _QueueInternalResult,
)
from kafka_helper import create_kafka_topics_if_needed

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "log"))
from custom_log import AbstractLogger


class KafkaQueueInitializer(AbstractQueueInitializer):
    def __init__(self, config: QueueConfig, logger: AbstractLogger) -> None:
        super().__init__(config, logger)
        self.__client_id = "youyaku_queue_initializer"

    def initialize(self):
        # TOPICの作成対応
        topic_name = self._config.optional_param["topic_name"]
        self._logger.info(
            "create new kafka topic if needed. name is {}".format(topic_name)
        )
        create_kafka_topics_if_needed(
            topics=[topic_name],
            client_id=self.__client_id,
            config=self._config,
        )


class KafkaQueueProducer(AbstractQueueProducer):
    def produce(self, messages: list):
        loop = asyncio.new_event_loop()
        internal_result = loop.run_until_complete(
            self.__produce(loop=loop, messages=messages)
        )
        loop.close()

        if internal_result._e is not None:
            raise QueueError(
                "QueueProducer Error! error detail is {}".format(
                    internal_result._e
                )
            )

    async def produce_task(self, loop, messages: list):
        internal_result = await self.__produce(loop=loop, messages=messages)

        if internal_result._e is not None:
            raise QueueError(
                "QueueProducer Error! error detail is {}".format(
                    internal_result._e
                )
            )

    async def __produce(self, loop, messages: list) -> _QueueInternalResult:
        def serializer(value):
            return json.dumps(value).encode()

        try:
            producer = AIOKafkaProducer(
                loop=loop,
                bootstrap_servers=self._config.get_url(),
                value_serializer=serializer,
                compression_type="gzip",
            )
            await producer.start()
            # TODO : send_batchを使った方が良い
            for message in messages:
                await producer.send(
                    self._config.optional_param["topic_name"], message
                )
            await producer.stop()
        except Exception as e:
            return _QueueInternalResult(None, e=e)
        return _QueueInternalResult(None, e=None)


class KafkaQueueConsumer(AbstractQueueConsumer):
    def consume(self) -> list:
        loop = asyncio.new_event_loop()
        internal_result = loop.run_until_complete(self.__consume(loop))
        loop.close()

        if internal_result._e is not None:
            raise QueueError(
                "QueueConsumer Error! error detail is {}".format(
                    internal_result._e
                )
            )
        return internal_result._result

    async def consume_task(self, loop) -> list:
        internal_result = self.__consume(loop=loop)

        if internal_result._e is not None:
            raise QueueError(
                "QueueConsumer Error! error detail is {}".format(
                    internal_result._e
                )
            )
        return internal_result._result

    async def __consume(self, loop) -> _QueueInternalResult:
        def deserializer(serialized):
            return json.loads(serialized)

        try:
            consumer = AIOKafkaConsumer(
                self._config.optional_param["topic_name"],
                loop=loop,
                group_id="youyaku_ai_group",
                # isolation_level="read_committed",
                bootstrap_servers=self._config.get_url(),
                value_deserializer=deserializer,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
            )
            await consumer.start()

            # messageのpositionとoffset（終端）を確認し、データがなければ、空のデータを返す
            # TODO : 1パーティションの対応のみなので、パーティションが複数の対応が必要
            partition = list(consumer.assignment())[0]
            position = await consumer.position(partition=partition)
            offset_dict = await consumer.end_offsets(partitions=[partition])
            end = offset_dict[partition]
            if position == end:
                return _QueueInternalResult(result=[], e=None)

            # データを一つ取得
            data = await consumer.getone()
            messages = [data.value]
            await consumer.commit()
        except Exception as e:
            return _QueueInternalResult(result=None, e=e)
        finally:
            await consumer.stop()
        return _QueueInternalResult(result=messages, e=None)
