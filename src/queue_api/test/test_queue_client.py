import unittest
import uuid

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import kafka_helper
from queue_client import QueueConfig
from queue_factory import QueueProducerCreator, QueueConsumerCreator

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "log"))
from custom_log import LoggerFactory


class TestQueueClient(unittest.TestCase):
    def setUp(self) -> None:
        self.__config = QueueConfig(
            host=os.environ.get("QUEUE_HOST"),
            port=int(os.environ.get("QUEUE_PORT")),
            optional_param={
                "topic_name": "test_queue_helper",
                "timeout": 5000,
            },
        )

        # Topicが残っているケースがあるので、事前にチェックして、削除する
        # TODO : アドホックな対応なので、なぜTearDownでTopicが削除されてないのか？を調査して対処する
        exist_topic = kafka_helper.exist_kafka_topic(
            self.__config.optional_param["topic_name"],
            client_id="test",
            config=self.__config,
        )
        if exist_topic:
            kafka_helper.delete_kafka_topics(
                topics=[self.__config.optional_param["topic_name"]],
                client_id="test",
                config=self.__config,
            )

        # テスト用のtopicの作成
        kafka_helper.create_kafka_topics_if_needed(
            topics=[self.__config.optional_param["topic_name"]],
            client_id="test",
            config=self.__config,
        )

    def tearDown(self) -> None:
        # テスト用のtopicの削除
        kafka_helper.delete_kafka_topics(
            topics=[self.__config.optional_param["topic_name"]],
            client_id="test",
            config=self.__config,
        )

    def test_send_and_receive_body(self):
        # データの作成
        body_text = "これはテストです"

        # 送信
        TestUtil.send_body_into_queue(
            body_text=body_text, config=self.__config
        )

        # 受信
        messages = TestUtil.receive_body_from_queue(config=self.__config)
        self.assertEqual(1, len(messages))
        message = messages[0]
        self.assertEqual(body_text, message["body"])
        try:
            _ = uuid.UUID(message["id"])
        except Exception:
            self.fail("id type don't equal UUID type")

        # 再度受信処理をして、データないことを確認
        messages = TestUtil.receive_body_from_queue(config=self.__config)
        self.assertEqual(0, len(messages))


class TestUtil:
    @staticmethod
    def send_body_into_queue(body_text: str, config: QueueConfig):
        # QueueProducerのインスタンスを取得
        logger = LoggerFactory.get_logger(
            logger_type="print", logger_name="test"
        )
        producer = QueueProducerCreator.create_producer(
            producer_type="kafka", config=config, logger=logger
        )

        # uuidを作成し、本文情報と一緒にjsonに格納
        id = str(uuid.uuid4())
        message = {"id": id, "body": body_text}

        # uuidと本文情報をjsonに格納し、Queueに送付
        producer.produce(messages=[message])

    @staticmethod
    def receive_body_from_queue(config: QueueConfig) -> list:
        # QueueConsumerのインスタンスを取得
        logger = LoggerFactory.get_logger(
            logger_type="print", logger_name="test"
        )
        consumer = QueueConsumerCreator.create_consumer(
            consumer_type="kafka", config=config, logger=logger
        )

        # jsonデータを取得
        messages = consumer.consume()
        return messages


if __name__ == "__main__":
    unittest.main()
