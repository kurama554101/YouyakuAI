import uuid

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "queue_api"))
from queue_client import AbstractQueueProducer, AbstractQueueConsumer


def send_body_into_queue(body_text: str, producer: AbstractQueueProducer):
    # uuidを作成し、本文情報と一緒にjsonに格納
    id = str(uuid.uuid4())
    message = {"id": id, "body": body_text}

    # uuidと本文情報をjsonに格納し、Queueに送付
    # TODO : エラーハンドリング（例えば。Queueが稼働していないケース、高負荷状態にmessageを受け付けないケース）
    producer.produce(messages=[message])

    # uuidを返す
    return id


async def send_body_into_queue_task(
    loop, body_text: str, producer: AbstractQueueProducer
):
    # uuidを作成し、本文情報と一緒にjsonに格納
    id = str(uuid.uuid4())
    message = {"id": id, "body": body_text}

    # uuidと本文情報をjsonに格納し、Queueに送付
    # TODO : エラーハンドリング（例えば。Queueが稼働していないケース、高負荷状態にmessageを受け付けないケース）
    await producer.produce_task(loop=loop, messages=[message])

    # uuidを返す
    return id


def receive_body_from_queue(consumer: AbstractQueueConsumer) -> list:
    # jsonデータを取得
    messages = consumer.consume()
    return messages
