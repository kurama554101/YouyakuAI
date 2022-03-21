from google.cloud import pubsub_v1
from concurrent import futures
from typing import Callable, List
import json
from multiprocessing import Manager
import time
import argparse
import os
from concurrent.futures import TimeoutError


def create_sample_messages(count: int = 5) -> List[dict]:
    messages = []
    for i in range(count):
        message = {"id": i, "body": "sample message No{}".format(i)}
        messages.append(message)
    return messages


def publish_message(project_id: str, topic_name: str, messages: List[dict]):
    print("start to publish message...")
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project=project_id, topic=topic_name,)

    def seriarize(value: dict) -> str:
        return json.dumps(value).encode()

    def get_callback(
        data: str,
    ) -> Callable[[pubsub_v1.publisher.futures.Future], None]:
        def callback(
            publish_future: pubsub_v1.publisher.futures.Future,
        ) -> None:
            try:
                publish_future.result(timeout=60)
            except futures.TimeoutError:
                print(f"publishing {data} time out...")
            except Exception as e:
                print(f"publish error is occured! datail is {e}")

        return callback

    publish_futures = []
    with publisher:
        for message in messages:
            data = seriarize(message)
            publish_future = publisher.publish(topic_path, data)
            publish_future.add_done_callback(get_callback(data))
            publish_futures.append(publish_future)
        futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)

    print("end to publish message!")


def consume_message(project_id: str, topic_name: str):
    print("start to consume message...")
    subscriber = pubsub_v1.SubscriberClient()
    subscription_name = "{}-sub".format(topic_name)
    subscription_path = subscriber.subscription_path(
        project=project_id, subscription=subscription_name
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
            timeout_sec = 5
            subscribe_future.result(timeout=timeout_sec)
        except TimeoutError:
            subscribe_future.cancel()
            subscribe_future.result()
            print("subscription timeout is occured!")
    return datas


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcp_project_id", default="youyaku-ai")
    parser.add_argument("--topic_name", default="youyaku_ai_topic")
    parser.add_argument(
        "--gcp_service_account_file",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "credentials",
            "youyaku-ai-service-account.json",
        ),
    )
    return parser.parse_args()


def main():
    # パラメーターの設定
    args = get_args()
    project_id = args.gcp_project_id
    topic_name = args.topic_name

    # 環境変数の設定
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"
    ] = args.gcp_service_account_file

    # サンプルの送信メッセージの作成
    messages = create_sample_messages()

    # 送信処理
    publish_message(
        project_id=project_id, topic_name=topic_name, messages=messages
    )
    time.sleep(5)

    # 受信処理
    datas = consume_message(project_id=project_id, topic_name=topic_name)
    print("receive data is : {}".format(datas))


if __name__ == "__main__":
    main()
