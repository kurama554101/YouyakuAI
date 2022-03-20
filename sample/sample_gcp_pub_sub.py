from google.cloud import pubsub_v1
from concurrent import futures
from typing import Callable, List
import json
from multiprocessing import Manager
import time
import argparse
import os


def create_sample_messages(count: int = 5) -> List[dict]:
    messages = []
    for i in range(count):
        message = {"id": i, "body": "sample message No{}".format(i)}
        messages.append(message)
    return messages


def publish_message(project_id: str, topic_name: str, messages: List[dict]):
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


def consume_message(project_id: str, topic_name: str):
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
        # debug
        print("pub/sub consume callback")

        data = deseriarize(message.data)

        # debug
        print("data is : {}".format(data))

        datas.append(data)
        message.ack()

    with subscriber:
        try:
            subscribe_future = subscriber.subscribe(
                subscription_path, callback
            )
            timeout_sec = 5

            # debug
            print("wait cosume process...")

            subscribe_future.result(timeout=timeout_sec)

            # debug
            print("consume process is done")
        except TimeoutError as e:
            subscribe_future.cancel()
            subscribe_future.result()

            # debug
            print(e)
            print("consume process is timeout! detail is {}".format(e))
        except Exception as e:
            subscribe_future.cancel()
            subscribe_future.result()

            # debug
            print(f"consumer error : {e}")
            print("consume process occured error! detail is {}".format(e))

    # debug
    print("final consume datas is {}".format(datas))

    return datas


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gcp_project_id", default=os.environ.get("GCP_PROJECT_ID")
    )
    parser.add_argument("--topic_name", default="youyaku_ai_topic")
    parser.add_argument(
        "--gcp_service_account_file",
        default=os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE"),
    )
    return parser.parse_args()


def main():
    # パラメーターの設定
    args = get_args()
    project_id = args.gcp_project_id
    topic_name = args.topic_name

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
