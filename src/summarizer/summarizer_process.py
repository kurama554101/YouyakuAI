

import sys
import os
from datetime import datetime
from enum import Enum
import time

from summarizer import SummarizerError, T5Summarizer
from summarizer_util import create_db_config, create_queue_config, create_logger, create_db_instance, create_queue_instance
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "log"))
from log import AbstractLogger, LoggerFactory
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "queue"))
from queue_client import AbstractQueueConsumer, QueueConfig, QueueError
from queue_factory import QueueConsumerCreator
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "db"))
from db_wrapper import BodyInfo, DBFactory, DBConfig, AbstractDB, InferenceStatus, SummarizeJobInfo, SummarizeResult


class SummarizerProcessResult(Enum):
    complete = (0, "summarizer process is completed!")
    queue_is_empty = (1, "summarizer process did not execute becuase queue is empty.")
    error_of_queue = (100, "summarizer process is failed because unknown queue error is occured.")
    error_of_summarizer = (200, "summarizer process is failed because some summarizer error are occured.")

    def get_id(self) -> int:
        return self.value[0]

    def get_text(self) -> str:
        return self.value[1]


def main():
    # Queueクライアントの取得
    queue_config = create_queue_config()
    logger = create_logger(logger_name="youyaku_ai_summarizer")
    queue_consumer = QueueConsumerCreator.create_consumer(consumer_type="kafka",
                                                          config=queue_config,
                                                          logger=logger)

    # 要約処理を行うインスタンスの取得
    hparams = {
        "max_input_length": 512,
        "max_target_length": 64,
        "model_dir": os.path.join(os.path.dirname(__file__), "model")
    }
    summarizer = T5Summarizer(hparams=hparams)

    # DBインスタンスの取得
    db_config = create_db_config()
    db_instance = create_db_instance(config=db_config, logger=logger)

    # 処理を永続的に実施
    while True:
        start_time = datetime.now()
        logger.info("summarize process is started. time is {}".format(start_time))
        result = loop_process(summarizer=summarizer,
                              queue_consumer=queue_consumer,
                              db_instance=db_instance,
                              logger=logger)
        end_time = datetime.now()
        process_time = end_time - start_time
        logger.info("summarize process is end. process time is {}".format(process_time))
        logger.info("summarize process result is {}".format(result.get_text()))

        # interval
        time.sleep(0.5)

def loop_process(summarizer:T5Summarizer,
                 queue_consumer:AbstractQueueConsumer,
                 db_instance:AbstractDB,
                 logger:AbstractLogger):

    # Queueからmessageを取得（なければ処理終了）
    try:
        messages = queue_consumer.consume()
    except QueueError as e:
        logger.error(e)
        return SummarizerProcessResult.error_of_queue

    # メッセージデータがあれば、推論処理を実施
    if len(messages) == 0:
        logger.info("summarize request is not found. summarize process is not called.")
        return SummarizerProcessResult.queue_is_empty
    ## 本文情報をDBに追加
    body_infos_with_id = {}
    for message in messages:
        # 本文情報の取得
        id   = message["id"]
        body = message["body"]
        logger.info("input text is {}".format(message))

        # DBに情報を登録
        body_info = BodyInfo(body=body, created_at=datetime.now()) # TODO : APIリクエストが実施された時間を入れた方が良いかも（jsonデータに含める）
        body_infos_with_id[id] = body_info
    db_instance.insert_body_infos(body_infos=list(body_infos_with_id.values()))

    ## 推論処理の実施とDBに結果登録
    ### 推論処理の実施
    try:
        # TODO : forループで回すのは遅いので、処理の見直しが必要
        results = []
        for body_info in body_infos_with_id.values():
            predicted_texts = summarizer.predict(inputs=[body_info.get_body()])
            results.append(predicted_texts[0])  # T5Summarizerでは、要約候補の文章が10個出てくるが、もっともスコアが高いもののみを採用
    except SummarizerError as e:
        logger.error(e)
        return SummarizerProcessResult.error_of_summarizer
    if len(body_infos_with_id) != len(results):
        logger.error("summarize result count(= {}) is not equal to request body count(= {}). summarize result is not saved into DB.".format(len(body_infos_with_id),
                                                                                                                                    len(results)))
        return SummarizerProcessResult.error_of_summarizer

    ### 推論処理の結果をDBに保存
    summarize_results = []
    for (_, body_info), predicted_text in zip(body_infos_with_id.items(), results):
        summarize_result = SummarizeResult(body_id=body_info.id, inference_status=InferenceStatus.complete.value, predicted_text=predicted_text, label_text=None)
        summarize_results.append(summarize_result)
    db_instance.insert_summarize_results(result_infos=summarize_results)

    ### DBの推論ジョブのステータスを更新
    job_infos = []
    for message_id, result in zip(body_infos_with_id.keys(), summarize_results):
        job_info = SummarizeJobInfo(job_id=message_id, result_id=result.id)
        job_infos.append(job_info)
    db_instance.insert_summarize_job_info(job_infos=job_infos)

    return SummarizerProcessResult.complete


if __name__ == "__main__":
    main()
