import unittest
from fastapi.testclient import TestClient
from typing import List
import uuid
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api import SummarizerApi, ResponseInferenceStatus, ResponseSetCorrectedResult
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "db"))
from db_wrapper import AbstractDB, BodyInfo, DBConfig, InferenceStatus, SummarizeJobInfo, SummarizeResult
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "queue"))
from queue_client import AbstractQueueProducer, QueueConfig
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "log"))
from log import AbstractLogger, LoggerFactory


class TestAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.queue_config = QueueConfig(host="dummy",
                                        port=0,
                                        optional_param={"timeout": 5000, "topic_name": "test"})
        self.db_config = DBConfig(host="test_db",
                                  port="0",
                                  username="test",
                                  password="test",
                                  db_name="test")
        self.logger = LoggerFactory.get_logger(logger_type="print", logger_name="test_api")

    def test_request_summarize(self):
        # セットアップ
        queue_producer = QueueProducerForTest(config=self.queue_config, logger=self.logger)
        db_instance = DBForTest(config=self.db_config, log_instance=self.logger)
        client = self.__create_client(queue_producer=queue_producer,
                                      db_instance=db_instance,
                                      logger=self.logger)

        # 検証
        body = "これはテストです。"
        response = client.post(
            "/request_summarize/",
            json={"body": body}
        )
        self.assertEqual(response.status_code, 200)
        message = response.json()
        self.assertTrue("message_id" in message)
        try:
            id = uuid.UUID(message["message_id"])
        except Exception as e:
            self.fail("id type don't equal UUID type. error detail is {}".format(e))

        ## DBにジョブが登録されていること
        job_info = db_instance.fetch_summarize_job_info(job_id=id)
        self.assertEqual(job_info.job_id, id)

        ## Queueにメッセージが送られているかを検証
        queue = queue_producer._get_queue()
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["id"], str(id))
        self.assertEqual(queue[0]["body"], body)

    # TODO : complete以外のステータスパターンのテスト
    def test_get_summarize_result_complete(self):
        # テスト用のDBクラスの作成
        job_id = uuid.uuid4()
        result_info = SummarizeResult(body_id=1, inference_status=InferenceStatus.complete.value, predicted_text="てすとです", label_text=None)
        result_info.id = 1
        job_info = SummarizeJobInfo(job_id=job_id, result_id=result_info.id)
        db_instance = DBForTest(config=self.db_config,
                                log_instance=self.logger,
                                dummy_result_infos=[result_info],
                                dummy_job_infos=[job_info])

        # テスト用のQueueクラスの作成
        queue_producer = QueueProducerForTest(config=self.queue_config, logger=self.logger)

        # テスト用のAPIクライアントの作成
        client = self.__create_client(queue_producer=queue_producer,
                                      db_instance=db_instance,
                                      logger=self.logger)

        # 検証
        response = client.get(
            "/summarize_result/?job_id={}".format(job_id)
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status_id"], ResponseInferenceStatus.complete_job.get_id())
        self.assertEqual(result["status_detail"], ResponseInferenceStatus.complete_job.get_detail())
        self.assertEqual(result["result_detail"]["predicted_text"], result_info.predicted_text)

    # TODO : complete以外のステータスパターンのテスト
    def test_set_correct_summarize_result_complete(self):
        # テスト用のDBクラスの作成
        job_id = uuid.uuid4()
        result_info = SummarizeResult(body_id=1, inference_status=InferenceStatus.complete.value, predicted_text="てすとです", label_text=None)
        result_info.id = 1
        job_info = SummarizeJobInfo(job_id=job_id, result_id=result_info.id)
        db_instance = DBForTest(config=self.db_config,
                                log_instance=self.logger,
                                dummy_result_infos=[result_info],
                                dummy_job_infos=[job_info])

        # テスト用のQueueクラスの作成
        queue_producer = QueueProducerForTest(config=self.queue_config, logger=self.logger)

        # テスト用のAPIクライアントの作成
        client = self.__create_client(queue_producer=queue_producer,
                                      db_instance=db_instance,
                                      logger=self.logger)

        # 検証
        corrected_text = "これは正しい要約です。"
        response = client.post(
            "/set_correct_summarize_result/",
            json={"job_id": str(job_id), "corrected_text": corrected_text}
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status_id"], ResponseSetCorrectedResult.complete_job.get_id())
        self.assertEqual(result["status_detail"], ResponseSetCorrectedResult.complete_job.get_detail())

        # DBにジョブ情報が正しく保存されているかを検証
        actual_result_info = db_instance.fetch_summarize_result_by_id(result_id=result_info.id)
        self.assertEqual(actual_result_info.label_text, corrected_text)

    def __create_client(self,
                        queue_producer:AbstractQueueProducer,
                        db_instance:AbstractDB,
                        logger:AbstractLogger):
        app = SummarizerApi(queue_producer=queue_producer,
                            db_instance=db_instance,
                            logger=logger).create_app()
        client = TestClient(app)
        return client


class DBForTest(AbstractDB):
    def __init__(self,
                 config: DBConfig,
                 log_instance: AbstractLogger,
                 dummy_result_infos: List[SummarizeResult] = [],
                 dummy_job_infos: List[SummarizeJobInfo] = [],
                 dummy_body_infos: List[BodyInfo] = []):
        super().__init__(config, log_instance)

        # ダミーのDB作成
        self.__result_infos = dummy_result_infos
        self.__job_infos = dummy_job_infos
        self.__body_infos = dummy_body_infos

    def create_all_tables_if_needed(self):
        pass

    def drop_all_tables(self):
        pass

    def create_database_if_needed(self):
        pass

    def drop_database(self):
        pass

    def insert_body_infos(self, body_infos:List[BodyInfo]) -> List[int]:
        self.__body_infos.extend(body_infos)

    def insert_summarize_results(self, result_infos:List[SummarizeResult]) -> List[int]:
        self.__result_infos.extend(result_infos)

    def insert_summarize_job_info(self, job_infos:List[SummarizeJobInfo]) -> List[int]:
        self.__job_infos.extend(job_infos)

    def fetch_body_infos(self) -> List[BodyInfo]:
        return self.__body_infos

    def fetch_summarize_results(self) -> List[SummarizeResult]:
        return self.__result_infos

    def fetch_summarize_result_by_id(self, result_id:int) -> SummarizeResult:
        return [result_info for result_info in self.__result_infos if result_info.id == result_id][0]

    def fetch_summarize_job_info(self, job_id:uuid.UUID) -> SummarizeJobInfo:
        return [job_info for job_info in self.__job_infos if job_info.job_id == job_id][0]

    def update_label_text_of_result_by_id(self, result_id:int, label_text:str) -> int:
        for index, result_info in enumerate(self.__result_infos):
            if result_info.id == result_id:
                result_info.label_text = label_text
                self.__result_infos[index] = result_info
                return result_id

    def update_summarize_results(self, result_infos: List[SummarizeResult]) -> List[int]:
        pass

    def update_summarize_job_info(self, job_infos:List[SummarizeJobInfo]) -> List[int]:
        pass


class QueueProducerForTest(AbstractQueueProducer):
    def __init__(self, config: QueueConfig, logger:AbstractLogger) -> None:
        super().__init__(config=config, logger=logger)

        # テスト用のQueueの作成
        self.__queue = []

    def produce(self, messages:list):
        self.__queue.extend(messages)

    # for test
    def _get_queue(self) -> list:
        return self.__queue


if __name__ == "__main__":
    unittest.main()
