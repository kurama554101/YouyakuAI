import unittest
from typing import List
import uuid
from datetime import datetime

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from summarizer_model import T5Summarizer
from summarizer_process import SummarizerProcessResult, loop_process
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "db"))
from db_wrapper import AbstractDB, BodyInfo, DBConfig, SummarizeJobInfo, SummarizeResult
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "queue"))
from queue_client import AbstractQueueConsumer, QueueConfig
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "log"))
from custom_log import AbstractLogger, LoggerFactory



class TestSuammarizer(unittest.TestCase):
    def setUp(self) -> None:
        self.queue_config = QueueConfig(host="dummy",
                                        port=0,
                                        optional_param={"timeout": 5000, "topic_name": "test"})
        self.db_config = DBConfig(host="test_db",
                                  port="0",
                                  username="test",
                                  password="test",
                                  db_name="test")
        self.logger = LoggerFactory.get_logger(logger_type="print", logger_name="test_summarizer")
        self.db_instance = DBForTest(config=self.db_config,
                                     log_instance=self.logger,
                                     dummy_body_infos=[],
                                     dummy_job_infos=[],
                                     dummy_result_infos=[])
        self.queue = QueueConsumerForTest(config=self.queue_config, logger=self.logger)

    def tearDown(self) -> None:
        return super().tearDown()

    def test_summarize_loop_process(self):
        # テスト用のデータを作成
        job_id = uuid.uuid4()
        body = "これはテストの本文データです。試しに要約してみてね。"
        message = {"id": str(job_id), "body": body}
        body_info = BodyInfo(body=body, created_at=datetime.now())
        body_info.id = 1
        job_info = SummarizeJobInfo(job_id=job_id, result_id=None)
        self.db_instance.insert_body_infos(body_infos=[body_info])
        self.db_instance.insert_summarize_job_info(job_infos=[job_info])
        self.queue._add_data(messages=[message])

        # Summarizerインスタンスの作成
        hparams = {
            "max_input_length": 512,
            "max_target_length": 64,
            "model_dir": os.path.join(os.path.dirname(__file__), "..", "model")
        }
        summarizer_instance = T5Summarizer(hparams=hparams)

        # ループ処理の実施
        process_result = loop_process(summarizer=summarizer_instance,
                                      queue_consumer=self.queue,
                                      db_instance=self.db_instance,
                                      logger=self.logger)

        # 検証
        self.assertEqual(process_result, SummarizerProcessResult.complete)

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
        updated_indexes = []
        for result_info in result_infos:
            target_result_info_indexes = [key for key, info in enumerate(self.__result_infos) if result_info.id == info.id]
            if len(target_result_info_indexes) == 0:
                continue
            target_index = target_result_info_indexes[0]
            self.__result_infos[target_index] = result_info
            updated_indexes.append(target_index)
        return updated_indexes

    def update_summarize_job_info(self, job_infos:List[SummarizeJobInfo]) -> List[int]:
        updated_indexes = []
        for job_info in job_infos:
            target_job_info_indexes = [key for key, info in enumerate(self.__job_infos) if job_info.job_id == info.job_id]
            if len(target_job_info_indexes) == 0:
                continue
            target_index = target_job_info_indexes[0]
            self.__job_infos[target_index] = job_info
            updated_indexes.append(target_index)
        return updated_indexes

class QueueConsumerForTest(AbstractQueueConsumer):
    def __init__(self,
                 config: QueueConfig,
                 logger:AbstractLogger) -> None:
        super().__init__(config=config, logger=logger)

        # テスト用のQueueの作成
        self.__queue = []

    # for test
    def _add_data(self, messages:list):
        self.__queue.extend(messages)

    def consume(self) -> list:
        return self.__queue


if __name__ == "__main__":
    unittest.main()
