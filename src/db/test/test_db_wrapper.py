import unittest
from datetime import datetime
import uuid

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db_wrapper import (
    DBFactory,
    DBConfig,
    BodyInfo,
    SummarizeJobInfo,
    SummarizeResult,
    InferenceStatus,
)
from custom_log import LoggerFactory


class TestDBWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = DBConfig(
            host=os.environ.get("DB_HOST"),
            port=os.environ.get("DB_PORT"),
            username=os.environ.get("DB_USERNAME"),
            password=os.environ.get("DB_PASSWORD"),
            db_name="test_summarizer_db",
            # db_name=os.environ.get("DB_NAME")
            db_type="mysql",
        )
        log_instance = LoggerFactory.get_logger(
            logger_type="print", logger_name="test_db_wrapper"
        )
        cls.db_instance = DBFactory.get_db_instance(
            db_config=cls.config, log_instance=log_instance
        )
        cls.db_instance.create_all_tables_if_needed()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.db_instance.drop_all_tables()

    def test_insert_body_infos(self):
        # BodyInfoの追加処理
        expected_body_texts = ["てすとです", "てすとです2"]
        expected_body_infos = [
            BodyInfo(body=expected_body_texts[0], created_at=datetime.now()),
            BodyInfo(body=expected_body_texts[1], created_at=datetime.now()),
        ]
        ids = self.db_instance.insert_body_infos(
            body_infos=expected_body_infos
        )

        # BodyInfoの取得を行い、データが正しいかを検証
        actual_body_infos = self.db_instance.fetch_body_infos()
        for expected_body_info, expected_body_text in zip(
            expected_body_infos, expected_body_texts
        ):
            actual_body_info = [
                body_info
                for body_info in actual_body_infos
                if body_info.id == expected_body_info.id
            ][0]
            self.assertEqual(expected_body_info, actual_body_info)
            self.assertEqual(expected_body_text, actual_body_info.get_body())

    def test_insert_summarizer_results(self):
        # BodyInfoの追加処理
        body_infos = [
            BodyInfo(body="てすとです", created_at=datetime.now()),
            BodyInfo(body="てすとです2", created_at=datetime.now()),
        ]
        ids = self.db_instance.insert_body_infos(body_infos=body_infos)

        # SummarizerResultの追加処理
        expected_predicted_texts = ["予測です", "予測です2"]
        expected_summarizer_results = [
            SummarizeResult(
                body_id=ids[0],
                inference_status=InferenceStatus.complete.value,
                predicted_text=expected_predicted_texts[0],
                label_text=expected_predicted_texts[0],
            ),
            SummarizeResult(
                body_id=ids[1],
                inference_status=InferenceStatus.complete.value,
                predicted_text=expected_predicted_texts[1],
                label_text=expected_predicted_texts[1],
            ),
        ]
        result_ids = self.db_instance.insert_summarize_results(
            result_infos=expected_summarizer_results
        )

        # SummarizerResultの取得を行い、データが正しいかを検証
        actual_results = self.db_instance.fetch_summarize_results()
        for expected_result, expected_predicted_text in zip(
            expected_summarizer_results, expected_predicted_texts
        ):
            actual_result = [
                result
                for result in actual_results
                if result.id == expected_result.id
            ][0]
            self.assertEqual(expected_result, actual_result)
            self.assertEqual(
                expected_predicted_text, actual_result.get_predicted_text()
            )
            self.assertEqual(
                expected_predicted_text, actual_result.get_label_text()
            )

    def test_fetch_summarize_result_by_id(self):
        # BodyInfoの追加処理
        body_infos = [
            BodyInfo(body="てすとです", created_at=datetime.now()),
            BodyInfo(body="てすとです2", created_at=datetime.now()),
        ]
        ids = self.db_instance.insert_body_infos(body_infos=body_infos)

        # SummarizerResultの追加処理
        expected_predicted_texts = ["予測です", "予測です2"]
        expected_summarizer_results = [
            SummarizeResult(
                body_id=ids[0],
                inference_status=InferenceStatus.complete.value,
                predicted_text=expected_predicted_texts[0],
                label_text=expected_predicted_texts[0],
            ),
            SummarizeResult(
                body_id=ids[1],
                inference_status=InferenceStatus.complete.value,
                predicted_text=expected_predicted_texts[1],
                label_text=expected_predicted_texts[1],
            ),
        ]
        result_ids = self.db_instance.insert_summarize_results(
            result_infos=expected_summarizer_results
        )

        # SummarizeResultの取得処理
        actual_result = self.db_instance.fetch_summarize_result_by_id(
            result_id=result_ids[0]
        )
        self.assertEqual(actual_result, expected_summarizer_results[0])

    def test_fetch_summarize_job_info(self):
        # SummarizeJobInfoの追加処理
        expected_job_info = SummarizeJobInfo(
            job_id=uuid.uuid4(), result_id=uuid.uuid4()
        )
        job_infos = [
            expected_job_info,
            SummarizeJobInfo(job_id=uuid.uuid4(), result_id=uuid.uuid4()),
        ]
        ids = self.db_instance.insert_summarize_job_info(job_infos=job_infos)

        # 指定したjob_idのレコードを取れるかを検証
        actual_job_info = self.db_instance.fetch_summarize_job_info(
            job_id=expected_job_info.job_id
        )
        self.assertEqual(actual_job_info, expected_job_info)


if __name__ == "__main__":
    unittest.main()
