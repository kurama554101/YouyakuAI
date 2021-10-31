import unittest

from datetime import datetime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db_wrapper import BodyInfo, DBConfig, DBFactory, DBUtil, SummarizeJobInfo, SummarizeJobLog, SummarizeResult
from custom_log import LoggerFactory
import bigquery_util
import uuid


class TestBigQueryDBWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DBConfig(host="",
                              port="",
                              username="",
                              password="",
                              db_name="test_youyaku_ai_db",
                              db_type="bigquery")
        cls.log_instance = LoggerFactory.get_logger(logger_type="print", logger_name="test_bq_db_wrapper")
        cls.db_instance = DBFactory.get_db_instance(db_config=cls.config, log_instance=cls.log_instance)
        cls.db_instance.create_database_if_needed()
        cls.db_instance.create_all_tables_if_needed()

    @classmethod
    def tearDownClass(cls) -> None:
        # データセットやテーブル作成完了タイミングはまちまちなので、テーブルの中身を削除するように修正する
        #cls.db_instance.drop_all_tables()
        #cls.db_instance.drop_database()

        # 削除しても、table is truncated、となってしまい、insertに失敗する
        #cls.db_instance.delete_all_body_infos()
        pass

    def test_insert_body_infos(self):
        body_infos = [
            BodyInfo(body="テストです1", created_at=datetime.now()),
            BodyInfo(body="テストです2", created_at=datetime.now())
        ]

        # insert処理の実行
        ids = self.db_instance.insert_body_infos(body_infos=body_infos)
        self.assertEqual(len(body_infos), len(ids))

        # 追加したデータが一致しているかを確認
        for i, id in enumerate(ids):
            actual_info = self.db_instance.fetch_body_info_by_id(body_info_id=id)
            self.assertIsNotNone(actual_info)
            self.assertEqual(body_infos[i].body, actual_info.body)
            self.assertEqual(body_infos[i].created_at, actual_info.created_at)

    def test_insert_summarize_results(self):
        result_infos = [
            SummarizeResult(body_id=uuid.uuid4(), inference_status=0, predicted_text="これはテストです1", label_text="間違いです1"),
            SummarizeResult(body_id=uuid.uuid4(), inference_status=0, predicted_text="これはテストです2", label_text="間違いです2")
        ]

        # insert処理の実行
        ids = self.db_instance.insert_summarize_results(result_infos=result_infos)
        self.assertEqual(len(result_infos), len(ids))

        # 追加したデータが一致しているかを確認
        for i, id in enumerate(ids):
            actual_info = self.db_instance.fetch_summarize_result_by_id(result_id=id)
            self.assertIsNotNone(actual_info)
            self.assertEqual(result_infos[i].body_id, actual_info.body_id)
            self.assertEqual(result_infos[i].inference_status, actual_info.inference_status)
            self.assertEqual(result_infos[i].label_text, actual_info.label_text)
            self.assertEqual(result_infos[i].predicted_text, actual_info.predicted_text)

    def test_insert_summarize_job_infos(self):
        job_infos = [
            SummarizeJobInfo(job_id=uuid.uuid4(), result_id=uuid.uuid4()),
            SummarizeJobInfo(job_id=uuid.uuid4(), result_id=uuid.uuid4())
        ]

        # insert処理の実行
        ids = self.db_instance.insert_summarize_job_info(job_infos=job_infos)
        self.assertEqual(len(job_infos), len(ids))

        # 追加したデータが一致しているかを確認
        for i, id in enumerate(ids):
            actual_info = self.db_instance.fetch_summarize_job_info(job_id=id)
            self.assertIsNotNone(actual_info)
            self.assertEqual(job_infos[i].job_id, actual_info.job_id)
            self.assertEqual(job_infos[i].result_id, actual_info.result_id)

    def test_insert_summarize_job_infos_with_result_id_is_none(self):
        job_infos = [
            SummarizeJobInfo(job_id=uuid.uuid4(), result_id=None)
        ]

        # insert処理の実行
        ids = self.db_instance.insert_summarize_job_info(job_infos=job_infos)
        self.assertEqual(len(job_infos), len(ids))

        # 追加したデータが一致しているかを確認
        for i, id in enumerate(ids):
            actual_info = self.db_instance.fetch_summarize_job_info(job_id=id)
            self.assertIsNotNone(actual_info)
            self.assertEqual(job_infos[i].job_id, actual_info.job_id)
            self.assertIsNone(actual_info.result_id)

    def test_insert_summarize_job_logs(self):
        job_logs = [
            SummarizeJobLog(job_id=uuid.uuid4())
        ]

        # insert処理の実行
        ids = self.db_instance.insert_summarize_job_log(job_logs=job_logs)
        self.assertEqual(len(job_logs), len(ids))

        # 追加したデータが取得可能かを確認
        for id in ids:
            actual_log = self.db_instance.fetch_summarize_job_log_by_id(job_id=id)
            self.assertIsNotNone(actual_log)
            self.assertEqual(id, actual_log.job_id)

    @unittest.skip("update method is skip because of bigquery limitation. detail is 'https://cloud.google.com/bigquery/docs/reference/standard-sql/data-manipulation-language#limitations'")
    def test_update_job_infos(self):
        # 追加するjob infoを定義する
        job_infos = [
            SummarizeJobInfo(job_id=str(uuid.uuid4()), result_id=None),
            SummarizeJobInfo(job_id=str(uuid.uuid4()), result_id=None),
            SummarizeJobInfo(job_id=str(uuid.uuid4()), result_id=None)
        ]

        # insert処理の実行
        ids_insert = self.db_instance.insert_summarize_job_info(job_infos=job_infos)

        # insertしたjob_infoに対して, update処理をするためのデータを定義
        for job_info in job_infos:
            job_info.result_id = str(uuid.uuid4())

        # update処理の実行
        ids_update = self.db_instance.update_summarize_job_info(job_infos=job_infos)

        # 値の検証処理
        self.assertEqual(len(ids_insert), len(ids_update))
        for i, id in enumerate(ids_update):
            self.assertEqual(ids_insert[i], id)
            actual_job_info = self.db_instance.fetch_summarize_job_info(job_id=id)
            self.assertIsNotNone(actual_job_info)
            self.assertIsNotNone(actual_job_info.result_id)
            self.assertEqual(job_infos[i].result_id, actual_job_info.result_id)

    @classmethod
    def __get_table_list(cls):
        client = bigquery_util.get_bq_client(project_id="youyaku-ai")
        l = bigquery_util.get_table_list(client=client , dataset_name="test_youyaku_ai_db")
        table_name_list = []
        for item in l:
            table_name_list.append(item.table_id)
        return table_name_list


if __name__ == "__main__":
    unittest.main()
