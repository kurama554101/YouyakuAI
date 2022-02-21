from abc import ABCMeta, abstractmethod
from typing import List
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, DateTime
from sqlalchemy import desc
import sqlalchemy
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import registry, sessionmaker
from sqlalchemy_utils import (
    create_database,
    drop_database,
    database_exists,
    UUIDType,
)
import uuid
from enum import Enum

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../log"))
from custom_log import AbstractLogger

import bigquery_util
from google.cloud import bigquery as bq
from more_itertools import chunked


class DBUtil:
    @staticmethod
    def convert_byte_to_str_if_needed(base_text) -> str:
        if base_text is None:
            return base_text
        if type(base_text) == str:
            return base_text
        return base_text.decode("utf-8")

    @staticmethod
    def convert_bq_table_schemas(metadata: sqlalchemy.schema.MetaData) -> dict:
        tables = {}
        for table in metadata.sorted_tables:
            table_name = table.name
            schema_list = DBUtil.convert_bq_table_schema(table=table)
            tables[table_name] = schema_list
        return tables

    @staticmethod
    def convert_bq_table_schema(
        table: sqlalchemy.schema.Table,
    ) -> List[bq.SchemaField]:
        schema_list = []
        for column in table.c:
            schema = DBUtil.convert_schema_field_from_alchemy_column(
                column=column
            )
            schema_list.append(schema)
        return schema_list

    @staticmethod
    def convert_data_type_of_bq(type_of_alchemy) -> str:
        if isinstance(type_of_alchemy, Integer):
            return "INT64"
        elif isinstance(type_of_alchemy, LONGTEXT):
            return "STRING"
        elif isinstance(type_of_alchemy, DateTime):
            return "DATETIME"
        elif type_of_alchemy == UUIDType:
            return "STRING"
        elif isinstance(type_of_alchemy, UUIDType):
            return "STRING"
        else:
            raise NotImplementedError(
                "{} type is not able to convert to bigquery column type!".format(
                    type_of_alchemy
                )
            )

    @staticmethod
    def convert_schema_field_from_alchemy_column(column) -> bq.SchemaField:
        name = column.name
        # TODO : id関連はUUIDで統一対応としたい（Integerを削除する）
        if name == "id" or name == "body_id" or name == "result_id":
            field_type = DBUtil.convert_data_type_of_bq(
                type_of_alchemy=UUIDType
            )
        else:
            type_of_alchemy = column.type
            field_type = DBUtil.convert_data_type_of_bq(
                type_of_alchemy=type_of_alchemy
            )
        mode = "NULLABLE" if column.nullable else "REQUIRED"
        schema = bq.SchemaField(name, field_type, mode)
        return schema

    @staticmethod
    def generate_uuid() -> str:
        id = str(uuid.uuid4())
        return id

    @staticmethod
    def get_converted_name_for_bq(name: str) -> str:
        # BigQueryでの禁止文字を削除する
        return name.replace("_", "-")


# モデルで利用する
mapper_registry = registry()


@mapper_registry.mapped
@dataclass
class BodyInfo:
    __table__ = Table(
        "body_info",
        mapper_registry.metadata,
        Column("id", UUIDType(binary=False), primary_key=True),
        Column("body", LONGTEXT, nullable=False),
        Column("created_at", DateTime, nullable=False),
    )
    id: UUIDType(binary=False) = field(init=False)
    body: str = field()
    created_at: datetime = field()

    def __post_init__(self):
        self.id = uuid.uuid4()

    def get_body(self) -> str:
        return DBUtil.convert_byte_to_str_if_needed(base_text=self.body)

    def convert_dict(self) -> dict:
        return {
            "id": str(self.id),
            "body": self.get_body(),
            "created_at": self.created_at,
        }


@mapper_registry.mapped
@dataclass
class SummarizeResult:
    __table__ = Table(
        "summarize_result",
        mapper_registry.metadata,
        Column("id", UUIDType(binary=False), primary_key=True),
        Column("body_id", UUIDType(binary=False), nullable=False),
        Column("inference_status", Integer, nullable=False),
        Column("predicted_text", LONGTEXT, nullable=True),
        Column("label_text", LONGTEXT, nullable=True),
    )
    id: UUIDType(binary=False) = field(init=False)
    body_id: UUIDType(binary=False) = field()
    inference_status: int = field()
    predicted_text: str = field()
    label_text: str = field()

    def __post_init__(self):
        self.id = uuid.uuid4()

    def get_predicted_text(self) -> str:
        return DBUtil.convert_byte_to_str_if_needed(
            base_text=self.predicted_text
        )

    def get_label_text(self) -> str:
        return DBUtil.convert_byte_to_str_if_needed(base_text=self.label_text)

    def convert_dict(self) -> dict:
        return {
            "id": str(self.id),
            "body_id": str(self.body_id),
            "inference_status": self.inference_status,
            "predicted_text": self.get_predicted_text(),
            "label_text": self.get_label_text(),
        }


@mapper_registry.mapped
@dataclass
class SummarizeJobInfo:
    __table__ = Table(
        "summarize_job_info",
        mapper_registry.metadata,
        Column("job_id", UUIDType(binary=False), primary_key=True),
        Column("result_id", UUIDType(binary=False), nullable=True),
    )
    job_id: UUIDType(binary=False) = field()
    result_id: int = field()

    def convert_dict(self) -> dict:
        return {
            "job_id": str(self.job_id),
            "result_id": str(self.result_id)
            if self.result_id is not None
            else None,
        }


# insertのみの処理とするため、job_idの登録用テーブルと、result_idを含んだテーブル(SummarizeJobInfo)は分離する
@mapper_registry.mapped
@dataclass
class SummarizeJobLog:
    __table__ = Table(
        "summarize_job_log",
        mapper_registry.metadata,
        Column("job_id", UUIDType(binary=False), primary_key=True),
    )
    job_id: UUIDType(binary=False) = field()

    def convert_dict(self) -> dict:
        return {"job_id": str(self.job_id)}


class InferenceStatus(Enum):
    complete = 1000
    complete_but_difference = 1001
    failed_unknown = 2000


class DBConfig:
    def __init__(
        self,
        host: str,
        port: str,
        username: str,
        password: str,
        db_name: str = "summarize_db",
        db_type: str = "mysql",
        db_date_format: str = "%Y-%m-%d %H:%M:%S",
        extra_params: dict = {},
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.db_name = db_name
        self.db_type = db_type
        self.db_date_format = db_date_format
        self.extra_params = extra_params

    def print(self):
        print(
            "{}, {}, {}, {}, {}, {}, {}, {}".format(
                self.host,
                self.port,
                self.username,
                self.password,
                self.db_name,
                self.db_type,
                self.db_date_format,
                self.extra_params,
            )
        )


class AbstractDB:
    __metadata__ = ABCMeta

    def __init__(self, config: DBConfig, log_instance: AbstractLogger):
        self._config = config
        self._log_instance = log_instance

    @abstractmethod
    def create_all_tables_if_needed(self):
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def drop_all_tables(self):
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def create_database_if_needed(self):
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def drop_database(self):
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def insert_body_infos(self, body_infos: List[BodyInfo]) -> List[int]:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def insert_summarize_results(
        self, result_infos: List[SummarizeResult]
    ) -> List[int]:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def insert_summarize_job_info(
        self, job_infos: List[SummarizeJobInfo]
    ) -> List[int]:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def insert_summarize_job_log(
        self, job_logs: List[SummarizeJobLog]
    ) -> List[int]:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def fetch_body_infos(self) -> List[BodyInfo]:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def fetch_body_info_by_id(self, body_info_id: uuid.UUID) -> BodyInfo:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def fetch_summarize_results(self) -> List[SummarizeResult]:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def fetch_summarize_result_by_id(
        self, result_id: uuid.UUID
    ) -> SummarizeResult:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def fetch_summarize_job_info(self, job_id: uuid.UUID) -> SummarizeJobInfo:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def fetch_summarize_job_log_by_id(
        self, job_id: uuid.UUID
    ) -> SummarizeJobLog:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def update_label_text_of_result_by_id(
        self, result_id: uuid.UUID, label_text: str
    ) -> int:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def update_summarize_results(
        self, result_infos: List[SummarizeResult]
    ) -> List[int]:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def update_summarize_job_info(
        self, job_infos: List[SummarizeJobInfo]
    ) -> List[uuid.UUID]:
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def delete_all_body_infos(self):
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def delete_all_summarize_results(self):
        raise NotImplementedError("this api is not implemented!")

    @abstractmethod
    def delete_all_job_infos(self):
        raise NotImplementedError("this api is not implemented!")


class DBFactory:
    @classmethod
    def get_db_instance(
        cls, db_config: DBConfig, log_instance: AbstractLogger
    ) -> AbstractDB:
        db_type = db_config.db_type
        if db_type == "mysql":
            return MySQLDB(config=db_config, log_instance=log_instance)
        elif db_type == "bigquery":
            return BigQueryDB(config=db_config, log_instance=log_instance)
        else:
            raise NotImplementedError(
                "{} type is not implemented DB type!!".format(db_type)
            )


class MySQLDB(AbstractDB):
    def __init__(self, config: DBConfig, log_instance: AbstractLogger):
        super().__init__(config=config, log_instance=log_instance)
        self.__metadata = mapper_registry.metadata
        self.__db_engine = create_engine(
            "mysql+mysqlconnector://{}:{}@{}:{}/{}?charset={}".format(
                self._config.username,
                self._config.password,
                self._config.host,
                self._config.port,
                self._config.db_name,
                "utf8",
            )
        )
        self.__session = sessionmaker(self.__db_engine)

    def create_all_tables_if_needed(self):
        self.__metadata.create_all(self.__db_engine)

    def drop_all_tables(self):
        self.__metadata.drop_all(self.__db_engine)

    def create_database_if_needed(self):
        if not database_exists(self.__db_engine.url):
            create_database(self.__db_engine.url)

    def drop_database(self):
        if database_exists(self.__db_engine.url):
            drop_database(self.__db_engine.url)

    def insert_body_infos(self, body_infos: List[BodyInfo]) -> List[int]:
        return self.__basic_insert_items(items=body_infos)

    def insert_summarize_results(
        self, result_infos: List[SummarizeResult]
    ) -> List[int]:
        return self.__basic_insert_items(items=result_infos)

    def insert_summarize_job_info(
        self, job_infos: List[SummarizeJobInfo]
    ) -> List[int]:
        # アイテムの追加処理
        with self.__session() as session:
            session.add_all(job_infos)
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                raise DBError(
                    "MySQLDB insert function is Error! error detail is {}".format(
                        e
                    )
                )

            # 追加したアイテムのIDリストを取得
            return [job_info.job_id for job_info in job_infos]

    def insert_summarize_job_log(
        self, job_logs: List[SummarizeJobLog]
    ) -> List[int]:
        # アイテムの追加処理
        with self.__session() as session:
            session.add_all(job_logs)
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                raise DBError(
                    "MySQLDB insert function is Error! error detail is {}".format(
                        e
                    )
                )

            # 追加したアイテムのIDリストを取得
            return [job_log.job_id for job_log in job_logs]

    def fetch_body_infos(self) -> List[BodyInfo]:
        return self.__basic_fetch_all(clazz=BodyInfo)

    def fetch_summarize_results(self) -> List[SummarizeResult]:
        return self.__basic_fetch_all(clazz=SummarizeResult)

    def fetch_summarize_result_by_id(
        self, result_id: uuid.UUID
    ) -> SummarizeResult:
        with self.__session() as session:
            stmt = session.query(SummarizeResult).filter(
                SummarizeResult.id == result_id
            )
            result = session.execute(stmt).fetchone()
            return result[0] if result is not None else None

    def fetch_summarize_job_info(self, job_id: uuid.UUID) -> SummarizeJobInfo:
        with self.__session() as session:
            stmt = session.query(SummarizeJobInfo).filter(
                SummarizeJobInfo.job_id == job_id
            )
            result = session.execute(stmt).fetchone()
            return result[0] if result is not None else None

    def fetch_summarize_job_log_by_id(
        self, job_id: uuid.UUID
    ) -> SummarizeJobLog:
        with self.__session() as session:
            stmt = session.query(SummarizeJobLog).filter(
                SummarizeJobLog.job_id == job_id
            )
            result = session.execute(stmt).fetchone()
            return result[0] if result is not None else None

    def update_label_text_of_result_by_id(
        self, result_id: uuid.UUID, label_text: str
    ) -> int:
        pass

    def update_summarize_results(
        self, result_infos: List[SummarizeResult]
    ) -> List[int]:
        pass

    def update_summarize_job_info(
        self, job_infos: List[SummarizeJobInfo]
    ) -> List[uuid.UUID]:
        with self.__session() as session:
            try:
                # TODO : もっと早いやり方があるはず。SQLAlchemyのupdate方法を確認
                for job_info in job_infos:
                    target = (
                        session.query(SummarizeJobInfo)
                        .filter(SummarizeJobInfo.job_id == job_info.job_id)
                        .first()
                    )
                    target.result_id = job_info.result_id
                session.commit()
            except Exception as e:
                session.rollback()
                raise DBError(
                    "MySQLDB update function({}) is Error! error detail is {}".format(
                        "update_summarize_job_info", e
                    )
                )
        return [job_info.job_id for job_info in job_infos]

    def __basic_insert_items(self, items: list) -> list:
        # アイテムの追加処理
        with self.__session() as session:
            session.add_all(items)
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                raise DBError(
                    "MySQLDB insert function is Error! error detail is {}".format(
                        e
                    )
                )

            # 追加したアイテムのIDリストを取得
            return [item.id for item in items]

    def __basic_fetch_all(self, clazz) -> list:
        # アイテムの取得処理
        with self.__session() as session:
            stmt = session.query(clazz).order_by(desc(clazz.id))
            result = session.execute(stmt)
            return [row[0] for row in result]


class BigQueryDB(AbstractDB):
    def __init__(self, config: DBConfig, log_instance: AbstractLogger):
        super().__init__(config=config, log_instance=log_instance)
        project_id = config.extra_params.get("gcp_project_id", None)
        credentials = config.extra_params.get("credentials", None)
        self.__max_chunked_size = config.extra_params.get(
            "max_chunked_size", 10000
        )
        self.__bq_client = bigquery_util.get_bq_client(
            project_id=project_id, credentials=credentials
        )

    def create_all_tables_if_needed(self):
        tables = DBUtil.convert_bq_table_schemas(mapper_registry.metadata)
        dataset_name = self._config.db_name
        for table_name, schemas in tables.items():
            if bigquery_util.exist_table(
                self.__bq_client, dataset_name, table_name
            ):
                continue
            bigquery_util.create_table(
                self.__bq_client, dataset_name, table_name, schemas
            )

    def drop_all_tables(self):
        tables = DBUtil.convert_bq_table_schemas(mapper_registry.metadata)
        dataset_name = self._config.db_name
        for table_name in tables.keys():
            bigquery_util.delete_table_if_needed(
                self.__bq_client, dataset_name, table_name
            )

    def create_database_if_needed(self):
        dataset_id = self._config.db_name
        if bigquery_util.exist_dataset(self.__bq_client, dataset_id):
            return
        bigquery_util.create_dataset(
            client=self.__bq_client, dataset_id=dataset_id
        )

    def drop_database(self):
        dataset_id = self._config.db_name
        bigquery_util.delete_dataset_if_exists(
            client=self.__bq_client, dataset_id=dataset_id
        )

    def insert_body_infos(self, body_infos: List[BodyInfo]) -> List[int]:
        items = []
        ids = []
        for body_info in body_infos:
            item = body_info.convert_dict()
            items.append(item)
            ids.append(uuid.UUID(item["id"]))

        self.__insert_infos(items, BodyInfo)
        return ids

    def insert_summarize_results(
        self, result_infos: List[SummarizeResult]
    ) -> List[int]:
        items = []
        ids = []
        for info in result_infos:
            item = info.convert_dict()
            items.append(item)
            ids.append(uuid.UUID(item["id"]))

        self.__insert_infos(items, SummarizeResult)
        return ids

    def insert_summarize_job_info(
        self, job_infos: List[SummarizeJobInfo]
    ) -> List[int]:
        items = []
        ids = []
        for info in job_infos:
            item = info.convert_dict()
            items.append(item)
            ids.append(info.job_id)

        self.__insert_infos(items, SummarizeJobInfo)
        return ids

    def insert_summarize_job_log(
        self, job_logs: List[SummarizeJobLog]
    ) -> List[int]:
        items = []
        ids = []
        for log in job_logs:
            item = log.convert_dict()
            items.append(item)
            ids.append(log.job_id)

        self.__insert_infos(items, SummarizeJobLog)
        return ids

    def __insert_infos(self, items: list, define_table):
        table_full_name = self.__get_table_full_name(
            table_name=define_table.__table__.name
        )
        bq_schema_list = DBUtil.convert_bq_table_schema(
            table=define_table.__table__
        )
        for chunked_items in chunked(items, self.__max_chunked_size):
            errors = self.__bq_client.insert_rows(
                table=table_full_name,
                rows=chunked_items,
                selected_fields=bq_schema_list,
            )
            if errors != []:
                raise DBError(
                    "insert error! error details is {}. insert items are {}".format(
                        errors, items
                    )
                )

    def fetch_body_infos(self) -> List[BodyInfo]:
        table_full_name = self.__get_table_full_name(
            table_name=BodyInfo.__table__.name
        )
        query = """
            SELECT * FROM `{}`
        """.format(
            table_full_name
        )
        query_job = self.__bq_client.query(query)
        infos = self.__convert_body_infos(query_job_of_bq=query_job)
        return infos

    def fetch_body_info_by_id(self, body_info_id: uuid.UUID) -> BodyInfo:
        table_full_name = self.__get_table_full_name(
            table_name=BodyInfo.__table__.name
        )
        query = """
            SELECT * FROM `{}`
            WHERE id = '{}'
        """.format(
            table_full_name, str(body_info_id)
        )
        query_job = self.__bq_client.query(query)
        infos = self.__convert_body_infos(query_job_of_bq=query_job)
        if len(infos) > 1:
            raise DBError(
                "{} is duplicated id in body_info!".format(body_info_id)
            )
        return infos[0] if len(infos) == 1 else None

    def __convert_body_infos(self, query_job_of_bq) -> List[BodyInfo]:
        infos = []
        for item in query_job_of_bq:
            info = BodyInfo(body=item["body"], created_at=item["created_at"])
            info.id = uuid.UUID(item["id"])
            infos.append(info)
        return infos

    def fetch_summarize_results(self) -> List[SummarizeResult]:
        table_full_name = self.__get_table_full_name(
            table_name=SummarizeResult.__table__.name
        )
        query = """
            SELECT * FROM `{}`
        """.format(
            table_full_name
        )
        query_job = self.__bq_client.query(query)
        infos = self.__convert_result_infos(query_job_of_bq=query_job)
        return infos

    def fetch_summarize_result_by_id(
        self, result_id: uuid.UUID
    ) -> SummarizeResult:
        table_full_name = self.__get_table_full_name(
            table_name=SummarizeResult.__table__.name
        )
        query = """
            SELECT * FROM `{}`
            WHERE id = '{}'
        """.format(
            table_full_name, str(result_id)
        )
        query_job = self.__bq_client.query(query)
        infos = self.__convert_result_infos(query_job_of_bq=query_job)
        if len(infos) > 1:
            raise DBError(
                "{} is duplicated id in summarize_result!".format(result_id)
            )
        return infos[0] if len(infos) == 1 else None

    def __convert_result_infos(self, query_job_of_bq) -> List[SummarizeResult]:
        infos = []
        for item in query_job_of_bq:
            info = SummarizeResult(
                body_id=uuid.UUID(item["body_id"]),
                inference_status=item["inference_status"],
                predicted_text=item["predicted_text"],
                label_text=item["label_text"],
            )
            info.id = uuid.UUID(item["id"])
            infos.append(info)
        return infos

    def fetch_summarize_job_info(self, job_id: uuid.UUID) -> SummarizeJobInfo:
        table_full_name = self.__get_table_full_name(
            table_name=SummarizeJobInfo.__table__.name
        )
        query = """
            SELECT * FROM `{}`
            WHERE job_id = '{}'
        """.format(
            table_full_name, str(job_id)
        )
        query_job = self.__bq_client.query(query)
        infos = self.__convert_job_infos(query_job_of_bq=query_job)
        if len(infos) > 1:
            raise DBError(
                "{} is duplicated id in summarize_job_info!".format(job_id)
            )
        return infos[0] if len(infos) == 1 else None

    def __convert_job_infos(self, query_job_of_bq) -> List[SummarizeJobInfo]:
        infos = []
        for item in query_job_of_bq:
            info = SummarizeJobInfo(
                job_id=uuid.UUID(item["job_id"]),
                result_id=uuid.UUID(item["result_id"])
                if item["result_id"] is not None
                else None,
            )
            infos.append(info)
        return infos

    def fetch_summarize_job_log_by_id(
        self, job_id: uuid.UUID
    ) -> SummarizeJobLog:
        table_full_name = self.__get_table_full_name(
            table_name=SummarizeJobLog.__table__.name
        )
        query = """
            SELECT * FROM `{}`
            WHERE job_id = '{}'
        """.format(
            table_full_name, str(job_id)
        )
        query_job = self.__bq_client.query(query)
        logs = self.__convert_job_logs(query_job_of_bq=query_job)
        if len(logs) > 1:
            raise DBError(
                "{} is duplicated id in summarize_job_log!".format(job_id)
            )
        return logs[0] if len(logs) == 1 else None

    def __convert_job_logs(self, query_job_of_bq) -> List[SummarizeJobLog]:
        logs = []
        for item in query_job_of_bq:
            log = SummarizeJobLog(job_id=uuid.UUID(item["job_id"]))
            logs.append(log)
        return logs

    def update_label_text_of_result_by_id(
        self, result_id: uuid.UUID, label_text: str
    ) -> int:
        pass

    def update_summarize_results(
        self, result_infos: List[SummarizeResult]
    ) -> List[int]:
        pass

    def update_summarize_job_info(
        self, job_infos: List[SummarizeJobInfo]
    ) -> List[uuid.UUID]:
        table_full_name = self.__get_table_full_name(
            table_name=SummarizeJobInfo.__table__.name
        )
        ids = []
        for job_info in job_infos:
            query = "UPDATE `{}` SET result_id = '{}' WHERE job_id = '{}'".format(
                table_full_name, job_info.result_id, job_info.job_id
            )
            self.__bq_client.query(query)
            ids.append(job_info.job_id)
        return ids

    def delete_all_body_infos(self):
        table_full_name = self.__get_table_full_name(
            table_name=BodyInfo.__table__.name
        )
        self.__delete_all_data_from_tabel(table_full_name)

    def delete_all_summarize_results(self):
        table_full_name = self.__get_table_full_name(
            table_name=SummarizeResult.__table__.name
        )
        self.__delete_all_data_from_tabel(table_full_name)

    def delete_all_job_infos(self):
        table_full_name = self.__get_table_full_name(
            table_name=SummarizeJobInfo.__table__.name
        )
        self.__delete_all_data_from_tabel(table_full_name)

    def __delete_all_data_from_tabel(self, target_table_name: str):
        query = """
            TRUNCATE TABLE `{}`
        """.format(
            target_table_name
        )
        self.__bq_client.query(query)

    def __get_table_full_name(self, table_name: str) -> str:
        table_full_name = bigquery_util.get_full_table_name(
            client=self.__bq_client,
            dataset_name=self._config.db_name,
            table_name=table_name,
        )
        return table_full_name


class DBError(Exception):
    pass
