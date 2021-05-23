from abc import ABCMeta, abstractmethod
from typing import List
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, ForeignKey, DateTime
from sqlalchemy import desc
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import registry, relationship, sessionmaker
from sqlalchemy_utils import create_database, drop_database, database_exists, UUIDType
import uuid
from enum import Enum

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../log'))
from log import AbstractLogger


class DBUtil:
    @staticmethod
    def convert_byte_to_str_if_needed(base_text) -> str:
        if base_text is None:
            return base_text
        if type(base_text) == str:
            return base_text
        return base_text.decode("utf-8")


# モデルで利用する
mapper_registry = registry()

@mapper_registry.mapped
@dataclass
class BodyInfo:
    __table__ = Table(
        "body_info",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("body", LONGTEXT, nullable=False),
        Column("created_at", DateTime, nullable=False)
    )
    id: int = field(init=False)
    body: str = field()
    created_at: datetime = field()

    def get_body(self) -> str:
        return DBUtil.convert_byte_to_str_if_needed(base_text=self.body)


@mapper_registry.mapped
@dataclass
class SummarizeResult:
    __table__ = Table(
        "summarize_result",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        #Column("body_id", Integer, ForeignKey("body_info.id")),
        Column("body_id", Integer, nullable=False),
        Column("inference_status", Integer, nullable=False),
        Column("predicted_text", LONGTEXT, nullable=True),
        Column("label_text", LONGTEXT, nullable=True)
    )
    id: int = field(init=False)
    body_id: int = field()
    inference_status: int = field()
    predicted_text: str = field()
    label_text: str = field()

    def get_predicted_text(self) -> str:
        return DBUtil.convert_byte_to_str_if_needed(base_text=self.predicted_text)

    def get_label_text(self) -> str:
        return DBUtil.convert_byte_to_str_if_needed(base_text=self.label_text)


@mapper_registry.mapped
@dataclass
class SummarizeJobInfo:
    __table__ = Table(
        "summarize_job_info",
        mapper_registry.metadata,
        Column("job_id", UUIDType(binary=False), primary_key=True, default=uuid.uuid4),
        Column("result_id", Integer, nullable=True)
    )
    job_id: UUIDType(binary=False) = field()
    result_id: int = field()
    #result_info = relationship("SummarizeResult")


class InferenceStatus(Enum):
    complete = 1000
    complete_but_difference = 1001
    failed_unknown = 2000


class DBConfig:
    def __init__(self, host:str, port:str, username:str, password:str, db_name:str="summarize_db", db_date_format:str='%Y-%m-%d %H:%M:%S'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.db_name = db_name
        self.db_date_format = db_date_format

    def print(self):
        print("{}, {}, {}, {}, {}".format(self.host, self.port, self.username, self.password, self.db_name))


class AbstractDB:
    __metadata__ = ABCMeta

    def __init__(self, config:DBConfig, log_instance:AbstractLogger):
        self._config = config
        self._log_instance = log_instance

    @abstractmethod
    def create_all_tables_if_needed(self):
        pass

    @abstractmethod
    def drop_all_tables(self):
        pass

    @abstractmethod
    def create_database_if_needed(self):
        pass

    @abstractmethod
    def drop_database(self):
        pass

    @abstractmethod
    def insert_body_infos(self, body_infos:List[BodyInfo]) -> List[int]:
        pass

    @abstractmethod
    def insert_summarize_results(self, result_infos:List[SummarizeResult]) -> List[int]:
        pass

    @abstractmethod
    def insert_summarize_job_info(self, job_infos:List[SummarizeJobInfo]) -> List[int]:
        pass

    @abstractmethod
    def fetch_body_infos(self) -> List[BodyInfo]:
        pass

    @abstractmethod
    def fetch_summarize_results(self) -> List[SummarizeResult]:
        pass

    @abstractmethod
    def fetch_summarize_result_by_id(self, result_id:int) -> SummarizeResult:
        pass

    @abstractmethod
    def fetch_summarize_job_info(self, job_id:uuid.UUID) -> SummarizeJobInfo:
        pass

    @abstractmethod
    def update_label_text_of_result_by_id(self, result_id:int, label_text:str) -> int:
        pass

    @abstractmethod
    def update_summarize_results(self, result_infos:List[SummarizeResult]) -> List[int]:
        pass

    @abstractmethod
    def update_summarize_job_info(self, job_infos:List[SummarizeJobInfo]) -> List[uuid.UUID]:
        pass


class DBFactory:
    @classmethod
    def get_db_instance(cls, db_config, log_instance, db_type="mysql") -> AbstractDB:
        if db_type == "mysql":
            return MySQLDB(config=db_config, log_instance=log_instance)
        else:
            raise NotImplementedError("{} type is not implemented DB type!!".format(db_type))


class MySQLDB(AbstractDB):
    def __init__(self, config:DBConfig, log_instance:AbstractLogger):
        super().__init__(config=config, log_instance=log_instance)
        self.__metadata  = mapper_registry.metadata
        self.__db_engine = create_engine("mysql+mysqlconnector://{}:{}@{}:{}/{}?charset={}".format(self._config.username,
                                                                                                   self._config.password,
                                                                                                   self._config.host,
                                                                                                   self._config.port,
                                                                                                   self._config.db_name,
                                                                                                   "utf8"))
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

    def insert_body_infos(self, body_infos:List[BodyInfo]) -> List[int]:
        return self.__basic_insert_items(items=body_infos)

    def insert_summarize_results(self, result_infos:List[SummarizeResult]) -> List[int]:
        return self.__basic_insert_items(items=result_infos)

    def insert_summarize_job_info(self, job_infos:List[SummarizeJobInfo]) -> List[int]:
        # アイテムの追加処理
        with self.__session() as session:
            session.add_all(job_infos)
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                raise DBError("MySQLDB insert function is Error! error detail is {}".format(e))

            # 追加したアイテムのIDリストを取得
            return [job_info.job_id for job_info in job_infos]

    def fetch_body_infos(self) -> List[BodyInfo]:
        return self.__basic_fetch_all(clazz=BodyInfo)

    def fetch_summarize_results(self) -> List[SummarizeResult]:
        return self.__basic_fetch_all(clazz=SummarizeResult)

    def fetch_summarize_result_by_id(self, result_id:int) -> SummarizeResult:
        with self.__session() as session:
            stmt = session.query(SummarizeResult).filter(SummarizeResult.id == result_id)
            result = session.execute(stmt).fetchone()
            return result[0] if result is not None else None

    def fetch_summarize_job_info(self, job_id:uuid.UUID) -> SummarizeJobInfo:
        with self.__session() as session:
            stmt = session.query(SummarizeJobInfo).filter(SummarizeJobInfo.job_id == job_id)
            result = session.execute(stmt).fetchone()
            return result[0] if result is not None else None

    def update_label_text_of_result_by_id(self, result_id:int, label_text:str) -> int:
        pass

    def update_summarize_results(self, result_infos:List[SummarizeResult]) -> List[int]:
        pass

    def update_summarize_job_info(self, job_infos:List[SummarizeJobInfo]) -> List[uuid.UUID]:
        with self.__session() as session:
            try:
                # TODO : もっと早いやり方があるはず。SQLAlchemyのupdate方法を確認
                for job_info in job_infos:
                    target = session.query(SummarizeJobInfo).filter(SummarizeJobInfo.job_id == job_info.job_id).first()
                    target.result_id = job_info.result_id
                session.commit()
            except Exception as e:
                session.rollback()
                raise DBError("MySQLDB update function({}) is Error! error detail is {}".format("update_summarize_job_info", e))
        return [job_info.job_id for job_info in job_infos]


    def __basic_insert_items(self, items:list) -> list:
        # アイテムの追加処理
        with self.__session() as session:
            session.add_all(items)
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                raise DBError("MySQLDB insert function is Error! error detail is {}".format(e))

            # 追加したアイテムのIDリストを取得
            return [item.id for item in items]

    def __basic_fetch_all(self, clazz) -> list:
        # アイテムの取得処理
        with self.__session() as session:
            stmt = session.query(clazz).order_by(desc(clazz.id))
            result = session.execute(stmt)
            return [row[0] for row in result]


class DBError(Exception):
    pass
