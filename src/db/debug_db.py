from dataclasses import dataclass
from dataclasses import field
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, ForeignKey, DateTime
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import registry, relationship, sessionmaker
import os
from datetime import datetime

import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from db_wrapper import DBFactory, DBConfig
from custom_log import LoggerFactory


mapper_registry = registry()


@mapper_registry.mapped
@dataclass
class BodyInfo:
    __table__ = Table(
        "body_info",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("body", LONGTEXT, nullable=False),
        Column("created_at", DateTime, nullable=False),
    )
    id: int = field(init=False)
    body: str = field()
    created_at: datetime = field()


@mapper_registry.mapped
@dataclass
class SummarizeResult:
    __table__ = Table(
        "summarize_result",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True),
        Column("body_id", Integer, ForeignKey("body_info.id")),
        Column("predicted_text", LONGTEXT),
        Column("label_text", LONGTEXT),
    )
    id: int = field(init=False)
    body_id: int = field()
    predicted_text: str = field()
    label_text: str = field()
    body_info = relationship("BodyInfo")


def main():
    meta = mapper_registry.metadata
    engine = create_engine(
        "mysql+mysqlconnector://{}:{}@{}:{}/{}?charset={}".format(
            os.environ.get("DB_USERNAME"),
            os.environ.get("DB_PASSWORD"),
            os.environ.get("DB_HOST"),
            os.environ.get("DB_PORT"),
            os.environ.get("DB_NAME"),
            "utf8",
        )
    )
    Session = sessionmaker(engine)

    # テーブル作成
    meta.create_all(engine)

    # body_infoの追加
    body_id = None
    with Session() as session:
        body_info = BodyInfo(
            body="てすとです",
            created_at=datetime(year=2021, month=1, day=1, hour=10, minute=0),
        )
        session.add(body_info)
        session.commit()
        body_id = body_info.id

    # 最新のidを取得
    # body_id = None
    # with Session() as session:
    #    stmt = session.query(BodyInfo).order_by(desc(BodyInfo.id))
    #    result = session.execute(stmt)
    #    latest_item = result.fetchone()[0]
    # for s in result:
    #    print(s)
    #    body_id = latest_item.id if latest_item is not None else None

    # resultの登録
    if body_id is None:
        print("latest_body id is not found!")
        return
    with Session() as session:
        session.add(
            SummarizeResult(
                body_id=body_id, predicted_text="予測です", label_text="外れます"
            )
        )
        session.commit()


def main2():
    config = DBConfig(
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT"),
        username=os.environ.get("DB_USERNAME"),
        password=os.environ.get("DB_PASSWORD"),
        db_name=os.environ.get("DB_NAME"),
    )
    logger = LoggerFactory.get_logger(
        logger_type="print", logger_name="debug_db"
    )
    db_instance = DBFactory.get_db_instance(
        db_config=config, log_instance=logger
    )

    # テーブルの作成
    db_instance.create_all_tables_if_needed()

    # BodyInfoの追加
    ids = db_instance.insert_body_infos(
        body_infos=[
            BodyInfo(body="てすとです", created_at=datetime.now()),
            BodyInfo(body="てすとです2", created_at=datetime.now()),
        ]
    )

    # SuammarizerResultの追加
    result_list = [
        SummarizeResult(body_id=id, predicted_text="予測です", label_text="外れます")
        for id in ids
    ]
    result_ids = db_instance.insert_summarize_results(result_infos=result_list)
    logger.info("result ids is {}".format(result_ids))


if __name__ == "__main__":
    # main()
    main2()
