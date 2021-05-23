import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from api import SummarizerApi
from api_gateway_util import create_db_config, create_queue_config, create_db_instance, create_logger, create_queue_initializer_instance, create_queue_instance

# 各インスタンスの作成
db_config = create_db_config()
queue_config = create_queue_config()
logger = create_logger("youyaku_ai_api_gateway")
db_instance = create_db_instance(config=db_config, logger=logger)
queue_instance = create_queue_instance(config=queue_config, logger=logger)
queue_initializer = create_queue_initializer_instance(config=queue_config, logger=logger)

# dbの初期化を実施
# TODO : ここでテーブル作成すべきかは検討が必要
def create_table_with_retry(db_instance):
    try:
        db_instance.create_all_tables_if_needed()
    except Exception as e:
        logger.error(e)
        time.sleep(1)
        create_table_with_retry(db_instance=db_instance)
    return
create_table_with_retry(db_instance=db_instance)

# Queueの初期化を実施
queue_initializer.initialize()

# APIの構築
app = SummarizerApi(queue_producer=queue_instance, db_instance=db_instance, logger=logger).create_app()
