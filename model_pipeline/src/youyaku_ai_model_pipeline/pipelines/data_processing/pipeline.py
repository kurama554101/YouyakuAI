import os
import pandas as pd

from .dataset_util import LivedoorDatasetUtil


# TODO : train, test, valでDFを返すように対応する
def preprocess(parameters:dict) -> dict:
    # フォルダの設定
    dataset_dir = os.path.join(os.path.dirname(__file__), parameters["dataset_dir"])
    os.makedirs(dataset_dir, exist_ok=True)

    # ライブドアデータのダウンロード処理
    livedoor_dataset_util = LivedoorDatasetUtil(data_folder=dataset_dir)
    livedoor_dataset_util.write_all_data_from_url()

    # datasetフォルダのパスを返す
    return {"dataset_dir": dataset_dir}
