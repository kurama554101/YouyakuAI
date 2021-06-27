from kedro.pipeline import Pipeline, node
import os
import pandas as pd

from .dataset_util import LivedoorDatasetUtil


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=get_raw_livedoor_dataset,
            inputs=["parameters"],
            outputs="livedoor_dataset",
            name="get_raw_livedoor_dataset"
        ),
        node(
            func=split_dataset,
            inputs=["parameters", "livedoor_dataset"],
            outputs=dict(train="train_dataset",
                         val="val_dataset",
                         test="test_dataset"),
            name="split_dataset"
        )
    ])


def get_raw_livedoor_dataset(parameters:dict) -> pd.DataFrame:
    # ライブドアデータのダウンロード処理
    data_dir = parameters["data_dir"]
    livedoor_dataset_util = LivedoorDatasetUtil(data_folder=data_dir)
    df = livedoor_dataset_util.download_and_get_all_data()

    # dataframeを返す
    return df


def split_dataset(parameters:dict, raw_dataset:pd.DataFrame) -> dict:
    data_dir = parameters["data_dir"]
    livedoor_dataset_util = LivedoorDatasetUtil(data_folder=data_dir)
    return livedoor_dataset_util.split_data(all_data=raw_dataset, random_seed=1000)
