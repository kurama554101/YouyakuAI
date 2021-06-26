from kedro.pipeline import Pipeline, node
import os
import torch
import pytorch_lightning as pl

from dataset_util import LivedoorDatasetUtil
from trainer import T5FineTunerWithLivedoorDataset


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=preprocess,
            inputs=["parameters"],
            outputs=["preprocess_outputs"],
            name="preprocess"
        ),
        node(
            func=train,
            inputs=["parameters", "preprocess_outputs"],
            outputs=["train_outputs"],
            name="train"
        ),
        node(
            func=postprocess,
            inputs=["parameters", "train_outputs"],
            outputs=None,
            name="postprocess"
        )
    ])


# TODO : ここは学習パイプラインとは別にデータ取得パイプラインを作って、対応する
def preprocess(parameters:dict) -> dict:
    # フォルダの設定
    dataset_dir = os.path.join(os.path.dirname(__file__), parameters["dataset"])
    os.makedirs(dataset_dir, exist_ok=True)

    # ライブドアデータのダウンロード処理
    livedoor_dataset_util = LivedoorDatasetUtil(data_folder=dataset_dir)
    livedoor_dataset_util.write_all_data_from_url()

    # datasetフォルダのパスを返す
    return {"dataset_dir": dataset_dir}


def train(parameters:dict, preprocess_outputs:dict) -> dict:
    model_dir = os.path.join(os.path.dirname(__file__), parameters["model"])
    os.makedirs(model_dir, exist_ok=True)

    dataset_dir  = preprocess_outputs["dataset_dir"]
    args_dict    = create_args_from_parameters(parameters=parameters, dataset_dir=dataset_dir)
    train_params = create_train_params(parameters=parameters)

    model   = T5FineTunerWithLivedoorDataset(hparams=args_dict)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    model.save(model_dir=model_dir)

    # モデルのパスを返す
    return {"model_dir": model_dir}


def postprocess(parameters:dict, train_outputs:dict):
    pass


def create_args_from_parameters(parameters:dict, dataset_dir:str) -> dict:
    USE_GPU = torch.cuda.is_available()
    h_param = parameters["hyper_parameters"]
    args_dict = dict(
        data_dir=dataset_dir,  # データセットのディレクトリ
        model_name_or_path=h_param["pretrained_model_name"],
        tokenizer_name_or_path=h_param["pretrained_model_name"],

        learning_rate=h_param["learning_rate"],
        weight_decay=h_param["weight_decay"],
        adam_epsilon=h_param["adam_epsilon"],
        warmup_steps=h_param["warmup_steps"],
        gradient_accumulation_steps=h_param["gradient_accumulation_steps"],

        n_gpu=1 if USE_GPU else 0,
        early_stop_callback=h_param["early_stop_callback"],
        fp_16=h_param["fp_16"],
        opt_level=h_param["opt_level"],
        max_grad_norm=h_param["max_grad_norm"],
        seed=h_param["seed"],
    )
    hparams = h_param["hparams"]
    args_dict.update(hparams)
    return args_dict


def create_train_params(parameters:dict) -> dict:
    USE_GPU = torch.cuda.is_available()
    h_param = parameters["hyper_parameters"]
    return dict(
        accumulate_grad_batches=h_param["gradient_accumulation_steps"],
        gpus=1 if USE_GPU else 0,
        max_epochs=h_param["num_train_epochs"],
        precision= 16 if h_param["fp_16"] else 32,
        amp_level=h_param["opt_level"],
        gradient_clip_val=h_param["max_grad_norm"],
    )
