from kedro.pipeline import Pipeline, node
import os
import torch
import pytorch_lightning as pl

from .dataset_util import LivedoorDatasetUtil
from .trainer import T5FineTunerWithLivedoorDataset


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=preprocess,
            inputs=["parameters"],
            outputs="preprocess_outputs",
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
    dataset_dir = os.path.join(os.path.dirname(__file__), parameters["dataset_dir"])
    os.makedirs(dataset_dir, exist_ok=True)

    # ライブドアデータのダウンロード処理
    livedoor_dataset_util = LivedoorDatasetUtil(data_folder=dataset_dir)
    livedoor_dataset_util.write_all_data_from_url()

    # datasetフォルダのパスを返す
    return {"dataset_dir": dataset_dir}


def train(parameters:dict, preprocess_outputs:dict) -> dict:
    model_dir = os.path.join(os.path.dirname(__file__), parameters["model_dir"])
    os.makedirs(model_dir, exist_ok=True)

    dataset_dir  = preprocess_outputs["dataset_dir"]
    args_dict    = create_args_from_parameters(parameters=parameters, dataset_dir=dataset_dir)
    train_params = create_train_params(parameters=parameters)
    args_dict.update(train_params)

    model   = T5FineTunerWithLivedoorDataset(hparams=args_dict)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    model.save(model_dir=model_dir)

    # モデルのパスを返す
    return {"model_dir": model_dir}


def postprocess(parameters:dict, train_outputs:dict):
    pass


# TODO : キーがtrain_paramsと重複しているので、まとめる
def create_args_from_parameters(parameters:dict, dataset_dir:str) -> dict:
    USE_GPU = torch.cuda.is_available()
    hyper_parameters = parameters["hyper_parameters"]
    args_dict = dict(
        data_dir=dataset_dir,  # データセットのディレクトリ
        model_name_or_path=hyper_parameters["pretrained_model_name"],
        tokenizer_name_or_path=hyper_parameters["pretrained_model_name"],

        learning_rate=hyper_parameters["learning_rate"],
        weight_decay=hyper_parameters["weight_decay"],
        adam_epsilon=hyper_parameters["adam_epsilon"],
        warmup_steps=hyper_parameters["warmup_steps"],
        gradient_accumulation_steps=hyper_parameters["gradient_accumulation_steps"],

        n_gpu=1 if USE_GPU else 0,
        early_stop_callback=hyper_parameters["early_stop_callback"],
        fp_16=hyper_parameters["fp_16"],
        opt_level=hyper_parameters["opt_level"],
        max_grad_norm=hyper_parameters["max_grad_norm"],
        seed=hyper_parameters["seed"],

        max_input_length=hyper_parameters["max_input_length"],
        max_target_length=hyper_parameters["max_target_length"],
        train_batch_size=hyper_parameters["train_batch_size"],
        eval_batch_size=hyper_parameters["eval_batch_size"],
        num_train_epochs=hyper_parameters["num_train_epochs"]
    )
    return args_dict


def create_train_params(parameters:dict) -> dict:
    USE_GPU = torch.cuda.is_available()
    hyper_parameters = parameters["hyper_parameters"]
    return dict(
        accumulate_grad_batches=hyper_parameters["gradient_accumulation_steps"],
        gpus=1 if USE_GPU else 0,
        max_epochs=hyper_parameters["num_train_epochs"],
        precision= 16 if hyper_parameters["fp_16"] else 32,
        amp_level=hyper_parameters["opt_level"],
        gradient_clip_val=hyper_parameters["max_grad_norm"],
    )
