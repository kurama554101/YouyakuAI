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


def preprocess(parameters:dict) -> dict:
    # フォルダの設定
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    # ライブドアデータのダウンロード処理
    livedoor_dataset_util = LivedoorDatasetUtil(data_folder=dataset_dir)
    livedoor_dataset_util.write_all_data_from_url()

    # datasetフォルダのパスを返す
    return {"dataset_dir": dataset_dir}


def train(parameters:dict, preprocess_outputs:dict) -> dict:
    model_dir = os.path.join(os.path.dirname(__file__), "model")
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
    args_dict = dict(
        data_dir=dataset_dir,  # データセットのディレクトリ
        model_name_or_path=parameters["pretrained_model_name"],
        tokenizer_name_or_path=parameters["pretrained_model_name"],

        learning_rate=parameters["learning_rate"],
        weight_decay=parameters["weight_decay"],
        adam_epsilon=parameters["adam_epsilon"],
        warmup_steps=parameters["warmup_steps"],
        gradient_accumulation_steps=parameters["gradient_accumulation_steps"],

        n_gpu=1 if USE_GPU else 0,
        early_stop_callback=parameters["early_stop_callback"],
        fp_16=parameters["fp_16"],
        opt_level=parameters["opt_level"],
        max_grad_norm=parameters["max_grad_norm"],
        seed=parameters["seed"],
    )
    hparams = parameters["hparams"]
    args_dict.update(hparams)
    return args_dict


def create_train_params(parameters:dict) -> dict:
    args_dict = create_args_from_parameters(parameters=parameters)
    return dict(
        accumulate_grad_batches=args_dict["gradient_accumulation_steps"],
        gpus=args_dict["n_gpu"],
        max_epochs=args_dict["num_train_epochs"],
        precision= 16 if args_dict["fp_16"] else 32,
        amp_level=args_dict["opt_level"],
        gradient_clip_val=args_dict["max_grad_norm"],
    )
