from kedro.pipeline import Pipeline, node
import os
import torch


def create_pipeline(**kwargs):
    return Pipeline([

    ])


def preprocess(parameters:dict) -> dict:
    # フォルダの設定
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)




def train(parameters:dict):
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    pass


def postprocess(parameters:dict):
    pass


def create_args_from_parameters(parameters:dict) -> dict:
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"
    USE_GPU = torch.cuda.is_available()
    args_dict = dict(
        data_dir=dataset_dir,  # データセットのディレクトリ
        model_name_or_path=PRETRAINED_MODEL_NAME,
        tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        gradient_accumulation_steps=1,

        n_gpu=1 if USE_GPU else 0,
        early_stop_callback=False,
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
    )
    hparams = {
        "max_input_length":  512,  # 入力文の最大トークン数
        "max_target_length": 64,  # 出力文の最大トークン数
        "train_batch_size":  8,
        "eval_batch_size":   8,
        "num_train_epochs":  8,
    }
    args_dict.update(hparams)
    return args_dict


def create_train_params(parameters:dict) -> dict:
    pass
