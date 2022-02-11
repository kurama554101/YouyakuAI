import argparse
from dataclasses import dataclass
from typing import get_type_hints
import os
import pandas as pd
import torch
import pytorch_lightning as pl
import yaml

from trainer_component import T5FineTunerWithLivedoorDataset

#
# COMPONENT ARGUMENTS
# ------------------------------------------------------------------------------
@dataclass
class ComponentArguments:
    """Argument of the component. Note: Data Generator has no inputs."""

    train_data_path: str
    val_data_path: str
    test_data_path: str
    suffix: str
    parameters: str


@dataclass
class OutputDestinations:
    """Outputs of the component."""

    trained_model: str


@dataclass
class Artifacts:
    component_arguments: ComponentArguments
    output_destinations: OutputDestinations

    @classmethod
    def arg_parser(cls) -> argparse.ArgumentParser:
        """Parse component argument and return as ComponentArguments."""
        parser = argparse.ArgumentParser()
        # generate argument parser based on ComponentArgument's definition
        for artifact in get_type_hints(cls).values():
            for arg_name, arg_type in get_type_hints(artifact).items():
                parser.add_argument(arg_name, type=arg_type)

        return parser

    @classmethod
    def from_args(cls) -> "Artifacts":
        args = vars(cls.arg_parser().parse_args())

        artifacts = {}
        for key, artifact_cls in get_type_hints(cls).items():
            existed_keys = get_type_hints(artifact_cls).keys()
            filtered_vars = {
                k: v for k, v in args.items() if k in existed_keys
            }

            artifacts[key] = artifact_cls(**filtered_vars)
        # parse args and convert into PipelineArguments
        return cls(**artifacts)


#
# MAIN FUNCTION
# ------------------------------------------------------------------------------
def main(args: ComponentArguments):
    parameters = yaml.load(args.parameters)
    train_dataset = pd.read_csv(args.train_data_path)
    val_dataset = pd.read_csv(args.val_data_path)
    test_dataset = pd.read_csv(args.test_data_path)
    print(f"train parameters are {parameters}")

    # 学習処理
    model = train(
        parameters=parameters,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )
    return model


def save_model(model: T5FineTunerWithLivedoorDataset, model_dir: str):
    print(f"saving model folder is {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir=model_dir)


def train(
    parameters: dict,
    train_dataset: pd.DataFrame,
    val_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
) -> T5FineTunerWithLivedoorDataset:
    dataset_params = dict(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )
    args_dict = create_args_from_parameters(parameters=parameters)
    train_params = create_train_params(parameters=parameters)
    args_dict.update(train_params)

    model = T5FineTunerWithLivedoorDataset(
        hparams=args_dict, dataset_params=dataset_params
    )
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    return model


# TODO : キーがtrain_paramsと重複しているので、まとめる
def create_args_from_parameters(
    parameters: dict, dataset_dir: str = None
) -> dict:
    USE_GPU = torch.cuda.is_available()
    hyper_parameters = parameters["hyper_parameters"]
    args_dict = dict(
        model_name_or_path=hyper_parameters["pretrained_model_name"],
        tokenizer_name_or_path=hyper_parameters["pretrained_model_name"],
        learning_rate=hyper_parameters["learning_rate"],
        weight_decay=hyper_parameters["weight_decay"],
        adam_epsilon=hyper_parameters["adam_epsilon"],
        warmup_steps=hyper_parameters["warmup_steps"],
        gradient_accumulation_steps=hyper_parameters[
            "gradient_accumulation_steps"
        ],
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
        num_train_epochs=hyper_parameters["num_train_epochs"],
    )
    return args_dict


def create_train_params(parameters: dict) -> dict:
    USE_GPU = torch.cuda.is_available()
    print("use_gpu is {}".format(USE_GPU))
    hyper_parameters = parameters["hyper_parameters"]
    return dict(
        accumulate_grad_batches=hyper_parameters[
            "gradient_accumulation_steps"
        ],
        gpus=1 if USE_GPU else 0,
        max_epochs=hyper_parameters["num_train_epochs"],
        precision=16 if hyper_parameters["fp_16"] else 32,
        amp_backend=hyper_parameters["amp_backend"],
        amp_level=hyper_parameters["opt_level"],
        gradient_clip_val=hyper_parameters["max_grad_norm"],
    )


#
# ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("---- train start -----")
    artifacts = Artifacts.from_args()
    model = main(artifacts.component_arguments)

    # モデルの保存処理
    save_model(model, artifacts.output_destinations.trained_model)
    print(f"model output path {artifacts.output_destinations.trained_model}")
    print("---- train end -----")
