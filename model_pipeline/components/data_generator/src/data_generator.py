import argparse
from dataclasses import dataclass
from typing import get_type_hints
import pandas as pd
from pathlib import Path

from dataset_util import LivedoorDatasetUtil

#
# COMPONENT ARGUMENTS
# ------------------------------------------------------------------------------


@dataclass
class ComponentArguments:
    """Argument of the component."""

    random_seed: int


@dataclass
class OutputDestinations:
    """Outputs of the component."""

    train_data_path: str
    val_data_path: str
    test_data_path: str


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


def main(args: ComponentArguments) -> dict:
    dataset_dir = "dataset"

    # ライブドアデータのダウンロード処理
    livedoor_dataset_util = LivedoorDatasetUtil(data_folder=dataset_dir)
    df = livedoor_dataset_util.download_and_get_all_data()

    # データをtrain, val, test用に分割
    return livedoor_dataset_util.split_data(
        all_data=df, random_seed=args.random_seed
    )


def write_csv(result_dict: dict, output_destinations: OutputDestinations):
    train_df = result_dict["train"]
    val_df = result_dict["val"]
    test_df = result_dict["test"]

    write_one_csv(train_df, output_destinations.train_data_path)
    write_one_csv(val_df, output_destinations.val_data_path)
    write_one_csv(test_df, output_destinations.test_data_path)


def write_one_csv(data: pd.DataFrame, destination: str):
    path = Path(destination)
    path.parent.mkdir(exist_ok=True, parents=True)
    data.to_csv(destination, index=False)


#
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    artifacts = Artifacts.from_args()
    result_dict = main(artifacts.component_arguments)

    # DataFrameをcsvに書き込む
    write_csv(result_dict, artifacts.output_destinations)
