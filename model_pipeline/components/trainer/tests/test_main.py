import unittest

from pandas.core import base
from trainer import main, save_model, ComponentArguments, OutputDestinations
import os
import yaml
import shutil


class TestTrainer(unittest.TestCase):
    def setUp(self) -> None:
        contents_folder = os.path.join(os.path.dirname(__file__), "test_contents")
        output_folder = os.path.join(os.path.dirname(__file__), "output")
        def load_yaml(path: str) -> dict:
            with open(path, "r") as f:
                return yaml.load(f)
        parameters = load_yaml(os.path.join(contents_folder, "test_parameters.yml"))
        self.args = ComponentArguments(train_data_path=os.path.join(contents_folder, "train.csv"),
                                       val_data_path=os.path.join(contents_folder, "val.csv"),
                                       test_data_path=os.path.join(contents_folder, "test.csv"),
                                       suffix="_test",
                                       parameters=parameters)
        self.out_dest = OutputDestinations(output_folder)

    def tearDown(self) -> None:
        shutil.rmtree(self.out_dest.trained_model_dir)

    def test_main(self):
        model = main(args=self.args)
        save_model(model=model, out_dest=self.out_dest)
