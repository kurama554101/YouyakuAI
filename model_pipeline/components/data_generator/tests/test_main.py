import unittest
from data_generator import (
    OutputDestinations,
    main,
    write_csv,
    ComponentArguments,
)
import os
import shutil
import pandas as pd


class TestDataGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.args = ComponentArguments(random_seed=10)
        self.output_folder = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_folder, exist_ok=True)
        self.output_destinations = OutputDestinations(
            train_data_path=os.path.join(self.output_folder, "train.csv"),
            val_data_path=os.path.join(self.output_folder, "val.csv"),
            test_data_path=os.path.join(self.output_folder, "test.csv"),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.output_folder)

    def test_main(self):
        # DataFrameの作成
        result_dict = main(args=self.args)

        # csvの作成
        write_csv(result_dict, self.output_destinations)

        # csvが出来ていることを検証
        self.assertTrue(
            os.path.exists(self.output_destinations.train_data_path)
        )
        self.assertTrue(os.path.exists(self.output_destinations.val_data_path))
        self.assertTrue(
            os.path.exists(self.output_destinations.test_data_path)
        )

        # 読み込み時にカラムが3つ（title, body, genre_id）となっていること
        expected_columns = ["title", "body", "genre_id"]
        train_df = pd.read_csv(self.output_destinations.train_data_path)
        self.assertEqual(train_df.columns.tolist(), expected_columns)
        val_df = pd.read_csv(self.output_destinations.val_data_path)
        self.assertEqual(val_df.columns.tolist(), expected_columns)
        test_df = pd.read_csv(self.output_destinations.test_data_path)
        self.assertEqual(test_df.columns.tolist(), expected_columns)
