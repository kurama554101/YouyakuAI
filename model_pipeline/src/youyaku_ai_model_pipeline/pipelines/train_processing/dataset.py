from torch.utils.data import Dataset
import os
import pandas as pd


class LivedoorDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 file_path=None,
                 data_frame:pd.DataFrame=None,
                 input_max_len=512,
                 target_max_len=512):
        # file_path か data_frameは設定する必要があるため、設定していない場合は例外を発火
        if file_path is None and data_frame is None:
            raise DatasetError("file_path and data_frame are None! please set file_path or data_frame!")

        self.file_path = file_path
        self.data_frame = data_frame

        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        def make_record(title, body):
            # ニュースタイトル生成タスク用の入出力形式に変換する。
            input = f"{body}"
            target = f"{title}"
            return input, target

        def append_item(title, body, genre_id):
            input, target = make_record(title, body)

            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input], max_length=self.input_max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )

            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.target_max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

        if self.file_path is not None:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip().split("\t")
                    assert len(line) == 3
                    assert len(line[0]) > 0
                    assert len(line[1]) > 0
                    assert len(line[2]) > 0

                    title = line[0]
                    body = line[1]
                    genre_id = line[2]

                    append_item(title=title, body=body, genre_id=genre_id)
        elif self.data_frame is not None:
            assert len(self.data_frame.columns) == 3
            for index, row in self.data_frame.iterrows():
                append_item(title=row["title"], body=row["body"], genre_id=row["genre_id"])
        else:
            raise DatasetError("file_path and data_frame are None! please set file_path or data_frame!")


class WikihowDataset(Dataset):
    pass


class DatasetError(Exception):
    pass
