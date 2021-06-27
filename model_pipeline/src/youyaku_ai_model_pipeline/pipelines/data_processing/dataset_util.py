from itertools import count
import tarfile
import re
import urllib.request
import os
import random
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src", "summarizer"))
# TODO : summarizerパッケージとの依存を無くしたいので、nlp_utilを別パッケージにして、summarizerとpipelineの方で読み込むようにする
from nlp_util import normalize_text


class LivedoorDatasetUtil:
    def __init__(self, data_folder:str, train_ratio:float=0.95, test_ratio:float=0.05, val_ratio:float=0.05) -> None:
        self.__target_genres = ["dokujo-tsushin",
                               "it-life-hack",
                               "kaden-channel",
                               "livedoor-homme",
                               "movie-enter",
                               "peachy",
                               "smax",
                               "sports-watch",
                               "topic-news"]
        self.__archive_url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
        self.__base_folder = data_folder
        os.makedirs(self.__base_folder, exist_ok=True)

        self.__train_ratio = train_ratio
        self.__val_ratio   = val_ratio
        self.__test_ratio  = test_ratio

    def write_all_data_from_url(self) -> dict:
        # ライブドアデータを取得し、本文・タイトル・ジャンルのデータを全て取得
        all_data = self.download_and_get_all_data()

        # 学習、検証、テスト用にデータセットを分割し、ファイルを作成
        return self.write_files_from_data(all_data=all_data, random_seed=1234)

    def download_and_get_all_data(self) -> list:
        # ライブドアのアーカイブファイルをダウンロード
        archive_path = os.path.join(self.__base_folder, os.path.basename(self.__archive_url))
        urllib.request.urlretrieve(self.__archive_url, archive_path)

        # データを取得
        all_data = self.get_all_data_from_archive_file(archive_path=archive_path)

        # archiveファイルの削除
        os.remove(archive_path)
        return all_data

    def get_all_data_from_archive_file(self, archive_path:str) -> pd.DataFrame:
        genre_files_list = [[] for genre in self.__target_genres]
        df = pd.DataFrame(index=[], columns=["title", "body", "genre_id"])

        with tarfile.open(archive_path) as archive_file:
            for archive_item in archive_file:
                for i, genre in enumerate(self.__target_genres):
                    if genre in archive_item.name and archive_item.name.endswith(".txt"):
                        genre_files_list[i].append(archive_item.name)

            for i, genre_files in enumerate(genre_files_list):
                for name in genre_files:
                    file = archive_file.extractfile(name)
                    title, body = self.__read_title_body(file)
                    title = normalize_text(title)
                    body = normalize_text(body)

                    if len(title) > 0 and len(body) > 0:
                        df.append(
                            pd.Series([title, body, i], index=df.columns), ignore_index=True
                        )
            return df

    def split_data(self, all_data:pd.DataFrame, random_seed:int=1000) -> dict:
        train_list = []
        val_list = []
        test_list = []
        all_dara_shuffle = all_data.sample(frac=1, random_state=random_seed)
        data_size = len(all_data)
        total_ratio = self.__train_ratio + self.__val_ratio + self.__test_ratio
        def get_item(row):
            return [row["title"], row["body"], row["genre_id"]]

        for i, row in all_dara_shuffle.iterrows():
            if i < self.__train_ratio / total_ratio * data_size:
                train_list.append(get_item(row))
            elif i < (self.__train_ratio + self.__val_ratio) / total_ratio * data_size:
                val_list.append(get_item(row))
            else:
                test_list.append(get_item(row))

        columns = ["title", "body", "genre_id"]
        return {
            "train": pd.DataFrame(train_list, columns=columns),
            "val": pd.DataFrame(val_list, columns=columns),
            "test": pd.DataFrame(test_list, columns=columns)
        }

    def write_files_from_data(self, all_data:pd.DataFrame, random_seed:int=1000) -> dict:
        random.seed(random_seed)
        random.shuffle(all_data)

        def to_line(data):
            title = data["title"]
            body = data["body"]
            genre_id = data["genre_id"]

            assert len(title) > 0 and len(body) > 0
            return f"{title}\t{body}\t{genre_id}\n"

        data_size = len(all_data)
        train_file_path = os.path.join(self.__base_folder, "train.tsv")
        val_file_path = os.path.join(self.__base_folder, "val.tsv")
        test_file_path = os.path.join(self.__base_folder, "test.tsv")
        total_ratio = self.__train_ratio + self.__val_ratio + self.__test_ratio
        with open(train_file_path, "w", encoding="utf-8") as f_train, \
             open(val_file_path, "w", encoding="utf-8") as f_dev, \
             open(test_file_path, "w", encoding="utf-8") as f_test:

            for i, data in tqdm(enumerate(all_data)):
                line = to_line(data)
                if i < self.__train_ratio / total_ratio * data_size:
                    f_train.write(line)
                elif i < (self.__train_ratio + self.__val_ratio) / total_ratio * data_size:
                    f_dev.write(line)
                else:
                    f_test.write(line)

        return {"train": train_file_path, "val": val_file_path, "test": test_file_path}

    def __remove_brackets(self, text):
        text = re.sub(r"(^【[^】]*】)|(【[^】]*】$)", "", text)
        return text

    def __read_title_body(self, file):
        next(file)
        next(file)
        title = next(file).decode("utf-8").strip()
        title = normalize_text(self.__remove_brackets(title))
        body = normalize_text(" ".join([line.decode("utf-8").strip() for line in file.readlines()]))
        return title, body
