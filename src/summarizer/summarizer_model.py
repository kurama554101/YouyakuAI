from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from nlp_util import normalize_text
import numpy as np


class T5Summarizer:
    def __init__(self, hparams: dict) -> None:
        self.__max_input_length = hparams["max_input_length"]
        self.__max_target_length = hparams["max_target_length"]
        self.__model_dir = hparams["model_dir"]
        self.__temperature = (
            hparams["temperature"] if "temperature" in hparams else 1.0
        )
        self.__num_beams = (
            hparams["num_beams"] if "num_beams" in hparams else 10
        )
        self.__diversity_penalty = (
            hparams["diversity_penalty"]
            if "diversity_penalty" in hparams
            else 1.0
        )
        self.__num_beam_groups = (
            hparams["num_beam_groups"] if "num_beam_groups" in hparams else 10
        )
        self.__num_return_sequences = (
            hparams["num_return_sequences"]
            if "num_return_sequences" in hparams
            else 10
        )
        self.__repetition_penalty = (
            hparams["repetition_penalty"]
            if "repetition_penalty" in hparams
            else 1.5
        )
        self.__tokenizer = T5Tokenizer.from_pretrained(
            self.__model_dir, is_fast=True
        )
        self.__model = T5ForConditionalGeneration.from_pretrained(
            self.__model_dir
        )

        # GPUが利用できる場合は、GPUモードとする
        self.__use_gpu = torch.cuda.is_available()
        if self.__use_gpu:
            self.__model.cuda()

        # モデルを推論モードに設定
        self.__model.eval()

    def predict(self, inputs: list) -> list:
        try:
            # 文字列に対する前処理の実施
            input_ids, input_mask = self.__preprocess(inputs=inputs)

            # 推論処理の実施
            outputs = self.__model.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                max_length=self.__max_target_length,
                temperature=self.__temperature,  # 生成にランダム性を入れる温度パラメータ
                num_beams=self.__num_beams,  # ビームサーチの探索幅
                diversity_penalty=self.__diversity_penalty,  # 生成結果の多様性を生み出すためのペナルティ
                num_beam_groups=self.__num_beam_groups,  # ビームサーチのグループ数
                num_return_sequences=self.__num_return_sequences,  # 生成する文の数
                repetition_penalty=self.__repetition_penalty,  # 同じ文の繰り返し（モード崩壊）へのペナルティ
            )

            # 後処理を行い、単語列を作成
            return self.__postprocess(outputs=outputs)
        except Exception as e:
            # 何かしらのエラーが発生した場合は、例外を発火させる
            raise SummarizerError(
                "Summarize Error is occured! error detail is {}".format(e)
            )

    def __preprocess(self, inputs: list) -> list:
        outputs = []

        # 正規化処理
        for input_text in inputs:
            outputs.append(normalize_text(input_text.replace("\n", " ")))

        # トークン化
        batch = self.__tokenizer.batch_encode_plus(
            inputs,
            max_length=self.__max_input_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        input_mask = batch["attention_mask"]
        if self.__use_gpu:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
        return input_ids, input_mask

    def __postprocess(self, outputs: list) -> list:
        tmp_generated_texts = [
            self.__tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for ids in outputs
        ]

        # 対象文に対して、生成される要訳文に応じて、リストを分割する
        generated_texts = (
            np.array(tmp_generated_texts)
            .reshape(-1, self.__num_return_sequences)
            .tolist()
        )
        return generated_texts


class SummarizerError(Exception):
    pass
