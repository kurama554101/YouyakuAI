from abc import ABCMeta, abstractmethod
from .dataset import LivedoorDataset
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams:dict):
        __metadata__ = ABCMeta

        super().__init__()
        self._hparams = hparams

        # 事前学習済みモデルの読み込み
        self._model = T5ForConditionalGeneration.from_pretrained(hparams["model_name_or_path"])

        # トークナイザーの読み込み
        self._tokenizer = T5Tokenizer.from_pretrained(hparams["tokenizer_name_or_path"], is_fast=True)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self._model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def save(self, model_dir:str):
        self._tokenizer.save_pretrained(model_dir)
        self._model.save_pretrained(model_dir)

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]

        # All labels set to -100 are ignored (masked),
        # the loss is only computed for labels in [0, ..., config.vocab_size]
        labels[labels[:, :] == self._tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_attention_mask=batch['target_mask'],
            labels=labels
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self._model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self._hparams["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self._hparams["learning_rate"],
                          eps=self._hparams["adam_epsilon"])
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self._hparams["warmup_steps"],
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    @abstractmethod
    def get_dataset(self, tokenizer, type_path):
        pass

    @abstractmethod
    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset,
                          batch_size=self._hparams["train_batch_size"],
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset,
                          batch_size=self._hparams["eval_batch_size"],
                          num_workers=4)

class T5FineTunerWithLivedoorDataset(T5FineTuner):
    def get_dataset(self, type_path):
        """データセットを作成する"""
        return LivedoorDataset(
            tokenizer=self._tokenizer,
            data_dir=self._hparams["data_dir"],
            type_path=type_path,
            input_max_len=self._hparams["max_input_length"],
            target_max_len=self._hparams["max_target_length"])

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            train_dataset = self.get_dataset(type_path="train.tsv")
            self.train_dataset = train_dataset

            val_dataset = self.get_dataset(type_path="val.tsv")
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) // (self._hparams["train_batch_size"] * max(1, self._hparams["n_gpu"])))
                // self._hparams["gradient_accumulation_steps"]
                * float(self._hparams["num_train_epochs"])
            )
