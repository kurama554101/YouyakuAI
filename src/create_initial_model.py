import os
import torch
import pytorch_lightning as pl
from torch.utils.data import dataset
from trainer import T5FineTunerWithLivedoorDataset
from dataset_util import LivedoorDatasetUtil


def main():
    # フォルダ設定
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    model_dir   = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 各種ハイパーパラメータの設定
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

    # 学習用のパラメーターの設定
    train_params = dict(
        accumulate_grad_batches=args_dict["gradient_accumulation_steps"],
        gpus=args_dict["n_gpu"],
        max_epochs=args_dict["num_train_epochs"],
        precision= 16 if args_dict["fp_16"] else 32,
        amp_level=args_dict["opt_level"],
        gradient_clip_val=args_dict["max_grad_norm"],
    )

    # ライブドアデータのダウンロード処理
    livedoor_dataset_util = LivedoorDatasetUtil(data_folder=dataset_dir)
    livedoor_dataset_util.write_all_data_from_url()

    # モデルの転移学習の実施
    model = T5FineTunerWithLivedoorDataset(hparams=args_dict)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    model.save(model_dir=model_dir)


if __name__ == "__main__":
    main()
