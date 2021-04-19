# YouyakuAI

要約を行うためのデモサービスを構築します。

## Setup

TBD

## Usage

### モデルの学習処理

日本語版T5モデルを下記コマンドでファインチューニングします。<br>
現状はライブドアデータを使ったファインチューングのみです。

```
$ python3 src/create_initial_model.py
```

### ダッシュボードの起動

構築したモデルを用いて、要約を試す場合はダッシュボードを起動します。

```
$ streamlit run src/dashboard.py
```
