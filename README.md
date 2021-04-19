# YouyakuAI

要約を行うためのデモサービスを構築します。

## Setup

必要なpythonパッケージを下記コマンドでインストールします。
```
$ pip3 install -r requirements.txt
```

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
