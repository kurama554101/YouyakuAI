# YouyakuAI

要約を行うためのデモサービスを構築します。

## Setup for local

### Training model

#### Create pipeline file

下記コマンドで、kubeflowで動作する学習パイプラインのファイル（yaml）を構築する
```
$ python3 pipeline.py
```

#### Update Docker Image

学習パイプラインで用いるDocker ImageをDocker Hubにアップロードする。

* model_pipeline/deploy.yamlの修正
  * docker_host_nameを、スクリプトを実行する方のホスト名に変更する
* Docker Imageのビルドとpush処理（下記コマンドの実行）

```
$ cd model_pipeline
$ python3 build_all_componens.py --deploy_type docker_hub
```

#### Execute training pipeline in Local

TBD

#### Execute training pipeline in VertexAI

TBD

### Create docker containers

ローカル環境で動作するサービスを下記で構築します。
```
$ make run
```

Docker Imageを作り直す場合は、下記コマンドでビルドも実施されます。
```
$ make run-build
```

※TODO : api_gateway側の立ち上がりが遅い（10秒程度）ため、api_gatewayの立ち上がりをまって、ダッシュボードを利用する必要がある。

### Stop docker containers

下記コマンドで停止が可能です。
```
$ make stop
```

## Usage

下記のローカルマシンのダッシュボードにアクセスして、要約処理を実行してください。
* http://localhost:8501

## Execute Test

### Execute All Test

下記のコマンドでテストを実行することが可能.
```
make test-start
```

テストの状況は下記コマンドで確認する.
```
make test-print
```

テスト終了後にはコンテナを停止させるため、下記コマンドを実行する.
```
make test-stop
```

### Execute Each Test

テストコードが存在するコンテナに入って、テストコードを実行する。<br>
※事前にdockerコンテナを全て起動する。

```
$ docker exec -it <コンテナ名> /bin/bash
$ python3 <テストしたいフォルダ>/test/<テストしたいファイル名>
```

## Change DB type

.envファイルを修正して、利用するDBを変更することが可能。

* DB_TYPE
  * mysql
  * bigquery

## Setup Delopment Environment

開発環境の構築手順について記載します。

### Used Development Tools

* VSCode
* Python 3.8以上
* black
  * formatter
* flake8
  * linter

### Setup for vscode

下記のツールをインストールします.

```
$ pip install flake8
$ pip install black
```

vscodeのsettings.jsonに下記を追加します.

```
    "python.analysis.extraPaths": [
        "./src/dashboard", "./src/db", "./src/log", "./src/summarizer", "./src/queue_api", "./src/api_gateway",
        "./model_pipeline/components/data_generator/src", "./model_pipeline/components/trainer/src"
    ],
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "79"],
    "editor.formatOnSave": true
```
