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
$ docker-compose up -d
```

※TODO : api_gateway側の立ち上がりが遅い（10秒程度）ため、api_gatewayの立ち上がりをまって、ダッシュボードを利用する必要がある。

## Usage

下記のローカルマシンのダッシュボードにアクセスして、要約処理を実行してください。
* http://localhost:8501
