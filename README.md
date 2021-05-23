# YouyakuAI

要約を行うためのデモサービスを構築します。

## Setup for local

### Training model

TODO : dockerコンテナ経由でモデルを学習するスクリプトを追加

### Create docker containers

ローカル環境で動作するサービスを下記で構築します。
```
$ docker-compose up -d
```

※TODO : api_gateway側の立ち上がりが遅い（10秒程度）ため、api_gatewayの立ち上がりをまって、ダッシュボードを利用する必要がある。

## Usage

下記のローカルマシンのダッシュボードにアクセスして、要約処理を実行してください。
* http://localhost:8501
