MAKEFILE_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMPOSE_FILES=docker-compose.yml:docker-compose-test.yml
# 下記はmake実行前に個別に設定すること
GOOGLE_PROJECT_ID=xxxx # need to set project_id
GOOGLE_SERVICE_ACCOUNT=youyaku-ai-account@${GOOGLE_PROJECT_ID}.iam.gserviceaccount.com
GOOGLE_APPLICATION_CREDENTIALS=${MAKEFILE_DIR}credentials/youyaku-ai-service-account.json
GOOGLE_LOCATION=us.gcr.io
GOOGLE_MAIL=xxx@gmail.com # need to set mail address
GOOBLE_BUCKET_NAME=youyaku_ai_pipeline
GOOGLE_BUCKET_LOCATION=US-CENTRAL1

setup-service-account:
	gcloud projects add-iam-policy-binding ${GOOGLE_PROJECT_ID} --member="serviceAccount:${GOOGLE_SERVICE_ACCOUNT}" --role="roles/aiplatform.user"
	gcloud projects add-iam-policy-binding ${GOOGLE_PROJECT_ID} --member="serviceAccount:${GOOGLE_SERVICE_ACCOUNT}" --role="roles/iam.serviceAccountUser"
	gcloud iam service-accounts add-iam-policy-binding ${GOOGLE_SERVICE_ACCOUNT} --member="user:${GOOGLE_MAIL}" --role="roles/iam.serviceAccountUser"

setup-account:
	export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}
	gcloud auth activate-service-account ${GOOGLE_SERVICE_ACCOUNT} --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

setup-bucket:
	gsutil mb -p ${GOOGLE_PROJECT_ID} -l ${GOOGLE_BUCKET_LOCATION} gs://${GOOBLE_BUCKET_NAME}

setup-docker:
	$(MAKE) setup-account
	gcloud auth configure-docker --project=${GOOGLE_PROJECT_ID}
	gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://${GOOGLE_LOCATION}

run:
	docker-compose up -d

run-build:
	docker-compose up -d --build

stop:
	docker-compose down

test-start:
	COMPOSE_FILE=${COMPOSE_FILES} docker-compose up -d

test-start-build:
	COMPOSE_FILE=${COMPOSE_FILES} docker-compose up -d --build

test-print:
	docker logs youyaku_ai_tester

test-stop:
	COMPOSE_FILE=${COMPOSE_FILES} docker-compose down

build-pipeline:
	python3 model_pipeline/pipeline.py

run-pipeline:
	python3 model_pipeline/pipeline.py --run_pipeline

build-push-component:
	python3 model_pipeline/build_all_components.py --docker_type gcr
