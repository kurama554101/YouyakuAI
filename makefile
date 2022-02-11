include .env
MAKEFILE_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
COMPOSE_FILES=docker-compose.yml:docker-compose-test.yml
GOOGLE_SERVICE_ACCOUNT=youyaku-ai-account@${GOOGLE_PROJECT_ID}.iam.gserviceaccount.com
GOOGLE_APPLICATION_CREDENTIALS=${MAKEFILE_DIR}credentials/youyaku-ai-service-account.json

echo-param:
	echo ${GOOGLE_PROJECT_ID}
	echo ${GOOGLE_SERVICE_ACCOUNT_FILE}

setup-service-account:
	gcloud projects add-iam-policy-binding ${GOOGLE_PROJECT_ID} --member="serviceAccount:${GOOGLE_SERVICE_ACCOUNT}" --role="roles/aiplatform.user"
	gcloud projects add-iam-policy-binding ${GOOGLE_PROJECT_ID} --member="serviceAccount:${GOOGLE_SERVICE_ACCOUNT}" --role="roles/iam.serviceAccountUser"
	gcloud iam service-accounts add-iam-policy-binding ${GOOGLE_SERVICE_ACCOUNT} --member="user:${GOOGLE_MAIL}" --role="roles/iam.serviceAccountUser"

#setup-account:
#	export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}
#	gcloud auth activate-service-account ${GOOGLE_SERVICE_ACCOUNT} --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

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
	python3 model_pipeline/pipeline.py --run_pipeline --pipeline_type gcp

build-push-component:
	python3 build_and_deploy_image.py --docker_type gcr
	python3 model_pipeline/build_all_components.py --docker_type gcr

rebuild-push-component:
	python3 build_and_deploy_image.py --docker_type gcr --rebuild
	python3 model_pipeline/build_all_components.py --docker_type gcr
