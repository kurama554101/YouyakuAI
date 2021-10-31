COMPOSE_FILES=docker-compose.yml:docker-compose-test.yml

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
