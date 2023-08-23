CURRENT_DIR = $(shell pwd)
PROJECT_NAME = gcs-workshop
include .env
export

prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data/service_data

build: prepare-dirs
	docker build -f Dockerfile -t ${DOCKERHUB_USER}/${PROJECT_NAME}:latest .

stop:
	docker rm -f ${PROJECT_NAME}_container && docker rm -f ${PROJECT_NAME}_container

run: stop
	docker run --rm \
		--env-file ${CURRENT_DIR}/.env  \
		-e "mode=train" \
	    -v "${CURRENT_DIR}/src:/srv/src" \
	    -v "${CURRENT_DIR}/data/service_data:/srv/data" \
	    -v "${CURRENT_DIR}/gcs_secret.json:/srv/gcs_secret.json" \
	    --name ${PROJECT_NAME}_container \
		${DOCKERHUB_USER}/${PROJECT_NAME}:latest

push: build
	docker push ${DOCKERHUB_USER}/${PROJECT_NAME}:latest

train:
	docker run -it --rm \
	    --env-file ${CURRENT_DIR}/.env \
	    -v "${CURRENT_DIR}/src:/srv/src" \
	    -v "${CURRENT_DIR}/data:/srv/data" \
		--network service_network \
	    --name ${PROJECT_NAME}_search_container \
	    ${DOCKERHUB_USER}/${PROJECT_NAME}:latest train

clean: stop
	docker image rm -f ${DOCKERHUB_USER}/${PROJECT_NAME}:latest && \
	docker image rm -f ${DOCKERHUB_USER}/${PROJECT_NAME}:latest