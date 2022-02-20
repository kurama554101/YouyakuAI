FROM python:3.8 AS build_base

ARG base_path=.

WORKDIR /workspace
ENV PYTHONPATH $PYTHONPATH:/workspace

#
# BUILD LOCAL IMAGE
# ------------------------------------------------------------------------------
FROM build_base AS local

# install python packages and build_mode
ADD ${base_path}/requirements.txt .
RUN pip3 install -r requirements.txt
ENV BUILD_MODE="local"

#
# BUILD PRODUCTION IMAGE
# ------------------------------------------------------------------------------
FROM build_base AS production

# install python packages and build_mode
ADD ${base_path}/docker/summarizer_processor/requirements.txt .
RUN pip3 install -r requirements.txt
ENV BUILD_MODE="production"

# copy the source files
COPY src/summarizer_processor summarizer_processor
COPY src/queue_api queue_api
COPY src/db db
COPY src/log log

# set environment parameters
ARG queue_host
ENV QUEUE_HOST=${queue_host}
ARG queue_port
ENV QUEUE_PORT=${queue_port}
ARG queue_name
ENV QUEUE_NAME=${queue_name}
ARG db_host
ENV DB_HOST=${db_host}
ARG db_port
ENV DB_PORT=${db_port}
ARG db_username
ENV DB_USERNAME=${db_username}
ARG db_password
ENV DB_PASSWORD=${db_password}
ARG db_name
ENV DB_NAME=${db_name}
ARG db_type
ENV DB_TYPE=${db_type}
ARG google_project_id
ENV GOOGLE_PROJECT_ID=${google_project_id}
ARG google_prediction_location
ENV GOOGLE_PREDICTION_LOCATION=${google_prediction_location}
ARG google_prediction_endpoint
ENV GOOGLE_PREDICTION_ENDPOINT=${google_prediction_endpoint}
ARG summarizer_internal_api_local_host
ENV SUMMARIZER_INTERNAL_API_LOCAL_HOST=${summarizer_internal_api_local_host}
ARG summarizer_internal_api_local_port
ENV SUMMARIZER_INTERNAL_API_LOCAL_PORT=${summarizer_internal_api_local_port}
ARG summarizer_internal_api_type
ENV SUMMARIZER_INTERNAL_API_TYPE=${summarizer_internal_api_type}

# check the pass of test
RUN cd summarizer_processor && python3 -m unittest

# set entrypoint
ENTRYPOINT [ "python3", "summarizer_processor/summarizer_process.py" ]
