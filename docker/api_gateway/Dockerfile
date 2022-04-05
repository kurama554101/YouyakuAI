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
ADD ${base_path}/docker/api_gateway/requirements.txt .
RUN pip3 install -r requirements.txt
ENV BUILD_MODE="production"

# open PORT
ARG api_port
EXPOSE ${api_port}

# copy the source files
COPY src/api_gateway api_gateway
COPY src/queue_api queue_api
COPY src/db db
COPY src/log log

# set environment values
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

# check the pass of test
RUN cd api_gateway && python3 -m unittest

# set entrypoint
ENTRYPOINT [ "uvicorn", "api_gateway.main:app", "--reload", "--host", "0.0.0.0", "--port", ${api_port} ]
