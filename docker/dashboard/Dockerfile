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
ADD ${base_path}/docker/dashboard/requirements.txt .
RUN pip3 install -r requirements.txt
ENV BUILD_MODE="production"

# open PORT
ARG port
EXPOSE ${port}

# copy the source files
COPY src/dashboard dashboard
COPY src/db db
COPY src/log log

# set environment values
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
ARG api_host
ENV API_HOST=${api_host}
ARG api_port
ENV API_PORT=${api_port}

# check the pass of test
RUN cd dashboard && python3 -m unittest

# set entrypoint
ENTRYPOINT [ "python3", "dashboard/app.py" ]
