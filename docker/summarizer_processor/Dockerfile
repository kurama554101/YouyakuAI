FROM python:3.8 AS build_base

WORKDIR /workspace
ADD requirements.txt .
RUN pip3 install -r requirements.txt


#
# BUILD LOCAL IMAGE For TEST
# ------------------------------------------------------------------------------
FROM build_base AS local

ENV BUILD_MODE="local"

#
# BUILD PRODUCTION IMAGE
# ------------------------------------------------------------------------------
FROM build_base AS production

ENV BUILD_MODE="production"
