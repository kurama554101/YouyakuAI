FROM pytorch/pytorch:latest AS build_base

ARG base_path=.

WORKDIR /workspace
ENV PYTHONPATH $PYTHONPATH:/workspace

#
# BUILD LOCAL IMAGE For TEST
# ------------------------------------------------------------------------------
FROM build_base AS local

ADD ${base_path}/requirements.txt .
RUN pip3 install -r requirements.txt
ENV BUILD_MODE="local"

#
# BUILD PRODUCTION IMAGE
# ------------------------------------------------------------------------------
FROM build_base AS production

ADD ${base_path}/docker/summarizer/requirements.txt .
RUN pip3 install -r requirements.txt
ENV BUILD_MODE="production"

ARG PORT=8001
EXPOSE $PORT

COPY src/summarizer summarizer
COPY src/log log

# TODO : portをARGのPORTを使うように修正
ENTRYPOINT [ "uvicorn", "summarizer.serve:app", "--reload", "--host", "0.0.0.0", "--port", "8001" ]
