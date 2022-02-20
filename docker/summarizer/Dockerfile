FROM pytorch/pytorch:latest AS build_base

ARG base_path=.

WORKDIR /workspace
ENV PYTHONPATH $PYTHONPATH:/workspace

#
# BUILD LOCAL IMAGE
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

ARG port
EXPOSE ${port}

COPY src/summarizer summarizer
COPY src/log log

# execute test
RUN cd summarizer && python3 -m unittest

# set entrypoint
ENTRYPOINT [ "uvicorn", "summarizer.serve:app", "--reload", "--host", "0.0.0.0", "--port", ${port} ]
