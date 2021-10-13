FROM python:3.8.10-slim-buster AS build_base

# install dependent packages
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    # for install
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install python packages
WORKDIR /component
COPY requirements.txt ./
RUN pip install -r requirements.txt

# copy base source code and add path into PYTHONPATH
ENV PYTHONPATH $PYTHONPATH:/component/src

#
# BUILD AND EXECUTE UNIT TEST
# ------------------------------------------------------------------------------
FROM build_base AS test_runner

# copy test directory and run test
COPY src/ src/
COPY tests/ tests/

# run unit tests
RUN python3 -m unittest

#
# BUILD PRODUCTION IMAGE
# ------------------------------------------------------------------------------
FROM build_base AS production

# apply the cache system, copy codes
COPY src/ src/

# execute python code as default entrypoint.
ENTRYPOINT [ "python", "/component/src/trainer.py" ]
