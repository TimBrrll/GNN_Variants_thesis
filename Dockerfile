FROM --platform=linux/amd64 nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 
ARG POETRY_EXTRA_ARGS=""
ARG POETRY_VERSION=1.3.1

ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && apt-get -y \
    install git make curl \
    libgl1-mesa-glx g++ pkg-config libcairo2-dev python3-dev

WORKDIR /project

# get latest poetry and install it locally.
RUN ["/bin/bash", "-c", "set -o pipefail && curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3 -"]
ENV PATH="/root/.poetry/bin:/root/.local/bin:${PATH}"

# do not create virtualenv for better caching
# the layer and install everything to system python
RUN poetry config virtualenvs.create false

COPY poetry.lock /project
COPY pyproject.toml /project

# the following two calls of poetry install are separated
# to avoid OOM problems when snapshotting the filesystem
# when building the image in CI/CD pipeline with kaniko
RUN poetry install --no-cache --no-root --without jobs,ai $POETRY_EXTRA_ARGS 

COPY . /project

# Just install current project. Dependencies had been installed with poetry.
RUN poetry install --only-root

RUN apt-get update && apt-get install -y libeigen3-dev pybind11-dev python3.10-dev

WORKDIR /project/code/main_methods/preprocessing
RUN g++ -O3 -shared -std=c++11 -fPIC -I/usr/include/eigen3 -I/usr/include/pybind11 -I/usr/include/python3.10  preprocessing.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix`
EXPOSE 3030

#For kernel models
# WORKDIR /project/code/main_methods/kernel_models
# RUN g++ -O3 -shared -std=c++11 -fPIC -I/usr/include/eigen3 -I/usr/include/pybind11 -I/usr/include/python3.10  kernel_models.cpp src/*cpp -o ../kernel_models`python3-config --extension-suffix`
# EXPOSE 3030


RUN --mount=type=cache,target=/root/.cache apt-get install -y --no-install-recommends python3-pip
RUN --mount=type=cache,target=/root/.cache pip3 install torch-scatter

WORKDIR /project
ENTRYPOINT ["poetry", "run", "python3", "-u", "code/main_methods/main_sparse_m_3_gnn.py"]