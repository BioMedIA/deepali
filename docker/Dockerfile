ARG BASE_IMAGE=pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM $BASE_IMAGE

# Packages
RUN apt-get update \
    && apt-get install -y \
        git \
        openssh-server \
        sudo \
        vim \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /run/sshd

# Python
ARG PYTHON_VERSION=
RUN [ -z $PYTHON_VERSION ] \
    || [ $PYTHON_VERSION = $(/opt/conda/bin/python --version | cut -d' ' -f2) ] \
    || conda install -c anaconda python=${PYTHON_VERSION}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Prerequisites
ARG CONDA_YAML="conda.yaml"
COPY "$CONDA_YAML" "/workspace/conda.yaml"
RUN conda update -n base -c defaults conda \
    && conda env update --name base --file "/workspace/conda.yaml" --prune \
    && conda clean -ay \
    && rm -f "/workspace/conda.yaml"

# Install deepali
COPY . "/workspace"
RUN pip install --editable /workspace

# User
ARG USER
ARG GROUP
ARG UID
ARG GID

RUN groupadd --gid "$GID" "$GROUP" \
    && useradd -l -u "$UID" -g "$GROUP" -m -d "/home/$USER" -s /bin/bash "$USER" \
    && usermod -a -G sudo "$USER" \
    && echo "$USER ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && chown -R "$USER:$GROUP" /opt/conda /workspace

USER $USER
