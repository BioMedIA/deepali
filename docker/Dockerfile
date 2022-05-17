ARG BASE_IMAGE=pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
FROM $BASE_IMAGE

ENV MAMBA_NO_BANNER=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Packages
RUN apt-get update \
    && apt-get install -y \
        git \
        openssh-server \
        sudo \
        vim \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /run/sshd

# Prerequisites
COPY conda/environment.linux-64.yml /tmp/environment.yml
RUN conda update -n base -c defaults conda \
    && conda env update --name base --file /tmp/environment.yml --prune \
    && conda clean -ay \
    && rm -f /tmp/environment.yml

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
