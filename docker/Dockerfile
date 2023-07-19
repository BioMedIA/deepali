ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Packages
#
# libgl1-mesa-glx is required by VTK
RUN apt-get update \
    && apt-get install -y \
        git \
        libgl1-mesa-glx \
        make \
        openssh-server \
        sudo \
        vim \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /run/sshd

# Conda / Mamba / Python
#
# - Wait for resolution of https://github.com/conda-incubator/conda-lock/issues/229 before upgrading to new conda-lock versions.
# - mamba>=1 is not compatible with conda-lock, including newer versions (e.g., conda-lock 1.3.0)
ARG CONDA_VERSION
ARG PYTHON_VERSION
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV MAMBA_NO_BANNER=1
ENV PATH="/opt/conda/bin:$PATH"
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/download/${CONDA_VERSION}/Mambaforge-${CONDA_VERSION}-Linux-x86_64.sh -O ~/mambaforge.sh \
    && /bin/bash ~/mambaforge.sh -b -p /opt/conda \
    && rm ~/mambaforge.sh \
    && \
    if [ ${PYTHON_VERSION} != $(/opt/conda/bin/python --version | cut -d' ' -f2) ]; then \
        mamba install --name base --channel conda-forge python=${PYTHON_VERSION}; \
    fi \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && mamba install --name base --channel conda-forge \
         conda-devenv=2.3.0 \
         conda-lock=1.0.5 \
    && mamba clean -ay

# Conda environment
#
# See https://github.com/pypa/setuptools_scm/tree/ca3855ba66fa4cb100f5039eea909932f815a4a4#usage-from-docker
# for information on how to give setuptools-scm access to the .git folder while retaining build caching.
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
COPY . /workspace
RUN make -C /workspace/conda env NAME=deepali EDITABLE=1 && conda clean -ay
RUN conda config --set auto_activate_base false
RUN echo "conda activate deepali" >> /etc/profile.d/conda.sh

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
