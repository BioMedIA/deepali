# Conda configuration

Conda environment configurations for `deepali` libraries.


## Create environment

If not done before, install either [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

A conda environment for this project can be created using the `env` Makefile target, i.e.,

```
make env [NAME=deepali] [EDITABLE=1]
```

This creates a conda environment with the specified `NAME`, and subsequently installs the `deepali` libraries. When `EDITABLE=1|true`, an editable installation of the libraries is performed using `pip install --editable`.

Alternatively, run the following `conda` commands intead of using `make`:

```
NAME=deepali
PLATFORM=linux-64

conda create --name $NAME --file environment.$PLATFORM.lock
conda run --name $NAME pip install [--editable] ..
```

where `PLATFORM` is one of the following values.

| **Platform**  | **Description**                             |
| ------------- | ------------------------------------------- |
| `linux-64`    | 64-bit Linux distribution (e.g., CentOS 7). |
| `osx-64`      | 64-bit macOS without CUDA device.           |
| `win-64`      | 64-bit Microsoft Windows with CUDA device.  |


## Manage dependencies

The following tools are required to update the generated dependency files.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html): Package dependency management tool.
2. Install [mamba](https://mamba.readthedocs.io/en/latest/): Faster dependency resolution and more informative error messages.
   - `conda install mamba=0.23 --name base --channel conda-forge`
3. Install [conda-devenv](https://conda-devenv.readthedocs.io/en/latest/): Advanced conda environment configuration such as conditional dependencies.
   - `conda install conda-devenv=2.3 --name base --channel conda-forge`
4. Install [conda-lock](https://conda-incubator.github.io/conda-lock/): Lock versions of dependencies and generate explicit lockfiles.
   - `conda install conda-lock=1.0 --name base --channel conda-forge`

After editing the `environment.devenv.yml` file to add, remove, or update required and optional dependencies, run the following command to re-generate the `environment.conda-lock.yml` and `environment.*.lock` files.

```
make all
```
