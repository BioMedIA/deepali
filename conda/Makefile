# Default name of conda environment
NAME = deepali

# Relative path of repository root directory
ROOT = ..

# Name of conda-devenv configuration file
DEVENV = environment.devenv.yml

# Name of conda-lock output file
LOCKFILE = environment.conda-lock.yml

# Detect operating system
ifeq '$(findstring ;,$(PATH))' ';'
    OS_NAME := Windows
else
    OS_NAME := Linux
    OS_NAME := $(shell uname 2>/dev/null || echo $(OS_NAME))
    OS_NAME := $(patsubst CYGWIN%,Linux,$(OS_NAME))
    OS_NAME := $(patsubst MSYS%,Windows,$(OS_NAME))
    OS_NAME := $(patsubst MINGW%,Windows,$(OS_NAME))
endif

# Default platform
ifeq ($(OS_NAME),Linux)
    PLATFORM = linux-64
endif
ifeq ($(OS_NAME),Darwin)
    PLATFORM = osx-64
endif
ifeq ($(OS_NAME),Windows)
    PLATFORM = win-64
endif

# Whether to use editable install of project libraries
EDITABLE = false
PIP_INSTALL_OPTS =
ifeq ($(EDITABLE),true)
	PIP_INSTALL_OPTS += --editable
endif
ifeq ($(EDITABLE),1)
	PIP_INSTALL_OPTS += --editable
endif


.PHONY: all linux osx win clear lock-linux lock-osx lock-win render env create-env update-env prune-env install


all: clear lock-linux lock-osx lock-win render

linux: clear lock-linux render

osx: clear lock-osx render

win: clear lock-win render

clear:
	@echo "Remove all generated files"
	@rm -f $(LOCKFILE) environment.devenv.{linux-64,osx-64,win-64}.yml environment.{linux-64,osx-64,win-64}.{lock,yml}

lock-linux:
	conda devenv --name $(NAME) --file $(DEVENV) --env-manager conda --env-var PLATFORM=linux-64 --print > environment.devenv.linux-64.yml
	conda lock --lockfile $(LOCKFILE) --platform linux-64 --file environment.devenv.linux-64.yml

lock-osx:
	conda devenv --name $(NAME) --file $(DEVENV) --env-manager conda --env-var PLATFORM=osx-64 --print > environment.devenv.osx-64.yml
	conda lock --lockfile $(LOCKFILE) --platform osx-64 --file environment.devenv.osx-64.yml

lock-win:
	conda devenv --name $(NAME) --file $(DEVENV) --env-manager conda --env-var PLATFORM=win-64 --print > environment.devenv.win-64.yml
	conda lock --lockfile $(LOCKFILE) --platform win-64 --file environment.devenv.win-64.yml

render:
	conda lock render $(LOCKFILE) --kind env --filename-template environment.{platform}
	conda lock render $(LOCKFILE) --kind explicit --filename-template environment.{platform}.lock
	@if ! grep -e 'pytorch.*cuda' environment.linux-64.lock > /dev/null 2> /dev/null; then \
		echo "Expected PyTorch with CUDA support for PLATFORM=linux-64. Check conda configuration."; \
	fi


env: create-env install

create-env:
	conda create --name $(NAME) --file environment.$(PLATFORM).lock

update-env:
	conda update --name $(NAME) --file environment.$(PLATFORM).lock --prune

install:
	conda run --name $(NAME) pip install $(PIP_INSTALL_OPTS) ./$(ROOT)[utils]
