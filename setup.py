#!/usr/bin/env python
# fmt: off

# Cannot use pyproject.toml to define project metadata using setuptools>=61
# because it causes an issue with PyTorch Lightning with PyTorch <1.11 
# (https://github.com/PyTorchLightning/pytorch-lightning/issues/12324).
#
# Either an earlier version of setuptools (<=59.5.0) is required, or PyTorch needs
# to be updated to version >=1.11. In order to support older PyTorch versions,
# use setup.py instead of the newer setuptools support of pyproject.toml for now
# (https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html).
#
# See earlier commit c6f684c4bcc7c7fd4820992b25d315d20680992c which replaced
# setup.py by pyproject.toml that needed to be reverted again due to this issue.

import os
from pathlib import Path

from setuptools import find_namespace_packages, setup


project_dir = Path(__file__).absolute().parent
os.chdir(project_dir)


namespace = "deepali"

long_description = Path("README.md").read_text()

packages = find_namespace_packages(where="src")
package_dir={"": "src"}

install_requires = [
    "dacite",
    "pandas",
    "pyyaml",
    "SimpleITK~=2.0",
    "torch>=1.9",
    "typing-extensions",
]

extras_require = {
    "dev": [
        "black",
        "flake8",
        "flake8-black",
        "pytest",
        "torchinfo",
    ],
    "utils": [
        "boto3",
        "pytorch-ignite>=0.4",
        "numpy",
        "scipy",
        "SimpleITK~=2.0",
        "vtk~=9.0",
    ],
}
extras_require["dev"] += extras_require["utils"]
extras_require["all"] = extras_require["dev"] + extras_require["utils"]


setup(
    # Unfortunately there exists a dummy PyPI package named deepali by a user whose
    # name is Deepali Sharma (deepali.sharma@sganalytics.com). The package appears to
    # be unmaintained and was only a test of how to publish a PyPI package. While that
    # package exists, use a package name that slightly differs from this project's name.
    name="deepali-hf",
    description="Image, point set, and surface registration library for PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    license_files=["LICENSE"],
    author="HeartFlow-Imperial College London",
    author_email="andreas.schuh@imperial.ac.uk",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Typing :: Typed",
    ],
    packages=packages,
    package_dir=package_dir,
    python_requires=">=3.7",
    use_scm_version=True,
    setup_requires=["setuptools-scm"],
    install_requires=install_requires,
    extras_require=extras_require,
)
