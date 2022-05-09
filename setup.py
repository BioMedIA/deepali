#!/usr/bin/env python
# fmt: off

import os
from pathlib import Path

from setuptools import find_namespace_packages, setup


project_dir = Path(__file__).absolute().parent
os.chdir(project_dir)


namespace = "deepali"

packages = find_namespace_packages(
    where=project_dir,
    include=[namespace + ".*"],
    exclude=["examples", "tests"]
)

install_requires = [
    "dacite",
    "pandas",
    "pyyaml",
    "SimpleITK~=2.0",
    "torch>=1.7",
    "typing-extensions",
]

extras_require = {
    "dev": [
        "black",
        "flake8",
        "pytest",
        "torchinfo",
    ],
    "utils": [
        "boto3",
        "ignite>=0.4",
        "numpy",
        "scipy",
        "SimpleITK~=2.0",
        "vtk~=9.0",  # as of May 2021, not available on PyPI for Python >=3.9
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
    version="0.1.0",
    description="Image, point set, and surface registration library for PyTorch.",
    author="HeartFlow-Imperial College London",
    author_email="andreas.schuh@imperial.ac.uk",
    license="Apache License 2.0",
    license_files=["LICENSE"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Typing :: Typed",
    ],
    namespace_packages=[namespace],
    packages=packages,
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require=extras_require,
)
