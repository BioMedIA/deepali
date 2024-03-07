# deepali

[![doi](https://zenodo.org/badge/DOI/10.5281/zenodo.8170161.svg)](https://doi.org/10.5281/zenodo.8170161) [![pypi](https://img.shields.io/pypi/v/hf-deepali)](https://pypi.org/project/hf-deepali/) [![downloads](https://img.shields.io/pypi/dm/hf-deepali)](https://pypistats.org/packages/hf-deepali) [![docs](https://github.com/BioMedIA/deepali/actions/workflows/docs.yml/badge.svg)](https://github.com/BioMedIA/deepali/actions/workflows/docs.yml) [![tests](https://github.com/BioMedIA/deepali/actions/workflows/tests.yml/badge.svg)](https://github.com/BioMedIA/deepali/actions/workflows/tests.yml) [![release](https://github.com/BioMedIA/deepali/actions/workflows/release.yml/badge.svg)](https://github.com/BioMedIA/deepali/actions/workflows/release.yml)

An [open source](https://github.com/BioMedIA/deepali/tree/main/LICENSE) research library for image, point set, and surface registration in [PyTorch], which is developed and maintained by the [HeartFlow-Imperial College London] research lab at the [Biomedical Image Analysis Group].

*[Deepali](https://en.wikipedia.org/wiki/Deepali)* is a Hindu/Sanskrit Indian given name, which means "joy" as in the gratification one may feel working with code built on a modern tensor library with support for automatic differentiation, and "chain of lamps" alluding to the application of the chain rule by *torch.autograd*, the concatenation of spatial coordinate transformations, and furthermore the (sequential) composition of PyTorch modules. In addition, the English words "deep" and "ali(-gnment)" partially contained in this name should highlight that this project is not only suitable for traditional non-learning based registration, but in particular facilitates deep learning based approaches to image alignment.

[HeartFlow-Imperial College London]: https://www.heartflow.com/newsroom/heartflow-announces-collaborative-research-agreement-with-imperial-college-london/
[Biomedical Image Analysis Group]: https://biomedia.doc.ic.ac.uk/


## Overview

At a granular level, *deepali* is a library that consists of the following components:

| **Component**  | **Description** |
| -------------- | --------------- |
| [deepali.core] | Common types, coordinate spaces, and tensor functions. |
| [deepali.data] | PyTorch tensor subclasses, data loader utilities, and datasets. |
| [deepali.losses] | Loss terms and evaluation metrics for image, point set, and surface registration. |
| [deepali.modules] | PyTorch modules without optimizable parameters built on core functions. |
| [deepali.networks] | Common building blocks of machine learning based registration models. We expect that most users may want to develop their own task-specific models. For this, the neural network components defined here may be used alongside ``torch.nn`` and other deep learning libraries (e.g. [MONAI]) to define these custom models. |
| [deepali.spatial] | Spatial transformation models whose parameters are either optimized directly as in traditional registration, or inferred by a machine learning model. |
| [deepali.utils] | Optional auxiliaries for interfacing with external libraries and tools. |


[deepali.core]: https://biomedia.github.io/deepali/reference/core/index.html
[deepali.data]: https://biomedia.github.io/deepali/reference/data/index.html
[deepali.losses]: https://biomedia.github.io/deepali/reference/losses/index.html
[deepali.modules]: https://biomedia.github.io/deepali/reference/modules/index.html
[deepali.networks]: https://biomedia.github.io/deepali/reference/networks/index.html
[deepali.spatial]: https://biomedia.github.io/deepali/reference/spatial/index.html
[deepali.utils]: https://biomedia.github.io/deepali/reference/utils/index.html


## Dependencies

The following lists the main dependencies of this project. For a complete list, please open file [setup.py](setup.py).

- [PyTorch]: The automatic differentiation and deep learning framework powering this project.
- [SimpleITK] (optional): Used by [deepali.data] to read and write images in file formats supported by ITK.
- [nibabel] (optional): Used by [deepali.data] to read and write images in NIfTI file formats if available.
- [NumPy] (optional): Used by [deepali.utils]. Other components use pure PyTorch.
- [VTK] (optional): May be used to read and write point sets and surface meshes.

## Installation

This library is available as [Python package on PyPI](https://pypi.org/project/hf-deepali/) and can be installed with [pip]:

```
pip install hf-deepali
```

The latest development version can be installed directly from the GitHub repository, i.e.,

```
pip install git+https://github.com/BioMedIA/deepali.git
```

Alternatively, it can be installed from a previously cloned local Git repository using

```
git clone https://github.com/BioMedIA/deepali.git && pip install ./deepali
```

This will install missing dependencies in the current Python environment from [PyPI]. To use [conda] for installing the required dependencies (recommended), create a conda environment with pre-installed dependencies **before** running `pip install`. For further information on how to create and manage project dependencies using conda, see [conda/README.md](https://github.com/BioMedIA/deepali/tree/main/conda/README.md).

Additional optional dependencies of the [deepali.utils] library can be installed with the command:

```
pip install hf-deepali[utils]
# or pip install git+https://github.com/BioMedIA/deepali.git#egg=deepali[utils]
```


[conda]: https://docs.conda.io/en/latest/
[pip]: https://pip.pypa.io/en/stable/
[PyPI]: https://pypi.org/
[Miniconda]: https://docs.conda.io/en/latest/miniconda.html

## Citation

Schuh, A., Qiu, H., and HeartFlow Research. *deepali: Image, point set, and surface registration in PyTorch* (2023). [doi:10.5281/zenodo.8170161](https://doi.org/10.5281/zenodo.8170161)

## Contributing

We appreciate all contributions. If you are planning to contribute bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, we would appreciate if you first open an issue and discuss the feature with us. Please also read the [CONTRIBUTING](https://github.com/BioMedIA/deepali/tree/main/CONTRIBUTING.md) notes. The participation in this open source project is subject to the [Code of Conduct](https://github.com/BioMedIA/deepali/tree/main/CODE_OF_CONDUCT.md).

By submitting a pull request to this project, you agree to license your contribution under the Apache license version 2.0 to this project.

Contributors to this project may want to install this package in development mode using

```
git clone https://github.com/BioMedIA/deepali.git
make -C deepali/conda env EDITABLE=1
conda activate deepali
```

This will link the source tree of the package in the Python environment.

## Related projects

Below we list a few projects which are either similar to *deepali* or complement its functionality. We encourage everyone interested in image registration to also explore these projects. You may especially be interested in combining the more general functionality available in *MONAI* with registration components provided by *deepali*.

- **[AIRLab]**: A non-learning based medical image registration framework that took advantage of [PyTorch]'s automatic differentiation and optimizers.
- **[DeepReg]**: A recent and actively developed framework for deep learning based medical image registration built on [TensorFlow]. Due to its YAML based configuration of different models and training settings within the scope of this framework, it should in particular attract users who are less interested in writing their own code, but train registration models provided by DeepReg on their data. As a community-supported open-source toolkit for research and education, you may also consider contributing your models to the framework. DeepReg also forms the basis for a benchmarking environment that will allow comparison of different deep learning models.
- **[Mermaid]**: This [PyTorch] based toolkit facilitates both traditional and learning based registration with a particular focus on diffeomorphic transformation models based on either static or time-dependent velocity fields, including scalar and vector momentum fields. It should be especially of interested to those familiar with the mathematical framework of [Large Deformation Metric Mapping] for [Computational Anatomy].
- **[MONAI]**: This excellent framework for deep learning in healthcare imaging is well maintained and part of the [PyTorch Ecosystem]. It is not specific to medical image registration. In particular, MONAI omits spatial transformation models for use in a registration method, whether optimized directly or integrated in a deep learning model, but contains advanced modules for sampling an image at deformed spatial locations. Common spatial transformations used for data augmentation and general neural network architectures for various tasks are also available in this framework.
- **[Neurite]**: A neural networks toolbox with a focus on medical image analysis in [TensorFlow]. Parts of it have been used in [VoxelMorph], for example.
- **[NITorch]**: A library written by post-docs in John Ashburner's group which is conceptually related to SPM and has a great overlap with *deepali*. Some low-level functionality for applying spatial transformations has more recently been contributed to MONAI as well.
- **[TorchIO]**: A library in the [PyTorch Ecosystem] for efficient loading, preprocessing, augmentation, and patch-based sampling of 3D medical images.
- **[TorchIR]**: PyTorch registration library for deep learning based image registration by Bob de Vos based on his related publications.


[AIRLab]: https://github.com/airlab-unibas/airlab
[DeepReg]: https://github.com/DeepRegNet/DeepReg
[Mermaid]: https://github.com/uncbiag/mermaid
[MONAI]: https://github.com/Project-MONAI/MONAI
[Neurite]: https://github.com/adalca/neurite
[nibabel]: https://nipy.org/nibabel/
[NITorch]: https://github.com/balbasty/nitorch
[NumPy]: https://numpy.org/
[PyTorch]: https://pytorch.org/
[PyTorch Ecosystem]: https://pytorch.org/ecosystem/
[SimpleITK]: https://simpleitk.org/
[TensorFlow]: https://www.tensorflow.org/
[TorchIO]: https://torchio.readthedocs.io/
[TorchIR]: https://github.com/BDdeVos/TorchIR/
[VoxelMorph]: https://github.com/voxelmorph/voxelmorph
[VTK]: https://vtk.org/

[Computational Anatomy]: https://en.wikipedia.org/wiki/Computational_anatomy
[Large Deformation Metric Mapping]: https://en.wikipedia.org/wiki/Large_deformation_diffeomorphic_metric_mapping
