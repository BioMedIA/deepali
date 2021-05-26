r"""Functions for collating dataset samples containing tensor decorators."""

from collections import abc
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, List, Mapping, NamedTuple, Sequence, overload

import torch
from torch.utils.data.dataloader import default_collate

from ..core.types import Batch, Dataclass, Sample, is_namedtuple

from .image import Image, ImageBatch
from .sample import sample_field_names, sample_field_value
from .sample import replace_all_sample_field_values


__all__ = ("collate_samples",)


@overload
def collate_samples(batch: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    ...


@overload
def collate_samples(batch: Sequence[Dataclass]) -> Dataclass:
    ...


@overload
def collate_samples(batch: Sequence[NamedTuple]) -> NamedTuple:
    ...


def collate_samples(batch: Sequence[Sample]) -> Batch:
    r"""Collate individual samples into a batch."""
    if not batch:
        raise ValueError("collate_samples() 'batch' must have at least one element")
    item0 = batch[0]
    names = sample_field_names(item0)
    values = []
    for name in names:
        elem0 = sample_field_value(item0, name)
        samples = [elem0]
        is_none = elem0 is None
        for item in batch[1:]:
            value = sample_field_value(item, name)
            if (is_none and value is not None) or (not is_none and value is None):
                raise ValueError(
                    f"collate_samples() 'batch' has some samples with '{name}' set and others without"
                )
            if not isinstance(value, type(elem0)):
                raise TypeError(
                    f"collate_samples() all 'batch' samples for field '{name}' must be of the same type"
                )
            samples.append(value)
        if is_none:
            values.append(None)
        elif isinstance(elem0, Image):
            images: List[Image] = samples
            data = default_collate([image.tensor() for image in images])
            grid = [image.grid() for image in images]
            values.append(ImageBatch(data, grid=grid))
        elif isinstance(elem0, ImageBatch):
            images: List[ImageBatch] = samples
            data = torch.cat([image.tensor() for image in images], dim=0)
            grid = tuple(grid for image in images for grid in image.grids())
            values.append(ImageBatch(data, grid))
        elif isinstance(elem0, (Path, str)):
            values.append(samples)
        elif isinstance(elem0, abc.Mapping) or is_dataclass(elem0) or is_namedtuple(elem0):
            values.append(collate_samples(samples))
        else:
            values.append(default_collate(samples))
    return replace_all_sample_field_values(item0, values)
