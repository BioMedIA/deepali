r"""Functions for collating dataset samples containing tensor decorators."""

from collections import abc
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, List, Mapping, NamedTuple, Sequence, overload

import torch
from torch.utils.data.dataloader import default_collate

from deepali.core.typing import Batch, Dataclass, Sample, is_namedtuple

from .flow import FlowField, FlowFields
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
        elif isinstance(elem0, FlowField):
            FlowFieldsType = type(elem0.batch())
            flow_fields: List[FlowField] = samples
            if any(flow_field.axes() != elem0.axes() for flow_field in flow_fields):
                raise ValueError(
                    f"collate_samples() 'batch' contains '{name}' flow fields with mixed axes"
                )
            data = default_collate([flow_field.tensor() for flow_field in flow_fields])
            grid = tuple(flow_field.grid() for flow_field in flow_fields)
            values.append(FlowFieldsType(data, grid, elem0.axes()))
        elif isinstance(elem0, FlowFields):
            FlowFieldsType = type(elem0)
            flow_fields: List[FlowFields] = samples
            if any(flow_field.axes() != elem0.axes() for flow_field in flow_fields):
                raise ValueError(
                    f"collate_samples() 'batch' contains '{name}' flow fields with mixed axes"
                )
            data = torch.cat([flow_field.tensor() for flow_field in flow_fields], dim=0)
            grid = tuple(grid for flow_field in flow_fields for grid in flow_field.grids())
            values.append(FlowFieldsType(data, grid, elem0.axes()))
        elif isinstance(elem0, Image):
            ImageBatchType = type(elem0.batch())
            images: List[Image] = samples
            data = default_collate([image.tensor() for image in images])
            grid = tuple(image.grid() for image in images)
            values.append(ImageBatchType(data, grid))
        elif isinstance(elem0, ImageBatch):
            ImageBatchType = type(elem0)
            images: List[ImageBatch] = samples
            data = torch.cat([image.tensor() for image in images], dim=0)
            grid = tuple(grid for image in images for grid in image.grids())
            values.append(ImageBatchType(data, grid))
        elif isinstance(elem0, (Path, str)):
            values.append(samples)
        elif isinstance(elem0, abc.Mapping) or is_dataclass(elem0) or is_namedtuple(elem0):
            values.append(collate_samples(samples))
        else:
            values.append(default_collate(samples))
    return replace_all_sample_field_values(item0, values)
