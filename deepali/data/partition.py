r"""Auxiliary types and functions to work with dataset partitions."""

from __future__ import annotations

from enum import Enum
from typing import Sequence, Tuple, Union


__all__ = ("Partition", "dataset_split_lengths")


class Partition(Enum):
    r"""Enumeration of dataset partitions / splits."""

    NONE = "none"
    EVAL = "eval"
    TRAIN = "train"
    VALID = "valid"

    @classmethod
    def from_arg(cls, arg: Union[Partition, str, None]) -> Partition:
        r"""Create enumeration value from function argument."""
        if arg is None:
            return cls.NONE
        if arg == "test":
            arg = "eval"
        return cls(arg)


def dataset_split_lengths(
    total: int, ratios: Union[float, Sequence[float]]
) -> Tuple[int, int, int]:
    r"""Split dataset in training, validation, and test subset.

    The output ``lengths`` of this function can be passed to ``torch.utils.data.random_split`` to obtain
    the ``torch.utils.data.dataset.Subset`` for each split.

    See also:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split

    Args:
        total: Total number of samples in dataset.
        ratios: Fraction of samples in each split. When a float or 1-tuple is given,
            the specified fraction of samples is used for training and all remaining
            samples for validation during training. When a 2-tuple is given, the test
            set is assigned no samples. Otherwise, a 3-tuple consisting of ratios
            for training, validation, and test set, respectively should be given.
            The ratios must sum to one.

    Returns:
        lengths: Number of dataset samples in each subset.

    """
    if not isinstance(ratios, float) and len(ratios) == 1:
        ratios = ratios[0]
    if isinstance(ratios, float):
        ratios = (ratios, 1.0 - ratios)
    if len(ratios) == 2:
        ratios += (0.0,)
    elif len(ratios) != 3:
        raise ValueError(
            "dataset_split_lengths() 'ratios' must be float or tuple of length 1, 2, or 3"
        )
    if ratios[0] <= 0 or ratios[0] > 1:
        raise ValueError("dataset_split_lengths() training split ratio must be in (0, 1]")
    if any([ratio < 0 or ratio > 1 for ratio in ratios]):
        raise ValueError("dataset_split_lengths() ratios must be in [0, 1]")
    if sum(ratios) != 1:
        raise ValueError("dataset_split_lengths() 'ratios' must sum to one")
    lengths = [int(round(ratio * total)) for ratio in ratios]
    lengths[2] = max(0, lengths[2] + (total - sum(lengths)))
    lengths[1] = max(0, lengths[1] + (total - sum(lengths)))
    assert sum(lengths) == total
    return tuple(lengths)
