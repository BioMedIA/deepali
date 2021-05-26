r"""Custom itertools functions."""

from itertools import repeat
from typing import Any, Iterable, Sequence, Union


def repeat_last(arg: Union[Any, Sequence[Any]], length: int) -> Sequence[Any]:
    r"""Repeat last element in sequence to extend it to the specified length."""
    if not isinstance(arg, str) and not isinstance(arg, Sequence):
        arg = (arg,)
    if not arg:
        raise ValueError("repeat_last() 'arg' must have at least one value to repeat")
    if len(arg) > length:
        raise ValueError("repeat_last() 'arg' sequence length must be at most '{length}'")
    arg = tuple(arg) + (arg[-1],) * (length - len(arg))
    return arg


def zip_longest_repeat_last(*args: Iterable):
    iterators = [iter(it) for it in args]
    num_active = len(iterators)
    if not num_active:
        return
    prev = None
    while True:
        values = []
        for i, it in enumerate(iterators):
            try:
                value = next(it)
            except StopIteration:
                num_active -= 1
                if not num_active:
                    return
                value = prev[i]
                iterators[i] = repeat(value)
            values.append(value)
        prev = tuple(values)
        yield prev
