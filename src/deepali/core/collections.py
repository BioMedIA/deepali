import re
from typing import Union

from torch import Tensor

from .typing import TensorCollection


__all__ = (
    "TensorCollection",
    "RE_OUTPUT_KEY_INDEX",
    "re_output_key_index",
    "tensor_collection_entry",
    "get_tensor",
)


RE_OUTPUT_KEY_INDEX = r"\[([0-9]+)\]"
re_output_key_index = re.compile(RE_OUTPUT_KEY_INDEX)


def tensor_collection_entry(output: TensorCollection, key: str) -> Union[TensorCollection, Tensor]:
    r"""Get specified output entry."""
    key = re_output_key_index.sub(r".\1", key)
    for index in key.split("."):
        if isinstance(output, (list, tuple)):
            try:
                index = int(index)
            except TypeError:
                raise KeyError(f"invalid output key {key}")
        elif not index or not isinstance(output, dict):
            raise KeyError(f"invalid output key {key}")
        output = output[index]
    return output


def get_tensor(output: TensorCollection, key: str) -> Tensor:
    r"""Get tensor at specified output entry."""
    item = tensor_collection_entry(output, key)
    if not isinstance(item, Tensor):
        raise TypeError(f"get_output_tensor() entry {key} must be Tensor")
    return item
