r"""Output conversion modules."""

from collections import namedtuple
from typing import Mapping, Sequence, Union

from torch import Tensor
from torch.nn import Module


class ToImmutableOutput(Module):
    r"""Convert input to immutable output container.

    For use with ``torch.utils.tensorboard.SummaryWriter.add_graph`` when model output is list or dict.
    See error message: "Encountering a dict at the output of the tracer might cause the trace to be incorrect,
    this is only valid if the container structure does not change based on the module's inputs. Consider using
    a constant container instead (e.g. for `list`, use a `tuple` instead. for `dict`, use a `NamedTuple` instead).
    If you absolutely need this and know the side effects, pass strict=False to trace() to allow this behavior."

    """

    def __init__(self, recursive: bool = True) -> None:
        super().__init__()
        self.recursive = recursive

    def forward(self, input: Union[Tensor, Sequence, Mapping]) -> Union[Tensor, Sequence, Mapping]:
        return as_immutable_container(input, recursive=self.recursive)

    def extra_repr(self) -> str:
        return f"recursive={self.recursive!r}"


def as_immutable_container(
    arg: Union[Tensor, Sequence, Mapping], recursive: bool = True
) -> Union[Tensor, Sequence, Mapping]:
    r"""Convert mutable container such as dict or list to an immutable container type.

    For use with ``torch.utils.tensorboard.SummaryWriter.add_graph`` when model output is list or dict.
    See error message: "Encountering a dict at the output of the tracer might cause the trace to be incorrect,
    this is only valid if the container structure does not change based on the module's inputs. Consider using
    a constant container instead (e.g. for `list`, use a `tuple` instead. for `dict`, use a `NamedTuple` instead).
    If you absolutely need this and know the side effects, pass strict=False to trace() to allow this behavior."

    """
    if recursive:
        if isinstance(arg, Mapping):
            arg = {key: as_immutable_container(value) for key, value in arg.items()}
        elif isinstance(arg, Sequence):
            arg = [as_immutable_container(value) for value in arg]
    if isinstance(arg, dict):
        output_type = namedtuple("Dict", sorted(arg.keys()))
        return output_type(**arg)
    if isinstance(arg, list):
        return tuple(arg)
    return arg
