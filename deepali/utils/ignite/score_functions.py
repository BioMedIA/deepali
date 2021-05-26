r"""Engine state output transformations for use as checkpoint score function."""

from typing import Mapping

from ignite.engine import Engine

from torch import Tensor

from ...core import get_tensor


def negative_loss_score_function(engine: Engine, key: str = "loss") -> Tensor:
    r"""Get negated loss value from ``engine.state.output``."""
    output = engine.state.output
    if isinstance(output, Mapping):
        output = get_tensor(output, key)
    assert isinstance(output, Tensor)
    return -output
