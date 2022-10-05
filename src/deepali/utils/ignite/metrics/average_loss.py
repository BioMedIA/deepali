from typing import Mapping, Union

from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

import torch
from torch import Tensor


class AverageLoss(Metric):
    r"""Calculates the average loss given batched loss values stored in ``engine.state.output``.

    This metric can be used instead of ``ignite.metrics.loss.Loss`` when the average loss values
    for each example in the input batch are stored as 1-dimensional tensor in the ``engine.state.output``.
    The (transformed) return value of the engines ``process_function`` must be either a tensor of the
    computed loss values, or a dictionary containing the key ``"loss"``.

    """

    @reinit__is_reduced
    def reset(self) -> None:
        self.accumulator = 0
        self.num_examples = 0

    @reinit__is_reduced
    def update(self, output: Union[Tensor, Mapping]) -> None:
        loss = output.get("loss") if isinstance(output, Mapping) else output
        if not isinstance(loss, Tensor):
            raise TypeError("AverageLoss.update() 'output' loss value must be torch.Tensor")
        if loss.ndim == 0:
            self.accumulator += loss.item()
            self.num_examples += 1
        elif loss.dim() == 1:
            self.accumulator += loss.sum().item()
            self.num_examples += loss.shape[0]
        else:
            raise ValueError(
                "AverageLoss.update() 'output' loss value must be scalar or 1-dimensional tensor"
            )

    @sync_all_reduce("accumulator", "num_examples")
    def compute(self) -> float:
        if self.num_examples == 0:
            raise NotComputableError(
                "Loss must have at least one example before it can be computed."
            )
        return self.accumulator / self.num_examples

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)
