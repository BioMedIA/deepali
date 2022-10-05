r"""Metrics for multi-label classification performance evaluation."""

from typing import Callable, Optional, Sequence, Union

from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

import torch
from torch import Tensor


class MultiLabelScore(Metric):
    r"""Compute a score for each class label."""

    def __init__(
        self,
        score_fn: Callable,
        num_classes: int,
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.score_fn = score_fn
        self.num_classes = num_classes
        self.accumulator = None
        self.num_examples = 0
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.accumulator = torch.zeros(self.num_classes, dtype=torch.float32, device=self._device)
        self.num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Tensor]) -> None:
        y_pred, y = output
        if y_pred.ndim < 2:
            raise ValueError(
                "MultiLabelScore.update() y_pred must have shape (N, C, ...),"
                f" but given {y_pred.shape}"
            )
        if y_pred.shape[1] not in (1, self.num_classes):
            raise ValueError(
                f"MultiLabelScore.update() expected y_pred to have 1 or {self.num_channels} channels"
            )
        if y.ndim + 1 == y_pred.ndim:
            y = y.unsqueeze(1)
        elif y.ndim != y_pred.ndim:
            raise ValueError(
                "MultiLabelScore.update() y_pred must have shape (N, C, ...) and y must have"
                f" shape (N, ...) or (N, 1, ...), but given {y.shape} vs {y_pred.shape}"
            )
        if y.shape != (y_pred.shape[0], 1) + y_pred.shape[2:]:
            raise ValueError("y and y_pred must have compatible shapes.")
        scores = multilabel_score(self.score_fn, y_pred, y, num_classes=self.num_classes)
        self.accumulator += scores
        self.num_examples += y_pred.shape[0]

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


def multilabel_score(
    score_fn, preds: Tensor, labels: Tensor, num_classes: Optional[int] = None
) -> Tensor:
    r"""Evaluate score for each class label."""
    assert labels.shape[1] == 1
    if num_classes is None:
        num_classes = preds.shape[1]
        if num_classes == 1:
            raise ValueError(
                "multilabel_score() 'num_classes' required when 'preds' is not one-hot encoded"
            )
    if preds.shape[1] == num_classes:
        preds = preds.argmax(dim=1, keepdim=True)
    elif preds.shape[1] != 1:
        raise ValueError("multilabel_score() 'preds' must have shape (N, C|1, ..., X)")
    result = torch.zeros(num_classes, dtype=torch.float, device=preds.device)
    for label in range(num_classes):
        y_pred = preds.eq(label).float()
        y = labels.eq(label).float()
        result[label] = score_fn(y_pred, y).mean()
    return result
