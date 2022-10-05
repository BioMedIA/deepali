r"""Fully connected layers."""

import math
from typing import Optional

from torch.nn import init
from torch.nn.modules import Linear as _Linear


class Linear(_Linear):
    r"""Fully connected layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: Optional[str] = "uniform",
    ) -> None:
        self.bias_init = "uniform" if isinstance(bias, bool) else bias
        self.weight_init = init
        super().__init__(in_features, out_features, bias=bool(bias))

    def reset_parameters(self) -> None:
        # Initialize weights
        if self.weight_init == "uniform":
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        elif self.weight_init == "constant":
            init.constant_(self.weight, 0.1)
        elif self.weight_init == "zeros":
            init.constant_(self.weight, 0.0)
        else:
            raise AssertionError(
                "Linear.reset_parameters() invalid 'init' value: {self.weight_init!r}"
            )
        # Initialize bias
        if self.bias is not None:
            if self.bias_init == "uniform":
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
            elif self.bias_init == "constant":
                init.constant_(self.bias, 0.1)
            elif self.bias_init == "zeros":
                init.constant_(self.bias, 0.0)
            else:
                raise AssertionError(
                    "Linear.reset_parameters() invalid 'bias' value: {self.bias_init!r}"
                )
