r"""Auxiliary functions for random sampling."""

from typing import Optional

import torch
from torch import Generator, LongTensor, Tensor
from torch.distributions import Gumbel


def multinomial(
    input: Tensor,
    num_samples: int,
    replacement: bool = False,
    generator: Optional[Generator] = None,
    out: Optional[LongTensor] = None,
) -> LongTensor:
    r"""Sample from a multinomial probability distribution.

    Args:
        input: Input vector of shape ``(N,)`` or matrix ``(M, N)``.
        num_samples: Number of random samples to draw.
        replacement: Whether to sample with or without replacement.
        generator: Random number generator to use.
        out: Pre-allocated output tensor.

    Returns:
        Indices of random samples. When ``input`` is a vector, a vector of ``num_sampled`` indices
        is returned. Otherwise, a matrix of shape ``(M, num_samples)`` is returned. When ``out``
        is given, the returned tensor is a reference to ``out``.

    """
    if input.ndim == 0 or input.ndim > 2:
        raise ValueError("multinomial() 'input' must be vector or matrix")
    num_candidates = input.size(-1)
    if not replacement and num_candidates < num_samples:
        raise ValueError("multinomial() 'num_samples' cannot be greater than number of categories")
    # torch.multinomial() works only for at most 2^24 categories.
    if num_candidates <= 2**24:
        out = torch.multinomial(
            input, num_samples, replacement=replacement, generator=generator, out=out
        )
    # Use inverse transform sampling if the number of candidates is large and replacement=True
    elif replacement:
        cdf = torch.cumsum(input / input.sum(dim=-1, keepdim=True), dim=-1)
        cdf = torch.divide(cdf, cdf[..., -1].unsqueeze(-1), out=cdf)
        val = torch.rand(input.shape[:-1] + (num_samples,), dtype=input.dtype, device=input.device)
        out = torch.searchsorted(cdf, val, out=out)
        out = torch.clip(out, 0, num_candidates - 1, out=out)
    # In case of replacement=False, use Gumbel-max trick instead of inverse transform sampling.
    else:
        gumbels: Tensor = Gumbel(0, 1).sample(input.shape[:-1] + (num_candidates,))
        out = torch.argsort(gumbels.add_(input), dim=-1, descending=True)
        out = out.narrow(-1, 0, num_samples)
    return out
