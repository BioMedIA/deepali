r"""Auxiliary functions for random sampling."""

from typing import Optional
import warnings

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
        Indices of random samples. When ``input`` is a vector, a vector of ``num_samples`` indices
        is returned. Otherwise, a matrix of shape ``(M, num_samples)`` is returned. When ``out``
        is given, the returned tensor is a reference to ``out``.

    """
    if input.ndim == 0 or input.ndim > 2:
        raise ValueError("multinomial() 'input' must be vector or matrix")
    num_candidates = input.size(-1)
    if not replacement and num_candidates < num_samples:
        raise ValueError("multinomial() 'num_samples' cannot be greater than number of categories")
    impl = _multinomial if num_candidates > 2**24 else torch.multinomial
    return impl(input, num_samples, replacement=replacement, generator=generator, out=out)


def _multinomial(
    input: Tensor,
    num_samples: int,
    replacement: bool = False,
    generator: Optional[Generator] = None,
    out: Optional[LongTensor] = None,
) -> LongTensor:
    r"""Sample from a multinomial probability distribution.

    This function can be used for inputs of any size and is unlike ``torch.multinomial`` not limited
    to 2**24 categories at the expense of a less efficient implementation.

    Args:
        input: Input vector of shape ``(N,)`` or matrix ``(M, N)``.
        num_samples: Number of random samples to draw.
        replacement: Whether to sample with or without replacement.
        generator: Random number generator to use.
        out: Pre-allocated output tensor.

    Returns:
        Indices of random samples. When ``input`` is a vector, a vector of ``num_samples`` indices
        is returned. Otherwise, a matrix of shape ``(M, num_samples)`` is returned. When ``out``
        is given, the returned tensor is a reference to ``out``.

    """
    if input.ndim == 0 or input.ndim > 2:
        raise ValueError("_multinomial() 'input' must be vector or matrix")
    num_candidates = input.size(-1)
    # Use inverse transform sampling if the number of candidates is large and replacement=True
    if replacement:
        cdf = input.type(torch.float64).cumsum(dim=-1)
        cdf = cdf.div_(cdf[..., -1:].clone())
        val = torch.rand(
            cdf.shape[:-1] + (num_samples,),
            generator=generator,
            dtype=cdf.dtype,
            device=cdf.device,
        )
        out = torch.searchsorted(cdf, val, out=out).clip_(0, num_candidates - 1)
    # In case of replacement=False, use Gumbel-max trick instead of inverse transform sampling.
    else:
        if num_samples > num_candidates:
            raise ValueError(
                "_multinomial() 'num_samples' cannot be greater than number of categories"
            )
        if generator is not None:
            warnings.warn(
                "_multinomial() with 'replacement=False' currently ignores 'generator' argument"
            )
        gumbels: Tensor = Gumbel(0, 1).sample(input.shape[:-1] + (num_candidates,))
        indices = torch.argsort(gumbels.to(input).add_(input), dim=-1, descending=True)
        indices = indices.narrow(-1, 0, num_samples)
        out = indices if out is None else out.copy_(indices)
    return out
