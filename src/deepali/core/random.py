r"""Auxiliary functions for random sampling."""

import random
from typing import Optional, Union

from pkg_resources import parse_version

import torch
from torch import Generator, LongTensor, Tensor


def manual_seed(seed: int):
    r"""Seed all pseudo-random number generators."""
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        ...
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        ...


def str_to_seed(seed: Union[int, str]) -> int:
    r"""Convert string argument to integer seed.

    Args:
        seed: Can be int, string of hexadecimal number, or other string.
            A hexadecimal number can optionally contain dashes. For example,
            can be data UUID (process ID) or MD5 image hash. When the input
            is neither an int nor hexadecimal number, the builtin ``hash``
            function is used to derive a seed from the given string.

    Returns:
        Seed value for random number generator.

    """
    if isinstance(seed, int):
        return seed
    if not isinstance(seed, str):
        raise TypeError("str_to_seed() 'seed' must be int or hexadecimal string")
    try:
        seed = int(seed, 0)
    except ValueError:
        try:
            seed = int(seed.replace("-", ""), 16)
        except ValueError:
            seed = abs(hash(seed))
    return seed


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
    if num_candidates > 2**24:
        impl = _multinomial
    else:
        impl = torch.multinomial
        input = input.float()
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
    out_shape = input.shape[:-1] + (num_samples,)
    if out is not None:
        if not isinstance(out, Tensor):
            raise TypeError("_multinomial() 'out' must be Tensor")
        if out.dtype != torch.int64:
            raise TypeError("_multinomial() 'out' must be int64 tensor")
        if out.shape != out_shape:
            raise ValueError(f"_multinomial() 'out' must have shape {out_shape}")
    # Use inverse transform sampling if the number of candidates is large and replacement=True
    if replacement:
        cdf = input.type(torch.float64).cumsum(dim=-1)
        cdf = cdf.div_(cdf[..., -1:].clone())
        val = torch.rand(out_shape, generator=generator, dtype=cdf.dtype, device=cdf.device)
        out = torch.searchsorted(cdf, val, out=out).clip_(0, num_candidates - 1)
    # In case of replacement=False, use Gumbel-max trick instead of inverse transform sampling.
    else:
        if num_samples > num_candidates:
            raise ValueError(
                "_multinomial() 'num_samples' cannot be greater than number of categories"
            )
        logit = input.log()
        value = torch.rand(
            input.shape[:-1] + (num_candidates,),
            generator=generator,
            dtype=logit.dtype,
            device=input.device,
        )
        value = value.log_().neg_().log_().neg_().add_(logit)
        if parse_version(torch.__version__) < parse_version("1.12"):
            _, index = torch.topk(value, num_samples, dim=-1, sorted=False)
            out = index if out is None else out.copy_(index)
        else:
            if out is None:
                out = torch.empty(out_shape, dtype=torch.int64, device=value.device)
            _ = torch.empty(out_shape, dtype=value.dtype, device=value.device)
            _, out = torch.topk(value, num_samples, dim=-1, sorted=False, out=(_, out))
    return out
