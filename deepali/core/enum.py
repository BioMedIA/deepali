r"""Definition of common enumerations."""

from __future__ import annotations

from enum import Enum, IntEnum
import itertools
import re
from typing import Sequence, Set, Tuple, Union


class Sampling(Enum):
    r"""Enumeration of image interpolation modes."""

    AREA = "area"
    BICUBIC = "bicubic"
    BSPLINE = "bspline"  # cubic B-spline
    LINEAR = "linear"  # bilinear or trilinear
    NEAREST = "nearest"

    @classmethod
    def from_arg(cls, arg: Union[Sampling, str, None]) -> Sampling:
        r"""Create enumeration value from function argument."""
        if isinstance(arg, str):
            arg = arg.lower()
        if arg is None or arg in ("default", "bilinear", "trilinear"):
            return cls.LINEAR
        if arg == "nn":
            return cls.NEAREST
        return cls(arg)

    def grid_sample_mode(self, num_spatial_dim: int) -> str:
        r"""Interpolation mode argument for torch.nn.functional.grid_sample() for given number of spatial dimensions."""
        if self == self.LINEAR:
            return "bilinear"
        if self == self.NEAREST:
            return "nearest"
        raise ValueError(
            f"torch.nn.functional.grid_sample() does not support padding mode '{self.value}' for {num_spatial_dim}-dimensional images"
        )

    def interpolate_mode(self, num_spatial_dim: int) -> str:
        r"""Interpolation mode argument for torch.nn.functional.interpolate() for given number of spatial dimensions."""
        if self == self.AREA:
            return "area"
        if self == self.BICUBIC:
            return "bicubic"
        if self == self.LINEAR:
            if num_spatial_dim == 1:
                return "linear"
            if num_spatial_dim == 2:
                return "bilinear"
            if num_spatial_dim == 3:
                return "trilinear"
        if self == self.NEAREST:
            return "nearest"
        raise ValueError(
            f"torch.nn.functional.interpolate() does not support padding mode '{self.value}' for {num_spatial_dim}-dimensional images"
        )


class PaddingMode(Enum):
    r"""Enumeration of image extrapolation modes."""

    NONE = "none"
    CONSTANT = "constant"
    BORDER = "border"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    ZEROS = "zeros"

    @classmethod
    def from_arg(cls, arg: Union[PaddingMode, str, None]) -> PaddingMode:
        r"""Create enumeration value from function argument."""
        if isinstance(arg, str):
            arg = arg.lower()
        if arg is None or arg == "default":
            return cls.ZEROS
        if arg in ("mirror", "reflection"):
            return cls.REFLECT
        if arg == "circular":
            return cls.REPLICATE
        return cls(arg)

    def conv_mode(self, num_spatial_dim: int = 3) -> str:
        r"""Padding mode argument for torch.nn.ConvNd()."""
        if self in (self.CONSTANT, self.ZEROS):
            return "zeros"
        elif self == self.REFLECT:
            return "reflect"
        elif self == self.REPLICATE:
            return "replicate"
        raise ValueError(
            f"torch.nn.Conv{num_spatial_dim}d() does not support padding mode '{self.value}'"
        )

    def grid_sample_mode(self, num_spatial_dim: int) -> str:
        r"""Padding mode argument for torch.nn.functional.grid_sample()."""
        if 2 <= num_spatial_dim <= 3:
            if self in (self.CONSTANT, self.ZEROS):
                return "zeros"
            if self == self.BORDER:
                return "border"
            if self == self.REFLECT:
                return "reflection"
        raise ValueError(
            f"torch.nn.functional.grid_sample() does not support padding mode '{self.value}' for {num_spatial_dim}-dimensional images"
        )

    def pad_mode(self, num_spatial_dim: int) -> str:
        r"""Padding mode argument for torch.nn.functional.pad() for given number of spatial dimensions."""
        if self == self.CONSTANT:
            return "constant"
        elif self == self.REFLECT:
            if 1 <= num_spatial_dim <= 2:
                return "reflect"
        elif self == self.REPLICATE:
            if 1 <= num_spatial_dim <= 3:
                return "replicate"
        raise ValueError(
            f"torch.nn.functional.pad() does not support padding mode '{self.value}' for {num_spatial_dim}-dimensional images"
        )


class SpatialDim(IntEnum):
    r"""Spatial image dimension selector."""

    X = 0
    Y = 1
    Z = 2
    T = 3

    @classmethod
    def from_arg(cls, arg: Union[int, str, SpatialDim]) -> SpatialDim:
        r"""Get enumeration value from function argument."""
        if arg in ("x", "X"):
            return cls.X
        if arg in ("y", "Y"):
            return cls.Y
        if arg in ("z", "Z"):
            return cls.Z
        if arg in ("t", "T"):
            return cls.T
        return cls(arg)

    def tensor_dim(self, ndim: int, channels_last: bool = False) -> int:
        r"""Map spatial dimension identifier to image data tensor dimension."""
        dim = ndim - (2 if channels_last else 1) - self.value
        if (
            channels_last
            and (dim < 1 or dim > ndim - 2)
            or (not channels_last and (dim < 2 or dim > ndim - 1))
        ):
            raise ValueError("SpatialDim.tensor_dim() is out-of-bounds")
        return dim

    def __str__(self) -> str:
        r"""Letter of spatial dimension."""
        return ("x", "y", "z", "t")[self.value]


SpatialDimArg = Union[int, str, SpatialDim]


class SpatialDerivativeKeys(object):
    r"""Auxiliary functions for identifying and enumerating spatial derivatives.

    Spatial derivatives are encoded by a sequence of letters, where each letter
    identifies the spatial dimension (cf. ``SpatialDim``) along which to take the
    derivative. The length of the string encoding determines the order of the
    derivative, i.e., how many times the input image is being derived with
    respect to one or more spatial dimensions.

    """

    @staticmethod
    def check(arg: Union[str, Sequence[str]]):
        r"""Check if given derivatives key is valid."""
        if isinstance(arg, str):
            arg = (arg,)
        for key in arg:
            if not isinstance(key, str):
                raise TypeError("Spatial derivatives key must be str")
            if re.search(r"[^xyzt]", key):
                raise ValueError(
                    "Spatial derivatives key must only contain letters 'x', 'y', 'z', or 't'"
                )

    @classmethod
    def is_valid(cls, arg: Union[str, Sequence[str]]) -> bool:
        r"""Check if given derivatives key is valid."""
        try:
            cls.check(arg)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def is_mixed(key: str) -> bool:
        r"""Whether derivative contains mixed terms."""
        return len(set(key)) > 1

    @staticmethod
    def all(ndim: int, order: Union[int, Sequence[int]]) -> Tuple[str, ...]:
        r"""Unmixed spatial derivatives of specified order."""
        if isinstance(order, int):
            order = (order,)
        keys = []
        dims = [str(SpatialDim(d)) for d in range(ndim)]
        for n in order:
            if n > 0:
                codes = dims
                for _ in range(1, n):
                    codes = [code + letter for code, letter in itertools.product(codes, dims)]
                keys.extend(codes)
        return keys

    @staticmethod
    def unmixed(ndim: int, order: int) -> Tuple[str, ...]:
        r"""Unmixed spatial derivatives of specified order."""
        if order <= 0:
            return ()
        return tuple((str(SpatialDim(d)) * order for d in range(ndim)))

    @classmethod
    def unique(cls, keys: Sequence[str]) -> Set[str]:
        r"""Unique spatial derivatives."""
        return set((cls.sorted(key) for key in keys))

    @classmethod
    def sorted(cls, key: str) -> str:
        r"""Sort letters of spatial dimensions in spatial derivative key."""
        return cls.join(sorted(cls.split(key)))

    @staticmethod
    def order(arg: str) -> int:
        r"""Order of the spatial derivative."""
        return len(arg)

    @classmethod
    def max_order(cls, keys: Sequence[str]) -> int:
        if not keys:
            return 0
        return max((cls.order(key) for key in keys))

    @staticmethod
    def split(arg: str) -> Tuple[SpatialDim, ...]:
        r"""Split spatial derivative key into spatial dimensions enum values."""
        return tuple((SpatialDim.from_arg(letter) for letter in arg))

    @staticmethod
    def join(arg: Sequence[SpatialDim]) -> str:
        r"""Join spatial dimensions to spatial derivative key."""
        return "".join(str(x) for x in arg)
