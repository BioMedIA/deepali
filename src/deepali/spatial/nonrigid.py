r"""Non-rigid transformation models."""

from __future__ import annotations

from copy import copy as shallow_copy
from typing import Callable, Optional, TypeVar, Union, cast

import torch
from torch import Tensor

from ..core.grid import Axes, Grid
from ..data.flow import FlowFields
from ..modules import ExpFlow

from .base import NonRigidTransform
from .parametric import ParametricTransform


TDenseVectorFieldTransform = TypeVar(
    "TDenseVectorFieldTransform", bound="DenseVectorFieldTransform"
)


class DenseVectorFieldTransform(ParametricTransform, NonRigidTransform):
    r"""Dense vector field transformation with linear interpolation at non-grid point locations."""

    @property
    def data_shape(self) -> torch.Size:
        r"""Get shape of transformation parameters tensor."""
        grid = self.grid()
        return (grid.ndim,) + grid.shape

    @torch.no_grad()
    def grid_(self: TDenseVectorFieldTransform, grid: Grid) -> TDenseVectorFieldTransform:
        r"""Set sampling grid of transformation domain and codomain.

        If ``self.params`` is a callable, only the grid attribute is updated, and
        the callable must return a tensor of matching size upon next evaluation.

        """
        params = self.params
        if isinstance(params, Tensor):
            prev_grid = self._grid
            grid_axes = Axes.from_grid(grid)
            flow_axes = self.axes()
            flow_grid = prev_grid.reshape(params.shape[2:])
            flow = FlowFields(params, grid=flow_grid, axes=flow_axes)
            flow = flow.sample(grid)
            flow = flow.axes(grid_axes)
            # Change self._grid before self.data_() as it defines self.data_shape
            super().grid_(grid)
            try:
                self.data_(flow.tensor())
            except Exception:
                self._grid = prev_grid
                raise
        else:
            super().grid_(grid)
        return self


class DisplacementFieldTransform(DenseVectorFieldTransform):
    r"""Dense displacement field transformation model."""

    def fit(self, flow: FlowFields, **kwargs) -> DisplacementFieldTransform:
        r"""Fit transformation to a given flow field.

        Args:
            flow: Flow fields to approximate.
            kwargs: Optional keyword arguments are ignored.

        Returns:
            Reference to this transformation.

        Raises:
            RuntimeError: When this transformation has no optimizable parameters.

        """
        params = self.params
        if params is None:
            raise AssertionError(f"{type(self).__name__}.data() 'params' must be set first")
        grid = self.grid()
        if not callable(params):
            grid = self.grid().resize(self.data_shape[:1:-1])
        flow = flow.to(self.device)
        flow = flow.sample(grid)
        flow = flow.axes(Axes.from_grid(grid))
        if callable(params):
            self._fit(flow, **kwargs)
        else:
            self.data_(flow.tensor())
        return self

    def update(self) -> DisplacementFieldTransform:
        r"""Update buffered displacement vector field."""
        super().update()
        u = self.data()
        u = u.view(*u.shape)  # such that named_buffers() returns both 'u' and 'p'
        self.register_buffer("u", u, persistent=False)
        return self


class StationaryVelocityFieldTransform(DenseVectorFieldTransform):
    r"""Dense stationary velocity field transformation."""

    def __init__(
        self,
        grid: Grid,
        groups: Optional[int] = None,
        params: Optional[Union[bool, Tensor, Callable[..., Tensor]]] = True,
        scale: Optional[float] = None,
        steps: Optional[int] = None,
    ) -> None:
        r"""Initialize transformation parameters.

        Args:
            grid: Grid on which to sample velocity field vectors.
            groups: Number of velocity fields. A given image batch can either be deformed by a
                single displacement field, or one separate displacement field for each image in the
                batch, e.g., for group-wise or batched registration. The default is one displacement
                field for all images in the batch, or the batch length N of ``params`` if provided.
            params: Initial parameters of velocity fields of shape ``(N, C, ...X)``, where N must match
                the value of ``groups``, and vector components are the image channels in the order x, y, z.
                Note that a tensor is only registered as optimizable module parameters when of type
                ``torch.nn.Parameter``. When a callable is given instead, it will be called each time the
                model parameters are accessed with the arguments set and returned by ``self.condition()``.
                When a boolean argument is given, a new zero-initialized tensor is created. If ``True``,
                it is registered as optimizable parameter.
            scale: Constant scaling factor of velocity fields.
            steps: Number of scaling and squaring steps.

        """
        super().__init__(grid, groups=groups, params=params)
        self.exp = ExpFlow(scale=scale, steps=steps, align_corners=grid.align_corners())

    def grid_(self, grid: Grid) -> StationaryVelocityFieldTransform:
        r"""Set sampling grid of transformation domain and codomain."""
        super().grid_(grid)
        self.exp.align_corners = grid.align_corners()
        return self

    def inverse(
        self, link: bool = False, update_buffers: bool = False
    ) -> StationaryVelocityFieldTransform:
        r"""Get inverse of this transformation.

        Args:
            link: Whether to inverse transformation keeps a reference to this transformation.
                If ``True``, the ``update()`` function of the inverse function will not recompute
                shared parameters, e.g., parameters obtained by a callable neural network, but
                directly access the parameters from this transformation. Note that when ``False``,
                the inverse transformation will still share parameters, modules, and buffers with
                this transformation, but these shared tensors may be replaced by a call of ``update()``
                (which is implicitly called as pre-forward hook when ``__call__()`` is invoked).
            update_buffers: Whether buffers of inverse transformation should be update after creating
                the shallow copy. If ``False``, the ``update()`` function of the returned inverse
                transformation has to be called before it is used.

        Returns:
            Shallow copy of this transformation with ``exp`` module which uses negative scaling factor
            to scale and square the stationary velocity field to computes the inverse displacement field.

        """
        inv = shallow_copy(self)
        if link:
            inv.link_(self)
        inv.exp = cast(ExpFlow, self.exp).inverse()
        if update_buffers:
            v = getattr(inv, "v", None)
            if v is not None:
                u = inv.exp(v)
                inv.register_buffer("u", u, persistent=False)
        return inv

    def update(self) -> StationaryVelocityFieldTransform:
        r"""Update buffered velocity and displacement vector fields."""
        super().update()
        v = self.data()
        u = self.exp(v)
        v = v.view(*v.shape)  # such that named_buffers() returns both 'v' and 'p'
        self.register_buffer("v", v, persistent=False)
        self.register_buffer("u", u, persistent=False)
        return self
