r"""Loss functions, evaluation metrics, and related utilities."""

import itertools
from typing import Optional, Sequence, Union

import math

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, logsigmoid

from ..core.bspline import evaluate_cubic_bspline
from ..core.enum import SpatialDerivativeKeys
from ..core.grid import Grid
from ..core.image import avg_pool, dot_channels, rand_sample, spatial_derivatives
from ..core.flow import denormalize_flow
from ..core.pointset import transform_grid
from ..core.pointset import transform_points
from ..core.tensor import as_one_hot_tensor, move_dim
from ..core.types import Array, ScalarOrTuple


__all__ = (
    "balanced_binary_cross_entropy_with_logits",
    "binary_cross_entropy_with_logits",
    "label_smoothing",
    "dice_score",
    "dice_loss",
    "kld_loss",
    "lcc_loss",
    "mse_loss",
    "ssd_loss",
    "mi_loss",
    "grad_loss",
    "bending_loss",
    "bending_energy",
    "be_loss",
    "bspline_bending_loss",
    "bspline_bending_energy",
    "bspline_be_loss",
    "curvature_loss",
    "diffusion_loss",
    "divergence_loss",
    "elasticity_loss",
    "focal_loss_with_logits",
    "total_variation_loss",
    "tv_loss",
    "tversky_index",
    "tversky_index_with_logits",
    "tversky_loss",
    "tversky_loss_with_logits",
    "inverse_consistency_loss",
    "masked_loss",
    "reduce_loss",
)


def label_smoothing(
    labels: Tensor,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    alpha: float = 0.1,
) -> Tensor:
    r"""Apply label smoothing to target labels.

    Implements label smoothing as proposed by Muller et al., (2019) in https://arxiv.org/abs/1906.02629v2.

    Args:
        labels: Scalar target labels or one-hot encoded class probabilities.
        num_classes: Number of target labels. If ``None``, use maximum value in ``target`` plus 1
            when a scalar label map is given.
        ignore_index: Ignore index to be kept during the expansion. The locations of the index
            value in the labels image is stored in the corresponding locations across all channels
            so that this location can be ignored across all channels later, e.g. in Dice computation.
            This argument must be ``None`` if ``labels`` has ``C == num_channels``.
        alpha: Label smoothing factor in [0, 1]. If zero, no label smoothing is done.

    Returns:
        Multi-channel tensor of smoothed target class probabilities.

    """
    if not isinstance(labels, Tensor):
        raise TypeError("label_smoothing() 'labels' must be Tensor")
    if labels.ndim < 4:
        raise ValueError("label_smoothing() 'labels' must be tensor of shape (N, C, ..., X)")
    if labels.shape[1] == 1:
        target = as_one_hot_tensor(
            labels, num_classes, ignore_index=ignore_index, dtype=torch.float32
        )
    else:
        target = labels.float()
    if alpha > 0:
        target = (1 - alpha) * target + alpha * (1 - target) / (target.size(1) - 1)
    return target


def balanced_binary_cross_entropy_with_logits(
    logits: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Balanced binary cross entropy.

    Shruti Jadon (2020) A survey of loss functions for semantic segmentation.
    https://arxiv.org/abs/2006.14822

    Args:
        logits: Logits of binary predictions as tensor of shape ``(N, 1, ..., X)``.
        target: Target label probabilities as tensor of shape ``(N, 1, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, 1, ..., X)``.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        Balanced binary cross entropy (bBCE). If ``reduction="none"``, the returned tensor has shape
        ``(N, 1, ..., X)`` with bBCE for each element. Otherwise, it is reduced into a scalar.

    """
    if logits.ndim < 3 or logits.shape[1] != 1:
        raise ValueError(
            "balanced_binary_cross_entropy_with_logits() 'logits' must have shape (N, 1, ..., X)"
        )
    if target.ndim < 3 or target.shape[1] != 1:
        raise ValueError(
            "balanced_binary_cross_entropy_with_logits() 'target' must have shape (N, 1, ..., X)"
        )
    if logits.shape[0] != target.shape[0]:
        raise ValueError(
            "balanced_binary_cross_entropy_with_logits() 'logits' and 'target' must have matching batch size"
        )
    neg_weight = target.flatten(1).mean(-1).reshape((-1,) + (1,) * (target.ndim - 1))
    pos_weight = 1 - neg_weight
    log_y_pred: Tensor = logsigmoid(logits)
    loss_pos = -log_y_pred.mul(target)
    loss_neg = logits.sub(log_y_pred).mul(1 - target)
    loss = loss_pos.mul(pos_weight).add(loss_neg.mul(neg_weight))
    loss = masked_loss(loss, weight, "balanced_binary_cross_entropy_with_logits", inplace=True)
    loss = reduce_loss(loss, reduction=reduction)
    return loss


def dice_score(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    epsilon: float = 1e-15,
    reduction: str = "mean",
) -> Tensor:
    r"""Soft Dice similarity of multi-channel classification result.

    Args:
        input: Normalized logits of binary predictions as tensor of shape ``(N, C, ..., X)``.
        target: Target label probabilities as tensor of shape ``(N, C, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, C, ..., X)``.
        epsilon: Small constant used to avoid division by zero.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        Dice similarity coefficient (DSC). If ``reduction="none"``, the returned tensor has shape
        ``(N, C)`` with DSC for each batch. Otherwise, the DSC scores are reduced into a scalar.

    """
    if not isinstance(input, Tensor):
        raise TypeError("dice_score() 'input' must be torch.Tensor")
    if not isinstance(target, Tensor):
        raise TypeError("dice_score() 'target' must be torch.Tensor")
    if input.dim() < 3:
        raise ValueError("dice_score() 'input' must be tensor of shape (N, C, ..., X)")
    if input.shape != target.shape:
        raise ValueError("dice_score() 'input' and 'target' must have identical shape")
    y_pred = input.float()
    y = target.float()
    intersection = dot_channels(y_pred, y, weight=weight)
    denominator = dot_channels(y_pred, y_pred, weight=weight) + dot_channels(y, y, weight=weight)
    loss = intersection.mul_(2).add_(epsilon).div(denominator.add_(epsilon))
    loss = reduce_loss(loss, reduction)
    return loss


def dice_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    epsilon: float = 1e-15,
    reduction: str = "mean",
) -> Tensor:
    r"""One minus soft Dice similarity of multi-channel classification result.

    Args:
        input: Normalized logits of binary predictions as tensor of shape ``(N, C, ..., X)``.
        target: Target label probabilities as tensor of shape ``(N, C, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, C, ..., X)``.
        epsilon: Small constant used to avoid division by zero.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        One minus Dice similarity coefficient (DSC). If ``reduction="none"``, the returned tensor has shape
        ``(N, C)`` with DSC for each batch. Otherwise, the DSC scores are reduced into a scalar.

    """
    dsc = dice_score(input, target, weight=weight, epsilon=epsilon, reduction="none")
    loss = reduce_loss(1 - dsc, reduction)
    return loss


def tversky_index(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    epsilon: float = 1e-15,
    normalize: bool = False,
    binarize: bool = False,
    reduction: str = "mean",
) -> Tensor:
    r"""Tversky index as described in https://arxiv.org/abs/1706.05721.

    Args:
        input: Binary predictions as tensor of shape ``(N, 1, ..., X)``
            or multi-class prediction tensor of shape ``(N, C, ..., X)``.
        target: Target labels as tensor of shape ``(N, ..., X)``, binary classification target
            of shape ``(N, 1, ..., X)``, or one-hot encoded tensor of shape ``(N, C, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, ..., X)`` or ``(N, 1|C, ..., X)``..
        alpha: False positives multiplier. Set to ``1 - beta`` if ``None``.
        beta: False negatives multiplier.
        epsilon: Constant used to avoid division by zero.
        normalize: Whether to normalize ``input`` predictions using ``sigmoid`` or ``softmax``.
        binarize: Whether to round normalized predictions to 0 or 1, respectively. If ``False``,
            soft normalized predictions (and target scores) are used. In order for the Tversky
            index to be identical to Dice, this option must be set to ``True`` and ``alpha=beta=0.5``.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        Tversky index (TI). If ``reduction="none"``, the returned tensor has shape ``(N, C)``
        with TI for each batch. Otherwise, the TI values are reduced into a scalar.

    """
    if alpha is None and beta is None:
        alpha = beta = 0.5
    elif alpha is None:
        alpha = 1 - beta
    elif beta is None:
        beta = 1 - alpha
    if not isinstance(input, Tensor):
        raise TypeError("tversky_index() 'input' must be torch.Tensor")
    if not isinstance(target, Tensor):
        raise TypeError("tversky_index() 'target' must be torch.Tensor")
    if input.ndim < 3 or input.shape[1] < 1:
        raise ValueError(
            "tversky_index() 'input' must be have shape (N, 1, ..., X) or (N, C>1, ..., X)"
        )
    if target.ndim < 2 or target.shape[1] < 1:
        raise ValueError(
            "tversky_index() 'target' must be have shape (N, ..., X), (N, 1, ..., X), or (N, C>1, ..., X)"
        )
    if target.shape[0] != input.shape[0]:
        raise ValueError(
            "tversky_index() 'input' and 'target' batch size must be identical"
            f", got {input.shape[0]} != {target.shape[0]}"
        )
    input = input.float()
    if input.shape[1] == 1:
        y_pred = input.sigmoid() if normalize else input
    else:
        y_pred = input.softmax(1) if normalize else input
    if binarize:
        y_pred = y_pred.round()
    num_classes = max(2, y_pred.shape[1])
    if target.ndim == input.ndim:
        y = target.type(y_pred.dtype)
        if target.shape[1] == 1:
            if num_classes > 2:
                raise ValueError(
                    "tversky_index() 'target' has shape (N, 1, ..., X)"
                    f", but 'input' is multi-class prediction (C={num_classes})"
                )
            if y_pred.shape[1] == 2:
                y_pred = y_pred.narrow(1, 1, 1)
        else:
            if target.shape[1] != num_classes:
                raise ValueError(
                    "tversky_index() 'target' has shape (N, C, ..., X)"
                    f", but C does not match 'input' with C={num_classes}"
                )
            if y_pred.shape[1] == 1:
                y = y.narrow(1, 1, 1)
        if binarize:
            y = y.round()
    elif target.ndim + 1 == y_pred.ndim:
        if num_classes == 2 and y_pred.shape[1] == 1:
            y = target.unsqueeze(1).ge(0.5).type(y_pred.dtype)
            if binarize:
                y = y.round()
        else:
            y = as_one_hot_tensor(target, num_classes, dtype=y_pred.dtype)
    else:
        raise ValueError(
            "tversky_index() 'target' must be tensor of shape (N, ..., X) or (N, C, ... X)"
        )
    if y.shape != y_pred.shape:
        raise ValueError("tversky_index() 'input' and 'target' shapes must be compatible")
    if weight is not None:
        if weight.ndim + 1 == y.ndim:
            weight = weight.unsqueeze(1)
        if weight.ndim != y.ndim:
            raise ValueError("tversky_index() 'weight' shape must be (N, ..., X) or (N, C, ..., X)")
        if weight.shape[0] != target.shape[0]:
            raise ValueError(
                "tversky_index() 'weight' batch size does not match 'input' and 'target'"
            )
        if weight.shape[1] == 1:
            weight = weight.repeat((1,) + (num_classes,) + (1,) * (weight.ndim - 2))
        if weight.shape != y.shape:
            raise ValueError(
                "tversky_index() 'weight' shape must be compatible with 'input' and 'target'"
            )
    intersection = dot_channels(y_pred, y, weight=weight)
    fps = dot_channels(y_pred, 1 - y, weight=weight).mul_(alpha)
    fns = dot_channels(1 - y_pred, y, weight=weight).mul_(beta)
    numerator = intersection.add_(epsilon)
    denominator = numerator.add(fps).add(fns)
    loss = numerator.div(denominator)
    loss = reduce_loss(loss, reduction)
    return loss


def tversky_index_with_logits(
    logits: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    epsilon: float = 1e-15,
    binarize: bool = False,
    reduction: str = "mean",
) -> Tensor:
    r"""Tversky index as described in https://arxiv.org/abs/1706.05721.

    Args:
        logits: Binary prediction logits as tensor of shape ``(N, 1, ..., X)``.
        target: Target labels as tensor of shape ``(N, ..., X)`` or ``(N, 1, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, ..., X)`` or ``(N, 1, ..., X)``.
        alpha: False positives multiplier. Set to ``1 - beta`` if ``None``.
        beta: False negatives multiplier.
        epsilon: Constant used to avoid division by zero.
        normalize: Whether to normalize ``input`` predictions using ``sigmoid`` or ``softmax``.
        binarize: Whether to round normalized predictions to 0 or 1, respectively. If ``False``,
            soft normalized predictions (and target scores) are used. In order for the Tversky
            index to be identical to Dice, this option must be set to ``True`` and ``alpha=beta=0.5``.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        Tversky index (TI). If ``reduction="none"``, the returned tensor has shape ``(N, 1)``
        with TI for each batch. Otherwise, the TI values are reduced into a scalar.

    """
    return tversky_index(
        logits,
        target,
        weight=weight,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        normalize=True,
        binarize=binarize,
        reduction=reduction,
    )


def tversky_loss(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    epsilon: float = 1e-15,
    normalize: bool = False,
    binarize: bool = False,
    reduction: str = "mean",
) -> Tensor:
    r"""Tversky loss as described in https://arxiv.org/abs/1706.05721.

    Args:
        input: Normalized logits of binary predictions as tensor of shape ``(N, C, ..., X)``.
        target: Target label probabilities as tensor of shape ``(N, C, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, ..., X)`` or ``(N, 1, ..., X)``.
        alpha: False positives multiplier. Set to ``1 - beta`` if ``None``.
        beta: False negatives multiplier.
        gamma: Exponent used for focal Tverksy loss.
        epsilon: Constant used to avoid division by zero.
        normalize: Whether to normalize ``input`` predictions using ``sigmoid`` or ``softmax``.
        binarize: Whether to round normalized predictions to 0 or 1, respectively. If ``False``,
            soft normalized predictions (and target scores) are used. In order for the Tversky
            index to be identical to Dice, this option must be set to ``True`` and ``alpha=beta=0.5``.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        One minus Tversky index (TI) to the power of gamma. If ``reduction="none"``, the returned
        tensor has shape ``(N, C)`` with the loss for each batch. Otherwise, a scalar is returned.

    """
    ti = tversky_index(
        input,
        target,
        weight=weight,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        epsilon=epsilon,
        normalize=normalize,
        binarize=binarize,
        reduction="none",
    )
    one = torch.tensor(1, dtype=ti.dtype, device=ti.device)
    loss = one.sub(ti)
    if gamma:
        if gamma > 1:
            loss = loss.pow_(gamma)
        elif gamma < 1:
            raise ValueError("tversky_loss() 'gamma' must be greater than or equal to 1")
    loss = reduce_loss(loss, reduction)
    return loss


def tversky_loss_with_logits(
    logits: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    epsilon: float = 1e-15,
    binarize: bool = False,
    reduction: str = "mean",
) -> Tensor:
    r"""Tversky loss as described in https://arxiv.org/abs/1706.05721.

    Args:
        logits: Binary prediction logits as tensor of shape ``(N, 1, ..., X)``.
        target: Target labels as tensor of shape ``(N, ..., X)`` or ``(N, 1, ..., X)``.
        weight: Voxelwise label weight tensor of shape ``(N, ..., X)`` or ``(N, 1, ..., X)``.
        alpha: False positives multiplier. Set to ``1 - beta`` if ``None``.
        beta: False negatives multiplier.
        gamma: Exponent used for focal Tverksy loss.
        epsilon: Constant used to avoid division by zero.
        normalize: Whether to normalize ``input`` predictions using ``sigmoid`` or ``softmax``.
        binarize: Whether to round normalized predictions to 0 or 1, respectively. If ``False``,
            soft normalized predictions (and target scores) are used. In order for the Tversky
            index to be identical to Dice, this option must be set to ``True`` and ``alpha=beta=0.5``.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        One minus Tversky index (TI) to the power of gamma. If ``reduction="none"``, the returned
        tensor has shape ``(N, C)`` with the loss for each batch. Otherwise, a scalar is returned.

    """
    return tversky_loss(
        logits,
        target,
        weight=weight,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        epsilon=epsilon,
        normalize=True,
        binarize=binarize,
        reduction=reduction,
    )


def focal_loss_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
) -> Tensor:
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        input: Logits of the predictions for each example.
        target: A tensor with the same shape as ``input``. Stores the binary classification
            label for each element in inputs (0 for the negative class and 1 for the positive class).
        weight: Multiplicative mask tensor with same shape as ``input``.
        alpha: Weighting factor in [0, 1] to balance positive vs negative examples or -1 for ignore.
        gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
        reduction: Either ``none``, ``mean``, or ``sum``.

    Returns:
        Loss tensor with the reduction option applied.

    """
    # https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Focal-Loss
    bce = binary_cross_entropy_with_logits(input, target, reduction="none")
    one = torch.tensor(1, dtype=bce.dtype, device=bce.device)
    loss = one.sub(torch.exp(-bce)).pow(gamma).mul(bce)
    if alpha >= 0:
        if alpha > 1:
            raise ValueError("focal_loss() 'alpha' must be in [0, 1]")
        loss = target.mul(alpha).add(one.sub(target).mul(1 - alpha)).mul(loss)
    loss = masked_loss(loss, weight, "focal_loss_with_logits", inplace=True)
    loss = reduce_loss(loss, reduction)
    return loss


def kld_loss(mean: Tensor, logvar: Tensor, reduction: str = "mean") -> Tensor:
    r"""Kullback-Leibler divergence in case of zero-mean and isotropic unit variance normal prior distribution.

    Kingma and Welling, Auto-Encoding Variational Bayes, ICLR 2014, https://arxiv.org/abs/1312.6114 (Appendix B).

    """
    loss = mean.square().add_(logvar.exp()).sub_(1).sub_(logvar)
    loss = reduce_loss(loss, reduction)
    loss = loss.mul_(0.5)
    return loss


def lcc_loss(
    input: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    kernel_size: ScalarOrTuple[int] = 7,
    epsilon: float = 1e-15,
    reduction: str = "mean",
) -> Tensor:
    r"""Local normalized cross correlation.

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask tensor with same shape as ``input``.
        kernel_size: Local rectangular window size in number of grid points.
        epsilon: Small constant added to denominator to avoid division by zero.
        reduction: Whether to compute "mean" or "sum" over all grid points. If "none",
            output tensor shape is equal to the shape of the input tensors given an odd
            kernel size.

    Returns:
        Negative local normalized cross correlation plus one.

    """

    def pool(data: Tensor) -> Tensor:
        return avg_pool(
            data,
            kernel_size=kernel_size,
            stride=1,
            padding=None,
            count_include_pad=False,
        )

    if not torch.is_tensor(input):
        raise TypeError("lcc_loss() 'input' must be tensor")
    if not torch.is_tensor(target):
        raise TypeError("lcc_loss() 'target' must be tensor")
    if input.shape != target.shape:
        raise ValueError("lcc_loss() 'input' must have same shape as 'target'")
    input = input.float()
    target = target.float()
    x = input - pool(input)
    y = target - pool(target)
    a = pool(x.mul(y))
    b = pool(x.square())
    c = pool(y.square())
    lcc = a.square().div_(b.mul(c).add_(epsilon))  # A^2 / BC cf. Avants et al., 2007, eq 5
    loss = lcc.mul_(-1).add_(1)  # minimize 1 - lcc, where loss range is [0, 1]
    loss = masked_loss(loss, mask, "lcc_loss")
    loss = reduce_loss(loss, reduction, mask)
    return loss


def mse_loss(
    input: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    norm: Optional[Union[float, Tensor]] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Average normalized squared differences.

    This loss is equivalent to `ssd_loss`, except that the default `reduction` is "mean".

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.

    Returns:
        Average normalized squared differences.

    """
    return ssd_loss(input, target, mask=mask, norm=norm, reduction=reduction)


def ssd_loss(
    input: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    norm: Optional[Union[float, Tensor]] = None,
    reduction: str = "sum",
) -> Tensor:
    r"""Sum of normalized squared differences.

    The SSD loss is equivalent to MSE, except that an optional overlap mask is supported and
    that the loss value is optionally multiplied by a normalization constant. Moreover, by default
    the sum instead of the mean of per-element loss values is returned (cf. ``reduction``).
    The value returned by ``max_difference(source, target).square()`` can be used as normalization
    factor, which is equvalent to first normalizing the images to [0, 1].

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.

    Returns:
        Sum of normalized squared differences.

    """
    if not isinstance(input, Tensor):
        raise TypeError("ssd_loss() 'input' must be tensor")
    if not isinstance(target, Tensor):
        raise TypeError("ssd_loss() 'target' must be tensor")
    if input.shape != target.shape:
        raise ValueError("ssd_loss() 'input' must have same shape as 'target'")
    loss = input.sub(target).square()
    loss = masked_loss(loss, mask, "ssd_loss")
    loss = reduce_loss(loss, reduction, mask)
    if norm is not None:
        norm = torch.as_tensor(norm, dtype=loss.dtype, device=loss.device).squeeze()
        if not norm.ndim == 0:
            raise ValueError("ssd_loss() 'norm' must be scalar")
        if norm > 0:
            loss = loss.div_(norm)
    return loss


def mi_loss(
    input: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    num_bins: Optional[int] = None,
    num_samples: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    normalized: bool = False,
) -> Tensor:
    r"""Calculate mutual information loss using Parzen window density and entropy estimations.

    References:
    - Qiu, H., Qin, C., Schuh, A., Hammernik, K.: Learning Diffeomorphic and Modality-invariant Registration using B-splines. Medical Imaging with Deep Learning. (2021).
    - Thévenaz, P., Unser, M.: Optimization of mutual information for multiresolution image registration. IEEE Trans. Image Process. 9, 2083–2099 (2000).

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Region of interest mask with same shape as ``input``.
        vmin: Minimal intensity value the joint and marginal density is estimated.
        vmax: Maximal intensity value the joint and marginal density is estimated.
        num_bins: Number of bin edges to discretize the density estimation.
        num_samples: Number of voxels in the image domain randomly sampled to compute the loss,
            ignored if `sample_ratio` is also set.
        sample_ratio: Ratio of voxels in the image domain randomly sampled to compute the loss.
        normalized: Calculate Normalized Mutual Information instead of Mutual Information if True.

    Returns:
        Negative mutual information. If ``normalized=True``, 2 is added such that the loss value is in [0, 1].

    """

    if target.ndim < 3:
        raise ValueError("mi_loss() 'target' must be tensor of shape (N, C, ..., X)")
    if input.shape != target.shape:
        raise ValueError("ssd_loss() 'input' must have same shape as 'target'")

    if vmin is None:
        vmin = torch.min(input.min(), target.min()).item()
    if vmax is None:
        vmax = torch.max(input.max(), target.max()).item()
    if num_bins is None:
        num_bins = 64
    elif num_bins == "auto":
        raise NotImplementedError("mi_loss() automatically setting num_bins based on dynamic range of input")

    # Flatten spatial dimensions of inputs
    shape = target.shape
    input = input.flatten(2)
    target = target.flatten(2)

    if mask is not None:
        if mask.ndim < 3 or mask.shape[2:] != shape[2:] or mask.shape[1] != 1:
            raise ValueError(
                "mi_loss() 'mask' must be tensor of shape (1|N, 1, ..., X) with spatial dimensions matching 'target'"
            )
        mask = mask.flatten(2)

    # Random image samples, optionally weighted by mask
    if sample_ratio is not None:
        if num_samples is not None:
            raise ValueError("mi_loss() 'num_samples' and 'sample_ratio' are mutually exclusive")
        if sample_ratio <= 0 or sample_ratio > 1:
            raise ValueError("mi_loss() 'sample_ratio' must be in open-closed interval (0, 1]")
        num_samples = max(1, int(sample_ratio * target.shape[2:].numel()))
    if num_samples is not None:
        input, target = rand_sample([input, target], num_samples, mask=mask, replacement=True)
    elif mask is not None:
        input = input.mul(mask)
        target = target.mul(mask)

    # set the bin edges and Gaussian kernel std
    bin_width = (vmax - vmin) / num_bins  # FWHM is one bin width
    bin_center = torch.linspace(vmin, vmax, num_bins, requires_grad=False)
    bin_center = bin_center.unsqueeze(1).type_as(input)

    # calculate Parzen window function response
    pw_sdev = bin_width * (1 / (2 * math.sqrt(2 * math.log(2))))
    pw_norm = 1 / math.sqrt(2 * math.pi) * pw_sdev

    def parzen_window_fn(x: Tensor) -> Tensor:
        return x.sub(bin_center).square().div(2 * pw_sdev**2).neg().exp().mul(pw_norm)

    pw_input = parzen_window_fn(input)  # (N, #bins, H*W*D)
    pw_target = parzen_window_fn(target)

    # calculate joint histogram
    hist_joint = pw_input.bmm(pw_target.transpose(1, 2))  # (N, #bins, #bins)
    hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5

    # joint and marginal distributions
    p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)
    p_input = torch.sum(p_joint, dim=2)
    p_target = torch.sum(p_joint, dim=1)

    # calculate entropy
    ent_input = -torch.sum(p_input * torch.log(p_input + 1e-5), dim=1)  # (N,1)
    ent_target = -torch.sum(p_target * torch.log(p_target + 1e-5), dim=1)  # (N,1)
    ent_joint = -torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

    if normalized:
        loss = 2 - torch.mean((ent_input + ent_target) / ent_joint)
    else:
        loss = torch.mean(ent_input + ent_target - ent_joint).neg()
    return loss


def nmi_loss(
    input: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    num_bins: Optional[int] = None,
    num_samples: Optional[int] = None,
    sample_ratio: Optional[float] = None,
) -> Tensor:
    return mi_loss(
        input,
        target,
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        num_bins=num_bins,
        num_samples=num_samples,
        sample_ratio=sample_ratio,
        normalized=True,
    )


def grad_loss(
    u: Tensor,
    p: Union[int, float] = 2,
    q: Optional[Union[int, float]] = 1,
    spacing: Optional[Array] = None,
    sigma: Optional[float] = None,
    mode: str = "central",
    which: Optional[Union[str, Sequence[str]]] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Loss term based on p-norm of spatial gradient of vector fields.

    The ``p`` and ``q`` parameters can be used to specify which norm to compute, i.e., ``sum(abs(du)**p)**q``,
    where ``du`` are the 1st order spatial derivative of the input vector fields ``u`` computed using a finite
    difference scheme and optionally normalized using a specified grid ``spacing``.

    This regularization loss is the basis, for example, for total variation and diffusion penalties.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        p: The order of the gradient norm. When ``p = 0``, the partial derivatives are summed up.
        q: Power parameter of gradient norm. If ``None``, then ``q = 1 / p``. If ``q == 0``, the
            absolute value of the sum of partial derivatives is computed at each grid point.
        spacing: Sampling grid spacing.
        sigma: Standard deviation of Gaussian in grid units.
        mode: Method used to approximate spatial derivatives. See ``spatial_derivatives()``.
        which: String codes of spatial deriviatives to compute. See ``SpatialDerivativeKeys``.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Spatial gradient loss of vector fields.

    """
    if u.ndim < 4:
        # No loss for homogeneous coordinate transformations
        if reduction == "none":
            raise NotImplementedError(
                "grad_loss() not implemented for linear transformation and 'reduction'='none'"
            )
        return torch.tensor(0, dtype=u.dtype, device=u.device)
    D = u.shape[1]
    if u.ndim - 2 != D:
        raise ValueError(f"grad_loss() 'u' must be tensor of shape (N, {u.ndim - 2}, ..., X)")
    if q is None:
        q = 1.0 / p
    derivs = spatial_derivatives(u, mode=mode, which=which, order=1, sigma=sigma, spacing=spacing)
    loss = torch.cat([deriv.unsqueeze(-1) for deriv in derivs.values()], dim=-1)
    if p == 1:
        loss = loss.abs()
    elif p != 0:
        if p % 2 == 0:
            loss = loss.pow(p)
        else:
            loss = loss.abs().pow_(p)
    loss = loss.sum(dim=-1)
    if q == 0:
        loss.abs_()
    elif q != 1:
        loss.pow_(q)
    loss = reduce_loss(loss, reduction)
    return loss


def bending_loss(
    u: Tensor,
    spacing: Optional[Array] = None,
    sigma: Optional[float] = None,
    mode: str = "sobel",
    reduction: str = "mean",
) -> Tensor:
    r"""Bending energy of vector fields.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        spacing: Sampling grid spacing.
        sigma: Standard deviation of Gaussian in grid units (cf. ``spatial_derivatives()``).
        mode: Method used to approximate spatial derivatives (cf. ``spatial_derivatives()``).
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Bending energy.

    """
    if u.ndim < 4:
        # No loss for homogeneous coordinate transformations
        if reduction == "none":
            raise NotImplementedError(
                "bending_energy() not implemented for linear transformation and 'reduction'='none'"
            )
        return torch.tensor(0, dtype=u.dtype, device=u.device)
    D = u.shape[1]
    if u.ndim - 2 != D:
        raise ValueError(f"bending_energy() 'u' must be tensor of shape (N, {u.ndim - 2}, ..., X)")
    which = SpatialDerivativeKeys.unique(SpatialDerivativeKeys.all(ndim=D, order=2))
    derivs = spatial_derivatives(u, mode=mode, which=which, sigma=sigma, spacing=spacing)
    derivs = torch.cat([deriv.unsqueeze(-1) for deriv in derivs.values()], dim=-1)
    derivs *= torch.tensor(
        [2 if SpatialDerivativeKeys.is_mixed(key) else 1 for key in which],
        device=u.device,
    )
    loss = derivs.pow(2).sum(-1)
    loss = reduce_loss(loss, reduction)
    return loss


be_loss = bending_loss
bending_energy = bending_loss


def bspline_bending_loss(
    data: Tensor, stride: ScalarOrTuple[int] = 1, reduction: str = "mean"
) -> Tensor:
    r"""Evaluate bending energy of cubic B-spline function, e.g., spatial free-form deformation.

    Args:
        data: Cubic B-spline interpolation coefficients as tensor of shape ``(N, C, ..., X)``.
        stride: Number of points between control points at which to evaluate bending energy, plus one.
            If a sequence of values is given, these must be the strides for the different spatial
            dimensions in the order ``(sx, ...)``. A stride of 1 is equivalent to evaluating bending
            energy only at the usually coarser resolution of the control point grid. It should be noted
            that the stride need not match the stride used to densely sample the spline deformation field
            at a given fixed target image resolution.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Bending energy of cubic B-spline.

    """
    if not isinstance(data, Tensor):
        raise TypeError("bspline_bending_loss() 'data' must be torch.Tensor")
    if not torch.is_floating_point(data):
        raise TypeError("bspline_bending_loss() 'data' must have floating point dtype")
    if data.ndim < 3:
        raise ValueError("bspline_bending_loss() 'data' must have shape (N, C, ..., X)")
    D = data.ndim - 2
    C = data.shape[1]
    if C != D:
        raise ValueError(
            f"bspline_bending_loss() 'data' number of channels ({C})"
            f" does not match number of spatial dimensions ({D})"
        )
    energy: Optional[Tensor] = None
    derivs = SpatialDerivativeKeys.all(D, order=2)
    derivs = SpatialDerivativeKeys.unique(derivs)
    npoints = 0
    for deriv in derivs:
        derivative = [0] * D
        for sdim in SpatialDerivativeKeys.split(deriv):
            derivative[sdim] += 1
        assert sum(derivative) == 2
        values = evaluate_cubic_bspline(data, stride=stride, derivative=derivative).square()
        if reduction != "none":
            npoints = values.shape[2:].numel()
            values = values.sum()
        if not SpatialDerivativeKeys.is_mixed(deriv):
            values = values.mul_(2)
        energy = values if energy is None else energy.add_(values)
    assert energy is not None
    assert npoints > 0
    if reduction == "mean" and npoints > 1:
        energy = energy.div_(npoints)
    return energy


bspline_be_loss = bspline_bending_loss
bspline_bending_energy = bspline_bending_loss


def curvature_loss(
    u: Tensor,
    spacing: Optional[Array] = None,
    sigma: Optional[float] = None,
    mode: str = "sobel",
    reduction: str = "mean",
) -> Tensor:
    r"""Loss term based on unmixed 2nd order spatial derivatives of vector fields.

    Fischer & Modersitzki (2003). Curvature based image registration. Journal Mathematical Imaging and Vision, 18(1), 81–85.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        spacing: Sampling grid spacing.
        sigma: Standard deviation of Gaussian in grid units (cf. ``spatial_derivatives()``).
        mode: Method used to approximate spatial derivatives (cf. ``spatial_derivatives()``).
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Curvature loss of vector fields.

    """
    if u.ndim < 4:
        # No loss for homogeneous coordinate transformations
        if reduction == "none":
            raise NotImplementedError(
                "curvature_loss() not implemented for linear transformation and 'reduction'='none'"
            )
        return torch.tensor(0, dtype=u.dtype, device=u.device)
    D = u.shape[1]
    if u.ndim - 2 != D:
        raise ValueError(f"curvature_loss() 'u' must be tensor of shape (N, {u.ndim - 2}, ..., X)")
    which = SpatialDerivativeKeys.unmixed(ndim=D, order=2)
    derivs = spatial_derivatives(u, mode=mode, which=which, sigma=sigma, spacing=spacing)
    derivs = torch.cat([deriv.unsqueeze(-1) for deriv in derivs.values()], dim=-1)
    loss = 0.5 * derivs.sum(-1).pow(2)
    loss = reduce_loss(loss, reduction)
    return loss


def diffusion_loss(
    u: Tensor,
    spacing: Optional[Tensor] = None,
    sigma: Optional[float] = None,
    mode: str = "central",
    reduction: str = "mean",
) -> Tensor:
    r"""Diffusion regularization loss."""
    loss = grad_loss(u, p=2, q=1, spacing=spacing, sigma=sigma, mode=mode, reduction=reduction)
    return loss.mul_(0.5)


def divergence_loss(
    u: Tensor,
    q: Optional[Union[int, float]] = 1,
    spacing: Optional[Array] = None,
    sigma: Optional[float] = None,
    mode: str = "central",
    reduction: str = "mean",
) -> Tensor:
    r"""Loss term encouraging divergence-free vector fields."""
    if u.ndim < 4:
        # No loss for homogeneous coordinate transformations
        if reduction == "none":
            raise NotImplementedError(
                "div_loss() not implemented for linear transformation and 'reduction'='none'"
            )
        return torch.tensor(0, dtype=u.dtype, device=u.device)
    D = u.shape[1]
    if u.ndim - 2 != D:
        raise ValueError(f"div_loss() 'u' must be tensor of shape (N, {u.ndim - 2}, ..., X)")
    which = SpatialDerivativeKeys.unmixed(ndim=D, order=1)
    loss = grad_loss(u, p=0, spacing=spacing, sigma=sigma, mode=mode, which=which, reduction="none")
    loss = loss.abs_() if q == 1 else loss.pow_(q)
    loss = reduce_loss(loss, reduction)
    return loss


def elasticity_loss(
    u: Tensor,
    spacing: Optional[Array] = None,
    sigma: Optional[float] = None,
    mode: str = "sobel",
    reduction: str = "mean",
) -> Tensor:
    r"""Loss term based on Navier-Cauchy PDE of linear elasticity.

    This linear elasticity loss includes only the term based on 1st order derivatives. The term of the
    Laplace operator, i.e., sum of unmixed 2nd order derivatives, is equivalent to the ``curvature_loss()``.
    This loss can be combined with the curvature regularization term to form a linear elasticity loss.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        spacing: Sampling grid spacing.
        sigma: Standard deviation of Gaussian in grid units (cf. ``spatial_derivatives()``).
        mode: Method used to approximate spatial derivatives (cf. ``spatial_derivatives()``).
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Linear elasticity loss of vector field.

    """
    if u.ndim < 4:
        # No loss for homogeneous coordinate transformations
        if reduction == "none":
            raise NotImplementedError(
                "elasticity_loss() not implemented for linear transformation and 'reduction'='none'"
            )
        return torch.tensor(0, dtype=u.dtype, device=u.device)
    N = u.shape[0]
    D = u.shape[1]
    if u.ndim - 2 != D:
        raise ValueError(f"elasticity_loss() 'u' must be tensor of shape (N, {u.ndim - 2}, ..., X)")
    derivs = spatial_derivatives(u, mode=mode, order=1, sigma=sigma, spacing=spacing)
    derivs = torch.cat([deriv.unsqueeze(-1) for deriv in derivs.values()], dim=-1)
    loss = torch.zeros((N,) + u.shape[2:], dtype=derivs.dtype, device=derivs.device)
    for a, b in itertools.product(range(D), repeat=2):
        loss += (0.5 * (derivs[:, a, ..., b] + derivs[:, b, ..., a])).square()
    if reduction == "none":
        loss = loss.unsqueeze(1)
    else:
        loss = reduce_loss(loss, reduction)
    return loss


def total_variation_loss(
    u: Tensor,
    spacing: Optional[Tensor] = None,
    sigma: Optional[float] = None,
    mode: str = "central",
    reduction: str = "mean",
) -> Tensor:
    r"""Total variation regularization loss."""
    return grad_loss(u, p=1, q=1, spacing=spacing, sigma=sigma, mode=mode, reduction=reduction)


tv_loss = total_variation_loss


def inverse_consistency_loss(
    forward: Tensor,
    inverse: Tensor,
    grid: Optional[Grid] = None,
    margin: Union[int, float] = 0,
    mask: Optional[Tensor] = None,
    units: str = "cube",
    reduction: str = "mean",
) -> Tensor:
    r"""Evaluate inverse consistency error.

    This function expects forward and inverse coordinate maps to be with respect to the unit cube
    of side length 2 as defined by the domain and codomain ``grid`` (see also ``Grid.axes()``).

    Args:
        forward: Tensor representation of spatial transformation.
        inverse: Tensor representation of inverse transformation.
        grid: Coordinate domain and codomain of forward transformation.
        margin: Number of ``grid`` points to ignore when computing mean error. If type of the
            argument is ``int``, this number of points are dropped at each boundary in each dimension.
            If a ``float`` value is given, it must be in [0, 1) and denote the percentage of sampling
            points to drop at each border. Inverse consistency of points near the domain boundary is
            affected by extrapolation and excluding these may be preferrable. See also ``mask``.
        mask: Foreground mask as tensor of shape ``(N, 1, ..., X)`` with size matching ``forward``.
            Inverse consistency errors at target grid points with a zero mask value are ignored.
        units: Compute mean inverse consistency error in specified units: ``cube`` with respect to
            normalized grid cube coordinates, ``voxel`` in voxel units, or in ``world`` units (mm).
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Inverse consistency error.

    """
    if not isinstance(forward, Tensor):
        raise TypeError("inverse_consistency_loss() 'forward' must be tensor")
    if not isinstance(inverse, Tensor):
        raise TypeError("inverse_consistency_loss() 'inverse' must be tensor")
    if not isinstance(margin, (int, float)):
        raise TypeError("inverse_consistency_loss() 'margin' must be int or float")
    if grid is None:
        if forward.ndim < 4:
            if inverse.ndim < 4:
                raise ValueError(
                    "inverse_consistency_loss() 'grid' required when both transforms are affine"
                )
            grid = Grid(shape=inverse.shape[2:])
        else:
            grid = Grid(shape=forward.shape[2:])
    # Compute inverse consistency error for each grid point
    x = grid.coords(dtype=forward.dtype, device=forward.device).unsqueeze(0)
    y = transform_grid(forward, x, align_corners=grid.align_corners())
    y = transform_points(inverse, y, align_corners=grid.align_corners())
    error = y - x
    # Set error outside foreground mask to zero
    if mask is not None:
        if not isinstance(mask, Tensor):
            raise TypeError("inverse_consistency_loss() 'mask' must be tensor")
        if mask.ndim != grid.ndim + 2:
            raise ValueError(
                f"inverse_consistency_loss() 'mask' must be {grid.ndim + 2}-dimensional"
            )
        if mask.shape[1] != 1:
            raise ValueError("inverse_consistency_loss() 'mask' must have shape (N, 1, ..., X)")
        if mask.shape[0] != 1 and mask.shape[0] != error.shape[0]:
            raise ValueError(
                f"inverse_consistency_loss() 'mask' batch size must be 1 or {error.shape[0]}"
            )
        error[move_dim(mask == 0, 1, -1).expand_as(error)] = 0
    # Discard error at grid boundary
    if margin > 0:
        if isinstance(margin, float):
            if margin < 0 or margin >= 1:
                raise ValueError(
                    f"inverse_consistency_loss() 'margin' must be in [0, 1), got {margin}"
                )
            m = [int(margin * n) for n in grid.size()]
        else:
            m = [max(0, int(margin))] * grid.ndim
        subgrid = tuple(reversed([slice(i, n - i) for i, n in zip(m, grid.size())]))
        error = error[(slice(0, error.shape[0]),) + subgrid + (slice(0, grid.ndim),)]
    # Scale differences by respective error units
    if units in ("voxel", "world"):
        error = denormalize_flow(error, size=grid.size(), channels_last=True)
        if units == "world":
            error *= grid.spacing().to(error)
    # Calculate error norm
    error: Tensor = error.norm(p=2, dim=-1)
    # Reduce error if requested
    if reduction != "none":
        count = error.numel()
        error = error.sum()
        if reduction == "mean" and mask is not None:
            count = (mask != 0).sum()
        error /= count
    return error


def masked_loss(
    loss: Tensor,
    mask: Optional[Tensor] = None,
    name: Optional[str] = None,
    inplace: bool = False,
) -> Tensor:
    r"""Multiply loss with an optionally specified spatial mask."""
    if mask is None:
        return loss
    if not name:
        name = "masked_loss"
    if not isinstance(mask, Tensor):
        raise TypeError(f"{name}() 'mask' must be tensor")
    if mask.shape[0] != 1 and mask.shape[0] != loss.shape[0]:
        raise ValueError(f"{name}() 'mask' must have same batch size as 'target' or batch size 1")
    if mask.shape[1] != 1 and mask.shape[1] != loss.shape[1]:
        raise ValueError(f"{name}() 'mask' must have same number of channels as 'target' or only 1")
    if mask.shape[2:] != loss.shape[2:]:
        raise ValueError(f"{name}() 'mask' must have same spatial shape as 'target'")
    if inplace:
        loss = loss.mul_(mask)
    else:
        loss = loss.mul(mask)
    return loss


def reduce_loss(loss: Tensor, reduction: str = "mean", mask: Optional[Tensor] = None) -> Tensor:
    r"""Reduce loss computed at each grid point."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduce_loss() 'reduction' must be 'mean', 'sum' or 'none'")
    if reduction == "none":
        return loss
    if mask is None:
        return loss.mean() if reduction == "mean" else loss.sum()
    value = loss.sum()
    if reduction == "mean":
        numel = mask.expand_as(loss).sum()
        value = value.div_(numel)
    return value
