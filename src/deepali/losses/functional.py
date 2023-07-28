r"""Loss functions, evaluation metrics, and related utilities."""

import itertools
from typing import Protocol, Optional, Sequence, Tuple, Union

import math

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, logsigmoid
import torch.nn.functional as F

from deepali.core.bspline import evaluate_cubic_bspline
from deepali.core.enum import SpatialDerivativeKeys, SpatialDim
from deepali.core.grid import Grid
from deepali.core.image import avg_pool, dot_channels, rand_sample, spatial_derivatives
from deepali.core.flow import denormalize_flow
from deepali.core.pointset import transform_grid
from deepali.core.pointset import transform_points
from deepali.core.tensor import as_one_hot_tensor, move_dim
from deepali.core.typing import Array, ScalarOrTuple


__all__ = (
    "balanced_binary_cross_entropy_with_logits",
    "binary_cross_entropy_with_logits",
    "label_smoothing",
    "dice_score",
    "dice_loss",
    "kld_loss",
    "lcc_loss",
    "mae_loss",
    "mse_loss",
    "ncc_loss",
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
    "wlcc_loss",
)


class ElementwiseLoss(Protocol):
    r"""Type annotation of a eleemntwise loss function."""

    def __call__(self, input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
        ...


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


def ncc_loss(
    source: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    epsilon: float = 1e-15,
    reduction: str = "mean",
) -> Tensor:
    r"""Normalized cross correlation.

    Args:
        source: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``source``.
        mask: Multiplicative mask tensor with same shape as ``source``.
        epsilon: Small constant added to denominator to avoid division by zero.
        reduction: Whether to compute "mean" or "sum" of normalized cross correlation
            of image pairs in batch. If "none", a 1-dimensional tensor is returned
            with length equal the batch size.

    Returns:
        Negative squared normalized cross correlation plus one.

    """

    if not isinstance(source, Tensor):
        raise TypeError("ncc_loss() 'source' must be tensor")
    if not isinstance(target, Tensor):
        raise TypeError("ncc_loss() 'target' must be tensor")
    if source.shape != target.shape:
        raise ValueError("ncc_loss() 'source' must have same shape as 'target'")

    source = source.reshape(source.shape[0], -1).float()
    target = target.reshape(source.shape[0], -1).float()

    source_mean = source.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)

    x = source.sub(source_mean)
    y = target.sub(target_mean)

    a = x.mul(y).sum(dim=1)
    b = x.square().sum(dim=1)
    c = y.square().sum(dim=1)

    loss = a.square_().div_(b.mul_(c).add_(epsilon)).neg_().add_(1)
    loss = masked_loss(loss, mask, "ncc_loss")
    loss = reduce_loss(loss, reduction, mask)
    return loss


def lcc_loss(
    source: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    kernel_size: ScalarOrTuple[int] = 7,
    epsilon: float = 1e-15,
    reduction: str = "mean",
) -> Tensor:
    r"""Local normalized cross correlation.

    References:
        Avants et al., 2008, Symmetric Diffeomorphic Image Registration with Cross Correlation:
            Evaluating Automated Labeling of Elderly and Neurodegenerative Brain,
            doi:10.1016/j.media.2007.06.004.

    Args:
        source: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``source``.
        mask: Multiplicative mask tensor with same shape as ``source``.
        kernel_size: Local rectangular window size in number of grid points.
        epsilon: Small constant added to denominator to avoid division by zero.
        reduction: Whether to compute "mean" or "sum" over all grid points. If "none", output
            tensor shape is equal to the shape of the input tensors given an odd kernel size.

    Returns:
        Negative local normalized cross correlation plus one.

    """

    if not isinstance(source, Tensor):
        raise TypeError("lcc_loss() 'source' must be tensor")
    if not isinstance(target, Tensor):
        raise TypeError("lcc_loss() 'target' must be tensor")
    if source.shape != target.shape:
        raise ValueError("lcc_loss() 'source' must have same shape as 'target'")

    def local_sum(data: Tensor) -> Tensor:
        return avg_pool(
            data,
            kernel_size=kernel_size,
            stride=1,
            padding=None,
            divisor_override=1,
        )

    def local_mean(data: Tensor) -> Tensor:
        return avg_pool(
            data,
            kernel_size=kernel_size,
            stride=1,
            padding=None,
            count_include_pad=False,
        )

    source = source.float()
    target = target.float()

    source_mean = local_mean(source)
    target_mean = local_mean(target)

    x = source.sub(source_mean)
    y = target.sub(target_mean)

    a = local_sum(x.mul(y))
    b = local_sum(x.square())
    c = local_sum(y.square())

    loss = a.square_().div_(b.mul_(c).add_(epsilon)).neg_().add_(1)
    loss = masked_loss(loss, mask, "lcc_loss")
    loss = reduce_loss(loss, reduction, mask)
    return loss


def wlcc_loss(
    source: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    source_mask: Optional[Tensor] = None,
    target_mask: Optional[Tensor] = None,
    kernel_size: ScalarOrTuple[int] = 7,
    epsilon: float = 1e-15,
    reduction: str = "mean",
) -> Tensor:
    r"""Weighted local normalized cross correlation.

    References:
        Lewis et al., 2020, Fast Learning-based Registration of Sparse 3D Clinical Images, arXiv:1812.06932.

    Args:
        source: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``source``.
        mask: Multiplicative mask tensor ``w_c`` with same shape as ``target`` and ``source``.
            This tensor is used for computing the weighted local correlation. If ``None`` and
            both ``source_mask`` and ``target_mask`` are given, it is set to the product of these.
            Otherwise, no mask is used to aggregate the local cross correlation values. When both
            ``source_mask`` and ``target_mask`` are ``None``, but ``mask`` is not, then the specified
            ``mask`` is used both as ``source_mask`` and ``target_mask``.
        source_mask: Multiplicative mask tensor ``w_m`` with same shape as ``source``.
            This tensor is used for computing the weighted local ``source`` mean. If ``None``,
            the local mean is computed over all ``source`` elements within each local neighborhood.
        target_mask: Multiplicative mask tensor ``w_f`` with same shape as ``source``.
            This tensor is used for computing the weighted local ``target`` mean. If ``None``,
            the local mean is computed over all ``target`` elements within each local neighborhood.
        kernel_size: Local rectangular window size in number of grid points.
        epsilon: Small constant added to denominator to avoid division by zero.
        reduction: Whether to compute "mean" or "sum" over all grid points. If "none", output
            tensor shape is equal to the shape of the input tensors given an odd kernel size.

    Returns:
        Negative local normalized cross correlation plus one.

    """

    if not isinstance(source, Tensor):
        raise TypeError("wlcc_loss() 'source' must be tensor")
    if not isinstance(target, Tensor):
        raise TypeError("wlcc_loss() 'target' must be tensor")

    if source.shape != target.shape:
        raise ValueError("wlcc_loss() 'source' must have same shape as 'target'")

    for t, t_name, w, w_name in zip(
        [target, source, target],
        ["target", "source", "target"],
        [mask, source_mask, target_mask],
        ["mask", "source_mask", "target_mask"],
    ):
        if w is None:
            continue
        if not isinstance(w, Tensor):
            raise TypeError(f"wlcc_loss() '{w_name}' must be tensor")
        if w.shape[0] not in (1, t.shape[0]):
            raise ValueError(
                f"wlcc_loss() '{w_name}' batch size ({w.shape[0]}) must be 1 or match '{t_name}' ({t.shape[0]})"
            )
        if w.shape[1] not in (1, t.shape[1]):
            raise ValueError(
                f"wlcc_loss() '{w_name}' number of channels ({w.shape[1]}) must be 1 or match '{t_name}' ({t.shape[1]})"
            )
        if w.shape[2:] != t.shape[2:]:
            raise ValueError(
                f"wlcc_loss() '{w_name}' grid shape ({w.shape[2:]}) must match '{t_name}' ({t.shape[2:]})"
            )

    def local_sum(data: Tensor) -> Tensor:
        return avg_pool(
            data,
            kernel_size=kernel_size,
            stride=1,
            padding=None,
            divisor_override=1,
        )

    def local_mean(data: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        if weight is None:
            return avg_pool(
                data,
                kernel_size=kernel_size,
                stride=1,
                padding=None,
                count_include_pad=False,
            )
        a = local_sum(data.mul(weight))
        b = local_sum(weight).add_(epsilon)
        return a.div_(b)

    if mask is not None and source_mask is None and target_mask is None:
        source_mask = mask.float()
        target_mask = source_mask
    else:
        if source_mask is not None:
            source_mask = source_mask.float()
        if target_mask is not None:
            target_mask = target_mask.float()

    source = source.float()
    target = target.float()

    source_mean = local_mean(source, source_mask)
    target_mean = local_mean(target, target_mask)

    x = source.sub(source_mean)
    y = target.sub(target_mean)

    if mask is None and source_mask is not None and target_mask is not None:
        mask = source_mask.mul(target_mask)
    if mask is not None:
        x = x.mul_(mask)
        y = y.mul_(mask)

    a = local_sum(x.mul(y))
    b = local_sum(x.square())
    c = local_sum(y.square())

    loss = a.square_().div_(b.mul_(c).add_(epsilon)).neg_().add_(1)
    loss = masked_loss(loss, mask, name="wlcc_loss")
    loss = reduce_loss(loss, reduction, mask)
    return loss


def huber_loss(
    input: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    norm: Optional[Union[float, Tensor]] = None,
    reduction: str = "mean",
    delta: float = 1.0,
) -> Tensor:
    r"""Normalized masked Huber loss.

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.
        delta: Specifies the threshold at which to change between delta-scaled L1 and L2 loss.

    Returns:
        Masked, aggregated, and normalized Huber loss.

    """

    def loss_fn(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
        return F.huber_loss(input, target, reduction=reduction, delta=delta)

    return elementwise_loss(
        "huber_loss", loss_fn, input, target, mask=mask, norm=norm, reduction=reduction
    )


def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    norm: Optional[Union[float, Tensor]] = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> Tensor:
    r"""Normalized masked smooth L1 loss.

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.
        delta: Specifies the threshold at which to change between delta-scaled L1 and L2 loss.

    Returns:
        Masked, aggregated, and normalized smooth L1 loss.

    """

    def loss_fn(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
        return F.smooth_l1_loss(input, target, reduction=reduction, beta=beta)

    return elementwise_loss(
        "smooth_l1_loss", loss_fn, input, target, mask=mask, norm=norm, reduction=reduction
    )


def l1_loss(
    input: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    norm: Optional[Union[float, Tensor]] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Normalized mean absolute error.

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.

    Returns:
        Normalized mean absolute error.

    """

    def loss_fn(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
        return F.l1_loss(input, target, reduction=reduction)

    return elementwise_loss(
        "l1_loss", loss_fn, input, target, mask=mask, norm=norm, reduction=reduction
    )


def mae_loss(
    input: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    norm: Optional[Union[float, Tensor]] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Normalized mean absolute error.

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.

    Returns:
        Normalized mean absolute error.

    """

    def loss_fn(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
        return F.l1_loss(input, target, reduction=reduction)

    return elementwise_loss(
        "mae_loss", loss_fn, input, target, mask=mask, norm=norm, reduction=reduction
    )


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
        Qiu, H., Qin, C., Schuh, A., Hammernik, K.: Learning Diffeomorphic and Modality-invariant
            Registration using B-splines. Medical Imaging with Deep Learning. (2021).
        Thévenaz, P., Unser, M.: Optimization of mutual information for multiresolution image registration.
            IEEE Trans. Image Process. 9, 2083–2099 (2000).

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
        raise NotImplementedError(
            "mi_loss() automatically setting num_bins based on dynamic range of input"
        )

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

    References:
        Fischer & Modersitzki (2003). Curvature based image registration.
            Journal Mathematical Imaging and Vision, 18(1), 81–85.

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
                "curvature_loss() not implemented for linear transformation and reduction='none'"
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
                "divergence_loss() not implemented for linear transformation and reduction='none'"
            )
        return torch.tensor(0, dtype=u.dtype, device=u.device)
    D = u.shape[1]
    if u.ndim - 2 != D:
        raise ValueError(f"divergence_loss() 'u' must be tensor of shape (N, {u.ndim - 2}, ..., X)")
    derivs = spatial_derivatives(u, mode=mode, order=1, sigma=sigma, spacing=spacing)
    derivs = torch.cat([deriv.unsqueeze(-1) for deriv in derivs.values()], dim=-1)
    loss = derivs.sum(dim=-1)
    loss = loss.abs_() if q < 2 else loss.pow_(q)
    loss = reduce_loss(loss, reduction)
    return loss


def lame_parameters(
    material_name: Optional[str] = None,
    first_parameter: Optional[float] = None,
    second_parameter: Optional[float] = None,
    shear_modulus: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    youngs_modulus: Optional[float] = None,
) -> Tuple[float, float]:
    r"""Get Lame parameters of linear elasticity given different quantities.

    Args:
        material_name: Name of material preset. Cannot be used in conjunction with other arguments.
        first_parameter: Lame's first parameter.
        second_parameter: Lame's second parameter, i.e., shear modulus.
        shear_modulus: Shear modulus, i.e., Lame's second parameter.
        poissons_ratio: Poisson's ratio.
        youngs_modulus: Young's modulus.

    Returns:
        lambda: Lame's first parameter.
        mu: Lame's second parameter.

    """
    RUBBER_POISSONS_RATIO = 0.4999
    RUBBER_SHEAR_MODULUS = 0.0006
    # Derive unspecified Lame parameters from any combination of two given quantities
    # (cf. conversion table at https://en.wikipedia.org/wiki/Young%27s_modulus#External_links)
    kwargs = {
        name: value
        for name, value in zip(
            [
                "first_parameter",
                "second_parameter",
                "shear_modulus",
                "poissons_ratio",
                "youngs_modulus",
            ],
            [first_parameter, second_parameter, poissons_ratio, youngs_modulus, shear_modulus],
        )
        if value is not None
    }
    # Default values for different materials (cf. Wikipedia pages for Poisson's ratio and shear modulus)
    if material_name:
        if kwargs:
            raise ValueError(
                "lame_parameters() 'material_name' cannot be specified in combination with other quantities"
            )
        if material_name == "rubber":
            poissons_ratio = RUBBER_POISSONS_RATIO
            shear_modulus = RUBBER_SHEAR_MODULUS
        else:
            raise ValueError(f"lame_parameters() unknown 'material_name': {material_name}")
    elif len(kwargs) != 2:
        raise ValueError(
            "lame_parameters() specify 'material_name' or exactly two parameters, got: "
            + ", ".join(f"{k}={v}" for k, v in kwargs.items())
        )
    if second_parameter is None:
        second_parameter = shear_modulus
    elif shear_modulus is None:
        shear_modulus = second_parameter
    else:
        raise ValueError(
            "lame_parameters() 'second_parameter' and 'shear_modulus' are mutually exclusive"
        )
    if first_parameter is None:
        if shear_modulus is None:
            if poissons_ratio is not None and youngs_modulus is not None:
                first_parameter = (
                    poissons_ratio * youngs_modulus / ((1 + poissons_ratio)(1 - 2 * poissons_ratio))
                )
                second_parameter = youngs_modulus / (2 * (1 + poissons_ratio))
        elif youngs_modulus is None:
            if poissons_ratio is None:
                poissons_ratio = RUBBER_POISSONS_RATIO
            first_parameter = 2 * shear_modulus * poissons_ratio / (1 - 2 * poissons_ratio)
        else:
            first_parameter = (
                shear_modulus
                * (youngs_modulus - 2 * shear_modulus)
                / (3 * shear_modulus - youngs_modulus)
            )
    elif second_parameter is None:
        if youngs_modulus is None:
            if poissons_ratio is None:
                poissons_ratio = RUBBER_POISSONS_RATIO
            second_parameter = first_parameter * (1 - 2 * poissons_ratio) / (2 * poissons_ratio)
        else:
            r = math.sqrt(
                youngs_modulus**2
                + 9 * first_parameter**2
                + 2 * youngs_modulus * first_parameter
            )
            second_parameter = youngs_modulus - 3 * first_parameter + r / 4
    if first_parameter is None or second_parameter is None:
        raise NotImplementedError(
            "lame_parameters() deriving Lame parameters from: "
            + ", ".join(f"'{name}'" for name in kwargs.keys())
        )
    if first_parameter < 0:
        raise ValueError("lame_parameter() 'first_parameter' is negative")
    if second_parameter < 0:
        raise ValueError("lame_parameter() 'second_parameter' is negative")
    return first_parameter, second_parameter


def elasticity_loss(
    u: Tensor,
    material_name: Optional[str] = None,
    first_parameter: Optional[float] = None,
    second_parameter: Optional[float] = None,
    shear_modulus: Optional[float] = None,
    poissons_ratio: Optional[float] = None,
    youngs_modulus: Optional[float] = None,
    spacing: Optional[Array] = None,
    sigma: Optional[float] = None,
    mode: str = "sobel",
    reduction: str = "mean",
) -> Tensor:
    r"""Loss term based on Navier-Cauchy PDE of linear elasticity.

    References:
        Fischer & Modersitzki, 2004, A unified approach to fast image registration and a new
            curvature based registration technique.

    Args:
        u: Batch of vector fields as tensor of shape ``(N, D, ..., X)``. When a tensor with less than
            four dimensions is given, it is assumed to be a linear transformation and zero is returned.
        material_name: Name of material preset. Cannot be used in conjunction with other arguments.
        first_parameter: Lame's first parameter.
        second_parameter: Lame's second parameter, i.e., shear modulus.
        shear_modulus: Shear modulus, i.e., Lame's second parameter.
        poissons_ratio: Poisson's ratio.
        youngs_modulus: Young's modulus.
        spacing: Sampling grid spacing.
        sigma: Standard deviation of Gaussian in grid units (cf. ``spatial_derivatives()``).
        mode: Method used to approximate spatial derivatives (cf. ``spatial_derivatives()``).
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Linear elasticity loss of vector field.

    """
    lambd, mu = lame_parameters(
        material_name=material_name,
        first_parameter=first_parameter,
        second_parameter=second_parameter,
        shear_modulus=shear_modulus,
        poissons_ratio=poissons_ratio,
        youngs_modulus=youngs_modulus,
    )
    if u.ndim < 4:
        # No loss for homogeneous coordinate transformations
        if reduction == "none":
            raise NotImplementedError(
                "elasticity_loss() not implemented for linear transformation and reduction='none'"
            )
        return torch.tensor(0, dtype=u.dtype, device=u.device)
    D = u.shape[1]
    if u.ndim - 2 != D:
        raise ValueError(f"elasticity_loss() 'u' must be tensor of shape (N, {u.ndim - 2}, ..., X)")
    derivs = spatial_derivatives(u, mode=mode, order=1, sigma=sigma, spacing=spacing)
    derivs = [derivs[str(SpatialDim(i))] for i in range(D)]
    loss = derivs[0].narrow(1, 0, 1).clone()
    for i in range(1, D):
        loss = loss.add_(derivs[i].narrow(1, i, 1))
    loss = loss.square_().mul_(lambd / 2)
    for j, k in itertools.product(range(D), repeat=2):
        temp = derivs[j].narrow(1, k, 1).add(derivs[k].narrow(1, j, 1))
        loss = loss.add_(temp.square_().mul_(mu / 4))
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


def elementwise_loss(
    name: str,
    loss_fn: ElementwiseLoss,
    input: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    norm: Optional[Union[float, Tensor]] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""Evaluate, aggregate, and normalize elementwise loss, optionally within masked region.

    Args:
        input: Source image sampled on ``target`` grid.
        target: Target image with same shape as ``input``.
        mask: Multiplicative mask with same shape as ``input``.
        norm: Positive factor by which to divide loss value.
        reduction: Whether to compute "mean" or "sum" over all grid points.
            If "none", output tensor shape is equal to the shape of the input tensors.

    Returns:
        Aggregated normalized loss value.

    """
    if not isinstance(input, Tensor):
        raise TypeError(f"{name}() 'input' must be tensor")
    if not isinstance(target, Tensor):
        raise TypeError(f"{name}() 'target' must be tensor")
    if input.shape != target.shape:
        raise ValueError(f"{name}() 'input' must have same shape as 'target'")
    if mask is None:
        loss = loss_fn(input, target, reduction=reduction)
    else:
        loss = loss_fn(input, target, reduction="none")
        loss = masked_loss(loss, mask, name)
        loss = reduce_loss(loss, reduction, mask)
    if norm is not None:
        norm = torch.as_tensor(norm, dtype=loss.dtype, device=loss.device).squeeze()
        if not norm.ndim == 0:
            raise ValueError(f"{name}() 'norm' must be scalar")
        if norm > 0:
            loss = loss.div_(norm)
    return loss


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
