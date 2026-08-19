"""Differentiable losses for ordered timestep-halving predictions."""

import torch

LOGITS_NDIM = 3
MIN_ORDERED_CLASSES = 2


def conditional_severity_rps_loss(
    member_logits: torch.Tensor,
    positive_labels: torch.Tensor,
) -> torch.Tensor:
    """Return the conditional ranked probability score for positive rows.

    Parameters
    ----------
    member_logits
        Conditional-severity logits with shape ``(rows, members, classes)``.
    positive_labels
        One-dimensional physical halving labels in ``[1, classes]``. The
        empty vector represents a batch containing no positive rows.

    Returns
    -------
    torch.Tensor
        A scalar mean over rows, ensemble members, and the ``classes - 1``
        cumulative class boundaries. An empty positive batch returns a
        differentiable zero connected to ``member_logits``.
    """
    if member_logits.ndim != LOGITS_NDIM:
        msg = "member_logits must have shape (rows, members, classes)"
        raise ValueError(msg)
    if not member_logits.is_floating_point():
        msg = "member_logits must have a floating-point dtype"
        raise TypeError(msg)

    n_rows, n_members, n_classes = member_logits.shape
    if n_members < 1:
        msg = "member_logits must contain at least one ensemble member"
        raise ValueError(msg)
    if n_classes < MIN_ORDERED_CLASSES:
        msg = "member_logits must contain at least two ordered classes"
        raise ValueError(msg)
    if positive_labels.ndim != 1 or len(positive_labels) != n_rows:
        msg = "positive_labels must contain one label per row"
        raise ValueError(msg)
    if positive_labels.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        msg = "positive_labels must have an integer dtype"
        raise TypeError(msg)

    if n_rows == 0:
        return member_logits.sum() * 0.0

    # Training labels are validated before device transfer. Retain a defensive
    # value check for direct CPU callers without forcing a CUDA synchronization
    # in every optimizer batch.
    if positive_labels.device.type == "cpu" and bool(torch.any((positive_labels < 1) | (positive_labels > n_classes))):
        msg = f"positive_labels must be between 1 and {n_classes}"
        raise ValueError(msg)
    labels = positive_labels.to(device=member_logits.device, dtype=torch.long)

    probabilities = torch.softmax(member_logits, dim=-1)
    predicted_cdf = probabilities[..., :-1].cumsum(dim=-1)

    zero_based_labels = labels - 1
    boundaries = torch.arange(n_classes - 1, device=member_logits.device)
    observed_cdf = (zero_based_labels[:, None] <= boundaries[None, :]).to(member_logits.dtype)

    return (predicted_cdf - observed_cdf[:, None, :]).square().mean()
