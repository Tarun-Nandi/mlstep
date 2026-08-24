"""Advanced training utilities for Phase 3 experiments."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MATRIX_NDIM = 2
MIN_CONTRASTIVE_ROWS = 2


def supervised_contrastive_loss(projections: Tensor, labels: Tensor, temperature: float) -> Tensor:
    """Return the stable supervised contrastive loss for one batch."""
    if projections.ndim != MATRIX_NDIM:
        msg = "contrastive projections must be a two-dimensional matrix"
        raise ValueError(msg)
    labels = labels.reshape(-1)
    if labels.shape[0] != projections.shape[0]:
        msg = "contrastive labels must contain one value per projection"
        raise ValueError(msg)
    if labels.device != projections.device:
        msg = "contrastive labels and projections must be on the same device"
        raise ValueError(msg)
    if projections.shape[0] < MIN_CONTRASTIVE_ROWS:
        msg = "contrastive batches must contain at least two rows"
        raise ValueError(msg)
    if not math.isfinite(temperature) or temperature <= 0.0:
        msg = "contrastive temperature must be finite and positive"
        raise ValueError(msg)

    normalized = F.normalize(projections, dim=1)
    logits = normalized @ normalized.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    candidate_mask = ~diagonal
    positive_mask = labels[:, None].eq(labels[None, :]) & candidate_mask
    valid_anchors = positive_mask.any(dim=1)
    if not bool(valid_anchors.all()):
        msg = "every contrastive row must have a same-class partner"
        raise ValueError(msg)

    log_denominator = torch.logsumexp(logits.masked_fill(~candidate_mask, float("-inf")), dim=1)
    log_probabilities = logits - log_denominator[:, None]
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    positive_log_probability = (
        torch.where(
            positive_mask,
            log_probabilities,
            torch.zeros_like(log_probabilities),
        ).sum(dim=1)
        / positive_counts
    )
    return -positive_log_probability[valid_anchors].mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1, weight: Optional[Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute smoothed cross-entropy loss."""
        n_classes = logits.shape[-1]

        # One-hot encoding
        one_hot = F.one_hot(labels, num_classes=n_classes).float()

        # Smooth labels
        smooth_labels = one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1)

        if self.weight is not None:
            loss = loss * self.weight[labels]

        return loss.mean()


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, logits: Tensor) -> Tensor:
        """Apply temperature scaling."""
        return logits / self.temperature


def ensemble_predictions(models: list, x: Tensor, device: torch.device) -> Tensor:
    """Generate ensemble predictions from multiple models."""
    _ = device  # Reserved for future use
    predictions = []

    for model in models:
        model.eval()
        with torch.no_grad():
            _, d, _, s = model(x)
            joint = model.joint_distribution(d, s)
            predictions.append(joint)

    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred


def apply_label_smoothing(labels: Tensor, n_classes: int, smoothing: float) -> Tensor:
    """Apply label smoothing to target labels."""
    smooth_labels = labels.float() * (1 - smoothing) + smoothing / n_classes
    return smooth_labels


class AdaptiveTemperatureLoss(nn.Module):
    """Loss with learnable temperature scaling."""

    def __init__(self, base_temp: float = 1.0, learnable: bool = True):
        super().__init__()
        if learnable:
            self.temperature = nn.Parameter(torch.tensor(base_temp))
        else:
            self.temperature = base_temp

    def forward(self, logits: Tensor, labels: Tensor, temperature: Optional[float] = None) -> Tensor:
        """Compute loss with temperature scaling."""
        temp = temperature if temperature is not None else self.temperature
        scaled_logits = logits / temp

        if len(labels.shape) == 1:
            return F.cross_entropy(scaled_logits, labels)
        else:
            # Soft labels
            log_probs = F.log_softmax(scaled_logits, dim=-1)
            return -(labels * log_probs).sum(dim=-1).mean()


def hard_negative_mining_loss(
    model, features: Tensor, labels: Tensor, hard_negative_indices: Tensor, device: torch.device
) -> Tensor:
    """Compute loss with hard negative mining."""
    _ = device  # Reserved for future use
    # Get standard predictions
    d_logit, _s_logits = model.forward_logits(features)
    standard_loss = F.binary_cross_entropy_with_logits(d_logit.squeeze(0), labels.float())

    # Focus on hard negatives
    if len(hard_negative_indices) > 0:
        hard_features = features[hard_negative_indices]
        hard_labels = labels[hard_negative_indices]

        hard_d_logit, _hard_s_logits = model.forward_logits(hard_features)
        hard_loss = F.binary_cross_entropy_with_logits(hard_d_logit.squeeze(0), hard_labels.float())

        # Weight hard negatives more heavily
        return standard_loss + 2.0 * hard_loss

    return standard_loss
