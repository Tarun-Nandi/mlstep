"""Advanced training utilities for Phase 3 experiments."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ContrastivePretrainer(nn.Module):
    """Contrastive pre-training for tabular data."""

    def __init__(self, n_features: int, hidden_dim: int = 256, projection_dim: int = 128, temperature: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim)
        )
        self.temperature = temperature

    def forward(self, x: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass with contrastive loss."""
        projections = self.encoder(x)
        # Normalize projections
        projections = F.normalize(projections, dim=1)

        # Compute contrastive loss
        batch_size = x.shape[0]
        labels = labels.view(-1, 1)

        # Create positive mask (same class)
        mask = (labels == labels.t()).float()

        # Compute similarity matrix
        sim_matrix = torch.mm(projections, projections.t()) / self.temperature

        # Remove diagonal (self-similarity)
        sim_matrix.fill_diagonal_(float("-inf"))

        # Contrastive loss
        exp_sim = torch.exp(sim_matrix)
        sum_exp = exp_sim.sum(dim=1, keepdim=True)

        # Positive pairs
        pos_mask = mask - torch.eye(batch_size, device=x.device)
        pos_sim = (sim_matrix * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

        loss = -torch.log(torch.exp(pos_sim) / (sum_exp.squeeze() + 1e-8))
        loss = loss.mean()

        return projections, loss


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
