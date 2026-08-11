"""Piecewise-linear numerical embeddings for tabular models."""

import math

import torch
from torch import nn

MATRIX_NDIM = 2


def _positive_integer(value: int, name: str) -> None:
    """Reject booleans and non-positive integer settings."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)


class PiecewiseLinearEmbedding(nn.Module):
    """Encode continuous features with cumulative piecewise-linear bases.

    For feature ``f`` with non-decreasing boundaries
    ``b[f, 0], ..., b[f, n_bins]``, interval ``j`` contributes

    ``clip((x - b[f, j]) / (b[f, j + 1] - b[f, j]), 0, 1)``.

    These contributions form the usual cumulative piecewise-linear encoding:
    intervals below the value are one, the active interval is fractional, and
    intervals above the value are zero. Each feature has an independent learned
    linear projection from this encoding to ``embedding_dim`` outputs.

    Repeated quantile boundaries are permitted. A zero-width interval is
    defined as the step ``x >= boundary``, which keeps the encoding finite and
    deterministic for sparse or constant features.

    Parameters
    ----------
    n_features:
        Number of continuous input features.
    n_bins:
        Number of intervals, and therefore encoding components, per feature.
    embedding_dim:
        Output width of each feature-specific projection.
    version:
        ``"B"`` returns the linear projection. ``"A"`` applies ReLU to it.
    """

    def __init__(
        self,
        n_features: int,
        n_bins: int = 48,
        embedding_dim: int = 12,
        version: str = "B",
    ) -> None:
        super().__init__()
        _positive_integer(n_features, "n_features")
        _positive_integer(n_bins, "n_bins")
        _positive_integer(embedding_dim, "embedding_dim")
        if version not in ("A", "B"):
            msg = "version must be 'A' or 'B'"
            raise ValueError(msg)

        self.n_features = n_features
        self.n_bins = n_bins
        self.embedding_dim = embedding_dim
        self.version = version

        # This is a batched collection of feature-specific Linear layers. The
        # historical ``embeddings`` name is retained for the public/state API.
        self.embeddings = nn.Parameter(torch.empty(n_features, n_bins, embedding_dim))
        self.bias = nn.Parameter(torch.empty(n_features, embedding_dim))

        self.register_buffer("bin_boundaries", torch.empty(n_features, n_bins + 1))
        self._boundaries_fitted = False
        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        """Return the flattened output width."""
        return self.n_features * self.embedding_dim

    def reset_parameters(self) -> None:
        """Initialize each feature projection like ``nn.Linear``."""
        bound = 1.0 / math.sqrt(self.n_bins)
        nn.init.uniform_(self.embeddings, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def set_bin_boundaries(self, boundaries: torch.Tensor) -> None:
        """Validate and store fitted boundaries.

        Parameters
        ----------
        boundaries:
            Finite, non-decreasing tensor with shape
            ``(n_features, n_bins + 1)``. Duplicate values are allowed.
        """
        if not isinstance(boundaries, torch.Tensor):
            msg = "boundaries must be a torch tensor"
            raise TypeError(msg)
        expected_shape = (self.n_features, self.n_bins + 1)
        if tuple(boundaries.shape) != expected_shape:
            msg = f"Expected boundaries shape {expected_shape}, got {tuple(boundaries.shape)}"
            raise ValueError(msg)
        if not torch.isfinite(boundaries).all():
            msg = "boundaries must be finite"
            raise ValueError(msg)
        if torch.any(boundaries[:, 1:] < boundaries[:, :-1]):
            msg = "boundaries must be non-decreasing for every feature"
            raise ValueError(msg)

        with torch.no_grad():
            self.bin_boundaries.copy_(
                boundaries.detach().to(
                    device=self.bin_boundaries.device,
                    dtype=self.bin_boundaries.dtype,
                )
            )
            self._boundaries_fitted = True

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Restore the fitted flag when boundaries arrive through a parent module."""
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if f"{prefix}bin_boundaries" in state_dict:
            self._boundaries_fitted = True

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return cumulative piecewise-linear coordinates."""
        left = self.bin_boundaries[:, :-1]
        right = self.bin_boundaries[:, 1:]
        width = right - left
        positive_width = width > 0
        safe_width = torch.where(positive_width, width, torch.ones_like(width))

        values = x.unsqueeze(-1)
        linear = ((values - left.unsqueeze(0)) / safe_width.unsqueeze(0)).clamp(0.0, 1.0)
        collapsed = (values >= right.unsqueeze(0)).to(dtype=linear.dtype)
        return torch.where(positive_width.unsqueeze(0), linear, collapsed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a matrix with shape ``(batch, n_features)``."""
        if x.ndim != MATRIX_NDIM or x.shape[1] != self.n_features:
            msg = f"Expected input shape (batch, {self.n_features}), got {tuple(x.shape)}"
            raise ValueError(msg)
        if not x.is_floating_point():
            msg = "PiecewiseLinearEmbedding requires floating-point inputs"
            raise TypeError(msg)
        if not self._boundaries_fitted:
            msg = "bin boundaries must be fitted before calling forward"
            raise RuntimeError(msg)

        encoded = self._encode(x)
        projected = torch.einsum("bfn,fnd->bfd", encoded, self.embeddings)
        projected = projected + self.bias.unsqueeze(0)
        if self.version == "A":
            projected = torch.relu(projected)
        return projected.reshape(x.shape[0], self.output_dim)
