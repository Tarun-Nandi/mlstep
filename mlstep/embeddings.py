"""Piecewise-linear numerical embeddings for tabular models."""

import math

import torch
from torch import nn

from mlstep.data import PLE_N_BINS

MATRIX_NDIM = 2
FEATURE_BLOCK_SIZE = 32
PLE_EMBEDDING_DIM = 12


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
    """

    def __init__(
        self,
        n_features: int,
        n_bins: int = PLE_N_BINS,
        embedding_dim: int = PLE_EMBEDDING_DIM,
    ) -> None:
        """Allocate one learned projection per feature; bin boundaries are set later."""
        super().__init__()
        if min(n_features, n_bins, embedding_dim) < 1:
            raise ValueError("feature count, bin count and embedding width must be positive")
        self.n_features = n_features
        self.n_bins = n_bins
        self.embedding_dim = embedding_dim

        # This is a batched collection of feature-specific Linear layers. The
        # historical ``embeddings`` name is retained for the public/state API.
        self.embeddings = nn.Parameter(torch.empty(n_features, n_bins, embedding_dim))
        self.bias = nn.Parameter(torch.empty(n_features, embedding_dim))

        self.register_buffer("bin_boundaries", torch.empty(n_features, n_bins + 1))
        self._boundaries_fitted = False
        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        """Return the total embedded width used to size the following network layer."""
        return self.n_features * self.embedding_dim

    def reset_parameters(self) -> None:
        """Initialize weights and biases uniformly; leave fitted bin boundaries unchanged."""
        bound = 1.0 / math.sqrt(self.n_bins)
        nn.init.uniform_(self.embeddings, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def set_bin_boundaries(self, boundaries: torch.Tensor) -> None:
        """Copy training-fitted quantiles into the model and mark the layer ready.

        ``boundaries`` has shape ``(n_features, n_bins + 1)`` and must be finite
        and nondecreasing. Repeated quantiles are allowed. The copy uses the
        buffer's device and dtype and does not track gradients.
        """
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
            # copy_ already handles device/dtype conversion; no temporary is needed.
            self.bin_boundaries.copy_(boundaries)
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
        """Restore tensors and the ready flag when loading this layer or its parent model.

        The flag is a Python attribute, so PyTorch does not restore it from the
        state dictionary. This hook also runs when loading the containing Student.
        """
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

    def _encode(self, x: torch.Tensor, feature_slice: slice = slice(None)) -> torch.Tensor:
        """Encode a block of scalars as ``(batch, block_features, n_bins)`` coordinates.

        ``x`` contains only the columns selected by ``feature_slice``. Completed
        intervals contribute 1, the active interval a fraction, and later ones 0.
        For boundaries [0, 1, 3], a value of 2 therefore becomes [1, 0.5].
        """
        left = self.bin_boundaries[feature_slice, :-1]
        right = self.bin_boundaries[feature_slice, 1:]
        width = right - left
        positive_width = width > 0
        # Repeated quantiles use a step below; avoid dividing by zero first.
        safe_width = torch.where(positive_width, width, torch.ones_like(width))

        values = x.unsqueeze(-1)
        linear = ((values - left) / safe_width).clamp(0.0, 1.0)
        collapsed = (values >= right).to(dtype=linear.dtype)
        return torch.where(positive_width, linear, collapsed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map scalars ``(batch, n_features)`` to ``(batch, output_dim)`` learned values.

        Columns stay in feature order, with each feature's embedding contiguous.
        Bin arithmetic uses float32; projections follow the caller's autocast
        policy. Blocking limits individual temporaries, although backward can
        retain encodings from every block during training.
        """
        if x.ndim != MATRIX_NDIM or x.shape[1] != self.n_features:
            msg = f"Expected input shape (batch, {self.n_features}), got {tuple(x.shape)}"
            raise ValueError(msg)
        if not self._boundaries_fitted:
            msg = "bin boundaries must be fitted before calling forward"
            raise RuntimeError(msg)

        # Bound the large (rows, features, bins) temporary when embedding all
        # inputs. Quantile arithmetic stays FP32; learned projections use AMP.
        blocks = []
        for start in range(0, self.n_features, FEATURE_BLOCK_SIZE):
            selected = slice(start, start + FEATURE_BLOCK_SIZE)
            with torch.autocast(x.device.type, enabled=False):
                encoded = self._encode(x[:, selected].float(), selected)
            projected = torch.einsum("bfn,fnd->bfd", encoded, self.embeddings[selected])
            projected = projected + self.bias[selected].to(projected.dtype)
            blocks.append(projected)
        return torch.cat(blocks, dim=1).reshape(x.shape[0], self.output_dim)
