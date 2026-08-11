"""Piecewise-linear numerical embeddings for tabular models.

Based on the numerical embeddings paper and TabM documentation.
Provides PLE (Piecewise Linear Embedding) for continuous features.
"""

import math

import torch
from torch import nn

MATRIX_NDIM = 2


class PiecewiseLinearEmbedding(nn.Module):
    """Apply piecewise-linear embeddings to continuous features.

    Each feature is divided into quantile bins and assigned a learnable
    embedding vector. The embedding is selected based on which bin the
    feature value falls into, with linear interpolation between bins.

    Args:
        n_features: Number of continuous features to embed
        n_bins: Number of quantile bins per feature (default: 48)
        embedding_dim: Dimension of each embedding vector (default: 12)
        version: Embedding version ('A' with activation, 'B' without)

    Reference:
        "Numerical Embeddings: A collection of techniques for embedding
        continuous features in neural networks"
    """

    def __init__(
        self,
        n_features: int,
        n_bins: int = 48,
        embedding_dim: int = 12,
        version: str = "B",
    ) -> None:
        super().__init__()

        if n_features <= 0:
            msg = "n_features must be positive"
            raise ValueError(msg)
        if n_bins <= 1:
            msg = "n_bins must be at least 2"
            raise ValueError(msg)
        if embedding_dim <= 0:
            msg = "embedding_dim must be positive"
            raise ValueError(msg)
        if version not in ("A", "B"):
            msg = "version must be 'A' or 'B'"
            raise ValueError(msg)

        self.n_features = n_features
        self.n_bins = n_bins
        self.embedding_dim = embedding_dim
        self.version = version

        # Learnable embedding vectors for each bin of each feature
        # Shape: (n_features, n_bins, embedding_dim)
        self.embeddings = nn.Parameter(torch.empty(n_features, n_bins, embedding_dim))

        # Bin boundaries (quantiles) - learned from data during preprocessing
        # These are stored as buffers but fitted externally and loaded
        self.register_buffer("bin_boundaries", torch.zeros(n_features, n_bins + 1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.embeddings, a=math.sqrt(5))

    def set_bin_boundaries(self, boundaries: torch.Tensor) -> None:
        """Set quantile bin boundaries from preprocessing.

        Args:
            boundaries: Tensor of shape (n_features, n_bins + 1) containing
                        the bin edges for each feature.
        """
        if boundaries.shape != (self.n_features, self.n_bins + 1):
            msg = f"Expected boundaries shape ({self.n_features}, {self.n_bins + 1}), got {boundaries.shape}"
            raise ValueError(msg)
        self.bin_boundaries = boundaries

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply piecewise-linear embedding to input features.

        Args:
            x: Input tensor of shape (batch_size, n_features)

        Returns
        -------
            Embedded tensor of shape (batch_size, n_features * embedding_dim).

        Note:
            This uses linear interpolation between bins. For each feature,
            we find which bin the value falls into and interpolate between
            the embeddings of the bounding bin edges.
        """
        if x.ndim != MATRIX_NDIM or x.shape[1] != self.n_features:
            msg = f"Expected input shape (batch, {self.n_features}), got {x.shape}"
            raise ValueError(msg)

        batch_size = x.shape[0]
        device = x.device

        # Expand input for vectorized operations
        # x_expanded: (batch, n_features, 1)
        x_expanded = x.unsqueeze(-1)

        # boundaries: (n_features, n_bins + 1) -> (1, n_features, n_bins + 1, 1)
        boundaries = self.bin_boundaries.unsqueeze(0).to(device)

        # Find which bin each value belongs to
        # bin_indices: (batch, n_features)
        bin_indices = torch.searchsorted(boundaries.squeeze(-1), x_expanded.squeeze(-1), right=False)
        bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)

        # Get embeddings for the left and right bins
        # embeddings: (n_features, n_bins, embedding_dim)
        all_embeddings = self.embeddings.to(device)

        # Gather the embeddings for the bin indices
        # left_embeddings: (batch, n_features, embedding_dim)
        left_embeddings = all_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)
        left_embeddings = torch.gather(
            left_embeddings, 2, bin_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.embedding_dim)
        ).squeeze(2)

        # Get right bin embeddings (with clamping)
        right_bin_indices = torch.clamp(bin_indices + 1, 0, self.n_bins - 1)
        right_embeddings = torch.gather(
            left_embeddings, 2, right_bin_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.embedding_dim)
        ).squeeze(2)

        # Get bin edges for interpolation
        # left_edges: (batch, n_features)
        left_edges = torch.gather(boundaries.squeeze(-1), 2, bin_indices.unsqueeze(-1)).squeeze(-1)
        right_edges = torch.gather(boundaries.squeeze(-1), 2, right_bin_indices.unsqueeze(-1)).squeeze(-1)

        # Compute interpolation weight
        bin_width = right_edges - left_edges
        bin_width = torch.clamp(bin_width, min=1e-6)  # Avoid division by zero

        weight = (x - left_edges) / bin_width
        weight = weight.unsqueeze(-1)  # (batch, n_features, 1)

        # Linear interpolation between left and right embeddings
        interpolated = left_embeddings + weight * (right_embeddings - left_embeddings)

        # Flatten to (batch, n_features * embedding_dim)
        output = interpolated.reshape(batch_size, self.n_features * self.embedding_dim)

        if self.version == "A":
            output = torch.relu(output)

        return output
