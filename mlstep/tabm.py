"""Small PyTorch building blocks for a faithful TabM-mini mechanism."""

import torch
from torch import nn

# Re-export PiecewiseLinearEmbedding for use in student.py
from mlstep.embeddings import PiecewiseLinearEmbedding

__all__ = [
    "LinearEnsemble",
    "MiniEnsembleInputScaling",
    "PiecewiseLinearEmbedding",
    "SharedMLPBackbone",
]

ENSEMBLE_NDIM = 3
MIN_MLP_NDIM = 2


def _positive_integer(value: int, name: str) -> None:
    """Reject booleans and non-positive integer settings."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)


class MiniEnsembleInputScaling(nn.Module):
    """Apply a trainable, member-specific scaling before feature mixing."""

    def __init__(self, k: int, n_features: int) -> None:
        super().__init__()
        _positive_integer(k, "k")
        _positive_integer(n_features, "n_features")
        self.weight = nn.Parameter(torch.empty(k, n_features))
        self.reset_parameters()

    @property
    def k(self) -> int:
        """Return the number of ensemble members."""
        return self.weight.shape[0]

    @property
    def n_features(self) -> int:
        """Return the input feature count."""
        return self.weight.shape[1]

    def reset_parameters(self) -> None:
        """Initialize every member-feature scaling independently to -1 or 1."""
        with torch.no_grad():
            self.weight.bernoulli_(0.5).mul_(2.0).add_(-1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale an ensemble tensor with shape ``(batch, members, features)``."""
        if x.ndim != ENSEMBLE_NDIM or x.shape[1:] != self.weight.shape:
            msg = f"input scaling expects shape (batch, {self.k}, {self.n_features}), received {tuple(x.shape)}"
            raise ValueError(msg)
        return x * self.weight


class SharedMLPBackbone(nn.Module):
    """MLP backbone shared by every ensemble member."""

    def __init__(
        self,
        n_features: int,
        hidden_layers: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        _positive_integer(n_features, "n_features")
        if not hidden_layers:
            msg = "hidden_layers must contain at least one width"
            raise ValueError(msg)
        for width in hidden_layers:
            _positive_integer(width, "hidden layer width")
        if not 0.0 <= dropout < 1.0:
            msg = "dropout must be in [0, 1)"
            raise ValueError(msg)

        blocks: list[nn.Module] = []
        in_features = n_features
        for out_features in hidden_layers:
            blocks.extend(
                (
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            in_features = out_features

        self.layers = nn.Sequential(*blocks)
        self.in_features = n_features
        self.out_features = hidden_layers[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared MLP to the final feature dimension."""
        if x.ndim < MIN_MLP_NDIM or x.shape[-1] != self.in_features:
            msg = f"backbone expects final dimension {self.in_features}, received {tuple(x.shape)}"
            raise ValueError(msg)
        return self.layers(x)


class LinearEnsemble(nn.Module):
    """Apply ``k`` independent linear heads to ``k`` member representations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        k: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        _positive_integer(in_features, "in_features")
        _positive_integer(out_features, "out_features")
        _positive_integer(k, "k")

        self.weight = nn.Parameter(torch.empty(k, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(k, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @property
    def in_features(self) -> int:
        """Return the input width of every head."""
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        """Return the output width of every head."""
        return self.weight.shape[2]

    @property
    def k(self) -> int:
        """Return the number of independent heads."""
        return self.weight.shape[0]

    def reset_parameters(self) -> None:
        """Match the independent-linear initialization used by TabM."""
        bound = self.in_features**-0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each head to its corresponding ensemble member."""
        if x.ndim != ENSEMBLE_NDIM or x.shape[1] != self.k or x.shape[2] != self.in_features:
            msg = f"linear ensemble expects shape (batch, {self.k}, {self.in_features}), received {tuple(x.shape)}"
            raise ValueError(msg)

        output = (x.transpose(0, 1) @ self.weight).transpose(0, 1)
        if self.bias is not None:
            output = output + self.bias
        return output
