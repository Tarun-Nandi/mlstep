"""Member-specific scaling and heads around a shared TabM-mini backbone."""

import torch
from torch import nn

# Keep the model building blocks available from one module.
from mlstep.embeddings import PiecewiseLinearEmbedding

__all__ = [
    "LinearEnsemble",
    "MiniEnsembleInputScaling",
    "PiecewiseLinearEmbedding",
    "SharedMLPBackbone",
]

ENSEMBLE_NDIM = 3


def _positive_integer(value: int, name: str) -> None:
    """Check dimensions shared by the three components before allocating parameters."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)


class MiniEnsembleInputScaling(nn.Module):
    """Give each ensemble member its own trainable input scaling.

    This creates different member representations before their inputs pass
    through the same backbone weights. No members are averaged here.
    """

    def __init__(
        self,
        k: int,
        n_features: int,
        *,
        initialization: str = "random-signs",
        feature_sizes: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """Allocate (members, features) scales with sign or grouped-normal initialisation.

        feature_sizes partitions input coordinates into original-feature groups:
        an embedding contributes several coordinates, a bypass scalar just one.
        Groups share initial normal values, not parameters during training.
        """
        super().__init__()
        _positive_integer(k, "k")
        _positive_integer(n_features, "n_features")
        if initialization not in ("random-signs", "normal"):
            msg = "initialization must be 'random-signs' or 'normal'"
            raise ValueError(msg)
        self.initialization = initialization
        self.feature_sizes = (1,) * n_features if feature_sizes is None else tuple(feature_sizes)
        if feature_sizes is not None:
            for size in self.feature_sizes:
                _positive_integer(size, "feature size")
            if sum(self.feature_sizes) != n_features:
                raise ValueError("feature_sizes must sum to n_features")
        self.weight = nn.Parameter(torch.empty(k, n_features))
        self.reset_parameters()

    @property
    def k(self) -> int:
        """Read the member count from the scaling parameter, avoiding duplicated state."""
        return self.weight.shape[0]

    @property
    def n_features(self) -> int:
        """Return the coordinate count after any embedding expansion."""
        return self.weight.shape[1]

    def reset_parameters(self) -> None:
        """Set scales to random ±1 signs or one normal draw per member and feature group.

        Normal draws are repeated across a group's coordinates. Subsequent
        optimisation updates each coordinate separately.
        """
        with torch.no_grad():
            if self.initialization == "random-signs":
                self.weight.bernoulli_(0.5).mul_(2.0).add_(-1.0)
            else:
                chunk_values = torch.empty(
                    self.k,
                    len(self.feature_sizes),
                    device=self.weight.device,
                    dtype=self.weight.dtype,
                )
                nn.init.normal_(chunk_values)
                expanded = torch.repeat_interleave(
                    chunk_values,
                    torch.tensor(self.feature_sizes, device=self.weight.device),
                    dim=1,
                    # The known size avoids deriving it from device-side repeats.
                    output_size=self.n_features,
                )
                self.weight.copy_(expanded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply (B, K, F) inputs by (K, F) scales, preserving all three axes.

        Check the member/feature axes so broadcasting cannot silently reuse one
        member or one feature across the ensemble.
        """
        if x.ndim != ENSEMBLE_NDIM or x.shape[1:] != self.weight.shape:
            msg = f"input scaling expects shape (batch, {self.k}, {self.n_features}), received {tuple(x.shape)}"
            raise ValueError(msg)
        return x * self.weight


class SharedMLPBackbone(nn.Module):
    """Mix features with the same hidden-layer weights for every ensemble member."""

    def __init__(
        self,
        n_features: int,
        hidden_layers: list[int],
        dropout: float,
    ) -> None:
        """Build Linear → ReLU → Dropout blocks for each hidden width.

        The last block also includes dropout. Leading batch/member axes pass
        through unchanged because Linear operates on the final dimension.
        """
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
        """Map (..., in_features) to (..., out_features) using shared weights.

        Linear checks input-width compatibility. Training uses independent
        dropout masks across activations; evaluation disables dropout.
        """
        return self.layers(x)


class LinearEnsemble(nn.Module):
    """Apply a separate learned linear projection to each ensemble member.

    One batched weight tensor holds all heads; no Python loop or mixing between
    members is needed during the forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        k: int,
        bias: bool = True,
    ) -> None:
        """Allocate weights (K, input_width, output_width) and optional (K, output_width) bias."""
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
        """Read the shared head input width from the weight tensor."""
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        """Read the output width per member: one detector logit or four severity logits."""
        return self.weight.shape[2]

    @property
    def k(self) -> int:
        """Read the number of independent member heads from the weight tensor."""
        return self.weight.shape[0]

    def reset_parameters(self) -> None:
        """Initialise each head uniformly within ±1/sqrt(input_width), including bias.

        Training later replaces detector biases with the population log odds.
        """
        bound = self.in_features**-0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, K, input_width) to (B, K, output_width), keeping members separate."""
        if x.ndim != ENSEMBLE_NDIM or x.shape[1] != self.k or x.shape[2] != self.in_features:
            msg = f"linear ensemble expects shape (batch, {self.k}, {self.in_features}), received {tuple(x.shape)}"
            raise ValueError(msg)

        # Make K the batch axis for matmul: (K, B, I) @ (K, I, O).
        # Transposes are views; they do not copy the member activations.
        output = (x.transpose(0, 1) @ self.weight).transpose(0, 1)
        if self.bias is not None:
            output = output + self.bias
        return output
