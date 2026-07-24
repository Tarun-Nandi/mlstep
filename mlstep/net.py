"""Neural network used by the timestep-halving experiment."""

import torch
from torch import nn


class FCNN(nn.Module):
    """A small one-hidden-layer classifier."""

    def __init__(self, n_features, n_classes, n_hidden=64, init_logits=None):
        super().__init__()
        self.hidden = nn.Linear(n_features, n_hidden)
        self.output = nn.Linear(n_hidden, n_classes)

        if init_logits is not None:
            init_logits = torch.as_tensor(init_logits, dtype=torch.float32)
            if init_logits.numel() != n_classes:
                raise ValueError(
                    f"Expected {n_classes} initial logits, got {init_logits.numel()}."
                )
            with torch.no_grad():
                self.output.bias.copy_(init_logits)

    def forward(self, x):
        """Return raw logits; cross-entropy applies softmax internally."""
        return self.output(torch.relu(self.hidden(x)))
