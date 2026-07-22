"""Module containing neural network architectures."""

import torch
from torch import nn
from data_utils import N_CLASSES


class FCNN(nn.Module):
    """
    One-hidden-layer classifier for the timestep-halving class (0 to N_CLASSES-1).

    Returns one logit per class (raw, no softmax — CrossEntropyLoss applies
    log-softmax itself). The predicted halving count is logits.argmax(dim=1).
    If class_priors is given, the head is initialised (weights zeroed, bias =
    log-priors) so the untrained model predicts exactly the empirical class
    distribution, input-independently — this removes the epochs otherwise
    spent learning the base rate, and is verified analytically in run_checks.
    
    """

    def __init__(self, n_features, n_hidden=64, class_priors =None):
        """
        Initialise the FCNN.

        :param class_priors: use class priors to account to data imbalance 
        :param n_hidden: Size of the hidden layer (defaults to 64).
        """
        super().__init__()
        # 267 input features -> 64 hidden neurons -> ReLU activation
        self.body = nn.Sequential(nn.Linear(n_features, n_hidden), nn.ReLU())
        # produces 5 logits (model's initial, unscaled confidence in each possible outcome)
        self.head = nn.Linear(n_hidden, N_CLASSES)
        """
        We introduce class priors to prevent the nn to develop a built in bias to towards the majority class
        """
        if class_priors is not None:
            p = torch.as_tensor(class_priors, dtype=torch.float32)
            # since we have 5 classes its better to check we have 5 prior probabilities in total
            if p.numel() != N_CLASSES:
                raise ValueError(f"Expected {N_CLASSES} class priors but got {p.numel()}.")
            if not torch.isfinite(p).all() or (p<0).any():
                raise ValueError("Class Priors must be finite and positive.")
            if p.sum() <= 0:
                raise ValueError("Atleast one class prior must be positive.")
            p = p / p.sum() # normalise the priors
            with torch.no_grad():
                self.head.bias.copy_(p.clamp_min(1e-12).log())

    def forward(self, x):
        """
        Forward method for the FCNN.

        :param x: input vector for the model
        """
        return self.head(self.body(x))
