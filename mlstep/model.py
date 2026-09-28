"""Shared TabM-mini model for supervised training and distillation."""

from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn

from mlstep.data import N_CLASSES, N_FEATURES, PLE_N_BINS, PREPROCESSING_METHODS
from mlstep.embeddings import PLE_EMBEDDING_DIM
from mlstep.tabm import (
    LinearEnsemble,
    MiniEnsembleInputScaling,
    PiecewiseLinearEmbedding,
    SharedMLPBackbone,
)

N_MEMBERS: Final = 4
MIN_MEMBERS: Final = 2
HIDDEN_LAYERS: Final = (512, 256)
DROPOUT: Final = 0.05
N_SEVERITY_CLASSES: Final = N_CLASSES - 1
MATRIX_NDIM: Final = 2


class Student(nn.Module):
    """Predict exact halving counts through detection and conditional severity heads.

    The same network serves supervised and distilled training; teacher targets
    change the loss in student.py, not this architecture. Members share a backbone
    but have separate input scaling and output heads.
    """

    def __init__(
        self,
        n_features: int,
        hidden_layers: list[int] | tuple[int, ...] = HIDDEN_LAYERS,
        k: int = N_MEMBERS,
        dropout: float = DROPOUT,
        ple_indices: list[int] | tuple[int, ...] | np.ndarray | None = None,
    ) -> None:
        """Build the backbone and heads, optionally embedding selected input columns.

        PLE indices preserve cache column order. Embedded columns come first,
        followed by unembedded scalars in their original order. Use build_model
        for the fixed architecture expected by the training and loading commands.
        """
        super().__init__()
        if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
            msg = "n_features must be a positive integer"
            raise ValueError(msg)
        if k < MIN_MEMBERS:
            raise ValueError("TabM-mini requires at least two members")
        hidden = tuple(hidden_layers)
        # Backbone and scaling constructors validate widths, dropout and k's type.

        self.n_features = n_features
        self.k = k
        self.hidden_layers = hidden
        self.dropout = float(dropout)
        self.ple_indices: list[int] | None = None
        self.bypass_indices: list[int] | None = None
        self.ple: PiecewiseLinearEmbedding | None = None

        effective_n_features = n_features
        if ple_indices is not None:
            indices = np.asarray(ple_indices)
            if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("PLE indices must be a nonempty vector of integers")
            if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n_features):
                raise ValueError("PLE indices must be unique and in range")
            self.ple_indices = indices.tolist()
            selected_set = set(self.ple_indices)
            self.bypass_indices = [index for index in range(n_features) if index not in selected_set]
            self.ple = PiecewiseLinearEmbedding(
                n_features=len(self.ple_indices),
                n_bins=PLE_N_BINS,
                embedding_dim=PLE_EMBEDDING_DIM,
            )
            effective_n_features = self.ple.output_dim + len(self.bypass_indices)

        self.backbone = SharedMLPBackbone(effective_n_features, list(hidden), self.dropout)
        if self.ple is None:
            self.input_scaling = MiniEnsembleInputScaling(k, effective_n_features)
        else:
            feature_sizes = [PLE_EMBEDDING_DIM] * len(self.ple_indices)
            feature_sizes.extend([1] * len(self.bypass_indices))
            self.input_scaling = MiniEnsembleInputScaling(
                k,
                effective_n_features,
                initialization="normal",
                feature_sizes=feature_sizes,
            )
        self.detector_head = LinearEnsemble(hidden[-1], 1, k=k)
        self.severity_head = LinearEnsemble(hidden[-1], N_SEVERITY_CLASSES, k=k)

    def forward_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map preprocessed rows (B, F) to detector (K, B) and severity (K, B, 4) logits.

        Training supervises every member through these logits. Embeddings are
        shared and computed before expanding inputs across ensemble members.
        """
        if x.ndim != MATRIX_NDIM or x.shape[1] != self.n_features:
            msg = f"Student expects shape (batch, {self.n_features}), received {tuple(x.shape)}"
            raise ValueError(msg)

        if self.ple is not None:
            embedded = self.ple(x[:, self.ple_indices])
            x = torch.cat((embedded, x[:, self.bypass_indices]), dim=1)

        # expand creates a view; the subsequent scaling allocates member activations.
        members = self.input_scaling(x.unsqueeze(1).expand(-1, self.k, -1))
        representation = self.backbone(members)
        detector = self.detector_head(representation).squeeze(-1).transpose(0, 1)
        severity = self.severity_head(representation).transpose(0, 1)
        return detector, severity

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return detector logits, P(halving), severity logits and P(count | halving).

        Logits retain member-first shapes (K, B) and (K, B, 4); probabilities
        have shapes (B,) and (B, 4). Probability arithmetic stays float32 under
        mixed precision, even when the backbone and heads use lower precision.
        """
        detector_logits, severity_logits = self.forward_logits(x)
        member_detector = torch.sigmoid(detector_logits.float())
        member_severity = torch.softmax(severity_logits.float(), dim=-1)
        detector = member_detector.mean(dim=0)
        # Average each member's joint mass, not the two heads independently.
        positive_mass = (member_detector.unsqueeze(-1) * member_severity).mean(dim=0)
        # Avoid division by zero if all detector probabilities underflow to zero.
        severity = positive_mass / detector.unsqueeze(-1).clamp_min(torch.finfo(detector.dtype).tiny)
        return detector_logits, detector, severity_logits, severity

    def initialize_detector_bias(self, population_log_odds: float) -> None:
        """Set detector bias to log(positive / negative) from the full training population.

        This gives the rare-event detector a useful starting offset. Random
        weights still contribute, so initial predictions need not equal the prior.
        """
        if not np.isfinite(population_log_odds):
            msg = "population_log_odds must be finite"
            raise ValueError(msg)
        with torch.no_grad():
            self.detector_head.bias.fill_(population_log_odds)

    def set_ple_bin_boundaries(self, boundaries: np.ndarray | torch.Tensor) -> None:
        """Copy cache-fitted quantiles into the PLE layer before its first forward pass.

        Accept NumPy or torch boundaries in selected-feature order. The embedding
        layer checks shape, finiteness and ordering; no quantiles are fitted here.
        """
        if self.ple is None:
            msg = "model has no PLE layer"
            raise RuntimeError(msg)
        values = torch.as_tensor(boundaries, dtype=torch.float32)
        self.ple.set_bin_boundaries(values)


def build_model(preprocess: str, ple_indices: list[int] | tuple[int, ...] | np.ndarray | None = None) -> Student:
    """Build the fixed 267-input, four-member architecture used by training and loading.

    Physical and Stretch128 inputs use scalars; PLE requires selected column
    indices. Fitted transforms remain in the cache, outside the neural network.
    """
    if preprocess not in PREPROCESSING_METHODS:
        msg = f"preprocess must be one of {PREPROCESSING_METHODS}"
        raise ValueError(msg)
    if preprocess != "ple" and ple_indices is not None:
        msg = "only ple preprocessing uses embedding indices"
        raise ValueError(msg)
    if preprocess == "ple" and ple_indices is None:
        msg = "ple preprocessing requires embedding feature indices"
        raise ValueError(msg)
    return Student(N_FEATURES, HIDDEN_LAYERS, N_MEMBERS, DROPOUT, ple_indices)


def save_model(path: Path | str, model: Student, preprocess: str) -> Path:
    """Write a new weights artifact for a model built with the standard factory.

    Save CPU tensors, preprocessing name and PLE indices; never overwrite a file.
    Optimiser state and fitted scalar preprocessing are omitted, so keep the
    matching cache. Custom architecture settings are not stored in this format.
    """
    if preprocess not in PREPROCESSING_METHODS:
        msg = f"preprocess must be one of {PREPROCESSING_METHODS}"
        raise ValueError(msg)
    if (preprocess == "ple") != (model.ple is not None):
        raise ValueError("model embeddings do not match the preprocessing method")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    with destination.open("xb") as handle:
        torch.save(
            {"preprocess": preprocess, "ple_indices": model.ple_indices, "state_dict": state},
            handle,
        )
    return destination


def load_model(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[Student, str]:
    """Rebuild a standard model, restore its weights and return it in evaluation mode.

    Load tensors on CPU first, then move the model to the requested device.
    The factory checks preprocessing settings; strict state loading checks saved
    parameter names and shapes. The caller must supply the matching cache for
    preprocessing raw data. This does not resume optimiser or training state.
    """
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    preprocess = artifact["preprocess"]
    model = build_model(preprocess, artifact["ple_indices"])
    model.load_state_dict(artifact["state_dict"], strict=True)
    return model.to(torch.device(device)).eval(), preprocess
