"""Unified MLP and TabM-mini students with optional XGBoost distillation.

Both architectures use a shared backbone with a population-scale detector
head and a conditional four-class severity head. TabM-mini additionally uses
random member-specific input scaling and independent member heads.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mlstep.data import N_CLASSES, random_undersampling, training_index_pools
from mlstep.evaluation import (
    detection_ap,
    epsilon_frontier,
    evaluate,
    halving_action_metrics,
    output_paths,
    probability_metrics,
    threshold_at_recall,
    write_json,
)
from mlstep.policy import EpsilonPolicy, boundary_probabilities
from mlstep.tabm import LinearEnsemble, MiniEnsembleInputScaling, SharedMLPBackbone

# Default paths
DEFAULT_OUTPUT_DIR: Final = Path(__file__).resolve().parent / "runs"
# NOTE: change the directory (update to make all three directories accessible)
DEFAULT_CACHE_DIR: Final = Path("/rds/user/rc-nand1/hpc-work/mlstep/cache")

# Architecture defaults
ARCHITECTURES: Final = ("mlp", "tabm-mini")
DEFAULT_ARCHITECTURE: Final = "mlp"
MIN_TABM_MEMBERS: Final = 2
MODEL_FORMAT_VERSION: Final = 2
ARCHITECTURE_VERSIONS: Final = {
    "mlp": "mlp-dual-head-v1",
    "tabm-mini": "tabm-mini-dual-head-v1",
}
HIDDEN_LAYERS: Final = [512, 256]
DROPOUT: Final = 0.05
GRADIENT_CLIP_NORM: Final = 5.0
MATRIX_NDIM: Final = 2
N_SEVERITY_CLASSES: Final = N_CLASSES - 1

# Training defaults
DEFAULT_EPOCHS: Final = 120
DEFAULT_PATIENCE: Final = 12
DEFAULT_BATCH_SIZE: Final = 1024
DEFAULT_EVAL_BATCH_SIZE: Final = 65536
DEFAULT_NEGATIVE_RATIO: Final = 256
DEFAULT_TARGET_RECALL: Final = 0.97
CACHE_RESIDENCIES: Final = ("auto", "host", "cuda")
DEFAULT_CACHE_RESIDENCY: Final = "auto"
DEFAULT_LAMBDA_SEV: Final = 0.1
DEFAULT_ALPHA: Final = 1.0
DEFAULT_BETA: Final = 0.1
DEFAULT_KD_TEMPERATURE: Final = 1.0
GPU_DATA_RESERVE_BYTES: Final = 16 * 2**30
LOAD_CHUNK_BYTES: Final = 512 * 2**20
PROGRESS_UPDATES_PER_PHASE: Final = 10
EPOCH_SEED_STRIDE: Final = 10_000
SAMPLING_SEED_POLICY: Final = "run-seed-times-10000-plus-epoch-v1"
DETECTOR_BIAS_POLICY: Final = "population-log-odds-v1"
LOSS_COMPONENT_NAMES: Final = (
    "total",
    "hard_detector",
    "hard_severity",
    "kd_detector",
    "kd_severity",
)
MEMBER_DIAGNOSTIC_TARGET_ROWS: Final = 262_144
MEMBER_DIAGNOSTIC_SUBSET_POLICY: Final = "all-positives-plus-highest-ensemble-scores-v1"

# Checkpoints are selected before any hard action policy is fitted. RPS is the
# primary proper score because the class labels are ordered; NLL and detector
# AP are deterministic tie-breakers. The detector threshold and ASAD policy
# are selected only after restoring the chosen probabilistic model.
SELECTION_POLICY: Final = "ordinal-rps-nll-v1"
THRESHOLD_RULE: Final = "highest validation threshold achieving target recall"
THRESHOLD_SCORE: Final = "ensemble any-halving detector probability"
SELECTION_CRITERIA: Final = (
    "minimize probability.ranked_probability_score",
    "minimize probability.negative_log_likelihood",
    "maximize detector average precision",
)
SEVERITY_DECISION: Final = "argmax conditional severity probability"

# This optional checkpoint is a ranking diagnostic only. It never participates
# in early stopping, threshold selection, or the canonical final diagnostics.
AP_SELECTION_POLICY: Final = "detector-ap-rps-nll-diagnostic-v1"
AP_SELECTION_CRITERIA: Final = (
    "maximize detector average precision",
    "minimize probability.ranked_probability_score (tie-breaker)",
    "minimize probability.negative_log_likelihood (tie-breaker)",
)

# Development-only ASAD policy diagnostic. These probabilities are not yet
# calibrated on an independent chronological block, so no epsilon is selected
# for deployment in this version.
EPSILON_POLICY_VERSION: Final = "epsilon-boundary-v1"
WEEK_MAX_ACTION: Final = 3
POLICY_DIAGNOSTIC_CONFIDENCES: Final = (1.0, 0.999, 0.995, 0.99, 0.975, 0.95, 0.9, 0.8, 0.7, 0.5)
POLICY_DIAGNOSTIC_RECALLS: Final = (0.80, 0.90, 0.95, 0.97, 0.99, 1.0)

FeatureMatrix = np.ndarray | torch.Tensor


@dataclass(frozen=True, slots=True)
class TrainEpochResult:
    """Loss and phase timings for one unchanged undersampled epoch."""

    loss: float
    loss_components: dict[str, float]
    sampled_rows: int
    batches: int
    sampling_seconds: float
    staging_seconds: float
    optimization_seconds: float


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    """Weighted terms whose sum is the optimized scalar loss."""

    total: torch.Tensor
    hard_detector: torch.Tensor
    hard_severity: torch.Tensor
    kd_detector: torch.Tensor
    kd_severity: torch.Tensor

    def stacked(self) -> torch.Tensor:
        """Return terms in the stable order used by training reports."""
        return torch.stack(
            (
                self.total,
                self.hard_detector,
                self.hard_severity,
                self.kd_detector,
                self.kd_severity,
            )
        )


@dataclass(frozen=True, slots=True)
class PreparedEpoch:
    """Contiguous device tensors for one preselected epoch."""

    x: torch.Tensor
    y: torch.Tensor
    teacher_margin: torch.Tensor | None
    teacher_severity: torch.Tensor | None
    batch_has_positive: tuple[bool, ...]
    sampled_rows: int
    batches: int
    sampling_seconds: float
    staging_seconds: float


class Student(nn.Module):
    """Matched MLP or faithful TabM-mini two-head student.

    The public logits retain the member-first shapes used by ``compute_loss``:
    detector logits have shape ``(K, B)`` and severity logits have shape
    ``(K, B, 4)``. The MLP is the K=1 control. TabM-mini creates distinct
    representations before the first feature-mixing layer, shares the entire
    backbone, and uses independent member heads.
    """

    def __init__(
        self,
        n_features: int,
        hidden_layers: list[int] | None = None,
        k: int = 1,
        dropout: float = DROPOUT,
        architecture: str = DEFAULT_ARCHITECTURE,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            msg = f"architecture must be one of {ARCHITECTURES}"
            raise ValueError(msg)
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            msg = "k must be a positive integer"
            raise ValueError(msg)
        if architecture == "mlp" and k != 1:
            msg = "The MLP control requires exactly one member"
            raise ValueError(msg)
        if architecture == "tabm-mini" and k < MIN_TABM_MEMBERS:
            msg = "TabM-mini requires at least two ensemble members"
            raise ValueError(msg)

        hidden_layers = list(HIDDEN_LAYERS if hidden_layers is None else hidden_layers)
        if not hidden_layers:
            msg = "hidden_layers must contain at least one width"
            raise ValueError(msg)

        self.architecture = architecture
        self.architecture_version = ARCHITECTURE_VERSIONS[architecture]
        self.k = k
        self.n_features = n_features
        self.hidden_layers = tuple(hidden_layers)
        self.dropout = dropout

        self.backbone = SharedMLPBackbone(
            n_features,
            hidden_layers,
            dropout,
        )

        if architecture == "mlp":
            self.input_scaling = None
            self.detector_head: nn.Module = nn.Linear(hidden_layers[-1], 1)
            self.severity_head: nn.Module = nn.Linear(hidden_layers[-1], N_SEVERITY_CLASSES)
        else:
            self.input_scaling = MiniEnsembleInputScaling(k, n_features)
            self.detector_head = LinearEnsemble(hidden_layers[-1], 1, k=k)
            self.severity_head = LinearEnsemble(hidden_layers[-1], N_SEVERITY_CLASSES, k=k)

    def forward_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return member logits without computing unused training probabilities."""
        if x.ndim != MATRIX_NDIM or x.shape[1] != self.n_features:
            msg = f"Student expects shape (batch, {self.n_features}), received {tuple(x.shape)}"
            raise ValueError(msg)

        if self.architecture == "mlp":
            representation = self.backbone(x)
            d_logit = self.detector_head(representation).squeeze(-1).unsqueeze(0)
            s_logits = self.severity_head(representation).unsqueeze(0)
        else:
            # A copy-free view becomes K distinct representations through the
            # random-sign trainable scaling before any features are mixed.
            member_inputs = x.unsqueeze(1).expand(-1, self.k, -1)
            member_inputs = self.input_scaling(member_inputs)
            representation = self.backbone(member_inputs)
            d_logit = self.detector_head(representation).squeeze(-1).transpose(0, 1)
            s_logits = self.severity_head(representation).transpose(0, 1)

        return d_logit, s_logits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the model and combine ensemble probabilities."""
        d_logit, s_logits = self.forward_logits(x)

        detector_probability = torch.sigmoid(d_logit)
        severity_probability = torch.softmax(s_logits, dim=-1)

        # Average each member's joint distribution, rather than multiplying
        # independently averaged detector and severity probabilities.
        d = detector_probability.mean(dim=0)
        positive_mass = (detector_probability.unsqueeze(-1) * severity_probability).mean(dim=0)
        denominator = d.unsqueeze(-1).clamp_min(torch.finfo(d.dtype).tiny)
        s = positive_mass / denominator
        return d_logit, d, s_logits, s

    def joint_distribution(self, d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Construct full 5-class distribution from detection and severity."""
        return torch.cat(
            ((1.0 - d).unsqueeze(-1), d.unsqueeze(-1) * s),
            dim=-1,
        )

    def initialize_detector_bias(self, population_log_odds: float) -> None:
        """Initialize every detector member to the population class prior."""
        if not np.isfinite(population_log_odds):
            msg = "population_log_odds must be finite"
            raise ValueError(msg)
        bias = self.detector_head.bias
        if bias is None:
            msg = "The detector head must have a bias for prior initialization"
            raise RuntimeError(msg)
        with torch.no_grad():
            bias.fill_(population_log_odds)


def student_from_checkpoint(checkpoint: dict) -> Student:
    """Reconstruct a format-v2 student and strictly load its parameters.

    Unversioned checkpoints belong to the earlier shared-head adapter model and
    are intentionally not guessed to be either of the new architecture types.
    """
    if not isinstance(checkpoint, dict):
        msg = "Student checkpoint must be a dictionary"
        raise TypeError(msg)
    if checkpoint.get("format_version") != MODEL_FORMAT_VERSION:
        msg = (
            f"Unsupported student checkpoint format; expected version {MODEL_FORMAT_VERSION}. "
            "Unversioned checkpoints use the legacy adapter architecture."
        )
        raise ValueError(msg)

    config = checkpoint.get("config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        msg = "Student checkpoint must contain dictionary config and state_dict entries"
        raise ValueError(msg)

    required = {"architecture", "architecture_version", "n_features", "hidden", "k", "dropout"}
    missing = sorted(required - config.keys())
    if missing:
        msg = f"Student checkpoint config is missing: {', '.join(missing)}"
        raise ValueError(msg)

    architecture = config["architecture"]
    expected_architecture_version = ARCHITECTURE_VERSIONS.get(architecture)
    if config["architecture_version"] != expected_architecture_version:
        msg = "Student checkpoint architecture version is unsupported"
        raise ValueError(msg)

    model = Student(
        n_features=config["n_features"],
        hidden_layers=config["hidden"],
        k=config["k"],
        dropout=config["dropout"],
        architecture=architecture,
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def compute_loss(
    d_logit: torch.Tensor,
    s_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_margin: torch.Tensor | None = None,
    teacher_severity: torch.Tensor | None = None,
    delta_s: float = 0.0,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    lambda_sev: float = DEFAULT_LAMBDA_SEV,
    has_positive: bool | None = None,
    kd_temperature: float = DEFAULT_KD_TEMPERATURE,
    return_breakdown: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, LossBreakdown]:
    """Compute hard-label and optional response-distillation losses.

    Temperature applies to both response-distillation terms. The teacher
    detector probability used as the conditional-severity relevance weight
    remains untempered.
    """
    n_members = d_logit.shape[0]
    if not np.isfinite(kd_temperature) or kd_temperature <= 0.0:
        msg = "kd_temperature must be finite and positive"
        raise ValueError(msg)
    if (teacher_margin is None) != (teacher_severity is None):
        msg = "Teacher margin and severity targets must either both be present or both absent"
        raise ValueError(msg)

    # Detector ground truth
    binary_target = (labels > 0).to(d_logit.dtype)
    binary_target = binary_target.unsqueeze(0).expand_as(d_logit)
    # d_logit represents the population margin. Negative undersampling makes
    # the sampled-data log odds larger by delta_s, so this shift is used only
    # when comparing against sampled hard labels.
    sampled_margin = d_logit + delta_s
    detector_loss = F.binary_cross_entropy_with_logits(
        sampled_margin,
        binary_target,
    )
    # Severity ground truth, evaluated for every member and positive row
    severity_loss = d_logit.new_zeros(())
    positive_mask = labels > 0

    # The training loop supplies this CPU-derived flag to avoid synchronising
    # the GPU merely to decide whether an unusually sparse batch has positives.
    if has_positive is None:
        has_positive = bool(positive_mask.any())

    if has_positive:
        positive_logits = s_logits[:, positive_mask, :]
        severity_target = labels[positive_mask] - 1
        severity_target = severity_target.unsqueeze(0).expand(n_members, -1)

        severity_loss = F.cross_entropy(
            positive_logits.reshape(-1, positive_logits.shape[-1]),
            severity_target.reshape(-1),
        )

    hard_detector = detector_loss
    hard_severity = lambda_sev * severity_loss

    # Optional distillation
    kd_detector = d_logit.new_zeros(())
    kd_severity = d_logit.new_zeros(())

    if teacher_margin is not None and teacher_severity is not None:
        teacher_margin_members = teacher_margin.unsqueeze(0).expand_as(d_logit)

        # Both are population-scale margins. Distil their Bernoulli responses
        # instead of matching arbitrary logit magnitude in the far-negative
        # tail. The T^2 factor retains the standard temperature scaling.
        teacher_detector_target = torch.sigmoid(teacher_margin_members / kd_temperature)
        detector_kd_loss = (
            F.binary_cross_entropy_with_logits(
                d_logit / kd_temperature,
                teacher_detector_target,
            )
            * kd_temperature**2
        )

        # At T=1 this is exactly the original conditional-severity KD target.
        teacher_probability = teacher_severity.clamp_min(torch.finfo(s_logits.dtype).tiny)
        teacher_probability = teacher_probability / teacher_probability.sum(dim=-1, keepdim=True)
        if kd_temperature == 1.0:
            student_log_probability = F.log_softmax(s_logits, dim=-1)
        else:
            teacher_probability = torch.softmax(
                torch.log(teacher_probability) / kd_temperature,
                dim=-1,
            )
            student_log_probability = F.log_softmax(
                s_logits / kd_temperature,
                dim=-1,
            )
        teacher_probability = teacher_probability.unsqueeze(0).expand(n_members, -1, -1)

        severity_kl = F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="none",
        ).sum(dim=-1)

        # Average member KLs first so the objective is invariant to K, then
        # normalize by the teacher-positive mass represented in this batch.
        teacher_detection_probability = torch.sigmoid(teacher_margin)
        severity_kl = severity_kl.mean(dim=0)
        positive_mass = teacher_detection_probability.sum().clamp_min(torch.finfo(severity_kl.dtype).tiny)
        severity_kl = (teacher_detection_probability * severity_kl).sum() / positive_mass * kd_temperature**2

        kd_detector = alpha * detector_kd_loss
        kd_severity = beta * severity_kl

    total_loss = hard_detector + hard_severity + kd_detector + kd_severity
    breakdown = LossBreakdown(
        total=total_loss,
        hard_detector=hard_detector,
        hard_severity=hard_severity,
        kd_detector=kd_detector,
        kd_severity=kd_severity,
    )

    return (total_loss, breakdown) if return_breakdown else total_loss


def checkpoint_selection_key(metrics: dict) -> tuple[float, float, float]:
    """Return a threshold-free proper-score checkpoint key; larger is better."""
    probability = metrics.get("probability")
    if probability is None:
        msg = "Checkpoint selection requires joint probability metrics"
        raise ValueError(msg)

    return (
        -float(probability["ranked_probability_score"]),
        -float(probability["negative_log_likelihood"]),
        float(metrics["ap"]),
    )


def ap_checkpoint_selection_key(metrics: dict) -> tuple[float, float, float]:
    """Return the AP-first diagnostic checkpoint key; larger is better."""
    probability = metrics.get("probability")
    if probability is None:
        msg = "AP checkpoint selection requires joint probability metrics"
        raise ValueError(msg)

    return (
        float(metrics["ap"]),
        -float(probability["ranked_probability_score"]),
        -float(probability["negative_log_likelihood"]),
    )


def _synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work when recording a phase duration."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _progress_interval(n_batches: int) -> int:
    """Print roughly ten progress updates without synchronising every batch."""
    return max(1, n_batches // PROGRESS_UPDATES_PER_PHASE)


def _load_array_resident(path: Path) -> np.ndarray:
    """Read an NPY array sequentially into RAM with visible progress.

    The old student kept the feature matrices as RDS-backed memory maps and
    then accessed shuffled rows. Copying sequentially once avoids repeated
    remote page faults while keeping every cached value unchanged.
    """
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    if source.ndim == 0:
        return np.asarray(source).copy()

    destination = np.empty(source.shape, dtype=source.dtype)
    bytes_per_row = max(1, source[0:1].nbytes)
    rows_per_chunk = max(1, LOAD_CHUNK_BYTES // bytes_per_row)
    n_chunks = int(np.ceil(len(source) / rows_per_chunk))
    progress_every = _progress_interval(n_chunks)
    started = time.perf_counter()
    print(
        f"Loading {path.name}: {source.nbytes / 2**30:.2f} GiB in {n_chunks} sequential chunk(s)",
        flush=True,
    )
    for chunk_number, start in enumerate(range(0, len(source), rows_per_chunk), start=1):
        stop = min(start + rows_per_chunk, len(source))
        destination[start:stop] = source[start:stop]
        if chunk_number % progress_every == 0 or chunk_number == n_chunks:
            elapsed = time.perf_counter() - started
            print(
                f"  {path.name}: {stop:,}/{len(source):,} rows ({stop / len(source):.0%}) in {elapsed:.1f}s",
                flush=True,
            )
    return destination


def _place_feature_matrices(
    train_x: np.ndarray,
    val_x: np.ndarray,
    device: torch.device,
    requested_residency: str,
) -> tuple[FeatureMatrix, FeatureMatrix, str, float]:
    """Keep feature matrices in host RAM or place them once on the GPU."""
    if requested_residency not in CACHE_RESIDENCIES:
        msg = f"cache residency must be one of {CACHE_RESIDENCIES}"
        raise ValueError(msg)

    feature_bytes = train_x.nbytes + val_x.nbytes
    use_cuda = requested_residency == "cuda"
    if requested_residency == "auto" and device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        use_cuda = feature_bytes + GPU_DATA_RESERVE_BYTES <= free_bytes

    if use_cuda and device.type != "cuda":
        msg = "CUDA cache residency requires --device cuda"
        raise ValueError(msg)

    if not use_cuda:
        if requested_residency == "auto" and device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            print(
                "Feature cache will remain in host RAM: "
                f"need {feature_bytes / 2**30:.2f} GiB plus a "
                f"{GPU_DATA_RESERVE_BYTES / 2**30:.0f} GiB reserve, "
                f"but only {free_bytes / 2**30:.2f} GiB is free",
                flush=True,
            )
        return train_x, val_x, "host", 0.0

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    required_bytes = feature_bytes + GPU_DATA_RESERVE_BYTES
    if required_bytes > free_bytes:
        msg = (
            f"CUDA feature residency needs {feature_bytes / 2**30:.2f} GiB plus "
            f"a {GPU_DATA_RESERVE_BYTES / 2**30:.0f} GiB working reserve, but "
            f"only {free_bytes / 2**30:.2f}/{total_bytes / 2**30:.2f} GiB is free"
        )
        raise MemoryError(msg)

    print(
        f"Staging {feature_bytes / 2**30:.2f} GiB of features on {device} "
        f"(free before staging: {free_bytes / 2**30:.2f} GiB)",
        flush=True,
    )
    started = time.perf_counter()
    train_device = torch.from_numpy(train_x).to(device)
    print("  train_x copied to GPU", flush=True)
    val_device = torch.from_numpy(val_x).to(device)
    _synchronize(device)
    elapsed = time.perf_counter() - started
    allocated = torch.cuda.memory_allocated(device)
    print(
        f"  val_x copied to GPU; feature staging took {elapsed:.1f}s, CUDA allocated {allocated / 2**30:.2f} GiB",
        flush=True,
    )
    return train_device, val_device, "cuda", elapsed


def _selected_rows_to_device(
    values: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Gather NumPy rows once in the existing shuffled order and transfer once."""
    selected = np.ascontiguousarray(values[indices])
    tensor = torch.from_numpy(selected)
    return tensor.to(device=device, dtype=dtype if dtype is not None else tensor.dtype)


def _prepare_epoch(
    features: FeatureMatrix,
    labels: np.ndarray,
    teacher_margin: np.ndarray | None,
    teacher_severity: np.ndarray | None,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    negative_ratio: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> PreparedEpoch:
    """Sample once and gather the unchanged shuffled epoch into contiguous tensors."""
    sampling_start = time.perf_counter()
    indices = random_undersampling(
        positive_indices,
        negative_indices,
        negative_ratio,
        seed,
    )
    if not len(indices):
        msg = "An undersampled epoch must contain at least one training row"
        raise ValueError(msg)
    sampling_seconds = time.perf_counter() - sampling_start

    n_batches = int(np.ceil(len(indices) / batch_size))
    selected_labels = np.ascontiguousarray(labels[indices])
    batch_has_positive = tuple(
        bool(np.any(selected_labels[start : start + batch_size] > 0)) for start in range(0, len(indices), batch_size)
    )

    staging_start = time.perf_counter()
    if isinstance(features, torch.Tensor):
        wrong_device_type = features.device.type != device.type
        wrong_device_index = device.index is not None and features.device.index != device.index
        if wrong_device_type or wrong_device_index:
            msg = f"Feature tensor is on {features.device}, expected {device}"
            raise ValueError(msg)
        index_tensor = torch.from_numpy(indices).to(device)
        epoch_x = torch.index_select(features, 0, index_tensor)
        del index_tensor
    else:
        epoch_x = _selected_rows_to_device(features, indices, device)
    epoch_y = torch.from_numpy(selected_labels).to(device)

    epoch_teacher_margin = None
    epoch_teacher_severity = None
    if teacher_margin is not None and teacher_severity is not None:
        epoch_teacher_margin = _selected_rows_to_device(
            teacher_margin,
            indices,
            device,
            dtype=torch.float32,
        )
        epoch_teacher_severity = _selected_rows_to_device(
            teacher_severity,
            indices,
            device,
            dtype=torch.float32,
        )
    _synchronize(device)
    return PreparedEpoch(
        x=epoch_x,
        y=epoch_y,
        teacher_margin=epoch_teacher_margin,
        teacher_severity=epoch_teacher_severity,
        batch_has_positive=batch_has_positive,
        sampled_rows=len(indices),
        batches=n_batches,
        sampling_seconds=sampling_seconds,
        staging_seconds=time.perf_counter() - staging_start,
    )


def train_epoch(
    model: Student,
    optimizer: torch.optim.Optimizer,
    features: FeatureMatrix,
    labels: np.ndarray,
    teacher_margin: np.ndarray | None,
    teacher_severity: np.ndarray | None,
    device: torch.device,
    batch_size: int,
    negative_ratio: int,
    seed: int,
    delta_s: float,
    alpha: float,
    beta: float,
    positive_indices: np.ndarray | None = None,
    negative_indices: np.ndarray | None = None,
    return_details: bool = False,
    lambda_sev: float = DEFAULT_LAMBDA_SEV,
    kd_temperature: float = DEFAULT_KD_TEMPERATURE,
) -> float | TrainEpochResult:
    """Train for one epoch with random undersampling."""
    model.train()

    if positive_indices is None or negative_indices is None:
        positive_indices, negative_indices = training_index_pools(labels)

    prepared = _prepare_epoch(
        features,
        labels,
        teacher_margin,
        teacher_severity,
        positive_indices,
        negative_indices,
        negative_ratio,
        seed,
        batch_size,
        device,
    )

    loss_values = torch.empty(
        (prepared.batches, len(LOSS_COMPONENT_NAMES)),
        device=device,
        dtype=torch.float32,
    )
    progress_every = _progress_interval(prepared.batches)
    optimization_start = time.perf_counter()

    for batch_number, start_idx in enumerate(range(0, prepared.sampled_rows, batch_size), start=1):
        stop_idx = min(start_idx + batch_size, prepared.sampled_rows)
        batch_x = prepared.x[start_idx:stop_idx]
        batch_y = prepared.y[start_idx:stop_idx]

        # Get teacher targets for this batch (if distilling)
        batch_teacher_margin = None
        batch_teacher_severity = None
        if prepared.teacher_margin is not None and prepared.teacher_severity is not None:
            batch_teacher_margin = prepared.teacher_margin[start_idx:stop_idx]
            batch_teacher_severity = prepared.teacher_severity[start_idx:stop_idx]

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        d_logit, s_logits = model.forward_logits(batch_x)

        # Compute loss
        loss_result = compute_loss(
            d_logit,
            s_logits,
            batch_y,
            batch_teacher_margin,
            batch_teacher_severity,
            delta_s=delta_s,
            alpha=alpha,
            beta=beta,
            lambda_sev=lambda_sev,
            kd_temperature=kd_temperature,
            has_positive=prepared.batch_has_positive[batch_number - 1],
            return_breakdown=True,
        )
        if not isinstance(loss_result, tuple):
            msg = "Loss breakdown was requested but not returned"
            raise TypeError(msg)
        loss, breakdown = loss_result

        # Backward pass with gradient clipping
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

        loss_values[batch_number - 1] = breakdown.stacked().detach()
        if batch_number % progress_every == 0 or batch_number == prepared.batches:
            _synchronize(device)
            elapsed = time.perf_counter() - optimization_start
            rows_done = stop_idx
            print(
                f"  train {batch_number:,}/{prepared.batches:,} batches | "
                f"{rows_done:,}/{prepared.sampled_rows:,} rows | {elapsed:.1f}s",
                flush=True,
            )

    _synchronize(device)
    optimization_seconds = time.perf_counter() - optimization_start
    # One transfer/synchronisation preserves the former Python-float summation
    # order without forcing a device sync after every optimizer step.
    component_rows = loss_values.cpu().tolist()
    loss_components = {
        name: sum(row[column] for row in component_rows) / prepared.batches
        for column, name in enumerate(LOSS_COMPONENT_NAMES)
    }
    result = TrainEpochResult(
        loss=loss_components["total"],
        loss_components=loss_components,
        sampled_rows=prepared.sampled_rows,
        batches=prepared.batches,
        sampling_seconds=prepared.sampling_seconds,
        staging_seconds=prepared.staging_seconds,
        optimization_seconds=optimization_seconds,
    )
    return result if return_details else result.loss


@torch.inference_mode()
def predict_joint_probabilities(
    model: Student,
    features: FeatureMatrix,
    device: torch.device,
    batch_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict joint class probabilities and any-halving probabilities."""
    model.eval()

    n_rows = len(features)
    n_batches = int(np.ceil(n_rows / batch_size))
    progress_every = _progress_interval(n_batches)
    output_dtype = next(model.parameters()).dtype
    # One pinned result matrix allows non-blocking D2H copies on CUDA. Column
    # zero holds d and the remaining columns hold conditional severity s.
    raw_output = torch.empty(
        (n_rows, N_CLASSES),
        dtype=output_dtype,
        device="cpu",
        pin_memory=device.type == "cuda",
    )
    prediction_start = time.perf_counter()

    for batch_number, start_idx in enumerate(range(0, n_rows, batch_size), start=1):
        stop_idx = min(start_idx + batch_size, n_rows)
        if isinstance(features, torch.Tensor):
            batch_x = features[start_idx:stop_idx]
        else:
            batch_x = torch.from_numpy(features[start_idx:stop_idx]).to(device)

        _, d, _, s = model(batch_x)
        packed = torch.cat((d.unsqueeze(-1), s), dim=-1)
        raw_output[start_idx:stop_idx].copy_(
            packed,
            non_blocking=device.type == "cuda",
        )
        if batch_number % progress_every == 0 or batch_number == n_batches:
            _synchronize(device)
            elapsed = time.perf_counter() - prediction_start
            print(
                f"  validation {batch_number:,}/{n_batches:,} batches | {stop_idx:,}/{n_rows:,} rows | {elapsed:.1f}s",
                flush=True,
            )

    _synchronize(device)
    raw = raw_output.numpy()
    # Keep detector scores contiguous because they are scanned repeatedly by
    # AP, threshold and operating-point calculations.
    d = raw[:, 0].copy()
    s = raw[:, 1:]

    # Construct full 5-class distribution
    q = np.empty((len(d), N_CLASSES), dtype=d.dtype)
    q[:, 0] = 1.0 - d
    q[:, 1:] = d[:, None] * s
    return q, d


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate sigmoid without overflowing large teacher margins."""
    values = np.asarray(values)
    output = np.empty(values.shape, dtype=np.result_type(values.dtype, np.float32))
    nonnegative = values >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    output[~nonnegative] = exponential / (1.0 + exponential)
    return output


def _normalized_severity(probabilities: np.ndarray) -> np.ndarray:
    """Return finite normalized conditional-severity probabilities."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != MATRIX_NDIM or probabilities.shape[1] != N_SEVERITY_CLASSES:
        msg = f"severity probabilities must have shape (rows, {N_SEVERITY_CLASSES})"
        raise ValueError(msg)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        msg = "severity probabilities must be finite and non-negative"
        raise ValueError(msg)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        msg = "severity probability rows must have positive mass"
        raise ValueError(msg)
    return probabilities / row_sums


def teacher_validation_diagnostics(
    teacher_margin: np.ndarray,
    teacher_severity: np.ndarray,
    labels: np.ndarray,
    target_recall: float,
) -> dict:
    """Evaluate the population-corrected responses cached from the teacher."""
    labels = np.asarray(labels)
    teacher_margin = np.asarray(teacher_margin)
    if labels.ndim != 1 or teacher_margin.shape != labels.shape:
        msg = "teacher margin and labels must be matching one-dimensional arrays"
        raise ValueError(msg)
    teacher_detector = _stable_sigmoid(teacher_margin)
    teacher_conditional = _normalized_severity(teacher_severity)
    if len(teacher_conditional) != len(labels):
        msg = "teacher severity and labels must contain matching rows"
        raise ValueError(msg)
    teacher_joint = np.empty((len(labels), N_CLASSES), dtype=teacher_conditional.dtype)
    teacher_joint[:, 0] = 1.0 - teacher_detector
    teacher_joint[:, 1:] = teacher_detector[:, None] * teacher_conditional
    teacher_metrics = evaluate(
        teacher_joint,
        labels,
        target_recall=target_recall,
        threshold_source="selected on cached teacher validation outputs",
        detection_outputs=teacher_detector,
    )
    positive = labels > 0
    if not positive.any():
        msg = "teacher validation diagnostics require at least one positive label"
        raise ValueError(msg)
    conditional_exact = float(np.mean(teacher_conditional[positive].argmax(axis=1) + 1 == labels[positive]))
    return {
        "source": "population-prior-corrected cached XGBoost validation responses",
        "detector_margin_scale": "population log-odds after removing the XGBoost undersampling shift",
        "split": "validation",
        "held_out": False,
        "detector_ap": float(teacher_metrics["ap"]),
        "conditional_severity_exact_on_true_positives": conditional_exact,
        "metrics": teacher_metrics,
    }


def student_teacher_agreement(  # noqa: PLR0915
    student_probabilities: np.ndarray,
    student_detector: np.ndarray,
    teacher_margin: np.ndarray,
    teacher_severity: np.ndarray,
    labels: np.ndarray,
    student_threshold: float,
    teacher_threshold: float,
) -> dict:
    """Compare final student responses with cached teacher responses."""
    student_probabilities = np.asarray(student_probabilities)
    student_detector = np.asarray(student_detector)
    labels = np.asarray(labels)
    teacher_margin = np.asarray(teacher_margin)
    expected_joint_shape = (len(labels), N_CLASSES)
    if labels.ndim != 1 or student_probabilities.shape != expected_joint_shape:
        msg = f"student probabilities must have shape {expected_joint_shape}"
        raise ValueError(msg)
    if student_detector.shape != labels.shape or teacher_margin.shape != labels.shape:
        msg = "student detector, teacher margin, and labels must have matching rows"
        raise ValueError(msg)
    teacher_detector = _stable_sigmoid(teacher_margin)
    teacher_conditional = _normalized_severity(teacher_severity)
    if len(teacher_conditional) != len(labels):
        msg = "teacher severity and labels must contain matching rows"
        raise ValueError(msg)
    positive = labels > 0
    if not positive.any():
        msg = "student-teacher agreement requires at least one positive label"
        raise ValueError(msg)
    epsilon = np.finfo(np.float64).eps

    detector_absolute_error_sum = 0.0
    detector_absolute_error_positive_sum = 0.0
    bernoulli_kl_sum = 0.0
    bernoulli_kl_positive_sum = 0.0
    joint_kl_sum = 0.0
    joint_kl_positive_sum = 0.0
    conditional_kl_weighted_sum = 0.0
    teacher_positive_weight_sum = 0.0

    chunk_rows = 1_000_000
    for start in range(0, len(labels), chunk_rows):
        stop = min(start + chunk_rows, len(labels))
        student_d_raw = student_detector[start:stop].astype(np.float64)
        teacher_d_raw = teacher_detector[start:stop].astype(np.float64)
        chunk_positive = positive[start:stop]
        detector_error = np.abs(student_d_raw - teacher_d_raw)
        detector_absolute_error_sum += float(detector_error.sum())
        detector_absolute_error_positive_sum += float(detector_error[chunk_positive].sum())

        student_d = np.clip(student_d_raw, epsilon, 1.0 - epsilon)
        teacher_d = np.clip(teacher_d_raw, epsilon, 1.0 - epsilon)
        bernoulli_kl = teacher_d * np.log(teacher_d / student_d)
        bernoulli_kl += (1.0 - teacher_d) * np.log((1.0 - teacher_d) / (1.0 - student_d))
        bernoulli_kl_sum += float(bernoulli_kl.sum())
        bernoulli_kl_positive_sum += float(bernoulli_kl[chunk_positive].sum())

        teacher_s = teacher_conditional[start:stop].astype(np.float64, copy=False)
        student_joint_raw = student_probabilities[start:stop].astype(np.float64)
        student_positive_mass = student_joint_raw[:, 1:]
        student_positive_total = student_positive_mass.sum(axis=1, keepdims=True)
        student_s = np.divide(
            student_positive_mass,
            student_positive_total,
            out=np.full_like(student_positive_mass, 1.0 / N_SEVERITY_CLASSES),
            where=student_positive_total > 0.0,
        )
        student_s = np.clip(student_s, epsilon, 1.0)
        student_s /= student_s.sum(axis=1, keepdims=True)

        student_q = np.clip(student_joint_raw, epsilon, 1.0)
        student_q /= student_q.sum(axis=1, keepdims=True)
        teacher_q = np.empty_like(student_q)
        teacher_q[:, 0] = 1.0 - teacher_d
        teacher_q[:, 1:] = teacher_d[:, None] * teacher_s
        teacher_q = np.clip(teacher_q, epsilon, 1.0)
        teacher_q /= teacher_q.sum(axis=1, keepdims=True)
        joint_kl = np.sum(teacher_q * (np.log(teacher_q) - np.log(student_q)), axis=1)
        joint_kl_sum += float(joint_kl.sum())
        joint_kl_positive_sum += float(joint_kl[chunk_positive].sum())

        teacher_s_clipped = np.clip(teacher_s, epsilon, 1.0)
        teacher_s_clipped /= teacher_s_clipped.sum(axis=1, keepdims=True)
        conditional_kl = np.sum(
            teacher_s_clipped * (np.log(teacher_s_clipped) - np.log(student_s)),
            axis=1,
        )
        conditional_kl_weighted_sum += float((teacher_d * conditional_kl).sum())
        teacher_positive_weight_sum += float(teacher_d.sum())

    n_positive = int(positive.sum())
    student_severity = student_probabilities[:, 1:].argmax(axis=1) + 1
    teacher_severity_action = teacher_conditional.argmax(axis=1) + 1
    student_actions = np.where(student_detector >= student_threshold, student_severity, 0)
    teacher_actions = np.where(teacher_detector >= teacher_threshold, teacher_severity_action, 0)
    action_confusion = np.bincount(
        teacher_actions * N_CLASSES + student_actions,
        minlength=N_CLASSES**2,
    ).reshape(N_CLASSES, N_CLASSES)

    return {
        "split": "validation",
        "held_out": False,
        "detector_probability_mae_all": detector_absolute_error_sum / len(labels),
        "detector_probability_mae_on_true_positives": detector_absolute_error_positive_sum / n_positive,
        "detector_bernoulli_kl_teacher_to_student_all": max(0.0, bernoulli_kl_sum / len(labels)),
        "detector_bernoulli_kl_teacher_to_student_on_true_positives": max(
            0.0,
            bernoulli_kl_positive_sum / n_positive,
        ),
        "joint_kl_teacher_to_student_all": max(0.0, joint_kl_sum / len(labels)),
        "joint_kl_teacher_to_student_on_true_positives": max(
            0.0,
            joint_kl_positive_sum / n_positive,
        ),
        "teacher_positive_weighted_conditional_severity_kl": max(
            0.0,
            conditional_kl_weighted_sum / teacher_positive_weight_sum,
        ),
        "conditional_severity_top1_agreement_on_true_positives": float(
            np.mean(student_severity[positive] == teacher_severity_action[positive])
        ),
        "independently_thresholded_action_agreement_all": float(np.mean(student_actions == teacher_actions)),
        "independently_thresholded_action_agreement_on_true_positives": float(
            np.mean(student_actions[positive] == teacher_actions[positive])
        ),
        "teacher_action_student_action_confusion": action_confusion.tolist(),
    }


def _summary(values: list[float]) -> dict[str, float] | None:
    """Summarize a non-empty list of finite diagnostics."""
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


@torch.inference_mode()
def member_ensemble_diagnostics(  # noqa: PLR0915
    model: Student,
    features: FeatureMatrix,
    labels: np.ndarray,
    ensemble_detector: np.ndarray,
    ensemble_probabilities: np.ndarray,
    target_recall: float,
    device: torch.device,
    batch_size: int,
) -> dict:
    """Measure individual member quality and diversity after selection."""
    model.eval()
    labels = np.asarray(labels)
    positive = labels > 0
    positive_labels = labels[positive]
    member_scores = np.empty((len(labels), model.k), dtype=np.float32)
    member_severity = np.empty((len(labels), model.k), dtype=np.int8)

    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        if isinstance(features, torch.Tensor):
            batch_x = features[start:stop]
        else:
            batch_x = torch.from_numpy(features[start:stop]).to(device)
        detector_logits, severity_logits = model.forward_logits(batch_x)
        member_scores[start:stop] = torch.sigmoid(detector_logits).transpose(0, 1).cpu().numpy()
        severity = severity_logits.argmax(dim=-1).transpose(0, 1) + 1
        member_severity[start:stop] = severity.cpu().numpy()

    reconstructed_ensemble = member_scores.mean(axis=1)
    if not np.allclose(reconstructed_ensemble, ensemble_detector, rtol=1e-5, atol=1e-7):
        msg = "Individual member probabilities do not reproduce the ensemble detector"
        raise RuntimeError(msg)

    individual = []
    member_thresholds = []
    for member in range(model.k):
        member_threshold = threshold_at_recall(labels, member_scores[:, member], target_recall)
        member_thresholds.append(member_threshold)
        member_actions = np.where(
            member_scores[:, member] >= member_threshold,
            member_severity[:, member],
            0,
        )
        individual.append(
            {
                "member": member,
                "detector_ap_full_validation": detection_ap(member_scores[:, member], labels),
                "detector_threshold_at_target_recall": member_threshold,
                "conditional_severity_exact_on_true_positives": float(
                    np.mean(member_severity[positive, member] == positive_labels)
                ),
                "equal_recall_action_metrics": halving_action_metrics(
                    member_actions,
                    labels,
                    n_classes=N_CLASSES,
                ),
            }
        )

    subset_size = max(int(positive.sum()), min(len(labels), MEMBER_DIAGNOSTIC_TARGET_ROWS))
    positive_indices = np.flatnonzero(positive)
    negative_indices = np.flatnonzero(~positive)
    n_negative_subset = max(0, subset_size - len(positive_indices))
    if n_negative_subset == 0:
        negative_indices = np.empty(0, dtype=np.int64)
    elif n_negative_subset < len(negative_indices):
        negative_scores = np.asarray(ensemble_detector)[negative_indices]
        selected = np.argpartition(negative_scores, -n_negative_subset)[-n_negative_subset:]
        negative_indices = negative_indices[selected]
    subset = np.sort(np.concatenate((positive_indices, negative_indices)))
    subset_scores = member_scores[subset]

    correlations = []
    undefined_correlations = 0
    equal_recall_disagreements = []
    common_threshold_disagreements = []
    positive_severity_disagreements = []
    ensemble_threshold = threshold_at_recall(labels, ensemble_detector, target_recall)
    for first in range(model.k):
        for second in range(first + 1, model.k):
            first_scores = subset_scores[:, first]
            second_scores = subset_scores[:, second]
            first_float64 = first_scores.astype(np.float64)
            second_float64 = second_scores.astype(np.float64)
            first_centered = first_float64 - float(first_float64.mean())
            second_centered = second_float64 - float(second_float64.mean())
            denominator = float(
                np.sqrt(np.dot(first_centered, first_centered) * np.dot(second_centered, second_centered))
            )
            if denominator > 0.0:
                correlation = float(np.dot(first_centered, second_centered) / denominator)
                correlations.append(float(np.clip(correlation, -1.0, 1.0)))
            else:
                undefined_correlations += 1
            equal_recall_disagreements.append(
                float(
                    np.mean((first_scores >= member_thresholds[first]) != (second_scores >= member_thresholds[second]))
                )
            )
            common_threshold_disagreements.append(
                float(np.mean((first_scores >= ensemble_threshold) != (second_scores >= ensemble_threshold)))
            )
            positive_severity_disagreements.append(
                float(np.mean(member_severity[positive, first] != member_severity[positive, second]))
            )

    individual_aps = [entry["detector_ap_full_validation"] for entry in individual]
    ensemble_ap = detection_ap(ensemble_detector, labels)
    ensemble_severity = ensemble_probabilities[:, 1:].argmax(axis=1) + 1
    return {
        "split": "validation",
        "held_out": False,
        "target_recall": target_recall,
        "individual_action_policy": "each member's highest threshold achieving target recall",
        "individual_members": individual,
        "ensemble": {
            "detector_ap_full_validation": ensemble_ap,
            "conditional_severity_exact_on_true_positives": float(
                np.mean(ensemble_severity[positive] == positive_labels)
            ),
            "ap_gain_over_mean_member": ensemble_ap - float(np.mean(individual_aps)),
            "ap_gain_over_best_member": ensemble_ap - max(individual_aps),
        },
        "diversity_subset": {
            "policy": MEMBER_DIAGNOSTIC_SUBSET_POLICY,
            "target_rows": MEMBER_DIAGNOSTIC_TARGET_ROWS,
            "rows": len(subset),
            "positive_rows": int(positive[subset].sum()),
            "negative_rows": int((~positive[subset]).sum()),
            "pair_count": model.k * (model.k - 1) // 2,
            "detector_score_pearson": _summary(correlations),
            "detector_score_pearson_undefined_pairs": undefined_correlations,
            "equal_recall_binary_disagreement": _summary(equal_recall_disagreements),
            "common_ensemble_threshold_binary_disagreement": _summary(common_threshold_disagreements),
            "positive_conditional_severity_disagreement": _summary(positive_severity_disagreements),
        },
    }


def evaluate_model(
    model: Student,
    features: FeatureMatrix,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int = 65536,
    *,
    target_recall: float = DEFAULT_TARGET_RECALL,
    threshold: float | None = None,
    threshold_source: str | None = None,
    include_diagnostics: bool = True,
) -> dict:
    """Evaluate the joint prediction at a selected or frozen threshold."""
    q, d = predict_joint_probabilities(model, features, device, batch_size)

    return evaluate(
        q,
        labels,
        threshold=threshold,
        target_recall=target_recall,
        threshold_source=threshold_source,
        detection_outputs=d,
        include_diagnostics=include_diagnostics,
    )


def epsilon_policy_diagnostic(
    probabilities: np.ndarray,
    labels: np.ndarray,
    target_recall: float,
) -> dict:
    """Build an uncalibrated validation-only ASAD policy frontier."""
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(probabilities):
        msg = "labels must be a one-dimensional array matching probabilities"
        raise ValueError(msg)
    positive = labels > 0
    if not positive.any():
        msg = "at least one positive label is required for the policy diagnostic"
        raise ValueError(msg)

    # Use the exact boundary calculation used by EpsilonPolicy. An equivalent
    # sum in a different order can differ by one ULP at the selected cutoff.
    # Reuse this matrix for confidence selection, the frontier and the
    # reference action so the multi-million-row float64 allocation occurs once.
    boundaries = boundary_probabilities(probabilities, WEEK_MAX_ACTION)
    first_boundary = boundaries[:, 1]
    recall_confidences = {
        f"{recall:.0%}": threshold_at_recall(labels, first_boundary, recall) for recall in POLICY_DIAGNOSTIC_RECALLS
    }
    reference_confidence = threshold_at_recall(labels, first_boundary, target_recall)
    candidate_confidences = (
        *POLICY_DIAGNOSTIC_CONFIDENCES,
        *recall_confidences.values(),
        reference_confidence,
    )
    frontier = epsilon_frontier(
        probabilities,
        labels,
        candidate_confidences,
        max_action=WEEK_MAX_ACTION,
        boundaries=boundaries,
    )

    reference_policy = EpsilonPolicy(reference_confidence, WEEK_MAX_ACTION)
    reference_actions = (boundaries[:, 1:] >= reference_confidence).sum(
        axis=1,
        dtype=np.int8,
    )
    reference_metrics = halving_action_metrics(
        reference_actions,
        labels,
        n_classes=N_CLASSES,
    )
    if reference_metrics["detection_recall"] + np.finfo(float).eps < target_recall:
        msg = "Epsilon-policy reference does not satisfy its requested recall"
        raise RuntimeError(msg)
    return {
        "policy": EPSILON_POLICY_VERSION,
        "status": "development_only_uncalibrated",
        "deployable": False,
        "calibrated": False,
        "selected_epsilon": None,
        "probability_source": "mean ensemble-member joint distribution",
        "formula": "max a such that P(Y >= a) >= boundary_confidence",
        "boundary_comparison": ">=",
        "probability_interpretation": (
            "population-prior-corrected model scores; empirical calibration has not been established"
        ),
        "selected_on": ("same validation period, fitted only after threshold-free probabilistic checkpoint selection"),
        "reporting_status": "optimistic development diagnostic, not held-out performance",
        "max_action": WEEK_MAX_ACTION,
        "action_domain": list(range(WEEK_MAX_ACTION + 1)),
        "class4_handling": (
            "q4 is combined with q3 for policy decisions; original y=4 targets remain class 4 in metrics"
        ),
        "raw_class4_probability": {
            "mean_all": float(probabilities[:, 4].mean()),
            "max_all": float(probabilities[:, 4].max()),
            "mean_on_positive": float(probabilities[positive, 4].mean()),
        },
        "recall_derived_boundary_confidences": recall_confidences,
        "reference_policy": {
            "purpose": "comparison at the existing interim detector recall target; not automatically selected",
            "target_recall": target_recall,
            "boundary_confidence": reference_confidence,
            "epsilon": reference_policy.epsilon,
            "achieved_recall": reference_metrics["detection_recall"],
            "metrics": reference_metrics,
        },
        "frontier": frontier,
    }


def _validate_model_args(args: argparse.Namespace) -> None:
    """Validate architecture and cache placement settings."""
    if args.architecture not in ARCHITECTURES:
        msg = f"architecture must be one of {ARCHITECTURES}"
        raise ValueError(msg)
    if args.architecture == "mlp" and args.members != 1:
        msg = "The MLP control requires --members 1"
        raise ValueError(msg)
    if args.architecture == "tabm-mini" and args.members < MIN_TABM_MEMBERS:
        msg = "TabM-mini requires --members of at least 2"
        raise ValueError(msg)
    if args.cache_residency not in CACHE_RESIDENCIES:
        msg = f"cache-residency must be one of {CACHE_RESIDENCIES}"
        raise ValueError(msg)
    if args.cache_residency == "cuda" and args.device != "cuda":
        msg = "--cache-residency cuda requires --device cuda"
        raise ValueError(msg)
    if not args.hidden or any(
        isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in args.hidden
    ):
        msg = "hidden must contain positive integer widths"
        raise ValueError(msg)
    if not 0.0 <= args.dropout < 1.0:
        msg = "dropout must be in [0, 1)"
        raise ValueError(msg)


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid training settings before loading a large cache."""
    positive_integer_arguments = {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "negative_ratio": args.negative_ratio,
        "members": args.members,
    }
    for name, value in positive_integer_arguments.items():
        if value < 1:
            msg = f"{name.replace('_', '-')} must be at least one"
            raise ValueError(msg)

    _validate_model_args(args)
    if not np.isfinite(args.lr) or not np.isfinite(args.weight_decay) or args.lr <= 0.0 or args.weight_decay < 0.0:
        msg = "lr must be positive and weight-decay must be non-negative"
        raise ValueError(msg)
    if not all(np.isfinite(value) for value in (args.alpha, args.beta, args.lambda_sev)) or any(
        value < 0.0 for value in (args.alpha, args.beta, args.lambda_sev)
    ):
        msg = "distillation weights must be non-negative"
        raise ValueError(msg)
    if not np.isfinite(args.kd_temperature) or args.kd_temperature <= 0.0:
        msg = "kd-temperature must be finite and positive"
        raise ValueError(msg)
    if not 0.0 < args.target_recall <= 1.0:
        msg = "target-recall must be in (0, 1]"
        raise ValueError(msg)
    if args.seed < 0:
        msg = "seed must be non-negative"
        raise ValueError(msg)
    if args.epochs >= EPOCH_SEED_STRIDE:
        msg = f"epochs must be below {EPOCH_SEED_STRIDE} for collision-free sampling seeds"
        raise ValueError(msg)
    if args.device == "cuda" and not torch.cuda.is_available():
        msg = "CUDA was requested but is not available"
        raise RuntimeError(msg)


def _validate_teacher_cache(
    margin: np.ndarray,
    severity: np.ndarray,
    n_rows: int,
    split: str,
) -> None:
    """Validate one pair of cached teacher-response arrays."""
    if margin.shape != (n_rows,):
        msg = f"{split} teacher margin cache has the wrong shape"
        raise ValueError(msg)
    if severity.shape != (n_rows, N_SEVERITY_CLASSES):
        msg = f"{split} teacher severity cache has the wrong shape"
        raise ValueError(msg)
    if not np.isfinite(margin).all() or not np.isfinite(severity).all():
        msg = f"{split} teacher caches must contain finite values"
        raise ValueError(msg)
    if np.any(severity < 0.0) or np.any(severity.sum(axis=1) <= 0.0):
        msg = f"{split} teacher severity rows must contain non-negative positive mass"
        raise ValueError(msg)


def _validate_cache_shapes(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    train_teacher_margin: np.ndarray | None,
    train_teacher_severity: np.ndarray | None,
    val_teacher_margin: np.ndarray,
    val_teacher_severity: np.ndarray,
) -> None:
    """Check the inexpensive cache-shape invariants needed for training."""
    if train_x.ndim != MATRIX_NDIM or val_x.ndim != MATRIX_NDIM:
        msg = "Feature caches must be two-dimensional"
        raise ValueError(msg)
    if train_y.ndim != 1 or val_y.ndim != 1:
        msg = "Label caches must be one-dimensional"
        raise ValueError(msg)
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        msg = "Feature and label cache row counts do not match"
        raise ValueError(msg)
    if not len(train_y) or not len(val_y):
        msg = "Training and validation caches must not be empty"
        raise ValueError(msg)
    if not np.any(val_y > 0):
        msg = "Validation labels must contain at least one positive example"
        raise ValueError(msg)
    if train_x.shape[1] != val_x.shape[1]:
        msg = "Training and validation caches have different feature counts"
        raise ValueError(msg)
    if (train_teacher_margin is None) != (train_teacher_severity is None):
        msg = "Training teacher margin and severity caches must both be present"
        raise ValueError(msg)
    if train_teacher_margin is not None and train_teacher_severity is not None:
        _validate_teacher_cache(
            train_teacher_margin,
            train_teacher_severity,
            len(train_y),
            "Training",
        )
    _validate_teacher_cache(
        val_teacher_margin,
        val_teacher_severity,
        len(val_y),
        "Validation",
    )


def run(args: argparse.Namespace) -> dict:  # noqa: PLR0912, PLR0915
    """Execute training loop with early stopping."""
    # Preserve programmatic callers created before cache placement became a
    # configurable command-line option.
    backward_compatible_defaults = {
        "cache_residency": DEFAULT_CACHE_RESIDENCY,
        "lambda_sev": DEFAULT_LAMBDA_SEV,
        "kd_temperature": DEFAULT_KD_TEMPERATURE,
        "save_ap_best_checkpoint": False,
    }
    for name, default in backward_compatible_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    _validate_args(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Determine cache directory based on preprocessing method
    cache_dir = args.cache_dir / "week" / args.preprocess

    print(f"Loading cache from: {cache_dir}", flush=True)

    # Read each cache once in sequential chunks. Repeated random access to an
    # RDS-backed mmap was the dominant bottleneck for the week experiment.
    cache_load_start = time.perf_counter()
    train_x_host = _load_array_resident(cache_dir / "train_x.npy")
    train_y = _load_array_resident(cache_dir / "train_y.npy")
    val_x_host = _load_array_resident(cache_dir / "val_x.npy")
    val_y = _load_array_resident(cache_dir / "val_y.npy")

    print(f"Train: {train_x_host.shape}, {train_x_host.nbytes / 2**20:.1f} MiB (resident)")
    print(f"Val: {val_x_host.shape}, {val_x_host.nbytes / 2**20:.1f} MiB (resident)")

    # Validation responses are loaded for both controls so teacher fidelity and
    # student agreement are directly comparable. Training responses are needed
    # only by distilled cells.
    train_teacher_margin = None
    train_teacher_severity = None
    if args.distill:
        print("Loading training teacher targets...", flush=True)
        train_teacher_margin = _load_array_resident(cache_dir / "train_teacher_margin.npy")
        train_teacher_severity = _load_array_resident(cache_dir / "train_teacher_severity.npy")
    print("Loading validation teacher targets...", flush=True)
    val_teacher_margin = _load_array_resident(cache_dir / "val_teacher_margin.npy")
    val_teacher_severity = _load_array_resident(cache_dir / "val_teacher_severity.npy")

    cache_load_seconds = time.perf_counter() - cache_load_start
    print(f"Cache loaded into host RAM in {cache_load_seconds:.1f}s", flush=True)

    _validate_cache_shapes(
        train_x_host,
        train_y,
        val_x_host,
        val_y,
        train_teacher_margin,
        train_teacher_severity,
        val_teacher_margin,
        val_teacher_severity,
    )

    metadata_path = cache_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    delta_t = metadata["teacher"]["delta_t"]
    print(f"Teacher delta_T: {delta_t:.4f}")

    # The model emits a population margin. Adding delta_s maps it to the
    # sampled-data prior used by the detector's hard-label BCE.
    pool_start = time.perf_counter()
    positive_indices, negative_indices = training_index_pools(train_y)
    pool_seconds = time.perf_counter() - pool_start
    n_positives = len(positive_indices)
    n_negatives_full = len(negative_indices)
    if not n_positives or not n_negatives_full:
        msg = "Training data must contain both positive and negative examples"
        raise ValueError(msg)
    print(f"Train positives: {n_positives:,}")
    print(f"Val positives: {np.count_nonzero(val_y > 0):,}")
    print(f"Training index pools built once in {pool_seconds:.2f}s")
    n_negatives_sampled = min(
        n_negatives_full,
        args.negative_ratio * n_positives,
    )
    delta_s = float(np.log(n_negatives_full / n_negatives_sampled))
    population_prevalence = n_positives / (n_positives + n_negatives_full)
    initial_detector_bias = float(np.log(n_positives / n_negatives_full))
    print(f"Student delta_S: {delta_s:.4f}")
    print(f"Initial detector population log-odds: {initial_detector_bias:.4f}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_x, val_x, cache_residency, feature_staging_seconds = _place_feature_matrices(
        train_x_host,
        val_x_host,
        device,
        args.cache_residency,
    )
    if cache_residency == "cuda":
        # Device tensors own their copies; release the 37 GiB host duplicates.
        del train_x_host, val_x_host
    print(f"Feature cache residency: {cache_residency}", flush=True)

    # Create model
    model = Student(
        n_features=train_x.shape[1],
        hidden_layers=args.hidden,
        k=args.members,
        dropout=args.dropout,
        architecture=args.architecture,
    )
    model.initialize_detector_bias(initial_detector_bias)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
    model_metadata = {
        "format_version": MODEL_FORMAT_VERSION,
        "architecture": model.architecture,
        "architecture_version": model.architecture_version,
        "n_features": train_x.shape[1],
        "hidden": list(model.hidden_layers),
        "k": model.k,
        "dropout": model.dropout,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "cache_residency": cache_residency,
    }
    training_protocol = {
        "negative_sampling": {
            "policy": "uniform random undersampling without replacement per epoch",
            "epoch_seed_policy": SAMPLING_SEED_POLICY,
            "epoch_seed_stride": EPOCH_SEED_STRIDE,
            "full_negative_rows": n_negatives_full,
            "sampled_negative_rows_per_epoch": n_negatives_sampled,
            "positive_rows_per_epoch": n_positives,
            "negative_to_positive_ratio": args.negative_ratio,
            "population_prior_correction_delta_s": delta_s,
        },
        "detector_bias_initialization": {
            "policy": DETECTOR_BIAS_POLICY,
            "value": initial_detector_bias,
            "positive_rows": n_positives,
            "negative_rows": n_negatives_full,
            "population_prevalence": population_prevalence,
        },
    }
    objective = {
        "version": "dual-head-response-kd-v3",
        "distill": args.distill,
        "hard_detector": "population-margin BCE with sampled-prior shift",
        "hard_severity": "conditional cross entropy on true-positive rows",
        "detector_kd": ("T^2 * BCEWithLogits(student population margin / T, sigmoid(teacher population margin / T))"),
        "severity_kd": ("T^2 * teacher-positive-mass-normalized conditional KL, averaged over ensemble members"),
        "severity_kd_teacher_gate": "untempered sigmoid of teacher population margin",
        "alpha": args.alpha,
        "beta": args.beta,
        "lambda_sev": args.lambda_sev,
        "kd_temperature": args.kd_temperature,
        "loss_component_reporting": "weighted mean per optimizer batch",
    }

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    best_selection_key = None
    best_epoch = None
    patience_counter = 0
    best_state = None
    best_probabilities = None
    best_detector_probability = None
    best_ap_selection_key = None
    best_ap_epoch = None
    best_ap_state = None
    history = []

    print(f"\nTraining for {args.epochs} epochs with patience {args.patience}")
    print(f"Architecture: {args.architecture}")
    print(f"Distillation: {args.distill}")
    print(f"Members: {args.members}")
    print(f"Batch size: {args.batch_size}")
    print(f"Evaluation batch size: {args.eval_batch_size}")
    print(f"Feature cache residency: {cache_residency}")
    print(f"Negative ratio: {args.negative_ratio}")
    print(f"Post-checkpoint target detector recall: {args.target_recall:.1%}")
    print(f"Checkpoint policy: {SELECTION_POLICY}")
    print(f"Save AP-best diagnostic checkpoint: {args.save_ap_best_checkpoint}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Architecture/KD cells with the same run seed see identical samples;
        # different run seeds use disjoint RNG streams rather than shifted
        # copies of the same epoch sequence.
        epoch_seed = args.seed * EPOCH_SEED_STRIDE + epoch

        train_result = train_epoch(
            model,
            optimizer,
            train_x,
            train_y,
            train_teacher_margin,
            train_teacher_severity,
            device,
            args.batch_size,
            args.negative_ratio,
            epoch_seed,
            delta_s,
            args.alpha,
            args.beta,
            lambda_sev=args.lambda_sev,
            kd_temperature=args.kd_temperature,
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            return_details=True,
        )
        if not isinstance(train_result, TrainEpochResult):
            msg = "Detailed epoch training unexpectedly returned only a scalar loss"
            raise TypeError(msg)
        train_loss = train_result.loss

        # Evaluate on validation set
        inference_start = time.perf_counter()
        val_probabilities, val_detector_probability = predict_joint_probabilities(
            model,
            val_x,
            device,
            batch_size=args.eval_batch_size,
        )
        inference_seconds = time.perf_counter() - inference_start
        metrics_start = time.perf_counter()
        probability = probability_metrics(val_probabilities, val_y)
        val_ap = detection_ap(val_detector_probability, val_y)
        val_metrics = {"ap": val_ap, "probability": probability}
        metrics_seconds = time.perf_counter() - metrics_start
        selection_key = checkpoint_selection_key(val_metrics)
        ap_selection_key = ap_checkpoint_selection_key(val_metrics)
        val_rps = float(probability["ranked_probability_score"])
        val_nll = float(probability["negative_log_likelihood"])
        val_brier = float(probability["brier_score"])

        if not all(
            np.isfinite(value)
            for value in (
                train_loss,
                val_ap,
                val_rps,
                val_nll,
                val_brier,
                *train_result.loss_components.values(),
            )
        ):
            msg = (
                f"Non-finite value at epoch {epoch}: loss={train_loss}, "
                f"val_rps={val_rps}, val_nll={val_nll}, val_ap={val_ap}"
            )
            raise FloatingPointError(msg)

        epoch_time = time.perf_counter() - epoch_start
        gpu_peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        gpu_peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0
        history.append(
            {
                "epoch": epoch,
                "sampling_seed": epoch_seed,
                "training_loss": float(train_loss),
                "training_loss_components_weighted": train_result.loss_components,
                "validation_ap": val_ap,
                "validation_ranked_probability_score": val_rps,
                "validation_negative_log_likelihood": val_nll,
                "validation_brier_score": val_brier,
                "selection_key": list(selection_key),
                "ap_selection_key": list(ap_selection_key),
                "seconds": epoch_time,
                "sampled_rows": train_result.sampled_rows,
                "training_batches": train_result.batches,
                "sampling_seconds": train_result.sampling_seconds,
                "staging_seconds": train_result.staging_seconds,
                "optimization_seconds": train_result.optimization_seconds,
                "validation_inference_seconds": inference_seconds,
                "validation_metrics_seconds": metrics_seconds,
                "gpu_peak_allocated_mib": gpu_peak_allocated_mib,
                "gpu_peak_reserved_mib": gpu_peak_reserved_mib,
            }
        )

        print(
            f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"RPS {val_rps:.6g} | NLL {val_nll:.6g} | AP {val_ap:.4f} | "
            f"time {epoch_time:.1f}s "
            f"(stage {train_result.staging_seconds:.1f}s, "
            f"train {train_result.optimization_seconds:.1f}s, "
            f"infer {inference_seconds:.1f}s, metrics {metrics_seconds:.1f}s, "
            f"GPU peak {gpu_peak_allocated_mib / 1024:.1f} GiB)"
        )
        loss_parts = train_result.loss_components
        print(
            "  weighted loss components | "
            f"hard detector {loss_parts['hard_detector']:.5f} | "
            f"hard severity {loss_parts['hard_severity']:.5f} | "
            f"KD detector {loss_parts['kd_detector']:.5f} | "
            f"KD severity {loss_parts['kd_severity']:.5f}"
        )

        # AP is tracked independently for an opt-in ranking diagnostic. It
        # does not affect canonical early stopping or any final metrics.
        if best_ap_selection_key is None or ap_selection_key > best_ap_selection_key:
            best_ap_selection_key = ap_selection_key
            best_ap_epoch = epoch
            if args.save_ap_best_checkpoint:
                best_ap_state = {name: value.cpu().clone() for name, value in model.state_dict().items()}

        # Early stopping
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_probabilities = val_probabilities
            best_detector_probability = val_detector_probability
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if (
        best_state is None
        or best_selection_key is None
        or best_epoch is None
        or best_probabilities is None
        or best_detector_probability is None
    ):
        msg = "Training completed without producing a valid checkpoint"
        raise RuntimeError(msg)
    if best_ap_selection_key is None or best_ap_epoch is None:
        msg = "Training completed without producing an AP diagnostic selection"
        raise RuntimeError(msg)
    if args.save_ap_best_checkpoint and best_ap_state is None:
        msg = "AP-best checkpoint saving was enabled but no state was captured"
        raise RuntimeError(msg)
    model.load_state_dict(best_state)
    restored_state = model.state_dict()
    if restored_state.keys() != best_state.keys() or any(
        not torch.equal(restored_state[name].detach().cpu(), best_state[name]) for name in best_state
    ):
        msg = "Loaded model state does not match the selected checkpoint"
        raise RuntimeError(msg)
    threshold_source = "selected after probabilistic checkpoint selection on validation set"
    final_probabilities = best_probabilities
    final_detector_probability = best_detector_probability
    final_metrics = evaluate(
        final_probabilities,
        val_y,
        target_recall=args.target_recall,
        threshold_source=threshold_source,
        detection_outputs=final_detector_probability,
    )
    best_threshold = float(final_metrics["threshold"])
    final_selection_key = checkpoint_selection_key(final_metrics)
    if not np.allclose(final_selection_key, best_selection_key, rtol=1e-12, atol=1e-15):
        msg = "Cached selected-epoch predictions do not reproduce their validation selection key"
        raise RuntimeError(msg)
    if final_metrics["recall"] + np.finfo(float).eps < args.target_recall:
        msg = "Cached selected-epoch predictions do not satisfy the requested detector recall"
        raise RuntimeError(msg)

    selection = {
        "policy": SELECTION_POLICY,
        "canonical": True,
        "split": "validation",
        "uses_hard_actions": False,
        "prevalence_weighting": "natural validation prevalence; no class weighting",
        "criteria": list(SELECTION_CRITERIA),
        "best_epoch": best_epoch,
        "best_key": list(best_selection_key),
        "selected_checkpoint_ranked_probability_score": float(final_metrics["probability"]["ranked_probability_score"]),
        "selected_checkpoint_negative_log_likelihood": float(final_metrics["probability"]["negative_log_likelihood"]),
        "selected_checkpoint_brier_score": float(final_metrics["probability"]["brier_score"]),
        "selected_checkpoint_ap": float(final_metrics["ap"]),
        "min_rps_observed": min(row["validation_ranked_probability_score"] for row in history),
        "min_nll_observed": min(row["validation_negative_log_likelihood"] for row in history),
        "max_ap_observed": max(row["validation_ap"] for row in history),
    }
    threshold_policy = {
        "rule": THRESHOLD_RULE,
        "score": THRESHOLD_SCORE,
        "split": "validation",
        "selected_after_checkpoint": True,
        "target_recall": args.target_recall,
        "threshold": best_threshold,
        "achieved_recall": float(final_metrics["recall"]),
    }
    diagnostics_start = time.perf_counter()
    teacher_validation = teacher_validation_diagnostics(
        val_teacher_margin,
        val_teacher_severity,
        val_y,
        args.target_recall,
    )
    teacher_agreement = student_teacher_agreement(
        final_probabilities,
        final_detector_probability,
        val_teacher_margin,
        val_teacher_severity,
        val_y,
        best_threshold,
        float(teacher_validation["metrics"]["threshold"]),
    )
    member_diagnostics = member_ensemble_diagnostics(
        model,
        val_x,
        val_y,
        final_detector_probability,
        final_probabilities,
        args.target_recall,
        device,
        args.eval_batch_size,
    )
    policy_diagnostic = epsilon_policy_diagnostic(
        final_probabilities,
        val_y,
        args.target_recall,
    )
    final_diagnostics_seconds = time.perf_counter() - diagnostics_start

    # Save results
    task = "distilled" if args.distill else "supervised"
    architecture_slug = args.architecture.replace("-", "_")
    artifact_name = f"student_{architecture_slug}_k{args.members}_{task}_seed{args.seed}"
    model_path, result_path = output_paths(args.output_dir, artifact_name, ".pt")
    ap_model_path = model_path.with_name(f"{model_path.stem}_ap_best{model_path.suffix}")
    checkpoint_config = {
        **model_metadata,
        "delta_s": delta_s,
    }
    diagnostic_protocol = {
        "teacher_validation_targets": "required for all supervised and distilled cells",
        "member_diagnostics": "full-validation AP/action metrics plus hard-subset diversity",
        "epsilon_policy": EPSILON_POLICY_VERSION,
    }
    ap_history_row = next(row for row in history if row["epoch"] == best_ap_epoch)
    ap_best_metadata = {
        "enabled": bool(args.save_ap_best_checkpoint),
        "saved": bool(args.save_ap_best_checkpoint),
        "path": ap_model_path.name if args.save_ap_best_checkpoint else None,
        "policy": AP_SELECTION_POLICY,
        "canonical": False,
        "uses_hard_actions": False,
        "criteria": list(AP_SELECTION_CRITERIA),
        "best_epoch": best_ap_epoch,
        "best_key": list(best_ap_selection_key),
        "validation_ap": float(ap_history_row["validation_ap"]),
        "validation_ranked_probability_score": float(ap_history_row["validation_ranked_probability_score"]),
        "validation_negative_log_likelihood": float(ap_history_row["validation_negative_log_likelihood"]),
        "same_epoch_as_canonical": best_ap_epoch == best_epoch,
    }
    diagnostic_checkpoints = {"ap_best": ap_best_metadata}

    torch.save(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "checkpoint_role": "canonical",
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": checkpoint_config,
            "threshold": best_threshold,
            "threshold_source": threshold_source,
            "target_recall": args.target_recall,
            "severity_decision": SEVERITY_DECISION,
            "selection": selection,
            "threshold_policy": threshold_policy,
            "objective": objective,
            "training_protocol": training_protocol,
            "diagnostic_protocol": diagnostic_protocol,
        },
        model_path,
    )
    if args.save_ap_best_checkpoint:
        if best_ap_state is None:
            msg = "AP-best checkpoint state unexpectedly missing during save"
            raise RuntimeError(msg)
        torch.save(
            {
                "format_version": MODEL_FORMAT_VERSION,
                "checkpoint_role": "diagnostic",
                "diagnostic_name": "ap_best",
                "state_dict": best_ap_state,
                "config": checkpoint_config,
                "selection": ap_best_metadata,
                "canonical_selection_policy": SELECTION_POLICY,
                "objective": objective,
                "training_protocol": training_protocol,
                "source_report": result_path.name,
            },
            ap_model_path,
        )

    result = {
        "model_format_version": MODEL_FORMAT_VERSION,
        "task": task,
        "seed": args.seed,
        "config": vars(args),
        "model": model_metadata,
        "objective": objective,
        "training_protocol": training_protocol,
        "metrics": final_metrics,
        "threshold": best_threshold,
        "threshold_source": threshold_source,
        "target_recall": args.target_recall,
        "severity_decision": SEVERITY_DECISION,
        "selection": selection,
        "diagnostic_checkpoints": diagnostic_checkpoints,
        "threshold_policy": threshold_policy,
        "teacher_validation": teacher_validation,
        "student_teacher_agreement": teacher_agreement,
        "member_ensemble_diagnostics": member_diagnostics,
        "epsilon_policy_diagnostic": policy_diagnostic,
        "history": history,
        "performance": {
            "cache_load_seconds": cache_load_seconds,
            "index_pool_seconds": pool_seconds,
            "feature_staging_seconds": feature_staging_seconds,
            "cache_residency": cache_residency,
            "final_diagnostics_seconds": final_diagnostics_seconds,
            "max_gpu_allocated_mib": max(row["gpu_peak_allocated_mib"] for row in history),
            "max_gpu_reserved_mib": max(row["gpu_peak_reserved_mib"] for row in history),
        },
    }
    write_json(result_path, result)

    print(f"\nSaved model: {model_path}")
    if args.save_ap_best_checkpoint:
        print(f"Saved AP-best diagnostic model: {ap_model_path}")
    print(f"Saved results: {result_path}")
    print(f"Final AP: {final_metrics['ap']:.4f}")
    print(
        "Final proper scores: "
        f"RPS {final_metrics['probability']['ranked_probability_score']:.6g} | "
        f"NLL {final_metrics['probability']['negative_log_likelihood']:.6g}"
    )
    print(f"Best epoch: {best_epoch}")
    print(f"Frozen threshold: {best_threshold:.6g}")
    print(f"Exact on positives: {final_metrics['severity']['exact_on_positive']:.4f}")
    print(f"All-row overprediction rate: {final_metrics['severity']['overprediction_rate_all']:.6f}")
    print(
        "Cached teacher validation: "
        f"AP {teacher_validation['detector_ap']:.6f} | "
        "conditional severity exact "
        f"{teacher_validation['conditional_severity_exact_on_true_positives']:.4f}"
    )
    print(
        "Student/teacher agreement: "
        f"joint KL {teacher_agreement['joint_kl_teacher_to_student_all']:.6g} | "
        "positive severity agreement "
        f"{teacher_agreement['conditional_severity_top1_agreement_on_true_positives']:.4f}"
    )
    print(
        "Member ensemble diagnostic: "
        f"AP gain over mean member {member_diagnostics['ensemble']['ap_gain_over_mean_member']:.6g}"
    )
    reference_policy = policy_diagnostic["reference_policy"]
    print(
        "ASAD policy diagnostic: "
        f"confidence {reference_policy['boundary_confidence']:.6g} | "
        f"epsilon {reference_policy['epsilon']:.6g} | "
        f"exact+ {reference_policy['metrics']['exact_on_positive']:.4f} | "
        f"over(all) {reference_policy['metrics']['overprediction_rate_all']:.6f}"
    )

    return result


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train student model with optional distillation")

    # Data paths
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Cache directory")
    parser.add_argument(
        "--preprocess",
        default="physical",
        choices=("physical", "quantile", "raw"),
        help="Preprocessing variant",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--cache-residency",
        default=DEFAULT_CACHE_RESIDENCY,
        choices=CACHE_RESIDENCIES,
        help="Keep resident feature matrices in host RAM, CUDA memory, or choose automatically",
    )

    # Architecture
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        choices=ARCHITECTURES,
        help="Student architecture",
    )
    parser.add_argument("--hidden", type=int, nargs="+", default=HIDDEN_LAYERS, help="Hidden layer sizes")
    parser.add_argument("--members", type=int, default=1, help="Number of ensemble members (k)")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout probability")

    # Training
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help="Validation inference batch size",
    )
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=DEFAULT_NEGATIVE_RATIO,
        help="Negative:positive sampling ratio",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument(
        "--target-recall",
        type=float,
        default=DEFAULT_TARGET_RECALL,
        help="Minimum detector recall used to select the validation threshold",
    )

    # Distillation
    parser.add_argument("--distill", action="store_true", help="Use knowledge distillation")
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Detector distillation weight",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="Severity distillation weight",
    )
    parser.add_argument(
        "--lambda-sev",
        type=float,
        default=DEFAULT_LAMBDA_SEV,
        help="Hard-label conditional-severity loss weight",
    )
    parser.add_argument(
        "--kd-temperature",
        type=float,
        default=DEFAULT_KD_TEMPERATURE,
        help="Temperature shared by detector and conditional-severity distillation",
    )
    parser.add_argument(
        "--save-ap-best-checkpoint",
        action="store_true",
        help="Also save an AP-selected diagnostic checkpoint without changing canonical early stopping",
    )

    # Misc
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
