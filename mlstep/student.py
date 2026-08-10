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
    epsilon_frontier,
    evaluate,
    halving_action_metrics,
    output_paths,
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
DEFAULT_EPOCHS: Final = 80
DEFAULT_PATIENCE: Final = 12
DEFAULT_BATCH_SIZE: Final = 1024
DEFAULT_EVAL_BATCH_SIZE: Final = 65536
DEFAULT_NEGATIVE_RATIO: Final = 256
DEFAULT_TARGET_RECALL: Final = 0.97
CACHE_RESIDENCIES: Final = ("auto", "host", "cuda")
DEFAULT_CACHE_RESIDENCY: Final = "auto"
GPU_DATA_RESERVE_BYTES: Final = 16 * 2**30
LOAD_CHUNK_BYTES: Final = 512 * 2**20
PROGRESS_UPDATES_PER_PHASE: Final = 10

# Interim week-experiment policy. At each epoch, use the highest detector
# threshold that achieves the requested recall. Then compare checkpoints
# lexicographically using these criteria, rather than by detector AP alone.
SELECTION_POLICY: Final = "recall-floor-exact-over-v1"
THRESHOLD_RULE: Final = "highest validation threshold achieving target recall"
THRESHOLD_SCORE: Final = "ensemble any-halving detector probability"
SELECTION_CRITERIA: Final = (
    "maximize severity.exact_on_positive_count",
    "minimize severity.overprediction_count_all",
    "minimize severity.overprediction_level_sum_all",
    "minimize severity.underprediction_level_sum_all",
    "maximize ap",
)
SEVERITY_DECISION: Final = "argmax conditional severity probability"

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
    sampled_rows: int
    batches: int
    sampling_seconds: float
    staging_seconds: float
    optimization_seconds: float


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
    alpha: float = 1.0,
    beta: float = 0.5,
    lambda_sev: float = 0.1,
    has_positive: bool | None = None,
) -> torch.Tensor:
    """Compute combined loss with optional distillation."""
    n_members = d_logit.shape[0]
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

    gt_loss = detector_loss + lambda_sev * severity_loss

    # Optional distillation
    distill_loss = d_logit.new_zeros(())

    if teacher_margin is not None and teacher_severity is not None:
        teacher_margin_members = teacher_margin.unsqueeze(0).expand_as(d_logit)

        # Both are population-scale margins.
        margin_loss = F.smooth_l1_loss(
            d_logit,
            teacher_margin_members,
        )

        # Use a valid T=1 probability distribution for the initial baseline.
        teacher_probability = teacher_severity.clamp_min(torch.finfo(s_logits.dtype).tiny)
        teacher_probability = teacher_probability / teacher_probability.sum(dim=-1, keepdim=True)
        teacher_probability = teacher_probability.unsqueeze(0).expand(n_members, -1, -1)

        student_log_probability = F.log_softmax(s_logits, dim=-1)
        severity_kl = F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="none",
        ).sum(dim=-1)

        teacher_detection_probability = torch.sigmoid(teacher_margin).unsqueeze(0)
        severity_kl = (teacher_detection_probability * severity_kl).mean()

        distill_loss = alpha * margin_loss + beta * severity_kl

    total_loss = gt_loss + distill_loss

    return total_loss


def checkpoint_selection_key(metrics: dict) -> tuple[int, int, int, int, float]:
    """Return the lexicographic key used to select a student checkpoint.

    Every checkpoint is evaluated at the highest detector threshold satisfying
    the requested validation recall. Larger returned tuples are better.
    """
    severity = metrics.get("severity")
    if severity is None:
        msg = "Checkpoint selection requires multiclass severity metrics"
        raise ValueError(msg)

    return (
        int(severity["exact_on_positive_count"]),
        -int(severity["overprediction_count_all"]),
        -int(severity["overprediction_level_sum_all"]),
        -int(severity["underprediction_level_sum_all"]),
        float(metrics["ap"]),
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

    loss_values = torch.empty(prepared.batches, device=device, dtype=torch.float32)
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
        loss = compute_loss(
            d_logit,
            s_logits,
            batch_y,
            batch_teacher_margin,
            batch_teacher_severity,
            delta_s=delta_s,
            alpha=alpha,
            beta=beta,
            has_positive=prepared.batch_has_positive[batch_number - 1],
        )

        # Backward pass with gradient clipping
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

        loss_values[batch_number - 1] = loss.detach()
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
    total_loss = sum(loss_values.cpu().tolist())
    result = TrainEpochResult(
        loss=total_loss / prepared.batches,
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
        "selected_on": "same validation period used for checkpoint selection",
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
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        msg = "lr must be positive and weight-decay must be non-negative"
        raise ValueError(msg)
    if args.alpha < 0.0 or args.beta < 0.0:
        msg = "distillation weights must be non-negative"
        raise ValueError(msg)
    if not 0.0 < args.target_recall <= 1.0:
        msg = "target-recall must be in (0, 1]"
        raise ValueError(msg)
    if args.device == "cuda" and not torch.cuda.is_available():
        msg = "CUDA was requested but is not available"
        raise RuntimeError(msg)


def _validate_cache_shapes(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    teacher_margin: np.ndarray | None,
    teacher_severity: np.ndarray | None,
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
    if (teacher_margin is None) != (teacher_severity is None):
        msg = "Teacher margin and severity caches must both be present"
        raise ValueError(msg)
    if teacher_margin is not None and teacher_severity is not None:
        if teacher_margin.shape != (len(train_y),):
            msg = "Teacher margin cache has the wrong shape"
            raise ValueError(msg)
        if teacher_severity.shape != (len(train_y), N_SEVERITY_CLASSES):
            msg = "Teacher severity cache has the wrong shape"
            raise ValueError(msg)


def run(args: argparse.Namespace) -> dict:  # noqa: PLR0912, PLR0915
    """Execute training loop with early stopping."""
    # Preserve programmatic callers created before cache placement became a
    # configurable command-line option.
    if not hasattr(args, "cache_residency"):
        args.cache_residency = DEFAULT_CACHE_RESIDENCY
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

    # Load teacher targets if distilling
    teacher_margin = None
    teacher_severity = None
    if args.distill:
        print("Loading teacher targets...", flush=True)
        teacher_margin = _load_array_resident(cache_dir / "train_teacher_margin.npy")
        teacher_severity = _load_array_resident(cache_dir / "train_teacher_severity.npy")

    cache_load_seconds = time.perf_counter() - cache_load_start
    print(f"Cache loaded into host RAM in {cache_load_seconds:.1f}s", flush=True)

    _validate_cache_shapes(
        train_x_host,
        train_y,
        val_x_host,
        val_y,
        teacher_margin,
        teacher_severity,
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
    print(f"Student delta_S: {delta_s:.4f}")

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
    ).to(device)

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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    best_selection_key = None
    best_epoch = None
    best_threshold = None
    patience_counter = 0
    best_state = None
    best_probabilities = None
    best_detector_probability = None
    history = []

    print(f"\nTraining for {args.epochs} epochs with patience {args.patience}")
    print(f"Architecture: {args.architecture}")
    print(f"Distillation: {args.distill}")
    print(f"Members: {args.members}")
    print(f"Batch size: {args.batch_size}")
    print(f"Evaluation batch size: {args.eval_batch_size}")
    print(f"Feature cache residency: {cache_residency}")
    print(f"Negative ratio: {args.negative_ratio}")
    print(f"Target detector recall: {args.target_recall:.1%}")
    print(f"Checkpoint policy: {SELECTION_POLICY}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Use different seed each epoch for negative sampling
        epoch_seed = args.seed + epoch

        train_result = train_epoch(
            model,
            optimizer,
            train_x,
            train_y,
            teacher_margin,
            teacher_severity,
            device,
            args.batch_size,
            args.negative_ratio,
            epoch_seed,
            delta_s,
            args.alpha,
            args.beta,
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
        val_metrics = evaluate(
            val_probabilities,
            val_y,
            target_recall=args.target_recall,
            detection_outputs=val_detector_probability,
            include_diagnostics=False,
        )
        metrics_seconds = time.perf_counter() - metrics_start
        val_ap = float(val_metrics["ap"])
        severity_metrics = val_metrics["severity"]
        val_exact = float(severity_metrics["exact_on_positive"])
        val_overprediction_rate = float(severity_metrics["overprediction_rate_all"])
        achieved_recall = float(val_metrics["recall"])
        selection_key = checkpoint_selection_key(val_metrics)
        selected_threshold = float(val_metrics["threshold"])

        if not all(
            np.isfinite(value)
            for value in (
                train_loss,
                val_ap,
                val_exact,
                val_overprediction_rate,
                achieved_recall,
                selected_threshold,
            )
        ):
            msg = (
                f"Non-finite value at epoch {epoch}: loss={train_loss}, "
                f"val_ap={val_ap}, exact={val_exact}, "
                f"overprediction_rate={val_overprediction_rate}, "
                f"recall={achieved_recall}, "
                f"threshold={selected_threshold}"
            )
            raise FloatingPointError(msg)

        epoch_time = time.perf_counter() - epoch_start
        gpu_peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        gpu_peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(train_loss),
                "validation_ap": val_ap,
                "achieved_recall": achieved_recall,
                "exact_on_positive": val_exact,
                "exact_on_positive_count": int(severity_metrics["exact_on_positive_count"]),
                "overprediction_count_all": int(severity_metrics["overprediction_count_all"]),
                "overprediction_rate_all": val_overprediction_rate,
                "overprediction_level_sum_all": int(severity_metrics["overprediction_level_sum_all"]),
                "underprediction_level_sum_all": int(severity_metrics["underprediction_level_sum_all"]),
                "threshold": selected_threshold,
                "selection_key": list(selection_key),
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
            f"exact+ {val_exact:.4f} | over(all) {val_overprediction_rate:.6f} | "
            f"recall {achieved_recall:.4f} | AP {val_ap:.4f} | "
            f"threshold {selected_threshold:.6g} | "
            f"time {epoch_time:.1f}s "
            f"(stage {train_result.staging_seconds:.1f}s, "
            f"train {train_result.optimization_seconds:.1f}s, "
            f"infer {inference_seconds:.1f}s, metrics {metrics_seconds:.1f}s, "
            f"GPU peak {gpu_peak_allocated_mib / 1024:.1f} GiB)"
        )

        # Early stopping
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_epoch = epoch
            best_threshold = selected_threshold
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
        or best_threshold is None
        or best_probabilities is None
        or best_detector_probability is None
    ):
        msg = "Training completed without producing a valid checkpoint"
        raise RuntimeError(msg)
    model.load_state_dict(best_state)
    restored_state = model.state_dict()
    if restored_state.keys() != best_state.keys() or any(
        not torch.equal(restored_state[name].detach().cpu(), best_state[name]) for name in best_state
    ):
        msg = "Loaded model state does not match the selected checkpoint"
        raise RuntimeError(msg)
    threshold_source = "selected with checkpoint on validation set"
    final_probabilities = best_probabilities
    final_detector_probability = best_detector_probability
    final_metrics = evaluate(
        final_probabilities,
        val_y,
        threshold=best_threshold,
        target_recall=args.target_recall,
        threshold_source=threshold_source,
        detection_outputs=final_detector_probability,
    )
    final_selection_key = checkpoint_selection_key(final_metrics)
    integer_key_matches = final_selection_key[:-1] == best_selection_key[:-1]
    ap_matches = np.isclose(final_selection_key[-1], best_selection_key[-1])
    if not integer_key_matches or not ap_matches:
        msg = "Cached selected-epoch predictions do not reproduce their validation selection key"
        raise RuntimeError(msg)
    if final_metrics["recall"] + np.finfo(float).eps < args.target_recall:
        msg = "Cached selected-epoch predictions do not satisfy the requested detector recall"
        raise RuntimeError(msg)

    selection = {
        "policy": SELECTION_POLICY,
        "threshold_rule": THRESHOLD_RULE,
        "threshold_score": THRESHOLD_SCORE,
        "target_recall": args.target_recall,
        "criteria": list(SELECTION_CRITERIA),
        "best_epoch": best_epoch,
        "best_key": list(best_selection_key),
        "threshold": best_threshold,
        "achieved_recall": float(final_metrics["recall"]),
        "selected_checkpoint_ap": float(final_metrics["ap"]),
        "max_ap_observed": max(row["validation_ap"] for row in history),
    }
    policy_diagnostic = epsilon_policy_diagnostic(
        final_probabilities,
        val_y,
        args.target_recall,
    )

    # Save results
    task = "distilled" if args.distill else "supervised"
    architecture_slug = args.architecture.replace("-", "_")
    artifact_name = f"student_{architecture_slug}_k{args.members}_{task}_seed{args.seed}"
    model_path, result_path = output_paths(args.output_dir, artifact_name, ".pt")

    torch.save(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": {
                **model_metadata,
                "delta_s": delta_s,
            },
            "threshold": best_threshold,
            "threshold_source": threshold_source,
            "target_recall": args.target_recall,
            "severity_decision": SEVERITY_DECISION,
            "selection": selection,
        },
        model_path,
    )

    result = {
        "model_format_version": MODEL_FORMAT_VERSION,
        "task": task,
        "seed": args.seed,
        "config": vars(args),
        "model": model_metadata,
        "metrics": final_metrics,
        "threshold": best_threshold,
        "threshold_source": threshold_source,
        "target_recall": args.target_recall,
        "severity_decision": SEVERITY_DECISION,
        "selection": selection,
        "epsilon_policy_diagnostic": policy_diagnostic,
        "history": history,
        "performance": {
            "cache_load_seconds": cache_load_seconds,
            "index_pool_seconds": pool_seconds,
            "feature_staging_seconds": feature_staging_seconds,
            "cache_residency": cache_residency,
            "max_gpu_allocated_mib": max(row["gpu_peak_allocated_mib"] for row in history),
            "max_gpu_reserved_mib": max(row["gpu_peak_reserved_mib"] for row in history),
        },
    }
    write_json(result_path, result)

    print(f"\nSaved model: {model_path}")
    print(f"Saved results: {result_path}")
    print(f"Final AP: {final_metrics['ap']:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Frozen threshold: {best_threshold:.6g}")
    print(f"Exact on positives: {final_metrics['severity']['exact_on_positive']:.4f}")
    print(f"All-row overprediction rate: {final_metrics['severity']['overprediction_rate_all']:.6f}")
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
    parser.add_argument("--alpha", type=float, default=1.0, help="Detector distillation weight")
    parser.add_argument("--beta", type=float, default=0.5, help="Severity distillation weight")

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
