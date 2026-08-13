"""Post-hoc feature importance for frozen neural students.

The primary analysis is grouped permutation importance on the chronological
validation cache.  A secondary Expected Gradients analysis provides local,
signed attributions without adding SHAP as a runtime dependency.  Neither
analysis retrains a model or reads the held-out test period.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F

from mlstep.data import FEATURES, GRID_SHAPE, N_CLASSES, Feature, class_counts, feature_names
from mlstep.evaluation import detection_ap, detection_counts, probability_metrics, safe_divide, threshold_at_recall
from mlstep.student import Student, student_from_checkpoint

PERMUTATION_REPORT_VERSION: Final = "student-grouped-permutation-v1"
EXPECTED_GRADIENTS_REPORT_VERSION: Final = "student-expected-gradients-v2"
DEFAULT_BATCH_SIZE: Final = 65_536
DEFAULT_REPEATS: Final = 5
DEFAULT_BOOTSTRAP_REPEATS: Final = 10_000
DEFAULT_TARGET_RECALL: Final = 0.97
DEFAULT_BACKGROUND_ROWS: Final = 256
DEFAULT_POSITIVE_ROWS: Final = 1_024
DEFAULT_NEGATIVE_ROWS: Final = 2_048
DEFAULT_EXPECTED_GRADIENT_SAMPLES: Final = 64
DEFAULT_GRADIENT_BATCH_SIZE: Final = 1_024
MATRIX_NDIM: Final = 2
SHA256_HEX_LENGTH: Final = 64
DEFAULT_PERMUTATION_STRATUM_ROWS: Final = GRID_SHAPE[1] * GRID_SHAPE[2]
FROZEN_MANIFEST_VERSION: Final = "held-out-student-ensemble-v1"
FROZEN_MANIFEST_STATUS: Final = "frozen-before-held-out-test-access"

# These variables are present in the inspected 24-hour directory but are not
# inputs to the frozen 266-column checkpoint.  They must never be reported as
# zero-importance features: the model had no opportunity to use them.
UNASSESSED_AVAILABLE_GROUPS: Final = (
    {
        "name": "water_vapour",
        "layout": "(z=38, y=72, x=96)",
        "reason": "row-aligned but absent from the checkpoint input schema; requires an augmented retraining run",
    },
    {
        "name": "sza",
        "layout": "(y=72, x=96)",
        "reason": "absent from the checkpoint and requires an explicit vertical broadcast before retraining",
    },
    {
        "name": "dryrt",
        "layout": "(dry-deposition species=42, y=72, x=96)",
        "reason": (
            "absent from the checkpoint; requires a species mapping plus a scientifically defined 38-level "
            "mask/broadcast using nlev_with_ddep"
        ),
    },
    {
        "name": "nlev_with_ddep",
        "layout": "(y=72, x=96)",
        "reason": "auxiliary dry-deposition mask absent from the checkpoint input schema",
    },
    {
        "name": "latitude/longitude",
        "layout": "static (y=72, x=96)",
        "reason": "absent from the checkpoint and requires an explicit vertical broadcast before retraining",
    },
)


@dataclass(frozen=True, slots=True)
class FeatureGroup:
    """A contiguous, scientifically meaningful group of input columns."""

    name: str
    start: int
    stop: int

    @property
    def width(self) -> int:
        """Return the number of scalar columns in the group."""
        return self.stop - self.start

    def state(self) -> dict[str, int | str]:
        """Return a JSON-ready description."""
        return {"name": self.name, "start": self.start, "stop": self.stop, "width": self.width}


@dataclass(frozen=True, slots=True)
class AnalysisInputs:
    """Validated model and cache inputs shared by both analyses."""

    models: tuple[Student, ...]
    checkpoints: tuple[dict, ...]
    checkpoint_paths: tuple[Path, ...]
    cache_dir: Path
    metadata: dict
    validation_artifacts: dict[str, dict[str, str | int]]
    threshold: float | None
    threshold_source: str | None
    manifest_path: Path | None


@dataclass(frozen=True, slots=True)
class ExpectedGradientResult:
    """Attributions and two complementary Monte Carlo completeness checks."""

    attributions: np.ndarray
    sampled_reference_residuals: np.ndarray
    fixed_background_residuals: np.ndarray
    explained_outputs: np.ndarray
    sampled_reference_outputs: np.ndarray
    fixed_background_output: np.ndarray


@dataclass(frozen=True, slots=True)
class EnsemblePredictionResult:
    """Joint predictions plus a numerically stable conditional severity view."""

    joint_probabilities: np.ndarray
    conditional_severity: np.ndarray


class PermutedFeatureMatrix:
    """Lazily replace one feature group from permuted source rows.

    Only the current inference batch is copied.  This avoids allocating a
    second full validation matrix for every group permutation.
    """

    def __init__(self, source: np.ndarray, group: FeatureGroup, row_permutation: np.ndarray) -> None:
        """Initialize the view over a resident source matrix."""
        if source.ndim != MATRIX_NDIM:
            msg = "source must be a two-dimensional feature matrix"
            raise ValueError(msg)
        permutation = np.asarray(row_permutation)
        if permutation.ndim != 1 or len(permutation) != len(source):
            msg = "row_permutation must contain one index per source row"
            raise ValueError(msg)
        if not np.issubdtype(permutation.dtype, np.integer):
            msg = "row_permutation must contain integer indices"
            raise TypeError(msg)
        if permutation.size and (permutation.min() < 0 or permutation.max() >= len(source)):
            msg = "row_permutation contains an out-of-range row"
            raise IndexError(msg)
        if not 0 <= group.start < group.stop <= source.shape[1]:
            msg = "feature group is outside the source column range"
            raise ValueError(msg)
        self.source = source
        self.group = group
        self.row_permutation = permutation

    def __len__(self) -> int:
        """Return the number of source rows."""
        return len(self.source)

    def __getitem__(self, item: slice) -> np.ndarray:
        """Materialize one contiguous inference batch."""
        if not isinstance(item, slice):
            msg = "PermutedFeatureMatrix supports contiguous slices only"
            raise TypeError(msg)
        start, stop, step = item.indices(len(self))
        if step != 1:
            msg = "PermutedFeatureMatrix does not support strided slices"
            raise ValueError(msg)
        output = np.array(self.source[start:stop], copy=True, order="C")
        mapped_rows = self.row_permutation[start:stop]
        group = self.group
        output[:, group.start : group.stop] = self.source[mapped_rows, group.start : group.stop]
        return output


def feature_groups(features: tuple[Feature, ...] = FEATURES) -> tuple[FeatureGroup, ...]:
    """Return contiguous column ranges derived from the canonical feature schema."""
    groups = []
    start = 0
    for feature in features:
        stop = start + feature.channels
        groups.append(FeatureGroup(feature.name, start, stop))
        start = stop
    return tuple(groups)


def within_block_permutation(n_rows: int, block_rows: int, seed: int) -> np.ndarray:
    """Return a deterministic row permutation that never crosses block boundaries."""
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        msg = "n_rows must be a positive integer"
        raise ValueError(msg)
    if isinstance(block_rows, bool) or not isinstance(block_rows, int) or block_rows < 1:
        msg = "block_rows must be a positive integer"
        raise ValueError(msg)
    generator = np.random.default_rng(seed)
    permutation = np.empty(n_rows, dtype=np.int64)
    for start in range(0, n_rows, block_rows):
        stop = min(start + block_rows, n_rows)
        permutation[start:stop] = start + generator.permutation(stop - start)
    return permutation


def _utc_now() -> str:
    """Return an unambiguous UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 2**20) -> str:
    """Hash a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, hash_contents: bool) -> dict[str, str | int]:
    """Return stable provenance for an input file."""
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        msg = f"Expected a file artifact: {resolved}"
        raise ValueError(msg)
    result: dict[str, str | int] = {"path": str(resolved), "bytes": resolved.stat().st_size}
    if hash_contents:
        result["sha256"] = _sha256_file(resolved)
    return result


def _load_json(path: Path, label: str) -> dict:
    """Load a required JSON object."""
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        msg = f"{label} must contain a JSON object"
        raise ValueError(msg)
    return content


def _verify_artifact_record(record: object, label: str, *, parent: Path | None = None) -> Path:
    """Verify the path, size, and digest of one frozen-manifest artifact."""
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        msg = f"Invalid {label} artifact record"
        raise ValueError(msg)
    path = Path(record["path"]).expanduser().resolve(strict=True)
    if not path.is_file():
        msg = f"Expected a file for {label}: {path}"
        raise ValueError(msg)
    if parent is not None and path.parent != parent:
        msg = f"{label} escaped its frozen directory: {path}"
        raise ValueError(msg)
    expected_bytes = record["bytes"]
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
        msg = f"{label} size changed after manifest freeze: {path}"
        raise RuntimeError(msg)
    digest = record["sha256"]
    if not isinstance(digest, str) or len(digest) != SHA256_HEX_LENGTH or _sha256_file(path) != digest:
        msg = f"{label} SHA-256 changed after manifest freeze: {path}"
        raise RuntimeError(msg)
    return path


def _manifest_inputs(path: Path) -> tuple[tuple[Path, ...], Path, float, str]:  # noqa: PLR0912, PLR0915
    """Extract frozen validation inputs after verifying every recorded artifact."""
    manifest_path = path.expanduser().resolve(strict=True)
    sidecar_path = Path(f"{manifest_path}.sha256")
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = _load_json(sidecar_path, "manifest digest sidecar")
    digest = _sha256_file(manifest_path)
    expected_sidecar = {"algorithm": "sha256", "manifest": manifest_path.name, "sha256": digest}
    if sidecar != expected_sidecar:
        msg = "Manifest SHA-256 validation failed"
        raise RuntimeError(msg)

    manifest = _load_json(manifest_path, "frozen manifest")
    if manifest.get("format_version") != FROZEN_MANIFEST_VERSION:
        msg = f"Unsupported frozen manifest format: {manifest.get('format_version')!r}"
        raise ValueError(msg)
    if manifest.get("status") != FROZEN_MANIFEST_STATUS:
        msg = "Manifest is not in the frozen-before-held-out-access state"
        raise ValueError(msg)
    if manifest.get("selection_constraints", {}).get("held_out_test_used") is not False:
        msg = "Frozen manifest does not prove validation-only model selection"
        raise ValueError(msg)
    ensemble = manifest.get("ensemble", {})
    validation = manifest.get("validation", {})
    members = ensemble.get("members")
    if not isinstance(members, list) or not members:
        msg = "Frozen manifest has no ensemble members"
        raise ValueError(msg)
    if ensemble.get("aggregation") != "equal arithmetic mean of full joint class probabilities":
        msg = "Feature importance requires an equal-probability frozen ensemble"
        raise ValueError(msg)
    if ensemble.get("member_count") != len(members) or ensemble.get("checkpoint_role") != "canonical":
        msg = "Frozen manifest does not declare the expected canonical ensemble"
        raise ValueError(msg)

    paths = []
    seen_paths = set()
    expected_weight = 1.0 / len(members)
    for index, member in enumerate(members):
        if not isinstance(member, dict) or member.get("role") != "canonical":
            msg = f"Frozen member {index} is not canonical"
            raise ValueError(msg)
        weight = member.get("probability_weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not np.isclose(weight, expected_weight, rtol=0.0, atol=1e-15)
        ):
            msg = f"Frozen member {index} does not have equal probability weight"
            raise ValueError(msg)
        checkpoint_path = _verify_artifact_record(member.get("checkpoint"), f"frozen member {index} checkpoint")
        _verify_artifact_record(member.get("source_report"), f"frozen member {index} source report")
        if checkpoint_path in seen_paths:
            msg = f"Duplicate frozen ensemble checkpoint: {checkpoint_path}"
            raise ValueError(msg)
        seen_paths.add(checkpoint_path)
        paths.append(checkpoint_path)

    cache_dir = Path(validation.get("cache_dir", "")).expanduser().resolve(strict=True)
    artifacts = validation.get("artifacts")
    if not isinstance(artifacts, dict) or not {"val_x.npy", "val_y.npy", "metadata.json"} <= set(artifacts):
        msg = "Frozen manifest validation-cache artifact set is incomplete"
        raise ValueError(msg)
    for name, artifact_record in artifacts.items():
        _verify_artifact_record(artifact_record, f"frozen validation cache {name}", parent=cache_dir)

    threshold = validation.get("frozen_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        msg = "Frozen manifest does not contain a valid validation threshold"
        raise ValueError(msg)
    threshold_source = str(validation.get("threshold_source", "frozen validation manifest"))
    return tuple(paths), cache_dir, float(threshold), threshold_source


def _validate_checkpoint(checkpoint: dict, path: Path, metadata: dict, n_features: int) -> None:
    """Reject diagnostic or cache-incompatible checkpoints."""
    if checkpoint.get("checkpoint_role") != "canonical":
        msg = f"Only canonical checkpoints may be analyzed: {path}"
        raise ValueError(msg)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        msg = f"Checkpoint has no configuration dictionary: {path}"
        raise ValueError(msg)
    if config.get("n_features") != n_features:
        msg = f"Checkpoint feature count does not match the validation cache: {path}"
        raise ValueError(msg)
    recorded_preprocess = metadata.get("preprocess")
    if config.get("preprocess") != recorded_preprocess:
        msg = f"Checkpoint preprocessing does not match cache {recorded_preprocess!r}: {path}"
        raise ValueError(msg)


def load_analysis_inputs(  # noqa: PLR0912, PLR0915
    checkpoint_paths: Sequence[Path],
    cache_dir: Path | None,
    *,
    manifest_path: Path | None,
    device: torch.device,
) -> AnalysisInputs:
    """Load and validate the exact models and chronological validation cache."""
    threshold = None
    threshold_source = None
    resolved_manifest = None
    if manifest_path is not None:
        if checkpoint_paths:
            msg = "Use either --manifest or --checkpoint, not both"
            raise ValueError(msg)
        manifest_checkpoints, manifest_cache, threshold, threshold_source = _manifest_inputs(manifest_path)
        checkpoint_paths = manifest_checkpoints
        resolved_manifest = manifest_path.expanduser().resolve(strict=True)
        if cache_dir is not None and cache_dir.expanduser().resolve(strict=True) != manifest_cache:
            msg = "--cache-dir differs from the cache frozen in --manifest"
            raise ValueError(msg)
        cache_dir = manifest_cache
    elif not checkpoint_paths:
        msg = "At least one --checkpoint or a --manifest is required"
        raise ValueError(msg)

    if cache_dir is None:
        msg = "--cache-dir is required when --manifest is not used"
        raise ValueError(msg)
    resolved_cache = cache_dir.expanduser().resolve(strict=True)
    metadata_path = resolved_cache / "metadata.json"
    metadata = _load_json(metadata_path, "cache metadata")
    expected_names = feature_names(FEATURES)
    if metadata.get("features") != expected_names or metadata.get("n_features") != len(expected_names):
        msg = "Validation cache feature names/order do not match the current canonical schema"
        raise ValueError(msg)
    preprocessing = metadata.get("preprocessing", {})
    if (
        preprocessing.get("method") != metadata.get("preprocess")
        or preprocessing.get("fit_split") != "chronological training timesteps only"
        or preprocessing.get("output_dtype") != "float32"
    ):
        msg = "Cache does not prove train-only preprocessing into float32 model inputs"
        raise ValueError(msg)

    val_x = np.load(resolved_cache / "val_x.npy", mmap_mode="r", allow_pickle=False)
    val_y = np.load(resolved_cache / "val_y.npy", mmap_mode="r", allow_pickle=False)
    if val_x.ndim != MATRIX_NDIM or not len(val_x) or val_x.shape[1] != len(expected_names):
        msg = "Validation feature matrix has an unexpected shape"
        raise ValueError(msg)
    if val_x.dtype != np.float32:
        msg = f"Validation feature matrix must be float32, found {val_x.dtype}"
        raise ValueError(msg)
    if val_y.shape != (len(val_x),) or metadata.get("val", {}).get("shape") != list(val_x.shape):
        msg = "Validation labels/cache metadata do not match the feature matrix"
        raise ValueError(msg)
    if not np.issubdtype(val_y.dtype, np.integer) or val_y.min() < 0 or val_y.max() >= N_CLASSES:
        msg = f"Validation labels must be integer classes in [0, {N_CLASSES - 1}]"
        raise ValueError(msg)
    if metadata.get("val", {}).get("class_counts") != class_counts(val_y):
        msg = "Validation class counts differ from cache metadata"
        raise ValueError(msg)
    validation_artifacts = {
        name: _artifact(resolved_cache / name, hash_contents=True)
        for name in ("metadata.json", "val_x.npy", "val_y.npy")
    }

    models = []
    checkpoints = []
    resolved_paths = []
    seen_paths = set()
    expected_config = None
    expected_objective = None
    for checkpoint_path in checkpoint_paths:
        resolved_path = checkpoint_path.expanduser().resolve(strict=True)
        if resolved_path in seen_paths:
            msg = f"Duplicate ensemble checkpoint: {resolved_path}"
            raise ValueError(msg)
        seen_paths.add(resolved_path)
        checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            msg = f"Checkpoint is not a dictionary: {resolved_path}"
            raise ValueError(msg)
        _validate_checkpoint(checkpoint, resolved_path, metadata, val_x.shape[1])
        config = checkpoint["config"]
        comparable_config = {
            key: config.get(key)
            for key in (
                "architecture",
                "architecture_version",
                "n_features",
                "hidden",
                "k",
                "dropout",
                "preprocess",
                "ple_config",
            )
        }
        if expected_config is None:
            expected_config = comparable_config
        elif comparable_config != expected_config:
            msg = "All ensemble checkpoints must have identical model configurations"
            raise ValueError(msg)
        objective = checkpoint.get("objective")
        if not isinstance(objective, dict):
            msg = f"Checkpoint has no objective dictionary: {resolved_path}"
            raise ValueError(msg)
        if expected_objective is None:
            expected_objective = objective
        elif objective != expected_objective:
            msg = "All ensemble checkpoints must have identical training objectives"
            raise ValueError(msg)
        model = student_from_checkpoint(checkpoint).to(device)
        model.eval()
        models.append(model)
        checkpoints.append(checkpoint)
        resolved_paths.append(resolved_path)

    return AnalysisInputs(
        models=tuple(models),
        checkpoints=tuple(checkpoints),
        checkpoint_paths=tuple(resolved_paths),
        cache_dir=resolved_cache,
        metadata=metadata,
        validation_artifacts=validation_artifacts,
        threshold=threshold,
        threshold_source=threshold_source,
        manifest_path=resolved_manifest,
    )


def _load_array_resident(path: Path) -> np.ndarray:
    """Copy an NPY array sequentially into host RAM with bounded scratch space."""
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    destination = np.empty(source.shape, dtype=source.dtype)
    if source.ndim == 0:
        destination[...] = source
        return destination
    bytes_per_row = max(1, source[0:1].nbytes)
    rows_per_chunk = max(1, (512 * 2**20) // bytes_per_row)
    started = time.perf_counter()
    print(f"Loading {path.name} into RAM: {source.shape} ({source.nbytes / 2**30:.2f} GiB)", flush=True)
    for start in range(0, len(source), rows_per_chunk):
        stop = min(start + rows_per_chunk, len(source))
        destination[start:stop] = source[start:stop]
    print(f"Loaded {path.name} in {time.perf_counter() - started:.1f}s", flush=True)
    return destination


def _synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work before timing or host access."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _ensemble_joint_and_conditional(
    models: Sequence[Student],
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the deployed joint mixture and stable conditional severity.

    The joint distribution deliberately follows ``Student.forward`` in the
    model dtype.  Conditional severity is additionally reconstructed from raw
    member logits in float64 log space.  This represents the same
    detector-weighted equal-joint mixture, but it remains defined when an
    extremely negative detector logit makes float32 sigmoid underflow to zero.
    """
    if not models:
        msg = "At least one model is required"
        raise ValueError(msg)

    joint_sum = None
    log_positive_sum = None
    for model in models:
        detector_logits, severity_logits = model.forward_logits(inputs)

        # Reproduce Student.forward and joint_distribution in the deployed
        # model dtype so headline joint/detector metrics do not change.
        member_detector = torch.sigmoid(detector_logits)
        member_severity = torch.softmax(severity_logits, dim=-1)
        detector = member_detector.mean(dim=0)
        positive_mass = (member_detector.unsqueeze(-1) * member_severity).mean(dim=0)
        conditional = positive_mass / detector.unsqueeze(-1).clamp_min(torch.finfo(detector.dtype).tiny)
        joint = model.joint_distribution(detector, conditional)
        joint_sum = joint if joint_sum is None else joint_sum + joint

        # Each checkpoint has equal weight, and each of its K members has
        # equal weight.  Constants common to every severity class cancel in
        # the final softmax, but retaining 1/K makes the checkpoint semantics
        # explicit and remains correct if models with different K are used.
        member_log_positive = F.logsigmoid(detector_logits.to(torch.float64)).unsqueeze(-1)
        member_log_positive = member_log_positive + F.log_softmax(severity_logits.to(torch.float64), dim=-1)
        checkpoint_log_positive = torch.logsumexp(member_log_positive, dim=0) - math.log(detector_logits.shape[0])
        log_positive_sum = (
            checkpoint_log_positive
            if log_positive_sum is None
            else torch.logaddexp(log_positive_sum, checkpoint_log_positive)
        )

    if joint_sum is None or log_positive_sum is None:
        msg = "Ensemble prediction produced no outputs"
        raise RuntimeError(msg)
    joint_probabilities = joint_sum / len(models)
    stable_conditional = torch.softmax(log_positive_sum, dim=-1)
    return joint_probabilities, stable_conditional


@torch.inference_mode()
def predict_equal_ensemble(
    models: Sequence[Student],
    features: object,
    device: torch.device,
    batch_size: int,
    *,
    label: str,
) -> EnsemblePredictionResult:
    """Predict equal-joint probabilities and stable conditional severity."""
    if not models:
        msg = "At least one model is required"
        raise ValueError(msg)
    if batch_size < 1:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    n_rows = len(features)  # type: ignore[arg-type]
    joint_output = torch.empty(
        (n_rows, N_CLASSES),
        dtype=torch.float64,
        device="cpu",
        pin_memory=device.type == "cuda",
    )
    conditional_output = torch.empty(
        (n_rows, N_CLASSES - 1),
        dtype=torch.float64,
        device="cpu",
        pin_memory=device.type == "cuda",
    )
    n_batches = int(np.ceil(n_rows / batch_size))
    progress_every = max(1, n_batches // 10)
    started = time.perf_counter()
    for batch_number, start in enumerate(range(0, n_rows, batch_size), start=1):
        stop = min(start + batch_size, n_rows)
        batch_values = features[start:stop]  # type: ignore[index]
        if isinstance(batch_values, torch.Tensor):
            batch_x = batch_values.to(device)
        else:
            batch_x = torch.from_numpy(np.asarray(batch_values)).to(device)
        joint_probabilities, conditional_severity = _ensemble_joint_and_conditional(models, batch_x)
        joint_output[start:stop].copy_(joint_probabilities.to(torch.float64), non_blocking=device.type == "cuda")
        conditional_output[start:stop].copy_(conditional_severity, non_blocking=device.type == "cuda")
        if batch_number % progress_every == 0 or batch_number == n_batches:
            _synchronize(device)
            print(
                f"  {label}: {batch_number:,}/{n_batches:,} batches, {stop:,}/{n_rows:,} rows, "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    _synchronize(device)
    return EnsemblePredictionResult(
        joint_probabilities=joint_output.numpy(),
        conditional_severity=conditional_output.numpy(),
    )


def _conditional_severity_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    conditional_severity: np.ndarray | None,
) -> dict[str, float | int]:
    """Score the four-class conditional severity distribution on positive rows."""
    positive = labels > 0
    if not np.any(positive):
        msg = "Conditional severity metrics require at least one positive row"
        raise ValueError(msg)
    positive_mass = probabilities[positive, 1:]
    denominator = positive_mass.sum(axis=1, dtype=np.float64, keepdims=True)
    zero_mass_rows = int(np.count_nonzero(denominator[:, 0] == 0.0))
    if conditional_severity is None:
        if zero_mass_rows:
            msg = (
                "Conditional severity cannot be recovered from joint probabilities with zero positive-class mass; "
                "supply the stable log-space conditional distribution"
            )
            raise ValueError(msg)
        conditional = positive_mass / denominator
    else:
        conditional_severity = np.asarray(conditional_severity)
        expected_shape = (len(probabilities), N_CLASSES - 1)
        if conditional_severity.shape != expected_shape:
            msg = f"conditional_severity must have shape {expected_shape}"
            raise ValueError(msg)
        conditional = conditional_severity[positive]
    metrics = probability_metrics(conditional, labels[positive] - 1)
    return {
        **metrics,
        "joint_positive_mass_zero_rows": zero_mass_rows,
        "scored_rows": int(np.count_nonzero(positive)),
    }


def score_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    conditional_severity: np.ndarray | None = None,
) -> dict:
    """Return threshold-free and fixed-threshold scores for one prediction matrix."""
    detector = probabilities[:, 1:].sum(axis=1, dtype=np.float64)
    positive = labels > 0
    detected = detector >= threshold
    counts = detection_counts(detected, positive)
    precision = safe_divide(counts["true_positive"], counts["true_positive"] + counts["false_positive"])
    recall = safe_divide(counts["true_positive"], counts["true_positive"] + counts["false_negative"])
    return {
        "joint": probability_metrics(probabilities, labels),
        "detector": {
            "average_precision": detection_ap(detector, labels),
            "precision": precision,
            "recall": recall,
            "false_positives_per_million_negatives": safe_divide(
                counts["false_positive"] * 1_000_000,
                counts["false_positive"] + counts["true_negative"],
            ),
            **counts,
        },
        "conditional_severity_positive_rows": _conditional_severity_metrics(
            probabilities,
            labels,
            conditional_severity,
        ),
    }


def importance_deltas(baseline: dict, perturbed: dict) -> dict[str, float]:
    """Express every importance so that positive values mean performance worsened."""
    return {
        "joint_rps_increase": (
            perturbed["joint"]["ranked_probability_score"] - baseline["joint"]["ranked_probability_score"]
        ),
        "joint_nll_increase": perturbed["joint"]["negative_log_likelihood"]
        - baseline["joint"]["negative_log_likelihood"],
        "joint_brier_increase": perturbed["joint"]["brier_score"] - baseline["joint"]["brier_score"],
        "detector_ap_drop": baseline["detector"]["average_precision"] - perturbed["detector"]["average_precision"],
        "fixed_threshold_recall_drop": baseline["detector"]["recall"] - perturbed["detector"]["recall"],
        "fixed_threshold_false_positive_increase": float(
            perturbed["detector"]["false_positive"] - baseline["detector"]["false_positive"]
        ),
        "conditional_severity_rps_increase": (
            perturbed["conditional_severity_positive_rows"]["ranked_probability_score"]
            - baseline["conditional_severity_positive_rows"]["ranked_probability_score"]
        ),
        "conditional_severity_nll_increase": (
            perturbed["conditional_severity_positive_rows"]["negative_log_likelihood"]
            - baseline["conditional_severity_positive_rows"]["negative_log_likelihood"]
        ),
    }


def _block_rps(probabilities: np.ndarray, labels: np.ndarray, block_rows: int) -> np.ndarray:
    """Return one joint RPS value per chronological block."""
    values = []
    for start in range(0, len(labels), block_rows):
        stop = min(start + block_rows, len(labels))
        values.append(probability_metrics(probabilities[start:stop], labels[start:stop])["ranked_probability_score"])
    return np.asarray(values, dtype=np.float64)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize independent perturbation repeats without clipping negative values."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "repeats": len(array),
    }


def block_bootstrap_interval(values: np.ndarray, repeats: int, seed: int) -> dict[str, float | int]:
    """Bootstrap the mean importance over whole chronological blocks."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != MATRIX_NDIM or not array.shape[0] or not array.shape[1]:
        msg = "values must have shape (permutation repeats, chronological blocks)"
        raise ValueError(msg)
    if repeats < 1:
        msg = "bootstrap repeats must be positive"
        raise ValueError(msg)
    per_block = array.mean(axis=0)
    generator = np.random.default_rng(seed)
    selections = generator.integers(0, len(per_block), size=(repeats, len(per_block)))
    estimates = per_block[selections].mean(axis=1)
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return {
        "estimate": float(per_block.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "blocks": len(per_block),
        "bootstrap_repeats": repeats,
        "seed": seed,
    }


def _checkpoint_provenance(inputs: AnalysisInputs) -> list[dict]:
    """Describe every ensemble member and its training objective."""
    records = []
    for path, checkpoint in zip(inputs.checkpoint_paths, inputs.checkpoints, strict=True):
        records.append(
            {
                "artifact": _artifact(path, hash_contents=True),
                "checkpoint_role": checkpoint.get("checkpoint_role"),
                "config": checkpoint.get("config"),
                "objective": checkpoint.get("objective"),
                "selection": checkpoint.get("selection"),
            }
        )
    return records


def _common_report(inputs: AnalysisInputs, labels: np.ndarray, block_rows: int) -> dict:
    """Build shared provenance and interpretation constraints."""
    return {
        "created_utc": _utc_now(),
        "data": {
            "split": "checkpoint-matched validation-cache partition only",
            "held_out_test_used": False,
            "held_out_statement_scope": (
                "the analysis accepts only train/validation cache arrays and exposes no test-data input; semantic "
                "split identity additionally relies on the recorded cache provenance"
            ),
            "cache_dir": str(inputs.cache_dir),
            "preprocess": inputs.metadata.get("preprocess"),
            "rows": len(labels),
            "class_counts": class_counts(labels),
            "rows_per_timestep_block": block_rows,
            "chronological_blocks": int(np.ceil(len(labels) / block_rows)),
            "cache_artifacts": inputs.validation_artifacts,
        },
        "ensemble": {
            "aggregation": "equal arithmetic mean of full joint class probabilities",
            "members": _checkpoint_provenance(inputs),
        },
        "manifest": None if inputs.manifest_path is None else _artifact(inputs.manifest_path, hash_contents=True),
        "feature_groups": [group.state() for group in feature_groups()],
        "unassessed_available_feature_groups": list(UNASSESSED_AVAILABLE_GROUPS),
        "interpretation": {
            "scope": "reliance of the fitted neural ensemble, not causal or physical importance",
            "low_score": (
                "low unique reliance given all other fitted-model inputs; correlated substitutes may mask relevance"
            ),
            "unseen_inputs": "variables absent from the checkpoint cannot receive post-hoc importance",
            "input_coordinates": (
                "attributions and perturbations operate on the checkpoint-matched preprocessed columns; "
                "columns retain a one-to-one mapping to the named physical inputs"
            ),
        },
        "implementation": _artifact(Path(__file__), hash_contents=True),
    }


def _write_json(path: Path, content: dict) -> None:
    """Write a deterministic, human-readable JSON report."""
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_output_dir(path: Path, artifact_names: Sequence[str]) -> Path:
    """Create an output directory while refusing to replace analysis artifacts."""
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    existing = [resolved / name for name in artifact_names if (resolved / name).exists()]
    if existing:
        msg = f"Refusing to replace existing feature-importance artifacts: {', '.join(map(str, existing))}"
        raise FileExistsError(msg)
    return resolved


def _write_permutation_csv(path: Path, rows: Sequence[dict]) -> None:
    """Write one compact row per feature group and repeat."""
    fields = [
        "group",
        "width",
        "repeat",
        "permutation_seed",
        "joint_rps_increase",
        "joint_nll_increase",
        "joint_brier_increase",
        "detector_ap_drop",
        "fixed_threshold_recall_drop",
        "fixed_threshold_false_positive_increase",
        "conditional_severity_rps_increase",
        "conditional_severity_nll_increase",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _write_permutation_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    """Write presentation-friendly group summaries sorted by primary importance."""
    metric_names = tuple(rows[0]["metrics"])
    statistics = ("mean", "standard_deviation", "minimum", "maximum")
    fields = ["group", "start", "stop", "width"]
    fields.extend(f"{metric}_{statistic}" for metric in metric_names for statistic in statistics)
    fields.extend(("joint_rps_bootstrap_lower_95", "joint_rps_bootstrap_upper_95"))
    ordered = sorted(rows, key=lambda row: row["metrics"]["joint_rps_increase"]["mean"], reverse=True)
    flattened = []
    for row in ordered:
        output = {
            "group": row["name"],
            "start": row["start"],
            "stop": row["stop"],
            "width": row["width"],
            "joint_rps_bootstrap_lower_95": row["joint_rps_whole_timestep_bootstrap_95"]["lower_95"],
            "joint_rps_bootstrap_upper_95": row["joint_rps_whole_timestep_bootstrap_95"]["upper_95"],
        }
        for metric in metric_names:
            for statistic in statistics:
                output[f"{metric}_{statistic}"] = row["metrics"][metric][statistic]
        flattened.append(output)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def _plot_permutation(summary: Sequence[dict], path: Path, severity_counts: Sequence[int]) -> None:
    """Plot overall, detector, and positive-conditional severity reliance."""
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    panels = (
        ("joint_rps_increase", "Overall five-class RPS increase", True),
        ("detector_ap_drop", "Detector average-precision drop", False),
        ("conditional_severity_rps_increase", "Positive-conditional severity RPS increase", False),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for axis, (metric, title, use_bootstrap) in zip(axes, panels, strict=True):
        ordered = sorted(summary, key=lambda row: row["metrics"][metric]["mean"])
        names = [row["name"] for row in ordered]
        means = np.asarray([row["metrics"][metric]["mean"] for row in ordered])
        if use_bootstrap:
            intervals = [row["joint_rps_whole_timestep_bootstrap_95"] for row in ordered]
            lower = np.asarray([interval["lower_95"] for interval in intervals])
            upper = np.asarray([interval["upper_95"] for interval in intervals])
        else:
            deviations = np.asarray([row["metrics"][metric]["standard_deviation"] for row in ordered])
            lower = means - deviations
            upper = means + deviations
        positions = np.arange(len(names))
        axis.barh(positions, means)
        axis.hlines(positions, lower, upper, colors="black", linewidth=1.0)
        cap_half_height = 0.08
        axis.vlines(lower, positions - cap_half_height, positions + cap_half_height, colors="black", linewidth=1.0)
        axis.vlines(upper, positions - cap_half_height, positions + cap_half_height, colors="black", linewidth=1.0)
        axis.set_yticks(positions, labels=names)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("Performance degradation after permutation")
        axis.set_title(title)
    severity_support = ", ".join(f"class {index + 1}: {count}" for index, count in enumerate(severity_counts))
    figure.suptitle(
        "Grouped permutation importance (higher = more relied upon)\n"
        f"Severity support — {severity_support}\n"
        "Overall bars: descriptive whole-timestep bootstrap interval; detector/severity bars: +/-1 SD over repeats"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_permutation(args: argparse.Namespace) -> dict:  # noqa: PLR0912, PLR0915
    """Run grouped permutation importance on cached validation inputs."""
    if args.repeats < 1 or args.bootstrap_repeats < 1:
        msg = "repeats and bootstrap-repeats must be positive"
        raise ValueError(msg)
    if args.batch_size < 1 or args.block_rows < 1 or args.permutation_stratum_rows < 1:
        msg = "batch-size, block-rows, and permutation-stratum-rows must be positive"
        raise ValueError(msg)
    if args.block_rows % args.permutation_stratum_rows:
        msg = "permutation-stratum-rows must divide one chronological timestep block exactly"
        raise ValueError(msg)
    if not 0.0 < args.target_recall <= 1.0:
        msg = "target-recall must be in (0, 1]"
        raise ValueError(msg)
    if args.threshold is not None and (not np.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0):
        msg = "threshold must be finite and in [0, 1]"
        raise ValueError(msg)
    selected_names = set(args.groups or [group.name for group in feature_groups()])
    unknown = selected_names - {group.name for group in feature_groups()}
    if unknown:
        msg = f"Unknown feature groups: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    output_dir = _prepare_output_dir(
        args.output_dir,
        (
            "permutation_importance.json",
            "permutation_importance.csv",
            "permutation_importance_summary.csv",
            "permutation_importance.png",
        ),
    )
    device = torch.device(args.device)
    inputs = load_analysis_inputs(
        args.checkpoint or (),
        args.cache_dir,
        manifest_path=args.manifest,
        device=device,
    )

    features = _load_array_resident(inputs.cache_dir / "val_x.npy")
    labels = _load_array_resident(inputs.cache_dir / "val_y.npy")
    if len(features) % args.block_rows:
        msg = "Validation rows are not an exact number of declared timestep blocks"
        raise ValueError(msg)

    baseline_predictions = predict_equal_ensemble(
        inputs.models,
        features,
        device,
        args.batch_size,
        label="baseline",
    )
    threshold = inputs.threshold
    threshold_source = inputs.threshold_source
    if threshold is None:
        if args.threshold is None:
            baseline_detector = baseline_predictions.joint_probabilities[:, 1:].sum(axis=1, dtype=np.float64)
            threshold = threshold_at_recall(labels, baseline_detector, args.target_recall)
            threshold_source = "fitted once on unperturbed validation ensemble"
        else:
            threshold = args.threshold
            threshold_source = "provided on command line and fixed for all perturbations"
    elif args.threshold is not None and not np.isclose(args.threshold, threshold, rtol=0.0, atol=1e-15):
        msg = "--threshold differs from the validation threshold frozen in --manifest"
        raise ValueError(msg)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        msg = "threshold must be finite and in [0, 1]"
        raise ValueError(msg)

    baseline = score_predictions(
        baseline_predictions.joint_probabilities,
        labels,
        float(threshold),
        conditional_severity=baseline_predictions.conditional_severity,
    )
    baseline_block_rps = _block_rps(baseline_predictions.joint_probabilities, labels, args.block_rows)
    del baseline_predictions

    rows = []
    block_deltas: dict[str, list[np.ndarray]] = {name: [] for name in selected_names}
    for repeat in range(args.repeats):
        permutation_seed = args.seed + repeat
        row_permutation = within_block_permutation(len(features), args.permutation_stratum_rows, permutation_seed)
        for group in feature_groups():
            if group.name not in selected_names:
                continue
            print(f"Permutation repeat {repeat + 1}/{args.repeats}: {group.name}", flush=True)
            permuted = PermutedFeatureMatrix(features, group, row_permutation)
            predictions = predict_equal_ensemble(
                inputs.models,
                permuted,
                device,
                args.batch_size,
                label=f"repeat {repeat + 1} {group.name}",
            )
            perturbed = score_predictions(
                predictions.joint_probabilities,
                labels,
                float(threshold),
                conditional_severity=predictions.conditional_severity,
            )
            deltas = importance_deltas(baseline, perturbed)
            rows.append(
                {
                    "group": group.name,
                    "width": group.width,
                    "repeat": repeat,
                    "permutation_seed": permutation_seed,
                    "perturbed_metrics": perturbed,
                    **deltas,
                }
            )
            block_deltas[group.name].append(
                _block_rps(predictions.joint_probabilities, labels, args.block_rows) - baseline_block_rps
            )
            del predictions

    summaries = []
    metric_names = tuple(importance_deltas(baseline, baseline))
    for group in feature_groups():
        group_rows = [row for row in rows if row["group"] == group.name]
        if not group_rows:
            continue
        summaries.append(
            {
                **group.state(),
                "metrics": {name: _summary([row[name] for row in group_rows]) for name in metric_names},
                "joint_rps_whole_timestep_bootstrap_95": block_bootstrap_interval(
                    np.stack(block_deltas[group.name]),
                    args.bootstrap_repeats,
                    args.seed + 100_000 + group.start,
                ),
            }
        )

    if args.permutation_stratum_rows == DEFAULT_PERMUTATION_STRATUM_ROWS:
        stratum_definition = "one fixed model level over the 72 x 96 horizontal grid"
    elif args.permutation_stratum_rows == args.block_rows:
        stratum_definition = "one full timestep, including cross-model-level substitutions"
    else:
        stratum_definition = "contiguous flattened row strata nested within each timestep"
    report = {
        "format_version": PERMUTATION_REPORT_VERSION,
        "method": {
            "name": "grouped permutation importance",
            "permutation_scope": "independently within row strata that never cross chronological timesteps",
            "permutation_stratum_rows": args.permutation_stratum_rows,
            "permutation_strata_per_timestep": args.block_rows // args.permutation_stratum_rows,
            "stratum_definition": stratum_definition,
            "vector_group_policy": "one shared row permutation for every channel in a group",
            "paired_repeat_policy": "the same row permutation is reused across all groups within a repeat",
            "repeats": args.repeats,
            "seed": args.seed,
            "primary_importance": "increase in five-class ranked probability score",
            "negative_values_clipped": False,
            "conditioning_caveat": (
                "importance is conditional on the declared row strata and all remaining correlated inputs; "
                "features constant within a stratum cannot be assessed by this perturbation"
            ),
            "bootstrap_caveat": (
                "descriptive iid resampling of whole timesteps; temporal autocorrelation and alternative fitted "
                "model uncertainty are not represented"
            ),
            "conditional_severity_validation_class_support": {
                str(severity_class): count for severity_class, count in enumerate(class_counts(labels)[1:], start=1)
            },
            "conditional_severity_numerics": (
                "detector-weighted equal-joint ensemble conditioned from raw member logits in float64 log space; "
                "this preserves severity when deployed float32 detector probabilities underflow to zero"
            ),
        },
        **_common_report(inputs, labels, args.block_rows),
        "threshold": {
            "value": float(threshold),
            "source": threshold_source,
            "refitted_after_permutation": False,
        },
        "baseline_metrics": baseline,
        "importance": summaries,
        "raw_repeats": rows,
    }
    _write_json(output_dir / "permutation_importance.json", report)
    _write_permutation_csv(output_dir / "permutation_importance.csv", rows)
    _write_permutation_summary_csv(output_dir / "permutation_importance_summary.csv", summaries)
    _plot_permutation(
        summaries,
        output_dir / "permutation_importance.png",
        class_counts(labels)[1:],
    )
    print(f"Saved grouped permutation importance to {output_dir}", flush=True)
    return report


def _balanced_block_sample(  # noqa: PLR0912
    n_rows: int,
    sample_rows: int,
    block_rows: int,
    seed: int,
    *,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Sample without replacement while spreading rows across time blocks."""
    if n_rows < 1 or sample_rows < 1 or block_rows < 1:
        msg = "n_rows, sample_rows, and block_rows must be positive"
        raise ValueError(msg)
    if eligible is not None:
        eligible = np.asarray(eligible)
        if eligible.shape != (n_rows,) or eligible.dtype != np.bool_:
            msg = "eligible mask must contain one Boolean value per row"
            raise ValueError(msg)
    generator = np.random.default_rng(seed)
    bounds = []
    counts = []
    for start in range(0, n_rows, block_rows):
        stop = min(start + block_rows, n_rows)
        bounds.append((start, stop))
        counts.append(stop - start if eligible is None else int(eligible[start:stop].sum()))

    target = min(sample_rows, sum(counts))
    if target == 0:
        return np.empty(0, dtype=np.int64)
    quotas = np.zeros(len(bounds), dtype=np.int64)
    allocated = 0
    while allocated < target:
        progressed = False
        for block_index in generator.permutation(len(bounds)):
            if quotas[block_index] < counts[block_index]:
                quotas[block_index] += 1
                allocated += 1
                progressed = True
                if allocated == target:
                    break
        if not progressed:
            break

    chosen = []
    for (start, stop), quota in zip(bounds, quotas, strict=True):
        if not quota:
            continue
        if eligible is None:
            local = generator.choice(stop - start, size=quota, replace=False)
        else:
            candidates = np.flatnonzero(eligible[start:stop])
            local = candidates if quota == len(candidates) else generator.choice(candidates, size=quota, replace=False)
        chosen.append(start + local)
    return np.sort(np.concatenate(chosen).astype(np.int64, copy=False))


def _uniform_eligible_sample(eligible: np.ndarray, sample_rows: int, seed: int) -> np.ndarray:
    """Sample rows uniformly without replacement from one label cohort."""
    mask = np.asarray(eligible)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        msg = "eligible must be a one-dimensional Boolean mask"
        raise ValueError(msg)
    if sample_rows < 1:
        msg = "sample_rows must be positive"
        raise ValueError(msg)
    candidates = np.flatnonzero(mask)
    if not len(candidates):
        msg = "eligible mask contains no candidate rows"
        raise ValueError(msg)
    if sample_rows >= len(candidates):
        return candidates
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(candidates, size=sample_rows, replace=False))


def _selected_rows(path: Path, indices: np.ndarray) -> np.ndarray:
    """Load a bounded row sample from an NPY matrix."""
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    return np.array(source[indices], dtype=np.float32, order="C", copy=True)


def expected_gradients(
    forward_targets: Callable[[torch.Tensor], torch.Tensor],
    explained: np.ndarray,
    background: np.ndarray,
    *,
    samples: int,
    internal_batch_size: int,
    seed: int,
    device: torch.device,
) -> ExpectedGradientResult:
    """Approximate SHAP values with Expected Gradients.

    The sampled-reference residual isolates Monte Carlo integration error along
    the paths actually drawn for each row.  The fixed-background residual also
    includes reference-sampling error relative to the entire supplied
    background pool.  Drawing all references and interpolation positions before
    batching makes the estimator invariant to ``internal_batch_size``.
    """
    explained = np.asarray(explained, dtype=np.float32)
    background = np.asarray(background, dtype=np.float32)
    if explained.ndim != MATRIX_NDIM or background.ndim != MATRIX_NDIM:
        msg = "explained and background inputs must be two-dimensional"
        raise ValueError(msg)
    if explained.shape[1] != background.shape[1] or not len(explained) or not len(background):
        msg = "explained and background inputs must be non-empty with matching columns"
        raise ValueError(msg)
    if samples < 1 or internal_batch_size < samples:
        msg = "samples must be positive and internal_batch_size must be at least samples"
        raise ValueError(msg)

    generator = np.random.default_rng(seed)
    with torch.no_grad():
        probe = forward_targets(torch.from_numpy(explained[:1]).to(device))
    if probe.ndim != MATRIX_NDIM or probe.shape[0] != 1:
        msg = "forward_targets must return shape (batch, targets)"
        raise ValueError(msg)
    n_targets = probe.shape[1]
    attributions = np.zeros((len(explained), explained.shape[1], n_targets), dtype=np.float32)
    explained_outputs = np.zeros((len(explained), n_targets), dtype=np.float32)
    reference_outputs = np.zeros_like(explained_outputs)
    examples_per_batch = max(1, internal_batch_size // samples)

    with torch.no_grad():
        background_output = forward_targets(torch.from_numpy(background).to(device)).detach().cpu().numpy()
    fixed_background_output = background_output.mean(axis=0)
    reference_indices = generator.integers(0, len(background), size=(len(explained), samples))
    interpolation_positions = generator.random((len(explained), samples, 1), dtype=np.float32)

    started = time.perf_counter()
    for start in range(0, len(explained), examples_per_batch):
        stop = min(start + examples_per_batch, len(explained))
        batch = explained[start:stop]
        n_examples = len(batch)
        batch_reference_indices = reference_indices[start:stop]
        references = background[batch_reference_indices]
        differences = batch[:, None, :] - references
        alpha = interpolation_positions[start:stop]
        interpolated = references + alpha * differences
        interpolated_tensor = torch.from_numpy(interpolated.reshape(-1, explained.shape[1])).to(device)
        interpolated_tensor.requires_grad_(True)
        outputs = forward_targets(interpolated_tensor)
        for target in range(n_targets):
            gradient = torch.autograd.grad(
                outputs[:, target].sum(),
                interpolated_tensor,
                retain_graph=target + 1 < n_targets,
            )[0]
            contribution = gradient.detach().cpu().numpy().reshape(n_examples, samples, -1) * differences
            attributions[start:stop, :, target] = contribution.mean(axis=1)

        with torch.no_grad():
            explained_outputs[start:stop] = forward_targets(torch.from_numpy(batch).to(device)).detach().cpu().numpy()
        reference_outputs[start:stop] = background_output[batch_reference_indices].mean(axis=1)
        print(
            f"  expected gradients: {stop:,}/{len(explained):,} rows, {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    summed_attributions = attributions.sum(axis=1)
    sampled_reference_residuals = summed_attributions - (explained_outputs - reference_outputs)
    fixed_background_residuals = summed_attributions - (explained_outputs - fixed_background_output)
    return ExpectedGradientResult(
        attributions=attributions,
        sampled_reference_residuals=sampled_reference_residuals,
        fixed_background_residuals=fixed_background_residuals,
        explained_outputs=explained_outputs,
        sampled_reference_outputs=reference_outputs,
        fixed_background_output=fixed_background_output,
    )


def _ensemble_target_function(models: Sequence[Student]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return differentiable detector and conditional-severity outputs."""
    if not models:
        msg = "At least one model is required"
        raise ValueError(msg)
    first_parameter = next(models[0].parameters())
    positive_classes = torch.arange(1, N_CLASSES, dtype=torch.float64, device=first_parameter.device)

    def forward(inputs: torch.Tensor) -> torch.Tensor:
        probabilities, conditional = _ensemble_joint_and_conditional(models, inputs)
        detector = probabilities[:, 1:].sum(dim=1).to(torch.float64)
        conditional_expected_halvings = conditional @ positive_classes
        return torch.stack((detector, conditional_expected_halvings), dim=1)

    return forward


def _ensemble_conditional_severity_function(models: Sequence[Student]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return conditional expected halvings from the mean joint distribution."""
    targets = _ensemble_target_function(models)

    def forward(inputs: torch.Tensor) -> torch.Tensor:
        return targets(inputs)[:, 1:2]

    return forward


def _cohort_statistics(
    attributions: np.ndarray,
    cohort_mask: np.ndarray,
    *,
    target: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Aggregate one target into scalar-column and physical-group statistics."""
    values = attributions[cohort_mask, :, target]
    if not len(values):
        msg = "Attribution cohort is empty"
        raise ValueError(msg)
    mean_abs = np.mean(np.abs(values), axis=0)
    mean_signed = np.mean(values, axis=0)
    total = float(mean_abs.sum())
    group_rows = []
    for group in feature_groups():
        group_values = values[:, group.start : group.stop]
        group_total = float(mean_abs[group.start : group.stop].sum())
        group_rows.append(
            {
                **group.state(),
                "total_mean_absolute_attribution": group_total,
                "normalized_total_absolute_share": safe_divide(group_total, total),
                "mean_absolute_attribution_per_channel": group_total / group.width,
                "mean_absolute_net_group_attribution": float(np.mean(np.abs(group_values.sum(axis=1)))),
                "mean_signed_net_group_attribution": float(np.mean(group_values.sum(axis=1))),
            }
        )
    return mean_abs, mean_signed, group_rows


def _write_expected_gradient_csvs(
    output_dir: Path,
    feature_rows: Sequence[dict],
    group_rows: Sequence[dict],
) -> None:
    """Write scalar-column and physical-group attribution tables."""
    with (output_dir / "expected_gradients_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    with (output_dir / "expected_gradients_groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)


def _plot_expected_gradients(group_rows: Sequence[dict], path: Path) -> None:
    """Plot detector cohorts and positive-only conditional-severity attribution."""
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    panels = (
        ("detector_probability", "positive", "Detection: positive rows"),
        ("detector_probability", "natural_validation", "Detection: natural-prevalence estimate"),
        (
            "conditional_expected_halvings",
            "positive",
            "Conditional severity: positive rows only",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    indexed_panels = []
    for target, cohort, _ in panels:
        indexed_panels.append(
            {row["group"]: row for row in group_rows if row["target"] == target and row["cohort"] == cohort}
        )
    order = sorted(
        indexed_panels[0],
        key=lambda name: max(panel[name]["mean_absolute_attribution_per_channel"] for panel in indexed_panels),
    )
    positions = np.arange(len(order))
    for axis, (_, _, title), indexed in zip(axes, panels, indexed_panels, strict=True):
        axis.barh(
            positions,
            [indexed[name]["mean_absolute_attribution_per_channel"] for name in order],
        )
        axis.set_yticks(positions, labels=order)
        axis.set_xlabel("Mean |attribution| per scalar channel")
        axis.set_title(title)
    figure.suptitle("Expected Gradients attribution (group-width-normalized view)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_expected_gradient_features(feature_rows: Sequence[dict], path: Path, *, top_n: int = 20) -> None:
    """Plot the most relevant scalar channels for detection and severity."""
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    panels = (
        ("detector_probability", "natural_validation", "Detection: natural-prevalence estimate"),
        ("conditional_expected_halvings", "positive", "Conditional severity: positive rows"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    for axis, (target, cohort, title) in zip(axes, panels, strict=True):
        selected = [row for row in feature_rows if row["target"] == target and row["cohort"] == cohort]
        selected = sorted(selected, key=lambda row: row["mean_absolute_attribution"], reverse=True)[:top_n]
        selected.reverse()
        axis.barh(
            [row["feature"] for row in selected],
            [row["mean_absolute_attribution"] for row in selected],
        )
        axis.set_xlabel("Mean |attribution|")
        axis.set_title(title)
    figure.suptitle(f"Top {top_n} scalar features by Expected Gradients")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_expected_gradients(args: argparse.Namespace) -> dict:  # noqa: PLR0912, PLR0915
    """Run Expected Gradients for detection and positive-conditional severity."""
    for name in ("background_rows", "positive_rows", "negative_rows", "samples", "internal_batch_size", "block_rows"):
        if getattr(args, name) < 1:
            msg = f"{name.replace('_', '-')} must be positive"
            raise ValueError(msg)
    if args.internal_batch_size < args.samples:
        msg = "internal-batch-size must be at least samples"
        raise ValueError(msg)
    output_dir = _prepare_output_dir(
        args.output_dir,
        (
            "expected_gradients.json",
            "expected_gradients_features.csv",
            "expected_gradients_groups.csv",
            "expected_gradients_groups.png",
            "expected_gradients_features.png",
            "expected_gradients_attributions.npz",
        ),
    )
    device = torch.device(args.device)
    inputs = load_analysis_inputs(
        args.checkpoint or (),
        args.cache_dir,
        manifest_path=args.manifest,
        device=device,
    )
    for model in inputs.models:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    train_x_path = inputs.cache_dir / "train_x.npy"
    train_y_path = inputs.cache_dir / "train_y.npy"
    val_x_path = inputs.cache_dir / "val_x.npy"
    train_x = np.load(train_x_path, mmap_mode="r", allow_pickle=False)
    train_y = np.load(train_y_path, mmap_mode="r", allow_pickle=False)
    val_x = np.load(val_x_path, mmap_mode="r", allow_pickle=False)
    val_y = np.load(inputs.cache_dir / "val_y.npy", mmap_mode="r", allow_pickle=False)
    expected_names = feature_names(FEATURES)
    if (
        train_x.ndim != MATRIX_NDIM
        or train_x.shape[1] != len(expected_names)
        or train_x.dtype != np.float32
        or inputs.metadata.get("train", {}).get("shape") != list(train_x.shape)
        or train_y.shape != (len(train_x),)
        or not np.issubdtype(train_y.dtype, np.integer)
        or inputs.metadata.get("train", {}).get("class_counts") != class_counts(train_y)
    ):
        msg = "Training feature cache does not match metadata/model schema"
        raise ValueError(msg)
    if val_x.shape[1] != len(expected_names) or val_x.dtype != np.float32:
        msg = "Validation feature cache does not match the model schema"
        raise ValueError(msg)
    if len(train_x) % args.block_rows or len(val_x) % args.block_rows:
        msg = "Train/validation rows are not exact multiples of the declared timestep block"
        raise ValueError(msg)
    training_artifacts = {
        "train_x.npy": _artifact(train_x_path, hash_contents=True),
        "train_y.npy": _artifact(train_y_path, hash_contents=True),
    }

    background_indices = _balanced_block_sample(
        len(train_x),
        args.background_rows,
        args.block_rows,
        args.seed,
    )
    positive_indices = _uniform_eligible_sample(
        np.asarray(val_y > 0),
        args.positive_rows,
        args.seed + 1,
    )
    negative_indices = _uniform_eligible_sample(
        np.asarray(val_y == 0),
        args.negative_rows,
        args.seed + 2,
    )
    explained_indices = np.concatenate((positive_indices, negative_indices))
    cohort = np.concatenate(
        (
            np.full(len(positive_indices), "positive", dtype="U8"),
            np.full(len(negative_indices), "negative", dtype="U8"),
        )
    )
    background = _selected_rows(train_x_path, background_indices)
    explained = _selected_rows(val_x_path, explained_indices)
    if not np.isfinite(background).all() or not np.isfinite(explained).all():
        msg = "Expected Gradients samples contain non-finite model inputs"
        raise ValueError(msg)
    ensemble_targets = _ensemble_target_function(inputs.models)
    detector_result = expected_gradients(
        lambda values: ensemble_targets(values)[:, 0:1],
        explained,
        background,
        samples=args.samples,
        internal_batch_size=args.internal_batch_size,
        seed=args.seed + 3,
        device=device,
    )
    positive_background_indices = _balanced_block_sample(
        len(train_x),
        args.background_rows,
        args.block_rows,
        args.seed + 4,
        eligible=np.asarray(train_y > 0),
    )
    if not len(positive_background_indices):
        msg = "Conditional-severity Expected Gradients requires positive training-background rows"
        raise ValueError(msg)
    positive_background = _selected_rows(train_x_path, positive_background_indices)
    positive_explained = explained[: len(positive_indices)]
    severity_result = expected_gradients(
        _ensemble_conditional_severity_function(inputs.models),
        positive_explained,
        positive_background,
        samples=args.samples,
        internal_batch_size=args.internal_batch_size,
        seed=args.seed + 5,
        device=device,
    )
    result_arrays = (
        detector_result.attributions,
        detector_result.sampled_reference_residuals,
        detector_result.fixed_background_residuals,
        detector_result.explained_outputs,
        detector_result.sampled_reference_outputs,
        detector_result.fixed_background_output,
        severity_result.attributions,
        severity_result.sampled_reference_residuals,
        severity_result.fixed_background_residuals,
        severity_result.explained_outputs,
        severity_result.sampled_reference_outputs,
        severity_result.fixed_background_output,
    )
    if not all(np.isfinite(values).all() for values in result_arrays):
        msg = "Expected Gradients produced non-finite outputs or attributions"
        raise RuntimeError(msg)

    names = feature_names(FEATURES)
    parents = [group.name for group in feature_groups() for _ in range(group.width)]
    feature_rows = []
    group_rows = []
    detector_masks = {
        "positive": cohort == "positive",
        "negative": cohort == "negative",
    }
    prevalence = float(np.mean(val_y > 0))
    detector_stats = {
        cohort_name: _cohort_statistics(detector_result.attributions, mask, target=0)
        for cohort_name, mask in detector_masks.items()
    }
    positive_abs, positive_signed, positive_groups = detector_stats["positive"]
    negative_abs, negative_signed, negative_groups = detector_stats["negative"]
    natural_abs = prevalence * positive_abs + (1.0 - prevalence) * negative_abs
    natural_signed = prevalence * positive_signed + (1.0 - prevalence) * negative_signed

    severity_abs, severity_signed, severity_groups = _cohort_statistics(
        severity_result.attributions,
        np.ones(len(positive_indices), dtype=bool),
        target=0,
    )
    target_cohorts = (
        ("detector_probability", "positive", positive_abs, positive_signed),
        ("detector_probability", "negative", negative_abs, negative_signed),
        ("detector_probability", "natural_validation", natural_abs, natural_signed),
        ("conditional_expected_halvings", "positive", severity_abs, severity_signed),
    )
    for target_name, cohort_name, mean_abs, mean_signed in target_cohorts:
        total = float(mean_abs.sum())
        for index, name in enumerate(names):
            feature_rows.append(
                {
                    "target": target_name,
                    "cohort": cohort_name,
                    "feature": name,
                    "group": parents[index],
                    "column": index,
                    "mean_absolute_attribution": float(mean_abs[index]),
                    "mean_signed_attribution": float(mean_signed[index]),
                    "normalized_absolute_share": safe_divide(float(mean_abs[index]), total),
                }
            )

    for target_name, cohort_name, rows in (
        ("detector_probability", "positive", positive_groups),
        ("detector_probability", "negative", negative_groups),
        ("conditional_expected_halvings", "positive", severity_groups),
    ):
        for row in rows:
            group_rows.append({"target": target_name, "cohort": cohort_name, "group": row["name"], **row})
    for group in feature_groups():
        group_total = float(natural_abs[group.start : group.stop].sum())
        positive_group = next(row for row in positive_groups if row["name"] == group.name)
        negative_group = next(row for row in negative_groups if row["name"] == group.name)
        group_rows.append(
            {
                "target": "detector_probability",
                "cohort": "natural_validation",
                "group": group.name,
                **group.state(),
                "total_mean_absolute_attribution": group_total,
                "normalized_total_absolute_share": safe_divide(group_total, float(natural_abs.sum())),
                "mean_absolute_attribution_per_channel": group_total / group.width,
                "mean_absolute_net_group_attribution": prevalence
                * positive_group["mean_absolute_net_group_attribution"]
                + (1.0 - prevalence) * negative_group["mean_absolute_net_group_attribution"],
                "mean_signed_net_group_attribution": prevalence * positive_group["mean_signed_net_group_attribution"]
                + (1.0 - prevalence) * negative_group["mean_signed_net_group_attribution"],
            }
        )

    np.savez_compressed(
        output_dir / "expected_gradients_attributions.npz",
        detector_attributions=detector_result.attributions,
        detector_sampled_reference_completeness_residuals=detector_result.sampled_reference_residuals,
        detector_fixed_background_completeness_residuals=detector_result.fixed_background_residuals,
        detector_explained_outputs=detector_result.explained_outputs,
        detector_sampled_reference_outputs=detector_result.sampled_reference_outputs,
        detector_fixed_background_output=detector_result.fixed_background_output,
        detector_explained_indices=explained_indices,
        detector_background_indices=background_indices,
        detector_cohorts=cohort,
        conditional_severity_attributions=severity_result.attributions,
        conditional_severity_sampled_reference_completeness_residuals=(severity_result.sampled_reference_residuals),
        conditional_severity_fixed_background_completeness_residuals=severity_result.fixed_background_residuals,
        conditional_severity_explained_outputs=severity_result.explained_outputs,
        conditional_severity_sampled_reference_outputs=severity_result.sampled_reference_outputs,
        conditional_severity_fixed_background_output=severity_result.fixed_background_output,
        conditional_severity_explained_indices=positive_indices,
        conditional_severity_background_indices=positive_background_indices,
        feature_names=np.asarray(names),
        target_names=np.asarray(("detector_probability", "conditional_expected_halvings")),
    )
    _write_expected_gradient_csvs(output_dir, feature_rows, group_rows)
    _plot_expected_gradients(group_rows, output_dir / "expected_gradients_groups.png")
    _plot_expected_gradient_features(feature_rows, output_dir / "expected_gradients_features.png")

    report = {
        "format_version": EXPECTED_GRADIENTS_REPORT_VERSION,
        "method": {
            "name": "Expected Gradients",
            "description": "Monte Carlo expected gradients; a SHAP-value approximation for differentiable models",
            "background_rows": {
                "detector": len(background_indices),
                "conditional_severity": len(positive_background_indices),
            },
            "samples_per_explained_row": args.samples,
            "analysis_seed": args.seed,
            "monte_carlo_seeds": {
                "detector": args.seed + 3,
                "conditional_severity": args.seed + 5,
            },
            "internal_batch_size": args.internal_batch_size,
            "targets": ["detector_probability", "conditional_expected_halvings"],
            "target_definitions": {
                "detector_probability": "P(class > 0) from the mean full joint checkpoint distribution",
                "conditional_expected_halvings": (
                    "E[class | class > 0] over classes 1..4 after averaging full joint checkpoint distributions"
                ),
            },
            "conditional_severity_numerics": (
                "detector-weighted equal-joint ensemble conditioned from raw member logits in float64 log space "
                "to remain finite under detector-probability underflow"
            ),
            "model_mode": "evaluation; parameter gradients disabled; input gradients enabled",
            "interpolation_caveat": (
                "straight-line paths in preprocessed input space may traverse off-manifold combinations and "
                "intermediate stratflag values; use grouped permutation as the headline reliance result"
            ),
            "aggregation_views": {
                "tables": "total group attribution share, per-channel mean absolute attribution, and net group sums",
                "plot": (
                    "mean absolute attribution per scalar channel to avoid a mechanical advantage for wider groups; "
                    "collective wide-group influence remains available in the tables"
                ),
                "conditional_severity": (
                    "reported only for positive validation rows with a positive training-row background because "
                    "the severity head is trained conditionally on positive labels"
                ),
            },
            "signed_attribution_interpretation": (
                "the sign is the direction of output change relative to sampled training-background paths in "
                "preprocessed input space; it is not a causal effect or a raw-unit slope"
            ),
        },
        **_common_report(inputs, val_y, args.block_rows),
        "sampling": {
            "detector_background": "unlabelled training rows sampled equally across chronological timesteps",
            "conditional_severity_background": (
                "positive-label training rows sampled as evenly as possible across chronological timesteps"
            ),
            "training_source_artifacts": training_artifacts,
            "explained": "class-stratified simple random sampling without replacement from validation",
            "available_positive_rows": int(np.sum(val_y > 0)),
            "available_negative_rows": int(np.sum(val_y == 0)),
            "requested_positive_rows": args.positive_rows,
            "requested_negative_rows": args.negative_rows,
            "explained_positive_rows": len(positive_indices),
            "explained_negative_rows": len(negative_indices),
            "detector_positive_and_negative_panels_reported_separately": True,
            "conditional_severity_explained_on_positive_rows_only": True,
            "conditional_severity_training_background_rows": len(positive_background_indices),
            "conditional_severity_validation_class_support": {
                str(severity_class): count for severity_class, count in enumerate(class_counts(val_y)[1:], start=1)
            },
            "conditional_severity_support_caveat": (
                "the two most severe classes have extremely limited validation support; interpret their feature "
                "attributions descriptively rather than as stable class-specific conclusions"
            ),
            "natural_validation_aggregation_positive_weight": prevalence,
            "natural_validation_aggregation_negative_weight": 1.0 - prevalence,
        },
        "completeness": {
            "sampled_reference": {
                "interpretation": "path-integration Monte Carlo error conditional on each row's sampled references",
                "targets": {
                    "detector_probability": {
                        "mean_absolute_residual": float(
                            np.mean(np.abs(detector_result.sampled_reference_residuals[:, 0]))
                        ),
                        "maximum_absolute_residual": float(
                            np.max(np.abs(detector_result.sampled_reference_residuals[:, 0]))
                        ),
                    },
                    "conditional_expected_halvings": {
                        "mean_absolute_residual": float(
                            np.mean(np.abs(severity_result.sampled_reference_residuals[:, 0]))
                        ),
                        "maximum_absolute_residual": float(
                            np.max(np.abs(severity_result.sampled_reference_residuals[:, 0]))
                        ),
                    },
                },
            },
            "fixed_background_expectation": {
                "interpretation": "combined path and reference-sampling error against the complete background pool",
                "targets": {
                    "detector_probability": {
                        "mean_absolute_residual": float(
                            np.mean(np.abs(detector_result.fixed_background_residuals[:, 0]))
                        ),
                        "maximum_absolute_residual": float(
                            np.max(np.abs(detector_result.fixed_background_residuals[:, 0]))
                        ),
                    },
                    "conditional_expected_halvings": {
                        "mean_absolute_residual": float(
                            np.mean(np.abs(severity_result.fixed_background_residuals[:, 0]))
                        ),
                        "maximum_absolute_residual": float(
                            np.max(np.abs(severity_result.fixed_background_residuals[:, 0]))
                        ),
                    },
                },
            },
        },
        "group_importance": group_rows,
        "feature_importance": feature_rows,
        "raw_attributions": str((output_dir / "expected_gradients_attributions.npz").resolve()),
    }
    _write_json(output_dir / "expected_gradients.json", report)
    print(f"Saved Expected Gradients analysis to {output_dir}", flush=True)
    return report


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    """Add model/cache arguments shared by both subcommands."""
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, action="append", help="Canonical checkpoint; repeat for an ensemble")
    source.add_argument("--manifest", type=Path, help="Frozen ensemble manifest and SHA-256 sidecar")
    parser.add_argument("--cache-dir", type=Path, help="Checkpoint-matched cache leaf; inferred from --manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--block-rows", type=int, default=int(np.prod(GRID_SHAPE)))


def parse_args() -> argparse.Namespace:
    """Parse grouped-permutation or Expected Gradients arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    permutation = subparsers.add_parser("permutation", help="Grouped validation permutation importance")
    _add_common_inputs(permutation)
    permutation.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    permutation.add_argument("--groups", nargs="+", help="Optional subset of physical groups")
    permutation.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    permutation.add_argument("--seed", type=int, default=0)
    permutation.add_argument("--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS)
    permutation.add_argument(
        "--permutation-stratum-rows",
        type=int,
        default=DEFAULT_PERMUTATION_STRATUM_ROWS,
        help="Rows shuffled together; default is one 72x96 horizontal grid at a fixed timestep/model level",
    )
    permutation.add_argument("--threshold", type=float, help="Fixed detector threshold; otherwise fit once on baseline")
    permutation.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)

    gradients = subparsers.add_parser("expected-gradients", help="Sampled SHAP-style gradient attributions")
    _add_common_inputs(gradients)
    gradients.add_argument("--background-rows", type=int, default=DEFAULT_BACKGROUND_ROWS)
    gradients.add_argument("--positive-rows", type=int, default=DEFAULT_POSITIVE_ROWS)
    gradients.add_argument("--negative-rows", type=int, default=DEFAULT_NEGATIVE_ROWS)
    gradients.add_argument("--samples", type=int, default=DEFAULT_EXPECTED_GRADIENT_SAMPLES)
    gradients.add_argument("--internal-batch-size", type=int, default=DEFAULT_GRADIENT_BATCH_SIZE)
    gradients.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> dict:
    """Run the selected post-hoc importance analysis."""
    args = parse_args() if args is None else args
    if args.command == "permutation":
        return run_permutation(args)
    if args.command == "expected-gradients":
        return run_expected_gradients(args)
    msg = f"Unsupported feature-importance command: {args.command!r}"
    raise ValueError(msg)


if __name__ == "__main__":
    main()
