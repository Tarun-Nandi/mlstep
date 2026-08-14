"""Offline comparison plots for student ensembles and cached XGBoost outputs.

This module never trains a model and never reads the held-out test period.  The
``evaluate`` command runs existing canonical checkpoints over a prepared
validation cache, saves compact prediction bundles, and renders the same plots
for supervised, distilled, and cached-teacher predictions.  The ``plot``
command recreates the figures from those bundles without rerunning inference.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from mlstep.data import FEATURES, GRID_SHAPE, N_CLASSES, class_counts, feature_names
from mlstep.evaluation import evaluate, halving_action_metrics, threshold_at_recall
from mlstep.student import fp_recall_diagnostics, predict_joint_probabilities, student_from_checkpoint

BUNDLE_FORMAT_VERSION: Final = "mlstep-prediction-bundle-v1"
REPORT_FORMAT_VERSION: Final = "mlstep-offline-comparison-v1"
DEFAULT_TARGET_RECALL: Final = 0.97
DEFAULT_BATCH_SIZE: Final = 65_536
DEFAULT_EXAMPLE_COUNT: Final = 3
MAX_PR_POINTS: Final = 2_000
MATRIX_NDIM: Final = 2
GRID_NDIM: Final = 3
MIN_STUDENT_GROUP_VALUES: Final = 2
SHA256_CHUNK_BYTES: Final = 8 * 2**20
CONFUSION_TEXT_THRESHOLD: Final = 0.5
OPERATING_PRECISION_PADDING: Final = 0.05
ACTION_SUMMARY_LABEL_MINIMUM: Final = 0.08
CLASS_LABELS: Final = ("0", "1", "2", "3", "4")
CALIBRATION_PROBABILITY_EDGES: Final = (
    0.0,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    0.2,
    0.4,
    0.6,
    0.8,
    0.9,
    0.95,
    0.99,
    1.0,
)
ERROR_CATEGORY_LABELS: Final = (
    "Background",
    "Exact positive column",
    "Same column, different output",
    "Missed positive column",
    "Spurious positive column",
)


@dataclass(frozen=True, slots=True)
class StudentGroup:
    """One named equal-probability ensemble of canonical checkpoints."""

    name: str
    checkpoint_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PredictionBundle:
    """Compact arrays and headline metrics needed to recreate every plot."""

    name: str
    kind: str
    threshold: float
    target_recall: float
    threshold_source: str
    timesteps: tuple[int, ...]
    grid_shape: tuple[int, int, int]
    labels: np.ndarray
    actions: np.ndarray
    detector_scores: np.ndarray
    metrics: dict[str, float | int]
    fp_recall: dict[str, float | int]


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    msg = f"Object of type {type(value).__name__} is not JSON serializable"
    raise TypeError(msg)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(SHA256_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, *, hash_contents: bool = False) -> dict[str, str | int]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        msg = f"Expected a file: {resolved}"
        raise ValueError(msg)
    result: dict[str, str | int] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
    }
    if hash_contents:
        result["sha256"] = _sha256(resolved)
    return result


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        msg = "Model names must contain at least one letter or number"
        raise ValueError(msg)
    return slug


def _parse_timestep_range(value: str) -> tuple[int, ...]:
    match = re.fullmatch(r"([0-9]+):([0-9]+)", value)
    if match is None:
        msg = "--timesteps must use inclusive START:STOP syntax, for example 119:143"
        raise argparse.ArgumentTypeError(msg)
    start, stop = (int(part) for part in match.groups())
    if stop < start:
        msg = "--timesteps STOP must be greater than or equal to START"
        raise argparse.ArgumentTypeError(msg)
    return tuple(range(start, stop + 1))


def _parse_student_groups(values: Sequence[Sequence[str]] | None) -> tuple[StudentGroup, ...]:
    groups = []
    names = set()
    for raw_group in values or ():
        if len(raw_group) < MIN_STUDENT_GROUP_VALUES:
            msg = "Each --student requires NAME followed by at least one checkpoint"
            raise ValueError(msg)
        name, *raw_paths = raw_group
        if name in names:
            msg = f"Duplicate model name: {name}"
            raise ValueError(msg)
        names.add(name)
        groups.append(StudentGroup(name=name, checkpoint_paths=tuple(Path(path) for path in raw_paths)))
    return tuple(groups)


def _load_validation_cache(cache_dir: Path) -> tuple[Path, dict, np.ndarray, np.ndarray]:
    resolved = cache_dir.expanduser().resolve(strict=True)
    metadata_path = resolved / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        msg = "Cache metadata must contain a JSON object"
        raise ValueError(msg)

    expected_names = feature_names(FEATURES)
    if metadata.get("features") != expected_names or metadata.get("n_features") != len(expected_names):
        msg = "Validation cache feature names/order do not match the current schema"
        raise ValueError(msg)
    preprocessing = metadata.get("preprocessing")
    if (
        not isinstance(preprocessing, dict)
        or preprocessing.get("method") != metadata.get("preprocess")
        or preprocessing.get("fit_split") != "chronological training timesteps only"
        or preprocessing.get("output_dtype") != "float32"
    ):
        msg = "Cache does not record supported training-only float32 preprocessing"
        raise ValueError(msg)

    features = np.load(resolved / "val_x.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(resolved / "val_y.npy", mmap_mode="r", allow_pickle=False)
    if features.ndim != MATRIX_NDIM or not len(features) or features.shape[1] != len(expected_names):
        msg = "Validation feature matrix has an unexpected shape"
        raise ValueError(msg)
    if features.dtype != np.float32:
        msg = f"Validation features must be float32, found {features.dtype}"
        raise ValueError(msg)
    if labels.shape != (len(features),) or not np.issubdtype(labels.dtype, np.integer):
        msg = "Validation labels must be an integer vector matching the feature rows"
        raise ValueError(msg)
    if labels.min() < 0 or labels.max() >= N_CLASSES:
        msg = f"Validation labels must lie in [0, {N_CLASSES - 1}]"
        raise ValueError(msg)
    validation = metadata.get("val", {})
    if validation.get("shape") != list(features.shape) or validation.get("class_counts") != class_counts(labels):
        msg = "Validation cache arrays do not match their metadata"
        raise ValueError(msg)
    return resolved, metadata, features, labels


def _validate_layout(n_rows: int, timesteps: Sequence[int], grid_shape: Sequence[int]) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in grid_shape)
    if len(shape) != GRID_NDIM or any(value < 1 for value in shape):
        msg = "grid_shape must contain three positive dimensions (z, y, x)"
        raise ValueError(msg)
    if not timesteps or len(set(timesteps)) != len(timesteps):
        msg = "timesteps must be a non-empty sequence of unique values"
        raise ValueError(msg)
    expected_rows = len(timesteps) * math.prod(shape)
    if n_rows != expected_rows:
        msg = f"{n_rows:,} rows do not match {len(timesteps)} timesteps on grid {shape} ({expected_rows:,})"
        raise ValueError(msg)
    return shape[0], shape[1], shape[2]


def _checkpoint_comparison_config(checkpoint: dict) -> dict:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        msg = "Student checkpoint has no configuration dictionary"
        raise ValueError(msg)
    keys = (
        "architecture",
        "architecture_version",
        "n_features",
        "hidden",
        "k",
        "dropout",
        "preprocess",
        "ple_config",
    )
    return {key: config.get(key) for key in keys}


def _load_checkpoint_group(
    group: StudentGroup,
    metadata: dict,
    n_features: int,
) -> tuple[list[dict], list[dict]]:
    checkpoints = []
    artifacts = []
    expected_config = None
    expected_objective = None
    seen_paths = set()
    for requested_path in group.checkpoint_paths:
        path = requested_path.expanduser().resolve(strict=True)
        if path in seen_paths:
            msg = f"Duplicate checkpoint in {group.name}: {path}"
            raise ValueError(msg)
        seen_paths.add(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_role") != "canonical":
            msg = f"Only canonical student checkpoints may be plotted: {path}"
            raise ValueError(msg)
        config = _checkpoint_comparison_config(checkpoint)
        if config.get("n_features") != n_features or config.get("preprocess") != metadata.get("preprocess"):
            msg = f"Checkpoint does not match the validation cache: {path}"
            raise ValueError(msg)
        objective = checkpoint.get("objective")
        if not isinstance(objective, dict):
            msg = f"Checkpoint has no objective metadata: {path}"
            raise ValueError(msg)
        if expected_config is None:
            expected_config = config
            expected_objective = objective
        elif config != expected_config or objective != expected_objective:
            msg = f"All checkpoints in {group.name} must share one architecture and objective"
            raise ValueError(msg)
        checkpoints.append(checkpoint)
        artifacts.append(_artifact(path, hash_contents=True))
    return checkpoints, artifacts


def _predict_student_ensemble(
    checkpoints: Sequence[dict],
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Average full probabilities and detector scores across checkpoints."""
    if not checkpoints:
        msg = "At least one checkpoint is required"
        raise ValueError(msg)
    probability_sum = None
    detector_sum = None
    for index, checkpoint in enumerate(checkpoints, start=1):
        print(f"{label}: checkpoint {index}/{len(checkpoints)}", flush=True)
        model = student_from_checkpoint(checkpoint).to(device)
        probabilities, detector = predict_joint_probabilities(model, features, device, batch_size)
        if probability_sum is None:
            probability_sum = probabilities.astype(np.float64)
            detector_sum = detector.astype(np.float64)
        else:
            probability_sum += probabilities
            detector_sum += detector  # type: ignore[operator]
        del model, probabilities, detector
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if probability_sum is None or detector_sum is None:
        msg = "Student ensemble inference produced no outputs"
        raise RuntimeError(msg)
    probability_sum /= len(checkpoints)
    detector_sum /= len(checkpoints)
    return probability_sum, detector_sum


def _cached_teacher_predictions(cache_dir: Path, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    margin = np.load(cache_dir / "val_teacher_margin.npy", mmap_mode="r", allow_pickle=False)
    severity = np.load(cache_dir / "val_teacher_severity.npy", mmap_mode="r", allow_pickle=False)
    if margin.shape != (n_rows,) or severity.shape != (n_rows, N_CLASSES - 1):
        msg = "Cached XGBoost teacher arrays do not match validation rows/classes"
        raise ValueError(msg)
    if not np.isfinite(margin).all() or not np.isfinite(severity).all():
        msg = "Cached XGBoost teacher arrays must contain only finite values"
        raise ValueError(msg)
    if np.any(severity < 0) or not np.allclose(severity.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
        msg = "Cached XGBoost severity rows must be probability distributions"
        raise ValueError(msg)

    detector = np.empty(margin.shape, dtype=np.result_type(margin.dtype, np.float32))
    nonnegative = margin >= 0
    detector[nonnegative] = 1.0 / (1.0 + np.exp(-margin[nonnegative]))
    exponential = np.exp(margin[~nonnegative])
    detector[~nonnegative] = exponential / (1.0 + exponential)
    probabilities = np.empty((n_rows, N_CLASSES), dtype=np.result_type(severity.dtype, np.float32))
    probabilities[:, 0] = 1.0 - detector
    probabilities[:, 1:] = detector[:, None] * severity
    return probabilities, detector


def _hard_actions(probabilities: np.ndarray, detector: np.ndarray, threshold: float) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    detector = np.asarray(detector)
    if probabilities.shape != (len(detector), N_CLASSES):
        msg = f"probabilities must have shape (rows, {N_CLASSES})"
        raise ValueError(msg)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        msg = "threshold must be finite and in [0, 1]"
        raise ValueError(msg)
    severity = probabilities[:, 1:].argmax(axis=1).astype(np.uint8) + 1
    return np.where(detector >= threshold, severity, 0).astype(np.uint8)


def _headline_metrics(metrics: dict, actions: dict) -> dict[str, float | int]:
    probability = metrics["probability"]
    return {
        "average_precision": float(metrics["ap"]),
        "ranked_probability_score": float(probability["ranked_probability_score"]),
        "negative_log_likelihood": float(probability["negative_log_likelihood"]),
        "brier_score": float(probability["brier_score"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "true_positive": int(metrics["true_positive"]),
        "false_positive": int(metrics["false_positive"]),
        "false_negative": int(metrics["false_negative"]),
        "true_negative": int(metrics["true_negative"]),
        "exact_on_positive": float(actions["exact_on_positive"]),
        "mae_on_positive": float(actions["mae_on_positive"]),
    }


def _per_timestep(
    labels: np.ndarray,
    actions: np.ndarray,
    timesteps: Sequence[int],
    grid_shape: Sequence[int],
) -> list[dict]:
    truth = labels.reshape(len(timesteps), *grid_shape)
    predicted = actions.reshape(len(timesteps), *grid_shape)
    rows = []
    for index, timestep in enumerate(timesteps):
        actual_positive = truth[index] > 0
        predicted_positive = predicted[index] > 0
        rows.append(
            {
                "timestep": int(timestep),
                "actual_positive": int(actual_positive.sum()),
                "predicted_positive": int(predicted_positive.sum()),
                "true_positive": int(np.count_nonzero(actual_positive & predicted_positive)),
                "false_positive": int(np.count_nonzero(~actual_positive & predicted_positive)),
                "false_negative": int(np.count_nonzero(actual_positive & ~predicted_positive)),
            }
        )
    return rows


def _minimum_vertical_distances(source_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    distances = []
    for timestep, level, y_index, x_index in np.argwhere(source_mask):
        target_levels = np.flatnonzero(target_mask[timestep, :, y_index, x_index])
        if len(target_levels):
            distances.append(int(np.min(np.abs(target_levels - level))))
    return np.asarray(distances, dtype=np.int16)


def column_colocation(
    labels: np.ndarray,
    actions: np.ndarray,
    timesteps: Sequence[int],
    grid_shape: Sequence[int],
) -> dict:
    """Split strict grid-box errors by whether the opposite event shares a column."""
    _validate_layout(len(labels), timesteps, grid_shape)
    truth = labels.reshape(len(timesteps), *grid_shape)
    predicted = actions.reshape(len(timesteps), *grid_shape)
    actual_positive = truth > 0
    predicted_positive = predicted > 0
    actual_column = actual_positive.any(axis=1)
    predicted_column = predicted_positive.any(axis=1)

    false_positive = ~actual_positive & predicted_positive
    false_negative = actual_positive & ~predicted_positive
    fp_same_column = false_positive & actual_column[:, None, :, :]
    fn_same_column = false_negative & predicted_column[:, None, :, :]
    fp_distances = _minimum_vertical_distances(fp_same_column, actual_positive)
    fn_distances = _minimum_vertical_distances(fn_same_column, predicted_positive)

    def distance_summary(values: np.ndarray) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "mean_levels": float(values.mean()) if len(values) else None,
            "median_levels": float(np.median(values)) if len(values) else None,
            "maximum_levels": int(values.max()) if len(values) else None,
        }

    return {
        "strict_grid_box": {
            "true_positive": int(np.count_nonzero(actual_positive & predicted_positive)),
            "false_positive": int(false_positive.sum()),
            "false_negative": int(false_negative.sum()),
        },
        "false_positive_breakdown": {
            "same_horizontal_column_different_level": int(fp_same_column.sum()),
            "no_actual_positive_in_column": int((false_positive & ~actual_column[:, None, :, :]).sum()),
            "same_column_vertical_distance": distance_summary(fp_distances),
        },
        "false_negative_breakdown": {
            "another_prediction_in_same_horizontal_column": int(fn_same_column.sum()),
            "no_prediction_in_column": int((false_negative & ~predicted_column[:, None, :, :]).sum()),
            "same_column_vertical_distance": distance_summary(fn_distances),
        },
        "interpretation": (
            "Same-column errors remain strict grid-box errors. This descriptive split does not treat them as "
            "correct or lower-cost without an agreed UKCA action scope."
        ),
    }


def _bundle_metrics(evaluation: dict, action_metrics: dict) -> dict[str, float | int]:
    return _headline_metrics(evaluation, action_metrics)


def _save_bundle(path: Path, bundle: PredictionBundle) -> None:
    metric_arrays = {f"metric_{name}": np.asarray(value) for name, value in bundle.metrics.items()}
    fp_arrays = {f"fp_recall_{name}": np.asarray(value) for name, value in bundle.fp_recall.items()}
    np.savez_compressed(
        path,
        format_version=np.asarray(BUNDLE_FORMAT_VERSION),
        split=np.asarray("chronological validation"),
        held_out=np.asarray(False),
        model_name=np.asarray(bundle.name),
        model_kind=np.asarray(bundle.kind),
        threshold=np.asarray(bundle.threshold, dtype=np.float64),
        target_recall=np.asarray(bundle.target_recall, dtype=np.float64),
        threshold_source=np.asarray(bundle.threshold_source),
        timesteps=np.asarray(bundle.timesteps, dtype=np.int32),
        grid_shape=np.asarray(bundle.grid_shape, dtype=np.int16),
        labels=np.asarray(bundle.labels, dtype=np.uint8),
        actions=np.asarray(bundle.actions, dtype=np.uint8),
        detector_scores=np.asarray(bundle.detector_scores, dtype=np.float32),
        **metric_arrays,
        **fp_arrays,
    )


def _npz_scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    value = np.asarray(archive[name])
    if value.shape != ():
        msg = f"Prediction bundle field {name!r} must be scalar"
        raise ValueError(msg)
    return value.item()


def load_bundle(path: Path) -> PredictionBundle:
    """Load and validate one compact prediction bundle."""
    with np.load(path.expanduser().resolve(strict=True), allow_pickle=False) as archive:
        if _npz_scalar(archive, "format_version") != BUNDLE_FORMAT_VERSION:
            msg = f"Unsupported prediction bundle: {path}"
            raise ValueError(msg)
        if _npz_scalar(archive, "split") != "chronological validation" or bool(_npz_scalar(archive, "held_out")):
            msg = "Offline reporting bundles must be validation-only"
            raise ValueError(msg)
        timesteps = tuple(int(value) for value in archive["timesteps"])
        grid_shape = tuple(int(value) for value in archive["grid_shape"])
        if len(grid_shape) != GRID_NDIM:
            msg = "Prediction bundle grid_shape must contain (z, y, x)"
            raise ValueError(msg)
        labels = np.array(archive["labels"], dtype=np.uint8, copy=True)
        actions = np.array(archive["actions"], dtype=np.uint8, copy=True)
        scores = np.array(archive["detector_scores"], dtype=np.float32, copy=True)
        metrics = {
            name.removeprefix("metric_"): _npz_scalar(archive, name)
            for name in archive.files
            if name.startswith("metric_")
        }
        fp_recall = {
            name.removeprefix("fp_recall_"): _npz_scalar(archive, name)
            for name in archive.files
            if name.startswith("fp_recall_")
        }
        bundle = PredictionBundle(
            name=str(_npz_scalar(archive, "model_name")),
            kind=str(_npz_scalar(archive, "model_kind")),
            threshold=float(_npz_scalar(archive, "threshold")),
            target_recall=float(_npz_scalar(archive, "target_recall")),
            threshold_source=str(_npz_scalar(archive, "threshold_source")),
            timesteps=timesteps,
            grid_shape=(grid_shape[0], grid_shape[1], grid_shape[2]),
            labels=labels,
            actions=actions,
            detector_scores=scores,
            metrics=metrics,
            fp_recall=fp_recall,
        )
    _validate_bundle(bundle)
    return bundle


def _validate_bundle(bundle: PredictionBundle) -> None:
    shape = _validate_layout(len(bundle.labels), bundle.timesteps, bundle.grid_shape)
    if bundle.grid_shape != shape:
        msg = "Prediction bundle grid shape was not canonicalized"
        raise ValueError(msg)
    if bundle.actions.shape != bundle.labels.shape or bundle.detector_scores.shape != bundle.labels.shape:
        msg = "Prediction bundle labels, actions, and scores must have matching vector shapes"
        raise ValueError(msg)
    if bundle.labels.max() >= N_CLASSES or bundle.actions.max() >= N_CLASSES:
        msg = f"Prediction bundle actions and labels must lie in [0, {N_CLASSES - 1}]"
        raise ValueError(msg)
    if not np.isfinite(bundle.detector_scores).all() or np.any(
        (bundle.detector_scores < 0) | (bundle.detector_scores > 1)
    ):
        msg = "Prediction bundle detector scores must be finite probabilities"
        raise ValueError(msg)


def _validate_bundle_set(bundles: Sequence[PredictionBundle]) -> None:
    if not bundles:
        msg = "At least one prediction bundle is required"
        raise ValueError(msg)
    names = [bundle.name for bundle in bundles]
    if len(names) != len(set(names)):
        msg = "Prediction bundle model names must be unique"
        raise ValueError(msg)
    reference = bundles[0]
    for bundle in bundles:
        _validate_bundle(bundle)
        if (
            bundle.timesteps != reference.timesteps
            or bundle.grid_shape != reference.grid_shape
            or not np.array_equal(bundle.labels, reference.labels)
        ):
            msg = "All prediction bundles must describe identical rows, labels, timesteps, and grid shape"
            raise ValueError(msg)
        if not math.isclose(bundle.target_recall, reference.target_recall, rel_tol=0.0, abs_tol=1e-12):
            msg = "All prediction bundles must use the same target-recall operating policy"
            raise ValueError(msg)


def _plot_modules():
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    pyplot = importlib.import_module("matplotlib.pyplot")
    colors = importlib.import_module("matplotlib.colors")
    patches = importlib.import_module("matplotlib.patches")
    return pyplot, colors, patches


def _model_colors(pyplot, n_models: int) -> list:
    colormap = pyplot.get_cmap("tab10")
    return [colormap(index % colormap.N) for index in range(n_models)]


def _annotate_bars(axis, bars, *, formatter: str = ".3g") -> None:
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            format(height, formatter),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_metric_dashboard(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    pyplot, _, _ = _plot_modules()
    names = [bundle.name for bundle in bundles]
    positions = np.arange(len(bundles))
    colors = _model_colors(pyplot, len(bundles))
    figure, axes = pyplot.subplots(2, 4, figsize=(20, 9))

    panels = (
        ("metrics", "average_precision", "Detection average precision ↑", 1.0, ".5f"),
        ("metrics", "ranked_probability_score", "Ranked probability score x 10⁶ ↓", 1e6, ".3f"),
        ("metrics", "negative_log_likelihood", "Negative log-likelihood x 10⁵ ↓", 1e5, ".3f"),
        ("metrics", "brier_score", "Brier score x 10⁵ ↓", 1e5, ".3f"),
        ("fixed_recall", "fp_at_97", "False positives at 97% recall ↓", 1.0, ".0f"),
        ("fixed_recall", "fp_at_99", "False positives at 99% recall ↓", 1.0, ".0f"),
        ("metrics", "exact_on_positive", "Exact action on required-positive boxes ↑", 1.0, ".3f"),
        ("metrics", "mae_on_positive", "Action MAE on required-positive boxes ↓", 1.0, ".3f"),
    )
    for axis, (source, metric, title, scale, formatter) in zip(axes.ravel(), panels, strict=True):
        values = [
            float(bundle.metrics[metric] if source == "metrics" else bundle.fp_recall[metric]) * scale
            for bundle in bundles
        ]
        bars = axis.bar(positions, values, color=colors)
        axis.set_title(title)
        axis.set_xticks(positions, labels=names, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.2)
        _annotate_bars(axis, bars, formatter=formatter)
        if metric in {"average_precision", "exact_on_positive"}:
            lower = max(0.0, min(values) - 0.04)
            axis.set_ylim(lower, 1.005)
        if metric.startswith("fp_at_"):
            axis.set_yscale("symlog", linthresh=1)

    figure.suptitle(
        "Model comparison on the chronological validation period\n"
        "Probability scores are threshold-free; FP and action metrics use thresholds fitted separately "
        "on the same labels",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison_dashboard.png", dpi=200, bbox_inches="tight")
    pyplot.close(figure)


def _downsample_curve(recall: np.ndarray, precision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(recall) <= MAX_PR_POINTS:
        return recall, precision
    indices = np.linspace(0, len(recall) - 1, MAX_PR_POINTS, dtype=np.int64)
    return recall[indices], precision[indices]


def _high_recall_precision_floor(bundles: Sequence[PredictionBundle]) -> float:
    """Keep every fitted operating point visible in the high-recall zoom."""
    minimum_operating_precision = min(float(bundle.metrics["precision"]) for bundle in bundles)
    return max(0.0, minimum_operating_precision - OPERATING_PRECISION_PADDING)


def _plot_precision_recall(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    pyplot, _, _ = _plot_modules()
    colors = _model_colors(pyplot, len(bundles))
    figure, axes = pyplot.subplots(1, 2, figsize=(14, 5.5))
    positive = bundles[0].labels > 0
    for color, bundle in zip(colors, bundles, strict=True):
        precision, recall, _ = precision_recall_curve(positive, bundle.detector_scores)
        recall, precision = _downsample_curve(recall, precision)
        label = f"{bundle.name} (AP {float(bundle.metrics['average_precision']):.5f})"
        for axis in axes:
            axis.plot(recall, precision, color=color, linewidth=1.8, label=label)
            axis.scatter(
                float(bundle.metrics["recall"]),
                float(bundle.metrics["precision"]),
                color=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
    axes[0].set_title("Full precision-recall curve")
    axes[0].set_xlim(0, 1.01)
    axes[0].set_ylim(0, 1.01)
    axes[1].set_title("High-recall region")
    axes[1].set_xlim(0.9, 1.002)
    axes[1].set_ylim(_high_recall_precision_floor(bundles), 1.002)
    for axis in axes:
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Validation precision-recall "
        f"(markers show each model's validation-fitted {bundles[0].target_recall:.0%} operating point)"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "precision_recall.png", dpi=200, bbox_inches="tight")
    pyplot.close(figure)


def _plot_fp_recall(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    pyplot, _, _ = _plot_modules()
    colors = _model_colors(pyplot, len(bundles))
    recalls = (95, 97, 99)
    figure, axes = pyplot.subplots(1, 2, figsize=(13, 5))
    for color, bundle in zip(colors, bundles, strict=True):
        false_positives = [int(bundle.fp_recall[f"fp_at_{recall}"]) for recall in recalls]
        precision = [float(bundle.fp_recall[f"precision_at_{recall}"]) for recall in recalls]
        axes[0].plot(recalls, false_positives, marker="o", linewidth=1.8, color=color, label=bundle.name)
        axes[1].plot(recalls, precision, marker="o", linewidth=1.8, color=color, label=bundle.name)
    axes[0].set_title("False positives at fixed target recall")
    axes[0].set_ylabel("False-positive grid boxes")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[1].set_title("Precision at fixed target recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 1.01)
    for axis in axes:
        axis.set_xlabel("Target recall (%)")
        axis.set_xticks(recalls)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Descriptive operating points fitted on the same validation labels")
    figure.tight_layout()
    figure.savefig(output_dir / "fixed_recall_comparison.png", dpi=200, bbox_inches="tight")
    pyplot.close(figure)


def _detector_calibration_bins(labels: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray]:
    """Summarise binary detector calibration in fixed rare-event-aware probability bins."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape:
        msg = "Calibration labels and detector scores must be matching vectors"
        raise ValueError(msg)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        msg = "Calibration detector scores must be finite probabilities"
        raise ValueError(msg)

    edges = np.asarray(CALIBRATION_PROBABILITY_EDGES, dtype=np.float64)
    positive = labels > 0
    row_count, _ = np.histogram(scores, bins=edges)
    positive_count, _ = np.histogram(scores[positive], bins=edges)
    score_sum, _ = np.histogram(scores, bins=edges, weights=scores.astype(np.float64, copy=False))
    mean_score = np.divide(
        score_sum,
        row_count,
        out=np.full(len(row_count), np.nan, dtype=np.float64),
        where=row_count > 0,
    )
    observed_positive_rate = np.divide(
        positive_count,
        row_count,
        out=np.full(len(row_count), np.nan, dtype=np.float64),
        where=row_count > 0,
    )
    return {
        "lower_edge": edges[:-1],
        "upper_edge": edges[1:],
        "row_count": row_count,
        "positive_count": positive_count,
        "mean_score": mean_score,
        "observed_positive_rate": observed_positive_rate,
    }


def _plot_detector_calibration(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    pyplot, _, _ = _plot_modules()
    colors = _model_colors(pyplot, len(bundles))
    figure, axes = pyplot.subplots(
        2,
        len(bundles),
        figsize=(max(9.0, 4.5 * len(bundles)), 8.2),
        squeeze=False,
        sharex="col",
        constrained_layout=True,
    )
    diagonal = np.concatenate(([0.0], np.geomspace(1e-8, 1.0, 100)))
    for column, (bundle, color) in enumerate(zip(bundles, colors, strict=True)):
        summary = _detector_calibration_bins(bundle.labels, bundle.detector_scores)
        occupied = summary["row_count"] > 0
        mean_score = summary["mean_score"][occupied]
        observed = summary["observed_positive_rate"][occupied]
        row_count = summary["row_count"][occupied]
        positive_count = summary["positive_count"][occupied]

        calibration_axis = axes[0, column]
        calibration_axis.plot(diagonal, diagonal, color="0.45", linestyle="--", linewidth=1, label="Ideal")
        calibration_axis.plot(mean_score, observed, color=color, linewidth=1.4, alpha=0.8)
        calibration_axis.scatter(mean_score, observed, color=color, edgecolor="black", linewidth=0.4, zorder=3)
        calibration_axis.set_title(bundle.name)
        calibration_axis.set_ylabel("Observed positive fraction")
        calibration_axis.set_ylim(0, 1.02)
        calibration_axis.legend(fontsize=8)

        support_axis = axes[1, column]
        support_axis.plot(mean_score, row_count, color="0.45", marker="o", linewidth=1.2, label="All rows")
        support_axis.plot(
            mean_score,
            positive_count,
            color=color,
            marker="s",
            linewidth=1.4,
            label="Positive rows",
        )
        support_axis.set_ylabel("Rows in occupied bin")
        support_axis.set_yscale("symlog", linthresh=1)
        support_axis.legend(fontsize=8)

        for axis in (calibration_axis, support_axis):
            axis.set_xscale("symlog", linthresh=1e-8)
            axis.set_xlim(0, 1.02)
            axis.grid(alpha=0.2)
        support_axis.set_xlabel("Mean predicted any-halving probability")

    figure.suptitle(
        "Binary detector calibration on chronological validation\n"
        "Fixed log-aware probability bins; support is shown below. Descriptive only: this reused validation "
        "set is not independent.\nSeverity calibration and row-independence uncertainty intervals are not shown.",
        fontsize=13,
    )
    figure.savefig(output_dir / "detector_calibration.png", dpi=200, bbox_inches="tight")
    pyplot.close(figure)


def _confusion(labels: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.bincount(
        labels.astype(np.int64) * N_CLASSES + actions.astype(np.int64),
        minlength=N_CLASSES**2,
    ).reshape(N_CLASSES, N_CLASSES)


def _positive_action_diagnostics(labels: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Return classwise recall and signed ordinal errors for required-positive rows."""
    labels = np.asarray(labels)
    actions = np.asarray(actions)
    if labels.ndim != 1 or actions.shape != labels.shape:
        msg = "Action-diagnostic labels and predictions must be matching vectors"
        raise ValueError(msg)

    confusion = _confusion(labels, actions)
    support = confusion.sum(axis=1)[1:]
    exact = np.diag(confusion)[1:]
    recall = np.divide(
        exact,
        support,
        out=np.full(N_CLASSES - 1, np.nan, dtype=np.float64),
        where=support > 0,
    )
    positive = labels > 0
    error = actions[positive].astype(np.int64) - labels[positive].astype(np.int64)
    under_count = int(np.count_nonzero(error < 0))
    exact_count = int(np.count_nonzero(error == 0))
    over_count = int(np.count_nonzero(error > 0))
    total = len(error)
    proportions = np.asarray((under_count, exact_count, over_count), dtype=np.float64)
    if total:
        proportions /= total
    return {
        "support": support,
        "recall": recall,
        "under_count": under_count,
        "exact_count": exact_count,
        "over_count": over_count,
        "proportions": proportions,
        "under_level_sum": int(-error[error < 0].sum()),
        "over_level_sum": int(error[error > 0].sum()),
        "mae": float(np.abs(error).mean()) if total else math.nan,
    }


def _plot_classwise_diagnostics(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    pyplot, _, _ = _plot_modules()
    colors = _model_colors(pyplot, len(bundles))
    diagnostics = [_positive_action_diagnostics(bundle.labels, bundle.actions) for bundle in bundles]
    support = np.asarray(diagnostics[0]["support"])
    actions = np.arange(1, N_CLASSES)
    figure, axes = pyplot.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)

    recall_axis = axes[0]
    recall_axis.axvspan(3.65, 4.35, color="0.92", zorder=0)
    for bundle, color, diagnostic in zip(bundles, colors, diagnostics, strict=True):
        recall_axis.plot(
            actions,
            diagnostic["recall"],
            color=color,
            marker="o",
            linewidth=1.7,
            label=bundle.name,
        )
    recall_axis.set_title("Exact recall by required positive action")
    recall_axis.set_xlabel("Required halving action (validation support)")
    recall_axis.set_ylabel("Fraction assigned the exact action")
    recall_axis.set_xticks(
        actions,
        labels=[f"{action}\n(n={count:,})" for action, count in zip(actions, support, strict=True)],
    )
    recall_axis.set_xlim(0.7, 4.3)
    recall_axis.set_ylim(0, 1.02)
    recall_axis.grid(alpha=0.2)
    recall_axis.legend(fontsize=8)

    summary_axis = axes[1]
    positions = np.arange(len(bundles))
    category_names = ("Underpredicted", "Exact", "Overpredicted")
    category_colors = ("#d95f02", "#1b9e77", "#7570b3")
    left = np.zeros(len(bundles), dtype=np.float64)
    for category_index, (category, color) in enumerate(zip(category_names, category_colors, strict=True)):
        values = np.asarray([diagnostic["proportions"][category_index] for diagnostic in diagnostics])
        bars = summary_axis.barh(positions, values, left=left, color=color, label=category)
        if category == "Exact":
            summary_axis.bar_label(
                bars,
                labels=[f"{value:.1%}" if value >= ACTION_SUMMARY_LABEL_MINIMUM else "" for value in values],
                label_type="center",
                color="white",
                fontsize=8,
            )
        left += values
    for position, diagnostic in zip(positions, diagnostics, strict=True):
        summary_axis.text(
            1.02,
            position,
            (
                f"MAE {float(diagnostic['mae']):.3f}; "
                f"under/over levels {diagnostic['under_level_sum']}/{diagnostic['over_level_sum']}"
            ),
            va="center",
            fontsize=8,
        )
    summary_axis.axvline(1.0, color="0.4", linewidth=0.8)
    summary_axis.set_title("Ordinal outcome on all required-positive boxes")
    summary_axis.set_xlabel("Fraction of required-positive boxes")
    summary_axis.set_yticks(positions, labels=[bundle.name for bundle in bundles])
    summary_axis.set_xlim(0, 1.58)
    summary_axis.set_xticks(np.linspace(0, 1, 5))
    summary_axis.invert_yaxis()
    summary_axis.grid(axis="x", alpha=0.2)
    summary_axis.legend(fontsize=8, loc="lower right")

    figure.suptitle(
        "Positive-class action diagnostics at separately validation-fitted thresholds\n"
        "Class 4 has no training support and at most one current validation example; no class-4 capability is claimed.",
        fontsize=13,
    )
    figure.savefig(output_dir / "classwise_diagnostics.png", dpi=200, bbox_inches="tight")
    pyplot.close(figure)


def _plot_confusions(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    pyplot, _, _ = _plot_modules()
    figure, axes = pyplot.subplots(
        1,
        len(bundles),
        figsize=(5.0 * len(bundles), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for axis, bundle in zip(axes[0], bundles, strict=True):
        confusion = _confusion(bundle.labels, bundle.actions)
        row_totals = confusion.sum(axis=1, keepdims=True)
        normalized = np.divide(
            confusion,
            row_totals,
            out=np.zeros_like(confusion, dtype=np.float64),
            where=row_totals != 0,
        )
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for actual in range(N_CLASSES):
            for predicted in range(N_CLASSES):
                fraction = normalized[actual, predicted]
                axis.text(
                    predicted,
                    actual,
                    f"{confusion[actual, predicted]:,}\n{fraction:.1%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if fraction > CONFUSION_TEXT_THRESHOLD else "black",
                )
        axis.set_title(bundle.name)
        axis.set_xlabel("Predicted halving action")
        axis.set_ylabel("Required halving action")
        axis.set_xticks(np.arange(N_CLASSES), labels=CLASS_LABELS)
        axis.set_yticks(np.arange(N_CLASSES), labels=CLASS_LABELS)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Fraction of actual class")
    figure.suptitle("Validation action confusion (raw count and row-normalised fraction)")
    figure.savefig(output_dir / "action_confusion.png", dpi=200, bbox_inches="tight")
    pyplot.close(figure)


def _plot_timestep_counts(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    pyplot, _, _ = _plot_modules()
    colors = _model_colors(pyplot, len(bundles))
    timesteps = np.asarray(bundles[0].timesteps)
    truth = bundles[0].labels.reshape(len(timesteps), -1)
    actual = np.count_nonzero(truth > 0, axis=1)
    figure, axes = pyplot.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(timesteps, actual, color="black", marker="o", linewidth=2.2, label="Actual positive")
    for color, bundle in zip(colors, bundles, strict=True):
        predicted = np.count_nonzero(bundle.actions.reshape(len(timesteps), -1) > 0, axis=1)
        axes[0].plot(timesteps, predicted, color=color, marker=".", linewidth=1.4, label=bundle.name)
        timestep_rows = _per_timestep(bundle.labels, bundle.actions, timesteps, bundle.grid_shape)
        false_positive = [row["false_positive"] for row in timestep_rows]
        false_negative = [row["false_negative"] for row in timestep_rows]
        axes[1].plot(timesteps, false_negative, color=color, marker="o", linewidth=1.5, label=f"{bundle.name} FN")
        axes[1].plot(
            timesteps,
            false_positive,
            color=color,
            marker="x",
            linestyle="--",
            linewidth=1.2,
            label=f"{bundle.name} FP",
        )
    axes[0].set_ylabel("Positive grid boxes")
    axes[0].set_title("Actual and predicted hard-box counts")
    axes[1].set_ylabel("Strict grid-box errors")
    axes[1].set_xlabel("Validation timestep")
    axes[1].set_title("False negatives (solid) and false positives (dashed)")
    axes[1].set_yscale("symlog", linthresh=1)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, fontsize=8)
    axes[1].set_xticks(timesteps)
    figure.suptitle("Chronological validation behaviour")
    figure.tight_layout()
    figure.savefig(output_dir / "timestep_diagnostics.png", dpi=200, bbox_inches="tight")
    pyplot.close(figure)


def _plot_vertical_profile(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    pyplot, _, _ = _plot_modules()
    colors = _model_colors(pyplot, len(bundles))
    shape = (len(bundles[0].timesteps), *bundles[0].grid_shape)
    truth = bundles[0].labels.reshape(shape) > 0
    actual_by_level = np.count_nonzero(truth, axis=(0, 2, 3))
    levels = np.arange(bundles[0].grid_shape[0])
    figure, axes = pyplot.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(levels, actual_by_level, color="black", linewidth=2.3, label="Actual positive")
    for color, bundle in zip(colors, bundles, strict=True):
        predicted = bundle.actions.reshape(shape) > 0
        predicted_by_level = np.count_nonzero(predicted, axis=(0, 2, 3))
        false_positive = np.count_nonzero(~truth & predicted, axis=(0, 2, 3))
        false_negative = np.count_nonzero(truth & ~predicted, axis=(0, 2, 3))
        axes[0].plot(levels, predicted_by_level, color=color, linewidth=1.5, label=bundle.name)
        axes[1].plot(levels, false_negative, color=color, linewidth=1.5, label=f"{bundle.name} FN")
        axes[1].plot(levels, false_positive, color=color, linestyle="--", linewidth=1.2, label=f"{bundle.name} FP")
    axes[0].set_title("Positive occurrence by model level")
    axes[0].set_ylabel("Grid-box occurrences over validation")
    axes[1].set_title("Errors by model level")
    axes[1].set_ylabel("Strict grid-box errors")
    axes[1].set_yscale("symlog", linthresh=1)
    for axis in axes:
        axis.set_xlabel("Model level index")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    figure.suptitle("Vertical distribution (model level is not labelled as physical altitude)")
    figure.tight_layout()
    figure.savefig(output_dir / "vertical_distribution.png", dpi=200, bbox_inches="tight")
    pyplot.close(figure)


def _plot_spatial_frequency(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    pyplot, colors_module, _ = _plot_modules()
    n_models = len(bundles)
    n_timesteps = len(bundles[0].timesteps)
    shape = (n_timesteps, *bundles[0].grid_shape)
    truth = bundles[0].labels.reshape(shape)
    actual_frequency = np.count_nonzero(truth > 0, axis=(0, 1))
    predicted_frequencies = [np.count_nonzero(bundle.actions.reshape(shape) > 0, axis=(0, 1)) for bundle in bundles]
    maximum_frequency = max(1, int(max(actual_frequency.max(), *(values.max() for values in predicted_frequencies))))

    figure, axes = pyplot.subplots(
        2,
        n_models + 1,
        figsize=(4.1 * (n_models + 1), 8),
        squeeze=False,
        constrained_layout=True,
    )
    frequency_image = axes[0, 0].imshow(
        actual_frequency,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=maximum_frequency,
    )
    axes[0, 0].set_title("Actual positive frequency")
    for column, (bundle, frequency) in enumerate(zip(bundles, predicted_frequencies, strict=True), start=1):
        axes[0, column].imshow(
            frequency,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=maximum_frequency,
        )
        axes[0, column].set_title(f"{bundle.name}\npredicted frequency")

    actual_maximum = truth.max(axis=(0, 1))
    severity_cmap = colors_module.ListedColormap(("white", "#4daf4a", "#377eb8", "#984ea3", "#e41a1c"))
    severity_norm = colors_module.BoundaryNorm(np.arange(-0.5, N_CLASSES + 0.5), N_CLASSES)
    severity_image = axes[1, 0].imshow(
        actual_maximum,
        origin="lower",
        aspect="auto",
        cmap=severity_cmap,
        norm=severity_norm,
    )
    axes[1, 0].set_title("Maximum required action")

    differences = [frequency.astype(np.int64) - actual_frequency for frequency in predicted_frequencies]
    difference_limit = max(1, int(max(np.abs(values).max() for values in differences)))
    difference_norm = colors_module.TwoSlopeNorm(vmin=-difference_limit, vcenter=0, vmax=difference_limit)
    difference_image = None
    for column, (bundle, difference) in enumerate(zip(bundles, differences, strict=True), start=1):
        difference_image = axes[1, column].imshow(
            difference,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            norm=difference_norm,
        )
        axes[1, column].set_title(f"{bundle.name}\npredicted - actual")

    for row in axes:
        for axis in row:
            axis.set_xlabel("x grid index")
            axis.set_ylabel("y grid index")
    figure.colorbar(
        frequency_image,
        ax=axes[0].tolist(),
        fraction=0.025,
        pad=0.015,
        label="Positive grid-box occurrences summed over time and model level",
    )
    figure.colorbar(
        severity_image,
        ax=axes[1, 0],
        fraction=0.08,
        pad=0.12,
        ticks=np.arange(N_CLASSES),
        label="Halving action",
        orientation="horizontal",
    )
    if difference_image is not None:
        figure.colorbar(
            difference_image,
            ax=axes[1, 1:].tolist(),
            fraction=0.025,
            pad=0.015,
            label="Occurrence-count difference",
        )
    figure.suptitle("Time- and level-aggregated horizontal distribution on validation")
    figure.savefig(output_dir / "spatial_frequency.png", dpi=200, bbox_inches="tight")
    pyplot.close(figure)


def _column_error_categories(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    if truth.shape != predicted.shape or truth.ndim != GRID_NDIM:
        msg = "Timestep truth and prediction must share a (z, y, x) shape"
        raise ValueError(msg)
    actual_positive = (truth > 0).any(axis=0)
    predicted_positive = (predicted > 0).any(axis=0)
    exact_column = np.all(truth == predicted, axis=0)
    categories = np.zeros(actual_positive.shape, dtype=np.uint8)
    categories[actual_positive & predicted_positive & exact_column] = 1
    categories[actual_positive & predicted_positive & ~exact_column] = 2
    categories[actual_positive & ~predicted_positive] = 3
    categories[~actual_positive & predicted_positive] = 4
    return categories


def _timestep_figure(bundles: Sequence[PredictionBundle], timestep_index: int):
    pyplot, colors_module, patches_module = _plot_modules()
    n_models = len(bundles)
    shape = (len(bundles[0].timesteps), *bundles[0].grid_shape)
    truth = bundles[0].labels.reshape(shape)[timestep_index]
    predictions = [bundle.actions.reshape(shape)[timestep_index] for bundle in bundles]

    figure, axes = pyplot.subplots(
        2,
        n_models + 1,
        figsize=(4.0 * (n_models + 1), 7.6),
        squeeze=False,
        constrained_layout=True,
    )
    severity_cmap = colors_module.ListedColormap(("white", "#4daf4a", "#377eb8", "#984ea3", "#e41a1c"))
    severity_norm = colors_module.BoundaryNorm(np.arange(-0.5, N_CLASSES + 0.5), N_CLASSES)
    action_image = axes[0, 0].imshow(
        truth.max(axis=0),
        origin="lower",
        aspect="auto",
        cmap=severity_cmap,
        norm=severity_norm,
    )
    axes[0, 0].set_title("Actual maximum action")
    for column, (bundle, prediction) in enumerate(zip(bundles, predictions, strict=True), start=1):
        axes[0, column].imshow(
            prediction.max(axis=0),
            origin="lower",
            aspect="auto",
            cmap=severity_cmap,
            norm=severity_norm,
        )
        axes[0, column].set_title(f"{bundle.name}\nmaximum predicted action")

    category_colors = ("white", "#4daf4a", "#ffbf00", "#377eb8", "#e41a1c")
    category_cmap = colors_module.ListedColormap(category_colors)
    category_norm = colors_module.BoundaryNorm(np.arange(-0.5, len(category_colors) + 0.5), len(category_colors))
    axes[1, 0].axis("off")
    legend = [
        patches_module.Patch(facecolor=color, edgecolor="0.5", label=label)
        for color, label in zip(category_colors, ERROR_CATEGORY_LABELS, strict=True)
    ]
    axes[1, 0].legend(handles=legend, loc="center", frameon=False, fontsize=9)
    for column, (bundle, prediction) in enumerate(zip(bundles, predictions, strict=True), start=1):
        categories = _column_error_categories(truth, prediction)
        axes[1, column].imshow(
            categories,
            origin="lower",
            aspect="auto",
            cmap=category_cmap,
            norm=category_norm,
        )
        axes[1, column].set_title(f"{bundle.name}\ncolumn comparison")
    for row_index, row in enumerate(axes):
        for column_index, axis in enumerate(row):
            if row_index == 1 and column_index == 0:
                continue
            axis.set_xlabel("x grid index")
            axis.set_ylabel("y grid index")
    figure.colorbar(
        action_image,
        ax=axes[0].tolist(),
        fraction=0.018,
        pad=0.02,
        ticks=np.arange(N_CLASSES),
        label="Halving action",
    )
    timestep = bundles[0].timesteps[timestep_index]
    positive_count = int(np.count_nonzero(truth > 0))
    figure.suptitle(
        f"Validation timestep {timestep}: {positive_count} required-positive grid boxes\n"
        "Horizontal maps collapse model level by maximum action; bottom row preserves same-column mismatches",
        fontsize=13,
    )
    return figure


def _plot_timestep_examples(
    bundles: Sequence[PredictionBundle],
    output_dir: Path,
    example_count: int,
) -> list[int]:
    pyplot, _, _ = _plot_modules()
    n_timesteps = len(bundles[0].timesteps)
    if example_count < 1:
        msg = "example_count must be positive"
        raise ValueError(msg)
    truth = bundles[0].labels.reshape(n_timesteps, -1)
    positive_counts = np.count_nonzero(truth > 0, axis=1)
    order = sorted(range(n_timesteps), key=lambda index: (-int(positive_counts[index]), index))
    selected = order[: min(example_count, n_timesteps)]
    for timestep_index in selected:
        figure = _timestep_figure(bundles, timestep_index)
        timestep = bundles[0].timesteps[timestep_index]
        figure.savefig(output_dir / f"spatial_timestep_{timestep}.png", dpi=200, bbox_inches="tight")
        pyplot.close(figure)
    return [bundles[0].timesteps[index] for index in selected]


def _plot_all_timesteps_pdf(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    pyplot, _, _ = _plot_modules()
    backend_pdf = importlib.import_module("matplotlib.backends.backend_pdf")
    with backend_pdf.PdfPages(output_dir / "spatial_all_timesteps.pdf") as pdf:
        for timestep_index in range(len(bundles[0].timesteps)):
            figure = _timestep_figure(bundles, timestep_index)
            pdf.savefig(figure, bbox_inches="tight")
            pyplot.close(figure)


def _plot_column_colocation(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    pyplot, _, _ = _plot_modules()
    positions = np.arange(len(bundles))
    colors = _model_colors(pyplot, len(bundles))
    diagnostics = [
        column_colocation(bundle.labels, bundle.actions, bundle.timesteps, bundle.grid_shape) for bundle in bundles
    ]
    fp_same = [row["false_positive_breakdown"]["same_horizontal_column_different_level"] for row in diagnostics]
    fp_elsewhere = [row["false_positive_breakdown"]["no_actual_positive_in_column"] for row in diagnostics]
    fn_same = [row["false_negative_breakdown"]["another_prediction_in_same_horizontal_column"] for row in diagnostics]
    fn_elsewhere = [row["false_negative_breakdown"]["no_prediction_in_column"] for row in diagnostics]

    figure, axes = pyplot.subplots(1, 2, figsize=(13, 5.5))
    fp_bottom = axes[0].bar(positions, fp_same, color=colors, label="True event elsewhere in same column")
    fp_top = axes[0].bar(
        positions,
        fp_elsewhere,
        bottom=fp_same,
        color=colors,
        hatch="//",
        alpha=0.65,
        label="No true event in column",
    )
    fn_bottom = axes[1].bar(positions, fn_same, color=colors, label="Prediction elsewhere in same column")
    fn_top = axes[1].bar(
        positions,
        fn_elsewhere,
        bottom=fn_same,
        color=colors,
        hatch="//",
        alpha=0.65,
        label="No prediction in column",
    )
    for axis, title in zip(axes, ("Strict false-positive boxes", "Strict false-negative boxes"), strict=True):
        axis.set_title(title)
        axis.set_xticks(positions, labels=[bundle.name for bundle in bundles], rotation=18, ha="right")
        axis.set_ylabel("Grid boxes")
        axis.set_yscale("symlog", linthresh=1)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    for axis, first, second in ((axes[0], fp_bottom, fp_top), (axes[1], fn_bottom, fn_top)):
        axis.bar_label(first, labels=[str(value) if value else "" for value in first.datavalues], fontsize=8)
        axis.bar_label(second, labels=[str(value) if value else "" for value in second.datavalues], fontsize=8)
    figure.suptitle(
        "Column-aware error description on validation\n"
        "Same-column cases remain errors until UKCA's operational action scope is agreed"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "column_colocation.png", dpi=200, bbox_inches="tight")
    pyplot.close(figure)


def _write_metrics_csv(bundles: Sequence[PredictionBundle], output_dir: Path) -> None:
    metric_names = tuple(bundles[0].metrics)
    fp_names = tuple(bundles[0].fp_recall)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("model", "kind", "threshold", "target_recall", *metric_names, *fp_names)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bundle in bundles:
            writer.writerow(
                {
                    "model": bundle.name,
                    "kind": bundle.kind,
                    "threshold": bundle.threshold,
                    "target_recall": bundle.target_recall,
                    **bundle.metrics,
                    **bundle.fp_recall,
                }
            )

    with (output_dir / "timestep_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "model",
            "timestep",
            "actual_positive",
            "predicted_positive",
            "true_positive",
            "false_positive",
            "false_negative",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bundle in bundles:
            for row in _per_timestep(bundle.labels, bundle.actions, bundle.timesteps, bundle.grid_shape):
                writer.writerow({"model": bundle.name, **row})


def render_plots(
    bundles: Sequence[PredictionBundle],
    output_dir: Path,
    *,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
) -> dict:
    """Render the complete headless comparison suite from prediction bundles."""
    _validate_bundle_set(bundles)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric_dashboard(bundles, output_dir)
    _plot_precision_recall(bundles, output_dir)
    _plot_fp_recall(bundles, output_dir)
    _plot_detector_calibration(bundles, output_dir)
    _plot_classwise_diagnostics(bundles, output_dir)
    _plot_confusions(bundles, output_dir)
    _plot_timestep_counts(bundles, output_dir)
    _plot_vertical_profile(bundles, output_dir)
    _plot_spatial_frequency(bundles, output_dir)
    _plot_column_colocation(bundles, output_dir)
    selected_timesteps = _plot_timestep_examples(bundles, output_dir, example_count)
    _plot_all_timesteps_pdf(bundles, output_dir)
    _write_metrics_csv(bundles, output_dir)
    plot_manifest = {
        "format_version": REPORT_FORMAT_VERSION,
        "split": "chronological validation",
        "held_out": False,
        "models": [bundle.name for bundle in bundles],
        "timesteps": list(bundles[0].timesteps),
        "grid_shape_zyx": list(bundles[0].grid_shape),
        "example_selection": {
            "policy": "largest true-positive grid-box count; ties resolved chronologically",
            "timesteps": selected_timesteps,
        },
        "detector_calibration": {
            "target": "binary any-halving probability",
            "probability_bin_edges": list(CALIBRATION_PROBABILITY_EDGES),
            "support": "all-row and required-positive row counts per occupied bin",
            "uncertainty_intervals": None,
        },
        "figures": [
            "model_comparison_dashboard.png",
            "precision_recall.png",
            "fixed_recall_comparison.png",
            "detector_calibration.png",
            "classwise_diagnostics.png",
            "action_confusion.png",
            "timestep_diagnostics.png",
            "vertical_distribution.png",
            "spatial_frequency.png",
            "column_colocation.png",
            "spatial_all_timesteps.pdf",
            *(f"spatial_timestep_{timestep}.png" for timestep in selected_timesteps),
        ],
        "caveats": [
            "All thresholds are fitted on the same repeatedly used validation labels.",
            "Same-column errors remain strict grid-box errors and are only a descriptive diagnostic.",
            "The vertical axis is model-level index, not verified physical altitude.",
            "Class 4 has no training support in the current one-week dataset.",
            (
                "Detector calibration is binary-only and descriptive on repeatedly used validation; "
                "no row-independence uncertainty intervals are shown."
            ),
        ],
    }
    _write_json(output_dir / "plot_manifest.json", plot_manifest)
    return plot_manifest


def _evaluate_predictions(
    name: str,
    kind: str,
    probabilities: np.ndarray,
    detector: np.ndarray,
    labels: np.ndarray,
    timesteps: tuple[int, ...],
    grid_shape: tuple[int, int, int],
    target_recall: float,
    *,
    source: dict,
) -> tuple[PredictionBundle, dict]:
    threshold = threshold_at_recall(labels, detector, target_recall)
    threshold_source = f"fitted on shared validation labels at target recall {target_recall:.6g}"
    evaluation = evaluate(
        probabilities,
        labels,
        threshold=threshold,
        threshold_source=threshold_source,
        detection_outputs=detector,
    )
    actions = _hard_actions(probabilities, detector, threshold)
    action_metrics = halving_action_metrics(actions, labels, n_classes=N_CLASSES)
    fixed_recall = fp_recall_diagnostics(detector, labels)
    headline = _bundle_metrics(evaluation, action_metrics)
    bundle = PredictionBundle(
        name=name,
        kind=kind,
        threshold=float(threshold),
        target_recall=target_recall,
        threshold_source=threshold_source,
        timesteps=timesteps,
        grid_shape=grid_shape,
        labels=np.asarray(labels, dtype=np.uint8),
        actions=actions,
        detector_scores=np.asarray(detector, dtype=np.float32),
        metrics=headline,
        fp_recall=fixed_recall,
    )
    report = {
        "name": name,
        "kind": kind,
        "source": source,
        "decision_policy": {
            "detector_threshold": float(threshold),
            "threshold_source": threshold_source,
            "target_recall": target_recall,
            "comparison": ">=",
            "severity": "argmax of positive classes 1--4 after equal-probability ensemble averaging",
        },
        "evaluation": evaluation,
        "hard_action_metrics": action_metrics,
        "fixed_recall_diagnostics": fixed_recall,
        "per_timestep": _per_timestep(labels, actions, timesteps, grid_shape),
        "column_colocation": column_colocation(labels, actions, timesteps, grid_shape),
    }
    return bundle, report


def _prepare_output_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    protected = (
        resolved / "comparison.json",
        resolved / "plot_manifest.json",
        resolved / "metrics.csv",
    )
    existing = [str(candidate) for candidate in protected if candidate.exists()]
    if existing:
        msg = f"Refusing to replace an existing reporting run: {', '.join(existing)}"
        raise FileExistsError(msg)
    return resolved


def run_evaluate(args: argparse.Namespace) -> dict:  # noqa: PLR0915
    """Run validation-only inference, save compact bundles, and render plots."""
    if args.batch_size < 1 or args.example_count < 1:
        msg = "batch-size and example-count must be positive"
        raise ValueError(msg)
    if not 0.0 < args.target_recall <= 1.0:
        msg = "target-recall must lie in (0, 1]"
        raise ValueError(msg)
    student_groups = _parse_student_groups(args.student)
    if not student_groups and not args.include_cached_xgboost:
        msg = "Supply at least one --student group or --include-cached-xgboost"
        raise ValueError(msg)
    requested_names = [group.name for group in student_groups]
    if args.include_cached_xgboost:
        requested_names.append(args.xgboost_name)
    if len(requested_names) != len(set(requested_names)):
        msg = "Every requested model name must be unique"
        raise ValueError(msg)
    slugs = [_slug(name) for name in requested_names]
    if len(slugs) != len(set(slugs)):
        msg = "Requested model names collapse to duplicate output filenames"
        raise ValueError(msg)

    output_dir = _prepare_output_directory(args.output_dir)
    cache_dir, metadata, features, labels = _load_validation_cache(args.cache_dir)
    grid_shape = _validate_layout(len(labels), args.timesteps, args.grid_shape)
    timesteps = tuple(args.timesteps)
    device = torch.device(args.device)
    bundles = []
    model_reports = []

    for group in student_groups:
        checkpoints, checkpoint_artifacts = _load_checkpoint_group(group, metadata, features.shape[1])
        probabilities, detector = _predict_student_ensemble(
            checkpoints,
            features,
            device,
            args.batch_size,
            label=group.name,
        )
        objective = checkpoints[0]["objective"]
        kind = "student-distilled" if objective.get("distill") else "student-supervised"
        bundle, report = _evaluate_predictions(
            group.name,
            kind,
            probabilities,
            detector,
            labels,
            timesteps,
            grid_shape,
            args.target_recall,
            source={
                "aggregation": "equal arithmetic mean of canonical full five-class checkpoint probabilities",
                "member_count": len(checkpoints),
                "checkpoints": checkpoint_artifacts,
                "configuration": _checkpoint_comparison_config(checkpoints[0]),
                "objective": objective,
            },
        )
        bundle_path = output_dir / f"{_slug(group.name)}.npz"
        _save_bundle(bundle_path, bundle)
        report["prediction_bundle"] = _artifact(bundle_path)
        bundles.append(bundle)
        model_reports.append(report)
        del probabilities, detector, checkpoints
        gc.collect()

    if args.include_cached_xgboost:
        probabilities, detector = _cached_teacher_predictions(cache_dir, len(labels))
        bundle, report = _evaluate_predictions(
            args.xgboost_name,
            "xgboost-cached-teacher",
            probabilities,
            detector,
            labels,
            timesteps,
            grid_shape,
            args.target_recall,
            source={
                "prediction_source": "population-prior-corrected cached XGBoost validation responses",
                "detector_margin": _artifact(cache_dir / "val_teacher_margin.npy"),
                "conditional_severity": _artifact(cache_dir / "val_teacher_severity.npy"),
                "teacher_metadata": metadata.get("teacher"),
            },
        )
        bundle_path = output_dir / f"{_slug(args.xgboost_name)}.npz"
        _save_bundle(bundle_path, bundle)
        report["prediction_bundle"] = _artifact(bundle_path)
        bundles.append(bundle)
        model_reports.append(report)
        del probabilities, detector
        gc.collect()

    plot_manifest = render_plots(bundles, output_dir, example_count=args.example_count)
    report = {
        "format_version": REPORT_FORMAT_VERSION,
        "status": "validation-inference-and-offline-plots-complete",
        "data": {
            "split": "chronological validation cache only",
            "held_out": False,
            "held_out_test_used": False,
            "cache_dir": str(cache_dir),
            "cache_metadata": _artifact(cache_dir / "metadata.json", hash_contents=True),
            "preprocess": metadata.get("preprocess"),
            "rows": len(labels),
            "class_counts": class_counts(labels),
            "timesteps": list(timesteps),
            "grid_shape_zyx": list(grid_shape),
            "row_order": "timestep-major; within each timestep C-order (z, y, x), x fastest",
        },
        "comparison_policy": {
            "checkpoint_role": "canonical RPS-selected only",
            "student_aggregation": "equal mean of full five-class probabilities before threshold/action",
            "threshold_rule": "one separately fitted validation threshold per model at the same target recall",
            "target_recall": args.target_recall,
            "action_rule": "0 below threshold; otherwise argmax over positive classes plus one",
            "same_column_policy": "descriptive diagnostic only; strict grid-box metrics remain primary",
        },
        "models": model_reports,
        "plots": plot_manifest,
    }
    _write_json(output_dir / "comparison.json", report)
    print(f"Saved offline comparison to {output_dir}", flush=True)
    return report


def run_plot(args: argparse.Namespace) -> dict:
    """Regenerate figures from compact prediction bundles without inference."""
    bundles = tuple(load_bundle(path) for path in args.bundle)
    return render_plots(bundles, args.output_dir, example_count=args.example_count)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the offline evaluation or plot-only command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run existing models over a validation cache and create bundles/plots",
    )
    evaluate_parser.add_argument("--cache-dir", type=Path, required=True)
    evaluate_parser.add_argument(
        "--student",
        action="append",
        nargs="+",
        metavar="VALUE",
        help='Repeatable group: --student "Model name" checkpoint1.pt [checkpoint2.pt ...]',
    )
    evaluate_parser.add_argument(
        "--include-cached-xgboost",
        action="store_true",
        help="Compare population-corrected val_teacher_margin/severity arrays from the same cache",
    )
    evaluate_parser.add_argument("--xgboost-name", default="XGBoost")
    evaluate_parser.add_argument(
        "--timesteps",
        type=_parse_timestep_range,
        required=True,
        help="Inclusive validation range, for example 119:143",
    )
    evaluate_parser.add_argument("--grid-shape", type=int, nargs=3, default=GRID_SHAPE, metavar=("Z", "Y", "X"))
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    evaluate_parser.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    evaluate_parser.add_argument("--example-count", type=int, default=DEFAULT_EXAMPLE_COUNT)

    plot_parser = subparsers.add_parser("plot", help="Regenerate plots from saved compact prediction bundles")
    plot_parser.add_argument("--bundle", type=Path, action="append", required=True)
    plot_parser.add_argument("--output-dir", type=Path, required=True)
    plot_parser.add_argument("--example-count", type=int, default=DEFAULT_EXAMPLE_COUNT)
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> dict:
    """Run the requested offline reporting command."""
    args = parse_args() if args is None else args
    if args.command == "evaluate":
        return run_evaluate(args)
    if args.command == "plot":
        return run_plot(args)
    msg = f"Unsupported reporting command: {args.command!r}"
    raise ValueError(msg)


if __name__ == "__main__":
    main()
