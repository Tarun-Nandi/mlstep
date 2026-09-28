"""Validation operating points and the fixed held-out t307-t360 evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np

from mlstep.cache import load_cache_state
from mlstep.data import FEATURES, N_CLASSES, N_FEATURES, class_counts, load_timesteps
from mlstep.evaluation import (
    actions_at_threshold,
    detection_ap,
    evaluate_probabilities,
    halving_action_metrics,
    halving_confusion_metrics,
    predict_joint_probabilities,
    recall_action_frontier,
)
from mlstep.model import Student, load_model
from mlstep.precision import PRECISIONS, resolve_device

DEFAULT_BATCH_SIZE: Final = 4096
LOCKED_TIMESTEPS: Final = tuple(range(307, 361))
MATRIX_NDIM: Final = 2
PROPER_METRICS: Final = (
    "negative_log_likelihood",
    "brier_score",
    "ranked_probability_score",
)


def _write_json(path: Path, content: dict) -> None:
    """Write a new JSON report, refusing to overwrite an existing file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(content, indent=2, allow_nan=False) + "\n")


def _validate_locked_split(metadata: dict) -> None:
    """Require every held-out timestep to be recorded as unused by the cache.

    Cache loading already checks that training, validation and unused sets
    are disjoint. This checks the evaluation-specific exclusion, not whether
    the held-out data was inspected during earlier research.
    """
    locked = set(LOCKED_TIMESTEPS)
    used = set(metadata["train"]["timesteps"]) | set(metadata["val"]["timesteps"])
    unused = set(metadata["split"]["unused_timesteps"])
    if locked & used or not locked <= unused:
        raise ValueError("cache split must leave every held-out timestep unused")


def _load_evaluation_model(model_path: Path, metadata: dict, ple: dict | None, device: str) -> Student:
    """Load weights and check them against already-loaded cache state.

    Cache loading establishes the feature schema. Here check the model's
    preprocessing choice and, for PLE, exact indices and fitted boundaries.
    Scalar preprocessing still requires the exact cache used for training.
    """
    model, method = load_model(model_path, device=resolve_device(device))
    if method != metadata["preprocess"]:
        raise ValueError("model preprocessing does not match the cache")
    if ple is not None:
        if model.ple is None or not np.array_equal(model.ple_indices, ple["ple_indices"]):
            raise ValueError("model PLE indices do not match the cache")
        if not np.array_equal(model.ple.bin_boundaries.detach().cpu().numpy(), ple["bin_boundaries"]):
            raise ValueError("model PLE boundaries do not match the cache")
    return model


def _load_validation_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Open cached validation features and labels without loading them all into RAM.

    Check their shapes/dtypes here; cache-state loading checks the small metadata
    files, while prediction and metric functions check numerical values.
    """
    directory = Path(cache_dir)
    features = np.load(directory / "val_x.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(directory / "val_y.npy", mmap_mode="r", allow_pickle=False)
    if features.ndim != MATRIX_NDIM or features.shape[1] != N_FEATURES or features.dtype != np.float32:
        msg = f"val_x.npy must be a float32 matrix with {N_FEATURES} columns"
        raise ValueError(msg)
    if labels.shape != (len(features),) or labels.dtype != np.int64:
        msg = "val_y.npy must be a matching int64 vector"
        raise ValueError(msg)
    return features, labels


def evaluate_operating_points(
    model_path: Path,
    cache_dir: Path,
    target_recalls: Sequence[float],
    output_path: Path,
    *,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    precision: str = "auto",
) -> dict:
    """Write validation probability scores, exact-count metrics and recall thresholds.

    Use cached preprocessed rows and fit only the requested recall thresholds.
    These rows also select training epochs, so this is a validation report,
    not an independent estimate of generalisation. Return the written report.
    """
    if not target_recalls:
        msg = "target_recalls must be supplied explicitly"
        raise ValueError(msg)
    if output_path.exists():
        raise FileExistsError(output_path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    metadata, _, ple = load_cache_state(cache_dir)
    validation_x, validation_y = _load_validation_cache(cache_dir)
    model = _load_evaluation_model(model_path, metadata, ple, device)
    probabilities, detector = predict_joint_probabilities(model, validation_x, device, batch_size, precision)

    result = {
        "kind": "validation-operating-points",
        "preprocess": metadata["preprocess"],
        "timesteps": metadata["val"]["timesteps"],
        "rows": len(validation_y),
        "class_counts": class_counts(validation_y),
        "probability_metrics": evaluate_probabilities(
            probabilities,
            validation_y,
            detector,
        ),
        "joint_map_actions": halving_action_metrics(probabilities.argmax(axis=1), validation_y),
        "operating_points": recall_action_frontier(
            probabilities,
            detector,
            validation_y,
            target_recalls,
        ),
    }
    _write_json(output_path, result)
    return result


def _aggregate_probability_metrics(
    reports: Sequence[dict],
    row_counts: Sequence[int],
    detector_chunks: Sequence[np.ndarray],
    positive_chunks: Sequence[np.ndarray],
) -> dict:
    """Combine timestep scores using the correct denominator for each metric.

    Joint scores are weighted by all rows; conditional scores by positive rows.
    AP depends on ranking across timesteps, so recompute it from pooled scores
    and binary labels instead of averaging the per-timestep AP values.
    """
    total_rows = sum(row_counts)
    positive_counts = [int(report["positive_examples"]) for report in reports]
    total_positive = sum(positive_counts)

    def weighted(name: str, weights: Sequence[int]) -> dict | None:
        """Average one score group, skipping empty groups whose report is None."""
        total_weight = sum(weights)
        if not total_weight:
            return None
        return {
            metric: sum(
                float(report[name][metric]) * weight for report, weight in zip(reports, weights, strict=True) if weight
            )
            / total_weight
            for metric in PROPER_METRICS
        }

    detector = np.concatenate(detector_chunks)
    positive = np.concatenate(positive_chunks).astype(np.int64, copy=False)
    return {
        "ap": detection_ap(detector, positive),
        "prevalence": total_positive / total_rows,
        "positive_examples": total_positive,
        "probability": weighted("probability", row_counts),
        "conditional_positive_probability": weighted("conditional_positive_probability", positive_counts),
    }


def evaluate_locked_timesteps(
    model_path: Path,
    cache_dir: Path,
    data_dir: Path,
    output_path: Path,
    *,
    threshold: float | None = None,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    precision: str = "auto",
) -> dict:
    """Write aggregate metrics for t307–t360 using the training cache's fitted transform.

    Load and release one raw timestep at a time. Always report joint-argmax
    counts; optionally evaluate a threshold selected beforehand on validation.
    Never fit preprocessing or thresholds here. Return the written report.
    """
    if output_path.exists():
        raise FileExistsError(output_path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    metadata, preprocessor, ple = load_cache_state(cache_dir)
    _validate_locked_split(metadata)
    model = _load_evaluation_model(model_path, metadata, ple, device)

    reports = []
    row_counts = []
    detector_chunks = []
    positive_chunks = []
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    map_confusion = np.zeros_like(confusion)
    for timestep in LOCKED_TIMESTEPS:
        raw_features, labels = load_timesteps(Path(data_dir), (timestep,), FEATURES)
        probabilities, detector = predict_joint_probabilities(
            model, raw_features, device, batch_size, precision, preprocessor=preprocessor,
        )
        report = evaluate_probabilities(probabilities, labels, detector)
        reports.append(report)
        row_counts.append(len(labels))
        # Pooled AP needs only these two vectors, not every five-class matrix.
        detector_chunks.append(detector)
        positive_chunks.append(labels > 0)
        # Flat index = true count * N_CLASSES + chosen count. Accumulate counts
        # here and calculate rates once, across the entire held-out period.
        map_actions = probabilities.argmax(axis=1)
        map_confusion += np.bincount(
            labels * N_CLASSES + map_actions, minlength=N_CLASSES**2,
        ).reshape(N_CLASSES, N_CLASSES)
        if threshold is not None:
            actions = actions_at_threshold(probabilities, detector, threshold)
            confusion += np.bincount(
                labels * N_CLASSES + actions, minlength=N_CLASSES**2,
            ).reshape(N_CLASSES, N_CLASSES)
            del actions
        # Otherwise the previous timestep remains alive while the next loads.
        del raw_features, probabilities, labels, map_actions

    result = {
        "kind": "locked-t307-t360",
        "preprocess": metadata["preprocess"],
        "timesteps": list(LOCKED_TIMESTEPS),
        "rows": sum(row_counts),
        "class_counts": map_confusion.sum(axis=1).tolist(),
        "probability_metrics": _aggregate_probability_metrics(
            reports,
            row_counts,
            detector_chunks,
            positive_chunks,
        ),
        "joint_map_actions": halving_confusion_metrics(map_confusion),
        "threshold_evaluation": None,
    }
    if threshold is not None:
        result["threshold_evaluation"] = {
            "threshold": float(threshold),
            "metrics": halving_confusion_metrics(confusion),
        }
    _write_json(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    """Define separate commands for validation threshold fitting and held-out scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    operating = commands.add_parser("operating-points", help="evaluate caller-supplied validation recalls")
    locked = commands.add_parser("locked", help="evaluate the fixed held-out t307-t360 period")
    for command in (operating, locked):
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--cache-dir", type=Path, required=True, help="Prepared neural cache")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--precision", choices=PRECISIONS, default="auto")
        command.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    operating.add_argument("--recalls", type=float, nargs="+", required=True)
    locked.add_argument("--data-dir", type=Path, required=True)
    locked.add_argument("--threshold", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation or locked-period evaluation from the command line."""
    args = _parser().parse_args(argv)
    if args.command == "operating-points":
        evaluate_operating_points(
            args.model,
            args.cache_dir,
            args.recalls,
            args.output,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
        )
    else:
        evaluate_locked_timesteps(
            args.model,
            args.cache_dir,
            args.data_dir,
            args.output,
            threshold=args.threshold,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
