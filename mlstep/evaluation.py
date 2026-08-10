"""Shared metrics, run metadata, JSON output, and lightweight timing."""

import argparse
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score

from mlstep.data import Feature, class_counts, feature_names
from mlstep.policy import EpsilonPolicy, boundary_probabilities, validate_probabilities

TOP_FRACTION = 0.001
PROBABILITY_NDIM = 2  # a probability matrix is two dimensional: (rows, classes)
BINARY_COLUMNS = 2  # more columns than this means genuine multiclass output


def detection_scores(outputs: np.ndarray) -> np.ndarray:
    """Return the any-halving score from binary or multiclass output."""
    outputs = np.asarray(outputs)
    if outputs.ndim == 1:
        return outputs
    if outputs.ndim != PROBABILITY_NDIM:
        msg = "model output must be a vector or probability matrix"
        raise ValueError(msg)
    if outputs.shape[1] == 1:
        return outputs[:, 0]
    return outputs[:, 1:].sum(axis=1)


def detection_ap(outputs: np.ndarray, targets: np.ndarray) -> float:
    """Average precision for detecting any non-zero halving class."""
    positive = np.asarray(targets) > 0
    if not positive.any():
        return 0.0
    # Convert the physical labels into a binary target
    return float(average_precision_score(positive, detection_scores(outputs)))


def threshold_at_recall(targets: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """Return the highest score threshold that captures the target no of positives."""
    if not 0 < target_recall <= 1:
        msg = "target_recall must be in (0, 1]"
        raise ValueError(msg)
    # From the positive rows, we sort the detector scores in descending order
    positive_scores = np.sort(np.asarray(scores)[np.asarray(targets) > 0])[::-1]
    if not len(positive_scores):
        return 1.0

    required = int(np.ceil(target_recall * len(positive_scores)))
    return float(positive_scores[required - 1])


def evaluate(
    outputs: np.ndarray,
    targets: np.ndarray,
    threshold: float | None = None,
    target_recall: float = 1.0,
    threshold_source: str | None = None,
    detection_outputs: np.ndarray | None = None,
) -> dict:
    """Evaluate rare-event detection and optional multiclass severity."""
    outputs = np.asarray(outputs)
    targets = np.asarray(targets, dtype=np.int64)
    # Error checks on model output
    if targets.ndim != 1 or not len(targets):
        msg = "targets must be a non-empty one-dimensional array"
        raise ValueError(msg)
    if len(outputs) != len(targets):
        msg = "expected one model output per target"
        raise ValueError(msg)
    if not np.isfinite(outputs).all():
        msg = "model outputs must be finite"
        raise ValueError(msg)
    if outputs.ndim == PROBABILITY_NDIM and np.any((outputs < 0) | (outputs > 1)):
        msg = "class probabilities must be in [0, 1]"
        raise ValueError(msg)
    if (
        outputs.ndim == PROBABILITY_NDIM
        and outputs.shape[1] > 1
        and not np.allclose(outputs.sum(axis=1), 1, rtol=1e-5, atol=1e-7)
    ):
        msg = "class probabilities must sum to one"
        raise ValueError(msg)

    scores = detection_scores(outputs if detection_outputs is None else detection_outputs)
    if len(scores) != len(targets):
        msg = "expected one detection output per target"
        raise ValueError(msg)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        msg = "detection scores must be in [0, 1]"
        raise ValueError(msg)
    positive = targets > 0
    n_positive = int(positive.sum())
    # if no frozen threshold is provided we choose one from the targets and the scores
    selected_here = threshold is None
    if selected_here:
        threshold = threshold_at_recall(targets, scores, target_recall)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        msg = "threshold must be finite and in [0, 1]"
        raise ValueError(msg)
    if threshold_source is None:
        threshold_source = "selected on these data" if selected_here else "provided/frozen before evaluation"
    # Convert the scores into final hard box decisions
    detected = scores >= threshold
    # calculate true positives/negatives + false positives/negatives
    counts = detection_counts(detected, positive)
    # precision: which of the models predictions end up actially being true
    precision = safe_divide(counts["true_positive"], counts["true_positive"] + counts["false_positive"])
    # recall: from all the positive boxes that need halving, how many is the model able to identify
    recall = safe_divide(counts["true_positive"], counts["true_positive"] + counts["false_negative"])
    # Calculating top ranked examples
    top_n = min(len(targets), max(1, int(np.ceil(TOP_FRACTION * len(targets)))))
    top = np.argpartition(scores, len(scores) - top_n)[-top_n:]
    positives_at_top = int(positive[top].sum())

    # This is the main metrics dictionary
    metrics = {
        "ap": detection_ap(scores, targets),
        "prevalence": float(positive.mean()),  # natural fraction of hard boxes in the data
        "positive_examples": n_positive,
        "score_mean": float(scores.mean()),
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "target_recall_requested": target_recall if selected_here else None,
        "precision": precision,
        "recall": recall,
        "f0_5": f_score(precision, recall, beta=0.5),  # Combines precision + recall (precision has higher weight)
        "mcc": mcc(counts),
        **counts,  # matthews correlation coefficient (uses all 4 confusion matrix counts)
        "top_fraction": TOP_FRACTION,
        "top_n": top_n,
        "precision_at_top": positives_at_top / top_n,
        "recall_at_top": safe_divide(positives_at_top, n_positive),
    }

    if selected_here and n_positive:
        metrics["validation_recall_sweep"] = {
            f"{sweep_recall:.0%}": operating_point(targets, scores, sweep_recall)
            for sweep_recall in (0.80, 0.90, 0.95, 0.97, 0.99, 1.0)
        }
    # Some more optinal multiclass metrics
    if outputs.ndim == PROBABILITY_NDIM and outputs.shape[1] > BINARY_COLUMNS:
        metrics["severity"] = exact_halvings_metrics(outputs, targets, detected)

    return metrics


def benchmark(function, *, warmup: int = 2, repeats: int = 5) -> dict:
    """Time a prediction callable after short warm-up."""
    # Use a warmup to try to account for things like initial memory allocation, cpu cache effects and so on
    for _ in range(warmup):
        function()
    # Measure 5 individual predicition calls
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        durations.append(time.perf_counter() - start)

    return {
        "median_seconds": median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "repeats": repeats,
    }


def run_metadata(
    args: argparse.Namespace,
    features: tuple[Feature, ...],
    train_y: np.ndarray,
    val_y: np.ndarray,
    versions: dict[str, str],
    splits: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> dict:
    """Build the status/config/data blocks shared by both training scripts."""
    train_steps, validation_steps, test_steps = splits
    names = feature_names(features)
    groups = [feature.name for feature in features]
    return {
        "config": {
            **vars(args),
            "train_steps": train_steps,
            "validation_steps": validation_steps,
            "held_out_test_steps": test_steps,
            "test_set_used": False,
            "feature_groups": groups,
            "feature_names": names,
            "n_features": len(names),
            "versions": {
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                **versions,
            },
        },
        "data": {
            "train_rows": len(train_y),
            "train_class_counts": class_counts(train_y),
            "validation_rows": len(val_y),
            "validation_class_counts": class_counts(val_y),
        },
    }


def output_paths(output_dir: Path, name: str, model_suffix: str) -> tuple[Path, Path]:
    """Return timestamped model and report paths for one run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create a timestamped name
    run_name = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
    return output_dir / f"{run_name}{model_suffix}", output_dir / f"{run_name}.json"


def write_json(path: Path, content: dict) -> None:
    """Write human-readable run metadata without model-specific utilities."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def _integer_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Return an integer vector without silently truncating values."""
    values = np.asarray(values)
    if values.ndim != 1 or not len(values):
        msg = f"{name} must be a non-empty one-dimensional array"
        raise ValueError(msg)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        msg = f"{name} must contain finite numeric values"
        raise ValueError(msg)
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64, copy=False)
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        msg = f"{name} must contain integer values"
        raise ValueError(msg)
    return rounded.astype(np.int64, copy=False)


def halving_action_metrics(
    actions: np.ndarray,
    targets: np.ndarray,
    *,
    n_classes: int = 5,
) -> dict:
    """Evaluate explicit integer halving actions against true requirements."""
    if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)) or n_classes < BINARY_COLUMNS:
        msg = "n_classes must be an integer of at least two"
        raise ValueError(msg)

    actions = _integer_vector(actions, "actions")
    targets = _integer_vector(targets, "targets")
    if len(actions) != len(targets):
        msg = "actions and targets must contain the same number of rows"
        raise ValueError(msg)
    if actions.min() < 0 or actions.max() >= n_classes:
        msg = f"actions must be between 0 and {n_classes - 1}"
        raise ValueError(msg)
    if targets.min() < 0 or targets.max() >= n_classes:
        msg = f"targets must be between 0 and {n_classes - 1}"
        raise ValueError(msg)

    positive = targets > 0
    detected = actions > 0
    detected_positive = detected & positive
    errors = actions - targets
    positive_errors = errors[positive]
    conditional_errors = errors[detected_positive]
    overprediction = errors > 0
    underprediction = errors < 0
    overprediction_levels = np.maximum(errors, 0)
    underprediction_levels = np.maximum(-errors, 0)

    confusion = np.bincount(
        targets * n_classes + actions,
        minlength=n_classes**2,
    ).reshape(n_classes, n_classes)
    support = np.bincount(targets, minlength=n_classes)
    counts = detection_counts(detected, positive)

    predicted_substeps = np.left_shift(1, actions)
    required_substeps = np.left_shift(1, targets)
    overprediction_substeps = np.maximum(predicted_substeps - required_substeps, 0)
    underprediction_substeps = np.maximum(required_substeps - predicted_substeps, 0)
    rows_per_million = 1_000_000 / len(targets)

    return {
        "confusion": confusion.tolist(),
        "support_by_class": support.tolist(),
        "action_counts": np.bincount(actions, minlength=n_classes).tolist(),
        "recall_by_class": [
            float(confusion[label, label] / support[label]) if support[label] else None for label in range(n_classes)
        ],
        "exact_by_class": [
            float(confusion[label, label] / support[label]) if support[label] else None for label in range(n_classes)
        ],
        "exact_count_all": int(np.count_nonzero(errors == 0)),
        "exact_rate_all": float(np.mean(errors == 0)),
        "exact_on_positive_count": int(np.count_nonzero(positive_errors == 0)),
        "exact_on_positive": mean_or_none(positive_errors == 0),
        "overprediction_count_all": int(np.count_nonzero(overprediction)),
        "overprediction_rate_all": float(np.mean(overprediction)),
        "overprediction_level_sum_all": int(overprediction_levels.sum()),
        "overprediction_level_mean_all": float(np.mean(overprediction_levels)),
        "underprediction_count_all": int(np.count_nonzero(underprediction)),
        "underprediction_rate_all": float(np.mean(underprediction)),
        "underprediction_level_sum_all": int(underprediction_levels.sum()),
        "underprediction_level_mean_all": float(np.mean(underprediction_levels)),
        "overprediction_substep_proxy_sum_all": int(overprediction_substeps.sum()),
        "overprediction_substep_proxy_mean_all": float(np.mean(overprediction_substeps)),
        "underprediction_substep_proxy_sum_all": int(underprediction_substeps.sum()),
        "underprediction_substep_proxy_mean_all": float(np.mean(underprediction_substeps)),
        "false_positives_per_million": counts["false_positive"] * rows_per_million,
        "overpredictions_per_million": int(np.count_nonzero(overprediction)) * rows_per_million,
        "detection_precision": safe_divide(
            counts["true_positive"],
            counts["true_positive"] + counts["false_positive"],
        ),
        "detection_recall": safe_divide(
            counts["true_positive"],
            counts["true_positive"] + counts["false_negative"],
        ),
        **counts,
        "mae_on_positive": mean_or_none(np.abs(positive_errors)),
        "underprediction_rate": mean_or_none(positive_errors < 0),
        "overprediction_rate": mean_or_none(positive_errors > 0),
        "exact_given_detected_positive": mean_or_none(conditional_errors == 0),
    }


def exact_halvings_metrics(outputs: np.ndarray, targets: np.ndarray, detected: np.ndarray) -> dict:
    """Calculate legacy threshold-plus-argmax exact-halving metrics."""
    outputs = np.asarray(outputs)
    detected = np.asarray(detected, dtype=bool)
    if outputs.ndim != PROBABILITY_NDIM or outputs.shape[1] <= BINARY_COLUMNS:
        msg = "outputs must be a multiclass probability matrix"
        raise ValueError(msg)
    if detected.shape != (len(outputs),):
        msg = "detected must contain one boolean decision per row"
        raise ValueError(msg)

    actions = np.where(detected, outputs[:, 1:].argmax(axis=1) + 1, 0)
    return halving_action_metrics(actions, targets, n_classes=outputs.shape[1])


def epsilon_frontier(
    probabilities: np.ndarray,
    targets: np.ndarray,
    boundary_confidences: tuple[float, ...] | list[float],
    *,
    max_action: int = 3,
) -> list[dict]:
    """Evaluate a conservative epsilon-policy frontier."""
    probabilities = validate_probabilities(probabilities)
    boundary_confidences = tuple(boundary_confidences)
    if not boundary_confidences:
        msg = "at least one boundary confidence is required"
        raise ValueError(msg)

    confidences = []
    for confidence in boundary_confidences:
        policy = EpsilonPolicy(float(confidence), max_action)
        confidences.append(policy.boundary_confidence)
    confidences = sorted(set(confidences), reverse=True)

    boundaries = boundary_probabilities(probabilities, max_action)
    row_indices = np.arange(len(probabilities))
    previous_actions = None
    frontier = []

    for confidence in confidences:
        actions = (boundaries[:, 1:] >= confidence).sum(axis=1, dtype=np.int64)
        actions = actions.astype(np.int8, copy=False)
        if previous_actions is not None and np.array_equal(actions, previous_actions):
            continue

        chosen_tail_probability = boundaries[row_indices, actions]
        model_overprediction_risk = np.clip(1.0 - chosen_tail_probability, 0.0, 1.0)
        metrics = halving_action_metrics(actions, targets, n_classes=probabilities.shape[1])
        frontier.append(
            {
                "boundary_confidence": confidence,
                "epsilon": 1.0 - confidence,
                "model_overprediction_risk_mean": float(model_overprediction_risk.mean()),
                "model_overprediction_risk_max": float(model_overprediction_risk.max()),
                "cap_hit_count": int(np.count_nonzero(actions == max_action)),
                "metrics": metrics,
            }
        )
        previous_actions = actions.copy()

    return frontier


def operating_point(targets: np.ndarray, scores: np.ndarray, target_recall: float) -> dict:
    """Build one operating point for the validation recall sweep."""
    threshold = threshold_at_recall(targets, scores, target_recall)
    counts = detection_counts(scores >= threshold, targets > 0)
    return {
        "threshold": threshold,
        "precision": safe_divide(
            counts["true_positive"],
            counts["true_positive"] + counts["false_positive"],
        ),
        "recall": safe_divide(
            counts["true_positive"],
            counts["true_positive"] + counts["false_negative"],
        ),
        **counts,
    }


def detection_counts(detected: np.ndarray, positive: np.ndarray) -> dict[str, int]:
    """Count true positives/negatives + false positives/negatives."""
    return {
        "true_positive": int(np.count_nonzero(detected & positive)),
        "false_positive": int(np.count_nonzero(detected & ~positive)),
        "false_negative": int(np.count_nonzero(~detected & positive)),
        "true_negative": int(np.count_nonzero(~detected & ~positive)),
    }


def mean_or_none(values: np.ndarray) -> float | None:
    """Calculate a mean when atleast one example exists otherwise return None."""
    return float(np.mean(values)) if len(values) else None


def safe_divide(numerator: int, denominator: int) -> float:
    """Peforms division that doesnt crach when denominator is 0."""
    return numerator / denominator if denominator else 0.0


def f_score(precision: float, recall: float, beta: float) -> float:
    """Calculate the f0.5 metric."""
    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    if not denominator:
        return 0.0
    return (1 + beta_squared) * precision * recall / denominator


def mcc(counts: dict[str, int]) -> float:
    """Calculate the mcc metric."""
    tp = float(counts["true_positive"])
    fp = float(counts["false_positive"])
    fn = float(counts["false_negative"])
    tn = float(counts["true_negative"])
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def json_default(value):
    """Serialise Path and numpy values that json cannot handle natively."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    msg = f"{type(value).__name__} is not JSON serializable"
    raise TypeError(msg)
