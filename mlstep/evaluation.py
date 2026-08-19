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
from mlstep.policy import (
    AsymmetricExactOverpredictionPolicy,
    EpsilonPolicy,
    boundary_probabilities,
    validate_probabilities,
)

TOP_FRACTION = 0.001
PROBABILITY_NDIM = 2  # a probability matrix is two dimensional: (rows, classes)
BINARY_COLUMNS = 2  # more columns than this means genuine multiclass output
BOUNDARY_TOLERANCE = 1e-12
PROBABILITY_METRIC_CHUNK_ROWS = 1_000_000


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


def probability_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Evaluate a full ordinal class distribution with proper scores.

    The ranked probability score is averaged over the ``n_classes - 1``
    cumulative class boundaries, so it remains on a comparable scale when the
    number of supported actions changes. All reductions use float64 and are
    chunked to avoid large temporary matrices for the week validation split.
    """
    probabilities = validate_probabilities(probabilities)
    targets = np.asarray(targets)
    if targets.ndim != 1 or len(targets) != len(probabilities):
        msg = "targets must be a one-dimensional array matching probabilities"
        raise ValueError(msg)
    if np.iscomplexobj(targets) or not np.issubdtype(targets.dtype, np.number) or not np.isfinite(targets).all():
        msg = "targets must contain finite numeric values"
        raise ValueError(msg)
    integer_targets = targets.astype(np.int64, copy=False)
    if not np.array_equal(targets, integer_targets):
        msg = "targets must contain integer class labels"
        raise ValueError(msg)

    n_rows, n_classes = probabilities.shape
    if n_classes < BINARY_COLUMNS:
        msg = "probabilities must contain at least two classes"
        raise ValueError(msg)
    if integer_targets.min() < 0 or integer_targets.max() >= n_classes:
        msg = f"targets must be between 0 and {n_classes - 1}"
        raise ValueError(msg)

    nll_sum = 0.0
    brier_sum = 0.0
    rps_sum = 0.0
    boundaries = np.arange(n_classes - 1, dtype=np.int64)
    tiny = np.finfo(np.float64).tiny

    for start in range(0, n_rows, PROBABILITY_METRIC_CHUNK_ROWS):
        stop = min(start + PROBABILITY_METRIC_CHUNK_ROWS, n_rows)
        chunk = probabilities[start:stop].astype(np.float64, copy=False)
        chunk_targets = integer_targets[start:stop]
        row_indices = np.arange(stop - start)
        true_probability = chunk[row_indices, chunk_targets]
        nll_sum -= float(np.log(np.maximum(true_probability, tiny)).sum())

        # ||q - one_hot(y)||^2 = sum(q^2) - 2 q_y + 1.
        brier_sum += float((np.square(chunk).sum(axis=1) - 2.0 * true_probability + 1.0).sum())

        cumulative = np.cumsum(chunk[:, :-1], axis=1, dtype=np.float64)
        observed_cumulative = chunk_targets[:, None] <= boundaries
        np.subtract(cumulative, observed_cumulative, out=cumulative)
        rps_sum += float(np.square(cumulative).sum())

    return {
        "negative_log_likelihood": nll_sum / n_rows,
        "brier_score": brier_sum / n_rows,
        "ranked_probability_score": rps_sum / (n_rows * (n_classes - 1)),
    }


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


def _ranking_diagnostics(
    targets: np.ndarray,
    scores: np.ndarray,
    positive: np.ndarray,
    n_positive: int,
    selected_here: bool,
) -> dict:
    """Build report-only ranking diagnostics outside the hot selection path."""
    top_n = min(len(targets), max(1, int(np.ceil(TOP_FRACTION * len(targets)))))
    top = np.argpartition(scores, len(scores) - top_n)[-top_n:]
    positives_at_top = int(positive[top].sum())
    diagnostics = {
        "top_fraction": TOP_FRACTION,
        "top_n": top_n,
        "precision_at_top": positives_at_top / top_n,
        "recall_at_top": safe_divide(positives_at_top, n_positive),
    }
    if selected_here and n_positive:
        diagnostics["validation_recall_sweep"] = {
            f"{sweep_recall:.0%}": operating_point(targets, scores, sweep_recall)
            for sweep_recall in (0.80, 0.90, 0.95, 0.97, 0.99, 1.0)
        }
    return diagnostics


def evaluate(
    outputs: np.ndarray,
    targets: np.ndarray,
    threshold: float | None = None,
    target_recall: float = 1.0,
    threshold_source: str | None = None,
    detection_outputs: np.ndarray | None = None,
    include_diagnostics: bool = True,
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
    }

    if include_diagnostics:
        metrics.update(_ranking_diagnostics(targets, scores, positive, n_positive, selected_here))
    # Some more optinal multiclass metrics
    if outputs.ndim == PROBABILITY_NDIM and outputs.shape[1] > BINARY_COLUMNS:
        metrics["probability"] = probability_metrics(outputs, targets)
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

    confusion = np.bincount(
        targets * n_classes + actions,
        minlength=n_classes**2,
    ).reshape(n_classes, n_classes)
    support = confusion.sum(axis=1)
    action_counts = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    total = int(confusion.sum())
    positive_count = int(support[1:].sum())

    true_class = np.arange(n_classes, dtype=np.int64)[:, None]
    action_class = np.arange(n_classes, dtype=np.int64)[None, :]
    error = action_class - true_class
    absolute_error = np.abs(error)
    over_levels = np.maximum(error, 0)
    under_levels = np.maximum(-error, 0)
    substeps = np.left_shift(1, np.arange(n_classes, dtype=np.int64))
    substep_error = substeps[None, :] - substeps[:, None]
    over_substeps = np.maximum(substep_error, 0)
    under_substeps = np.maximum(-substep_error, 0)

    overprediction_count = int(confusion[error > 0].sum())
    underprediction_count = int(confusion[error < 0].sum())
    overprediction_level_sum = int(np.sum(confusion * over_levels))
    underprediction_level_sum = int(np.sum(confusion * under_levels))
    overprediction_substep_sum = int(np.sum(confusion * over_substeps))
    underprediction_substep_sum = int(np.sum(confusion * under_substeps))
    exact_count = int(diagonal.sum())
    exact_positive_count = int(diagonal[1:].sum())

    counts = {
        "true_positive": int(confusion[1:, 1:].sum()),
        "false_positive": int(confusion[0, 1:].sum()),
        "false_negative": int(confusion[1:, 0].sum()),
        "true_negative": int(confusion[0, 0]),
    }
    positive_confusion = confusion[1:, :]
    positive_underprediction_count = int(positive_confusion[error[1:, :] < 0].sum())
    positive_overprediction_count = int(positive_confusion[error[1:, :] > 0].sum())
    positive_absolute_error_sum = int(np.sum(positive_confusion * absolute_error[1:, :]))
    rows_per_million = 1_000_000 / total

    def ratio_or_none(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "confusion": confusion.tolist(),
        "support_by_class": support.tolist(),
        "action_counts": action_counts.tolist(),
        "recall_by_class": [
            float(confusion[label, label] / support[label]) if support[label] else None for label in range(n_classes)
        ],
        "exact_by_class": [
            float(confusion[label, label] / support[label]) if support[label] else None for label in range(n_classes)
        ],
        "exact_count_all": exact_count,
        "exact_rate_all": exact_count / total,
        "exact_on_positive_count": exact_positive_count,
        "exact_on_positive": ratio_or_none(exact_positive_count, positive_count),
        "overprediction_count_all": overprediction_count,
        "overprediction_rate_all": overprediction_count / total,
        "overprediction_level_sum_all": overprediction_level_sum,
        "overprediction_level_mean_all": overprediction_level_sum / total,
        "underprediction_count_all": underprediction_count,
        "underprediction_rate_all": underprediction_count / total,
        "underprediction_level_sum_all": underprediction_level_sum,
        "underprediction_level_mean_all": underprediction_level_sum / total,
        "overprediction_substep_proxy_sum_all": overprediction_substep_sum,
        "overprediction_substep_proxy_mean_all": overprediction_substep_sum / total,
        "underprediction_substep_proxy_sum_all": underprediction_substep_sum,
        "underprediction_substep_proxy_mean_all": underprediction_substep_sum / total,
        "false_positives_per_million": counts["false_positive"] * rows_per_million,
        "overpredictions_per_million": overprediction_count * rows_per_million,
        "detection_precision": safe_divide(
            counts["true_positive"],
            counts["true_positive"] + counts["false_positive"],
        ),
        "detection_recall": safe_divide(
            counts["true_positive"],
            counts["true_positive"] + counts["false_negative"],
        ),
        **counts,
        "mae_on_positive": ratio_or_none(positive_absolute_error_sum, positive_count),
        "underprediction_rate": ratio_or_none(positive_underprediction_count, positive_count),
        "overprediction_rate": ratio_or_none(positive_overprediction_count, positive_count),
        "exact_given_detected_positive": ratio_or_none(
            exact_positive_count,
            counts["true_positive"],
        ),
    }


def asymmetric_exact_overprediction_cost(
    actions: np.ndarray,
    targets: np.ndarray,
    overprediction_penalty: float,
) -> dict[str, float | int]:
    """Evaluate ``1[a != y] + lambda * max(a-y, 0)`` on hard actions."""
    policy = AsymmetricExactOverpredictionPolicy(overprediction_penalty)
    metrics = halving_action_metrics(actions, targets, n_classes=5)
    return _asymmetric_cost_from_action_metrics(metrics, policy.overprediction_penalty)


def _asymmetric_cost_from_action_metrics(
    metrics: dict,
    overprediction_penalty: float,
) -> dict[str, float | int]:
    """Compute one empirical objective from already-reduced action metrics."""
    total = int(sum(metrics["support_by_class"]))
    inexact_count = total - int(metrics["exact_count_all"])
    overprediction_level_sum = int(metrics["overprediction_level_sum_all"])
    total_cost = inexact_count + overprediction_penalty * overprediction_level_sum
    return {
        "rows": total,
        "inexact_count": inexact_count,
        "inexact_rate": inexact_count / total,
        "overprediction_level_sum": overprediction_level_sum,
        "overprediction_level_mean": overprediction_level_sum / total,
        "lambda": overprediction_penalty,
        "total_cost": total_cost,
        "mean_cost": total_cost / total,
    }


def asymmetric_policy_frontier(
    probabilities: np.ndarray,
    targets: np.ndarray,
    overprediction_penalties: tuple[float, ...] | list[float],
) -> list[dict]:
    """Evaluate and mark nondominated asymmetric posterior-risk policies.

    Nondominance is empirical on the supplied labels: a policy is dominated
    only when another makes no more inexact predictions on positive rows and
    no more excess-halving levels overall, with a strict improvement in at
    least one quantity. This avoids the meaningless all-zero optimum induced
    by the roughly 1:10,000 class prevalence. The helper does not select a
    deployment penalty or claim held-out performance.
    """
    raw_penalties = tuple(overprediction_penalties)
    if not raw_penalties:
        msg = "at least one overprediction penalty is required"
        raise ValueError(msg)

    policies_by_penalty = {
        policy.overprediction_penalty: policy
        for value in raw_penalties
        for policy in (AsymmetricExactOverpredictionPolicy(value),)
    }
    entries = []
    for penalty in sorted(policies_by_penalty):
        policy = policies_by_penalty[penalty]
        actions = policy.predict(probabilities)
        metrics = halving_action_metrics(actions, targets, n_classes=5)
        objective = _asymmetric_cost_from_action_metrics(
            metrics,
            policy.overprediction_penalty,
        )
        positive_rows = int(sum(metrics["support_by_class"][1:]))
        positive_inexact_count = positive_rows - int(metrics["exact_on_positive_count"])
        entries.append(
            {
                "policy": policy.to_descriptor(),
                "empirical_objective": objective,
                "pareto_coordinates": {
                    "positive_inexact_count": positive_inexact_count,
                    "overprediction_level_sum_all": int(metrics["overprediction_level_sum_all"]),
                },
                "metrics": metrics,
            }
        )

    for candidate in entries:
        candidate_inexact = candidate["pareto_coordinates"]["positive_inexact_count"]
        candidate_over = candidate["pareto_coordinates"]["overprediction_level_sum_all"]
        candidate["nondominated"] = not any(
            (
                other["pareto_coordinates"]["positive_inexact_count"] <= candidate_inexact
                and other["pareto_coordinates"]["overprediction_level_sum_all"] <= candidate_over
                and (
                    other["pareto_coordinates"]["positive_inexact_count"] < candidate_inexact
                    or other["pareto_coordinates"]["overprediction_level_sum_all"] < candidate_over
                )
            )
            for other in entries
            if other is not candidate
        )
    return entries


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


def _validated_frontier_boundaries(
    probabilities: np.ndarray,
    max_action: int,
    boundaries: np.ndarray | None,
) -> np.ndarray:
    """Build or validate cumulative boundaries used by a policy frontier."""
    if isinstance(max_action, bool) or not isinstance(max_action, (int, np.integer)):
        msg = "max_action must be an integer"
        raise ValueError(msg)
    if not 1 <= max_action < probabilities.shape[1]:
        msg = f"max_action must be between 1 and {probabilities.shape[1] - 1}"
        raise ValueError(msg)
    if boundaries is None:
        return boundary_probabilities(probabilities, max_action)

    boundaries = np.asarray(boundaries)
    expected_shape = (len(probabilities), max_action + 1)
    if boundaries.shape != expected_shape:
        msg = f"boundaries must have shape {expected_shape}"
        raise ValueError(msg)
    if not np.issubdtype(boundaries.dtype, np.number) or np.iscomplexobj(boundaries):
        msg = "boundaries must contain real numeric values"
        raise ValueError(msg)
    if not np.isfinite(boundaries).all():
        msg = "boundaries must be finite"
        raise ValueError(msg)
    if np.any((boundaries < 0.0) | (boundaries > 1.0)):
        msg = "boundaries must be in [0, 1]"
        raise ValueError(msg)
    if not np.allclose(boundaries[:, 0], 1.0, rtol=0.0, atol=BOUNDARY_TOLERANCE):
        msg = "the action-zero boundary must equal one"
        raise ValueError(msg)
    if np.any(np.diff(boundaries, axis=1) > BOUNDARY_TOLERANCE):
        msg = "boundaries must be non-increasing across actions"
        raise ValueError(msg)
    return boundaries


def epsilon_frontier(
    probabilities: np.ndarray,
    targets: np.ndarray,
    boundary_confidences: tuple[float, ...] | list[float],
    *,
    max_action: int = 3,
    boundaries: np.ndarray | None = None,
) -> list[dict]:
    """Evaluate a conservative epsilon-policy frontier.

    ``boundaries`` may be reused from :func:`boundary_probabilities` to avoid
    rebuilding a large float64 matrix during the final policy diagnostic.
    """
    probabilities = validate_probabilities(probabilities)
    boundaries = _validated_frontier_boundaries(probabilities, max_action, boundaries)
    boundary_confidences = tuple(boundary_confidences)
    if not boundary_confidences:
        msg = "at least one boundary confidence is required"
        raise ValueError(msg)

    confidences = []
    for confidence in boundary_confidences:
        policy = EpsilonPolicy(float(confidence), max_action)
        confidences.append(policy.boundary_confidence)
    confidences = sorted(set(confidences), reverse=True)

    previous_actions = None
    frontier = []

    for confidence in confidences:
        actions = (boundaries[:, 1:] >= confidence).sum(axis=1, dtype=np.int8)
        if previous_actions is not None and np.array_equal(actions, previous_actions):
            continue

        model_overprediction_risk = np.take_along_axis(
            boundaries,
            actions[:, None],
            axis=1,
        ).ravel()
        np.subtract(1.0, model_overprediction_risk, out=model_overprediction_risk)
        np.clip(model_overprediction_risk, 0.0, 1.0, out=model_overprediction_risk)
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
        previous_actions = actions

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
