"""Offline batched model evaluation, probability metrics and halving actions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from mlstep.data import N_CLASSES, FittedPreprocesser
from mlstep.model import Student
from mlstep.precision import autocast_dtype, resolve_device

FeatureMatrix = np.ndarray | torch.Tensor

METRIC_CHUNK_ROWS = 1_000_000
MATRIX_NDIM = 2
MIN_PROBABILITY_CLASSES = 2


@torch.inference_mode()
def predict_joint_probabilities(
    model: Student,
    features: FeatureMatrix,
    device: torch.device | str,
    batch_size: int = 4096,
    precision: str = "auto",
    *,
    preprocessor: FittedPreprocesser | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute batched probabilities for training validation and offline reports.

    Accept preprocessed NumPy/torch rows by default. Reserved-period
    evaluation may supply a fitted preprocessor with raw NumPy rows; apply
    it per batch to avoid a full transformed copy. The model must already
    be on device. Return CPU float32 five-class probabilities and detector
    scores. This helper is not an FTorch export or production interface.
    """
    device = resolve_device(device)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if features.ndim != MATRIX_NDIM or features.shape[1] != model.n_features:
        raise ValueError(f"evaluation features must have {model.n_features} columns")
    if preprocessor is not None and not isinstance(features, np.ndarray):
        raise TypeError("raw evaluation rows must be a NumPy array")
    n_rows = len(features)
    if next(model.parameters()).device != device:
        raise ValueError("model and evaluation device must match")
    model.eval()

    amp_dtype = autocast_dtype(device, precision)
    # Column zero temporarily holds d; columns one-to-four hold conditional
    # severity. The same allocation becomes the final joint distribution.
    raw_output = torch.empty(
        (n_rows, N_CLASSES),
        dtype=torch.float32,
        device="cpu",
        pin_memory=device.type == "cuda",
    )
    finite_inputs = torch.ones((), dtype=torch.bool, device=device)
    for start_idx in range(0, n_rows, batch_size):
        stop_idx = min(start_idx + batch_size, n_rows)
        if isinstance(features, torch.Tensor):
            source = features[start_idx:stop_idx]
        else:
            rows = features[start_idx:stop_idx]
            if preprocessor is not None:
                rows = preprocessor.transform(rows)
            elif not rows.flags.writeable or any(stride < 0 for stride in rows.strides):
                # Torch cannot safely wrap read-only or negative-stride arrays.
                rows = rows.copy()
            # A writable NumPy batch can be shared: prediction never mutates it.
            source = torch.from_numpy(rows)
        batch_x = source.to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        # Keep this reduction on-device to avoid a CPU/GPU sync for every batch.
        finite_inputs.logical_and_(torch.isfinite(batch_x).all())

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, detector, _, severity = model(batch_x)
        packed = torch.cat((detector.unsqueeze(-1), severity), dim=-1)
        raw_output[start_idx:stop_idx].copy_(packed, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if not bool(finite_inputs):
        raise ValueError("features must contain only finite values")
    raw = raw_output.numpy()
    # Keep d directly: recovering it later from 1 - P(0) loses tiny scores.
    detector = raw[:, 0].copy()
    raw[:, 1:] *= detector[:, None]
    raw[:, 0] = 1.0 - detector
    return raw, detector


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Check a nonempty row-by-class probability matrix without renormalising it."""
    values = np.asarray(probabilities)
    if values.ndim != MATRIX_NDIM or not len(values) or values.shape[1] < MIN_PROBABILITY_CLASSES:
        msg = "probabilities must be a non-empty matrix with at least two classes"
        raise ValueError(msg)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        msg = "probabilities must contain real numeric values"
        raise ValueError(msg)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        msg = "probabilities must be finite and in [0, 1]"
        raise ValueError(msg)
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
        msg = "probability rows must sum to one"
        raise ValueError(msg)
    return values


def _integer_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Return counts as int64, accepting integer-valued floats but not fractions.

    Callers check row alignment and class range for their particular metric.
    """
    array = np.asarray(values)
    if array.ndim != 1 or not len(array):
        msg = f"{name} must be a non-empty one-dimensional array"
        raise ValueError(msg)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        msg = f"{name} must contain finite numeric values"
        raise ValueError(msg)
    integer = array.astype(np.int64, copy=False)
    if not np.array_equal(array, integer):
        msg = f"{name} must contain integer values"
        raise ValueError(msg)
    return integer


def _validated_detector_scores(scores: np.ndarray, n_rows: int) -> np.ndarray:
    """Check one finite probability score per row; return the array without copying."""
    values = np.asarray(scores)
    if values.ndim != 1 or len(values) != n_rows:
        msg = "detector_scores must be a vector matching probabilities"
        raise ValueError(msg)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        msg = "detector_scores must contain real numeric values"
        raise ValueError(msg)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        msg = "detector_scores must be finite and in [0, 1]"
        raise ValueError(msg)
    return values


def detection_ap(scores: np.ndarray, targets: np.ndarray) -> float:
    """Measure how well scores rank true nonzero counts above zeros.

    Return 0 when there are no positive targets. This measures detection only,
    not whether the predicted number of halvings is correct.
    """
    labels = _integer_vector(targets, "targets")
    values = _validated_detector_scores(scores, len(labels))
    positive = labels > 0
    if not positive.any():
        return 0.0
    return float(average_precision_score(positive, values))


def probability_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Validate predictions and labels, then return mean NLL, Brier score and RPS.

    Accept any number of ordered classes, with labels starting at zero.
    Lower scores are better; Brier sums over classes, while RPS averages over
    the class boundaries. Computation uses float64 chunks to limit temporaries.
    """
    values = validate_probabilities(probabilities)
    labels = _integer_vector(targets, "targets")
    if len(labels) != len(values):
        msg = "targets must match probabilities"
        raise ValueError(msg)
    n_rows, n_classes = values.shape
    if labels.min() < 0 or labels.max() >= n_classes:
        msg = f"targets must be between 0 and {n_classes - 1}"
        raise ValueError(msg)

    return _probability_metrics(values, labels)


def _probability_metrics(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Calculate probability scores from already validated, aligned arrays.

    Shared by the standalone metric function and the combined report so the
    latter does not rescan the same arrays for validation.
    """
    n_rows, n_classes = values.shape
    nll_sum = 0.0
    brier_sum = 0.0
    rps_sum = 0.0
    boundaries = np.arange(n_classes - 1)
    tiny = np.finfo(np.float64).tiny
    for start in range(0, n_rows, METRIC_CHUNK_ROWS):
        stop = min(start + METRIC_CHUNK_ROWS, n_rows)
        chunk = values[start:stop].astype(np.float64, copy=False)
        chunk_labels = labels[start:stop]
        true_probability = chunk[np.arange(stop - start), chunk_labels]
        nll_sum -= float(np.log(np.maximum(true_probability, tiny)).sum())
        # Expand squared distance to a one-hot label without allocating one.
        brier_sum += float((np.square(chunk).sum(axis=1) - 2.0 * true_probability + 1.0).sum())
        cumulative = np.cumsum(chunk[:, :-1], axis=1)
        # The observed CDF at boundary j is 1 exactly when the true class <= j.
        cumulative -= chunk_labels[:, None] <= boundaries
        rps_sum += float(np.square(cumulative).sum())

    return {
        "negative_log_likelihood": nll_sum / n_rows,
        "brier_score": brier_sum / n_rows,
        "ranked_probability_score": rps_sum / (n_rows * (n_classes - 1)),
    }


def evaluate_probabilities(
    probabilities: np.ndarray,
    targets: np.ndarray,
    detector_scores: np.ndarray | None = None,
) -> dict:
    """Report detection AP and joint/positive-only probability scores for counts 0–4.

    Conditional scores use only rows with positive true labels. If predicted
    positive mass is zero, use a uniform conditional distribution; if no true
    positives exist, return None for those scores. No threshold is fitted here.
    """
    values = validate_probabilities(probabilities)
    if values.shape[1] != N_CLASSES:
        msg = f"halving probabilities must contain exactly {N_CLASSES} classes"
        raise ValueError(msg)
    labels = _integer_vector(targets, "targets")
    if len(labels) != len(values) or labels.min() < 0 or labels.max() >= N_CLASSES:
        msg = f"targets must match probabilities and be between 0 and {N_CLASSES - 1}"
        raise ValueError(msg)
    scores = 1.0 - values[:, 0] if detector_scores is None else detector_scores
    scores = _validated_detector_scores(scores, len(values))
    positive = labels > 0

    conditional = None
    if positive.any():
        severity = values[positive, 1:].astype(np.float64, copy=True)
        mass = severity.sum(axis=1, keepdims=True)
        zero_mass = mass[:, 0] == 0.0
        severity[zero_mass] = 1.0 / (N_CLASSES - 1)
        mass[zero_mass] = 1.0
        severity /= mass
        conditional = _probability_metrics(severity, labels[positive] - 1)

    return {
        "ap": float(average_precision_score(positive, scores)) if positive.any() else 0.0,
        "prevalence": float(positive.mean()),
        "positive_examples": int(positive.sum()),
        "probability": _probability_metrics(values, labels),
        "conditional_positive_probability": conditional,
    }


def threshold_at_recall(targets: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """Fit the highest score threshold achieving at least the requested recall.

    Decisions use score >= threshold, so ties can exceed the requested recall.
    Fit on validation labels, then reuse the threshold unchanged on held-out data.
    """
    if isinstance(target_recall, (bool, np.bool_)) or not np.isfinite(target_recall) or not 0 < target_recall <= 1:
        msg = "target_recall must be in (0, 1]"
        raise ValueError(msg)
    labels = _integer_vector(targets, "targets")
    values = _validated_detector_scores(scores, len(labels))
    positive_scores = values[labels > 0]
    if not len(positive_scores):
        msg = "at least one positive target is required"
        raise ValueError(msg)
    required = int(np.ceil(float(target_recall) * len(positive_scores)))
    # Only one order statistic is needed, so avoid sorting the whole vector.
    index = len(positive_scores) - required
    positive_scores.partition(index)
    return float(positive_scores[index])


def actions_at_threshold(probabilities: np.ndarray, detector_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Choose zero below threshold, otherwise the most probable positive count.

    Scores equal to the threshold count as detections. Positive-class ties
    choose the smaller count. This policy differs from five-class joint argmax.
    """
    values = validate_probabilities(probabilities)
    if values.shape[1] != N_CLASSES:
        msg = f"halving probabilities must contain exactly {N_CLASSES} classes"
        raise ValueError(msg)
    scores = _validated_detector_scores(detector_scores, len(values))
    if isinstance(threshold, (bool, np.bool_)) or not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        msg = "threshold must be finite and in [0, 1]"
        raise ValueError(msg)
    positive_action = values[:, 1:].argmax(axis=1).astype(np.int64) + 1
    return np.where(scores >= threshold, positive_action, 0)


def halving_action_metrics(actions: np.ndarray, targets: np.ndarray) -> dict:
    """Build a true-count by chosen-count confusion matrix and summarise it.

    Both vectors contain aligned integer counts 0–4. Detection metrics group
    counts 1–4 together; exact/under/overprediction retain their ordering.
    """
    chosen = _integer_vector(actions, "actions")
    labels = _integer_vector(targets, "targets")
    if len(chosen) != len(labels):
        msg = "actions and targets must contain the same number of rows"
        raise ValueError(msg)
    if chosen.min() < 0 or chosen.max() >= N_CLASSES or labels.min() < 0 or labels.max() >= N_CLASSES:
        msg = f"actions and targets must be between 0 and {N_CLASSES - 1}"
        raise ValueError(msg)
    # Encode each (true, chosen) pair as a flat cell index for a single count pass.
    confusion = np.bincount(labels * N_CLASSES + chosen, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)
    return halving_confusion_metrics(confusion)


def halving_confusion_metrics(confusion: np.ndarray) -> dict:
    """Summarise a 5×5 count matrix, including matrices accumulated across files.

    Rows are true counts and columns are chosen counts. Exact, under- and
    overprediction rates use all examples as their denominator. Detection
    precision/recall return zero when their denominator is zero.
    """
    matrix = np.asarray(confusion)
    if matrix.shape != (N_CLASSES, N_CLASSES) or not np.issubdtype(matrix.dtype, np.integer):
        msg = f"confusion must be a {N_CLASSES} by {N_CLASSES} integer matrix"
        raise ValueError(msg)
    total = int(matrix.sum())
    if np.any(matrix < 0) or not total:
        msg = "confusion must contain non-negative counts and at least one row"
        raise ValueError(msg)

    true_positive = int(matrix[1:, 1:].sum())
    false_positive = int(matrix[0, 1:].sum())
    false_negative = int(matrix[1:, 0].sum())
    true_negative = int(matrix[0, 0])
    row, column = np.indices(matrix.shape)
    under = int(matrix[column < row].sum())
    over = int(matrix[column > row].sum())
    exact = int(np.trace(matrix))

    def divide(numerator: int, denominator: int) -> float:
        """Use zero as the reporting convention for an undefined detection ratio."""
        return numerator / denominator if denominator else 0.0

    return {
        "confusion": matrix.tolist(),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "detection_precision": divide(true_positive, true_positive + false_positive),
        "detection_recall": divide(true_positive, true_positive + false_negative),
        "exact_rate": exact / total,
        "underprediction_rate": under / total,
        "overprediction_rate": over / total,
    }


def evaluate_threshold(
    probabilities: np.ndarray,
    detector_scores: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict:
    """Apply a fixed detector threshold and return it alongside full action metrics."""
    actions = actions_at_threshold(probabilities, detector_scores, threshold)
    return {"threshold": float(threshold), "metrics": halving_action_metrics(actions, targets)}


def recall_action_frontier(
    probabilities: np.ndarray,
    detector_scores: np.ndarray,
    targets: np.ndarray,
    target_recalls: Sequence[float],
) -> list[dict]:
    """Fit and report detection thresholds for increasing validation recall targets.

    Return threshold, achieved recall, precision and false-positive/negative
    counts for each requested point. These are fitted on the supplied labels;
    they are not held-out estimates or measures of severity accuracy.
    """
    values = validate_probabilities(probabilities)
    labels = _integer_vector(targets, "targets")
    scores = _validated_detector_scores(detector_scores, len(values))
    if len(labels) != len(values):
        msg = "probabilities, detector_scores, and targets must contain matching rows"
        raise ValueError(msg)

    if values.shape[1] != N_CLASSES or labels.min() < 0 or labels.max() >= N_CLASSES:
        raise ValueError(f"expected {N_CLASSES} probability columns and targets in 0–{N_CLASSES - 1}")
    recalls = tuple(float(value) for value in target_recalls)
    if not recalls or any(not np.isfinite(value) or not 0 < value <= 1 for value in recalls):
        raise ValueError("target recalls must be nonempty and in (0, 1]")
    if any(left >= right for left, right in zip(recalls, recalls[1:])):
        raise ValueError("target recalls must be strictly increasing")

    # Sort each group once. Severity choices do not affect detection counts.
    positive = labels > 0
    positive_scores = np.sort(scores[positive])
    negative_scores = np.sort(scores[~positive])
    n_positive = len(positive_scores)
    if not n_positive:
        raise ValueError("at least one positive target is required")

    frontier = []
    for target_recall in recalls:
        required = int(np.ceil(target_recall * n_positive))
        threshold = float(positive_scores[n_positive - required])
        # 'left' counts scores strictly below threshold, preserving >= and ties.
        false_negative = int(np.searchsorted(positive_scores, threshold, side="left"))
        true_positive = n_positive - false_negative
        false_positive = len(negative_scores) - int(np.searchsorted(negative_scores, threshold, side="left"))
        frontier.append(
            {
                "target_recall": target_recall,
                "threshold": threshold,
                "achieved_recall": true_positive / n_positive,
                "precision": true_positive / (true_positive + false_positive),
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )
    return frontier
