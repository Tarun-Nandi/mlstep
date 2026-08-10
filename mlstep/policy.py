"""Architecture-independent policies for converting class probabilities to actions."""

from dataclasses import dataclass

import numpy as np

PROBABILITY_NDIM = 2
MIN_N_CLASSES = 2


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Validate and return a matrix of class probabilities."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != PROBABILITY_NDIM or not probabilities.shape[0] or probabilities.shape[1] < MIN_N_CLASSES:
        msg = "probabilities must have shape (n_rows, n_classes) with at least two classes"
        raise ValueError(msg)
    if not np.issubdtype(probabilities.dtype, np.number) or np.iscomplexobj(probabilities):
        msg = "probabilities must be real numeric values"
        raise ValueError(msg)
    if not np.isfinite(probabilities).all():
        msg = "probabilities must be finite"
        raise ValueError(msg)
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        msg = "probabilities must be in [0, 1]"
        raise ValueError(msg)
    row_sums = probabilities.sum(axis=1, dtype=np.float64)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-7):
        msg = "probability rows must sum to one"
        raise ValueError(msg)
    return probabilities


def _validate_max_action(max_action: int, n_classes: int) -> None:
    """Validate an inclusive maximum action against the class count."""
    if isinstance(max_action, bool) or not isinstance(max_action, (int, np.integer)):
        msg = "max_action must be an integer"
        raise ValueError(msg)
    if not 1 <= max_action < n_classes:
        msg = f"max_action must be between 1 and {n_classes - 1}"
        raise ValueError(msg)


def collapse_tail_class(probabilities: np.ndarray, max_action: int) -> np.ndarray:
    """Combine probability above an action cap into the final allowed action."""
    probabilities = validate_probabilities(probabilities)
    _validate_max_action(max_action, probabilities.shape[1])

    if max_action == probabilities.shape[1] - 1:
        return probabilities

    tail = probabilities[:, max_action:].sum(axis=1, dtype=np.float64, keepdims=True)
    return np.concatenate((probabilities[:, :max_action], tail), axis=1)


def boundary_probabilities(probabilities: np.ndarray, max_action: int) -> np.ndarray:
    """Calculate ``P(Y >= a)`` for every allowed action boundary."""
    probabilities = validate_probabilities(probabilities)
    _validate_max_action(max_action, probabilities.shape[1])

    # Build only the boundaries the policy can act on. This avoids allocating
    # both a collapsed probability matrix and a cumulative matrix for the
    # multi-million-row validation cache.
    boundaries = np.empty((len(probabilities), max_action + 1), dtype=np.float64)
    tail = np.zeros(len(probabilities), dtype=np.float64)
    for label in range(probabilities.shape[1] - 1, -1, -1):
        tail += probabilities[:, label]
        if label <= max_action:
            boundaries[:, label] = tail
    boundaries /= probabilities.sum(axis=1, dtype=np.float64, keepdims=True)
    # Action zero cannot overpredict by definition, so its lower-boundary
    # probability is exactly one rather than an approximately normalized sum.
    boundaries[:, 0] = 1.0
    return boundaries


def confidence_at_recall(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_recall: float,
    max_action: int,
) -> float:
    """Return the highest first-boundary confidence achieving target recall."""
    if not np.isfinite(target_recall) or not 0.0 < target_recall <= 1.0:
        msg = "target_recall must be in (0, 1]"
        raise ValueError(msg)

    probabilities = validate_probabilities(probabilities)
    _validate_max_action(max_action, probabilities.shape[1])
    targets = np.asarray(targets)
    if targets.ndim != 1 or len(targets) != len(probabilities):
        msg = "targets must be a one-dimensional array matching probabilities"
        raise ValueError(msg)
    positive = targets > 0
    if not positive.any():
        msg = "at least one positive target is required to select confidence"
        raise ValueError(msg)

    first_boundary = boundary_probabilities(probabilities, max_action)[:, 1]
    positive_confidences = np.sort(first_boundary[positive])[::-1]
    required = int(np.ceil(target_recall * len(positive_confidences)))
    return float(positive_confidences[required - 1])


@dataclass(frozen=True, slots=True)
class EpsilonPolicy:
    """Conservative ordinal policy based on cumulative class probability."""

    boundary_confidence: float
    max_action: int

    def __post_init__(self) -> None:
        """Validate scalar policy settings."""
        if not np.isfinite(self.boundary_confidence) or not 0.0 <= self.boundary_confidence <= 1.0:
            msg = "boundary_confidence must be finite and in [0, 1]"
            raise ValueError(msg)
        if isinstance(self.max_action, bool) or not isinstance(self.max_action, (int, np.integer)):
            msg = "max_action must be an integer"
            raise ValueError(msg)
        if self.max_action < 1:
            msg = "max_action must be at least one"
            raise ValueError(msg)

    @property
    def epsilon(self) -> float:
        """Return the model-implied overprediction-risk limit."""
        return 1.0 - self.boundary_confidence

    @classmethod
    def from_epsilon(cls, epsilon: float, max_action: int) -> "EpsilonPolicy":
        """Construct a policy from an epsilon risk limit."""
        if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            msg = "epsilon must be finite and in [0, 1]"
            raise ValueError(msg)
        return cls(boundary_confidence=1.0 - float(epsilon), max_action=max_action)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Convert joint probabilities into integer halving actions."""
        boundaries = boundary_probabilities(probabilities, self.max_action)
        passed = boundaries[:, 1:] >= self.boundary_confidence
        return passed.sum(axis=1, dtype=np.int64)
