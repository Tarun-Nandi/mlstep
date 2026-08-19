"""Architecture-independent policies for converting class probabilities to actions."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

PROBABILITY_NDIM = 2
MIN_N_CLASSES = 2
HALVING_N_CLASSES: Final = 5
HALVING_ACTION_DOMAIN: Final = tuple(range(HALVING_N_CLASSES))
POLICY_PREDICTION_CHUNK_ROWS: Final = 1_000_000
JOINT_MAP_POLICY_VERSION: Final = "joint-map-v1"
ASYMMETRIC_EXACT_OVERPREDICTION_POLICY_VERSION: Final = "asymmetric-exact-overprediction-v1"


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


def _validate_halving_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Validate the fixed five-class distribution used by the UKCA action policies."""
    probabilities = validate_probabilities(probabilities)
    if probabilities.shape[1] != HALVING_N_CLASSES:
        msg = f"halving probabilities must contain exactly {HALVING_N_CLASSES} classes"
        raise ValueError(msg)
    return probabilities


def _validate_overprediction_penalty(overprediction_penalty: float) -> float:
    """Return a finite non-negative overprediction penalty."""
    if isinstance(overprediction_penalty, (bool, np.bool_)):
        msg = "overprediction_penalty must be a finite non-negative number"
        raise ValueError(msg)
    try:
        value = float(overprediction_penalty)
    except (TypeError, ValueError) as error:
        msg = "overprediction_penalty must be a finite non-negative number"
        raise ValueError(msg) from error
    if not np.isfinite(value) or value < 0.0:
        msg = "overprediction_penalty must be a finite non-negative number"
        raise ValueError(msg)
    return value


def asymmetric_posterior_risks(
    probabilities: np.ndarray,
    overprediction_penalty: float,
) -> np.ndarray:
    """Return posterior risk for every exact-halving action.

    The per-outcome decision cost is

    ``1[action != target] + penalty * max(action - target, 0)``.

    Thus ordinary inexact predictions cost one, while overpredictions receive
    an additional penalty proportional to the number of excess halvings.  The
    output column for action ``a`` is the posterior expectation of that cost.
    """
    probabilities = _validate_halving_probabilities(probabilities)
    penalty = _validate_overprediction_penalty(overprediction_penalty)
    return _asymmetric_posterior_risks_validated(probabilities, penalty)


def _asymmetric_posterior_risks_validated(
    probabilities: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """Compute risks after the public boundary has validated its inputs."""
    # The exact-error contribution is sum_y q_y - q_a. Using the actual
    # float64 row sum preserves the defining expectation for float32 rows that
    # pass validation but differ from one by a rounding ULP.
    risks = np.empty(probabilities.shape, dtype=np.float64)
    np.negative(probabilities, out=risks, dtype=np.float64)
    risks += probabilities.sum(axis=1, dtype=np.float64)[:, None]
    if penalty == 0.0:
        return risks

    # E[(a-Y)_+] obeys the recurrence
    # E[(a-Y)_+] = E[((a-1)-Y)_+] + P(Y < a).
    expected_excess = np.zeros(len(probabilities), dtype=np.float64)
    cumulative_lower = probabilities[:, 0].astype(np.float64, copy=True)
    for action in range(1, HALVING_N_CLASSES):
        expected_excess += cumulative_lower
        risks[:, action] += penalty * expected_excess
        cumulative_lower += probabilities[:, action]
    return risks


@dataclass(frozen=True, slots=True)
class JointMapPolicy:
    """Choose the most probable exact halving class, including action zero."""

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Return joint maximum-a-posteriori actions with lower-action ties."""
        probabilities = _validate_halving_probabilities(probabilities)
        # NumPy returns the first maximum, which is the required deterministic
        # lower-action tie break because columns are ordered 0, ..., 4.
        return probabilities.argmax(axis=1).astype(np.int64, copy=False)

    def to_descriptor(self) -> dict[str, object]:
        """Return a JSON-serializable description of the frozen policy."""
        return {
            "version": JOINT_MAP_POLICY_VERSION,
            "action_domain": list(HALVING_ACTION_DOMAIN),
            "decision_rule": "argmax_a P(Y=a|x)",
            "tie_break": "lowest action",
            "prediction_requires_labels": False,
        }


@dataclass(frozen=True, slots=True)
class AsymmetricExactOverpredictionPolicy:
    """Minimize posterior exact-error plus excess-halving risk."""

    overprediction_penalty: float
    prediction_chunk_rows: int = POLICY_PREDICTION_CHUNK_ROWS

    def __post_init__(self) -> None:
        """Validate and canonicalize policy settings."""
        penalty = _validate_overprediction_penalty(self.overprediction_penalty)
        if (
            isinstance(self.prediction_chunk_rows, (bool, np.bool_))
            or not isinstance(self.prediction_chunk_rows, (int, np.integer))
            or self.prediction_chunk_rows < 1
        ):
            msg = "prediction_chunk_rows must be a positive integer"
            raise ValueError(msg)
        object.__setattr__(self, "overprediction_penalty", penalty)
        object.__setattr__(self, "prediction_chunk_rows", int(self.prediction_chunk_rows))

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Return posterior-risk-minimizing actions with lower-action ties."""
        probabilities = _validate_halving_probabilities(probabilities)
        actions = np.empty(len(probabilities), dtype=np.int64)
        for start in range(0, len(probabilities), self.prediction_chunk_rows):
            stop = min(start + self.prediction_chunk_rows, len(probabilities))
            risks = _asymmetric_posterior_risks_validated(
                probabilities[start:stop],
                self.overprediction_penalty,
            )
            # NumPy returns the first minimum, preserving the specified lower
            # action tie break without a floating-point perturbation.
            actions[start:stop] = risks.argmin(axis=1)
        return actions

    def to_descriptor(self) -> dict[str, object]:
        """Return a JSON-serializable description of the frozen policy."""
        return {
            "version": ASYMMETRIC_EXACT_OVERPREDICTION_POLICY_VERSION,
            "action_domain": list(HALVING_ACTION_DOMAIN),
            "cost": "1[action != target] + lambda * max(action - target, 0)",
            "lambda": self.overprediction_penalty,
            "decision_rule": "argmin_a E[cost(Y,a)|x]",
            "tie_break": "lowest action",
            "prediction_requires_labels": False,
        }


def policy_from_descriptor(
    descriptor: Mapping[str, object],
) -> JointMapPolicy | AsymmetricExactOverpredictionPolicy:
    """Reconstruct a supported recall-independent policy descriptor."""
    if not isinstance(descriptor, Mapping):
        msg = "policy descriptor must be a mapping"
        raise TypeError(msg)
    version = descriptor.get("version")
    if version == JOINT_MAP_POLICY_VERSION:
        return JointMapPolicy()
    if version == ASYMMETRIC_EXACT_OVERPREDICTION_POLICY_VERSION:
        if "lambda" not in descriptor:
            msg = "asymmetric policy descriptor is missing lambda"
            raise ValueError(msg)
        return AsymmetricExactOverpredictionPolicy(descriptor["lambda"])
    msg = f"unsupported action policy version: {version!r}"
    raise ValueError(msg)


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
    # Normalize with the same accumulated total used to build the boundaries.
    # A separate reduction can differ by one ULP and produce a boundary just
    # above one when the action-zero probability is exactly zero.
    boundaries /= tail[:, None]
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
