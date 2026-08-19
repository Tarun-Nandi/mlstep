"""Deterministic negative samplers for extremely imbalanced training data.

The samplers in this module select row indices only.  They deliberately know
nothing about models, feature matrices, or hard-negative mining.  A caller can
therefore construct a fixed hard-negative pool separately and test the sampling
protocol without loading any training features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, gcd, log
from typing import Final

import numpy as np
import numpy.typing as npt

IndexArray = npt.NDArray[np.int64]
WeightArray = npt.NDArray[np.float32]

_ORDER_STREAM: Final = 1
_HARD_ORDER_STREAM: Final = 2
_EASY_ORDER_STREAM: Final = 3
_EPOCH_SHUFFLE_STREAM: Final = 4
_MONOTONICITY_CHECK_ROWS: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class SamplingState:
    """Serializable state needed to reproduce one epoch sample."""

    policy: str
    epoch: int
    shuffle_seed: int
    negative_offset: int | None = None
    hard_offset: int | None = None
    easy_offset: int | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        """Return a JSON-compatible representation of this epoch state."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpochSample:
    """One shuffled epoch of row indices and aligned detector-loss weights."""

    indices: IndexArray
    detector_weights: WeightArray
    state: SamplingState
    positive_rows: int
    negative_rows: int
    hard_negative_rows: int
    easy_negative_rows: int

    def __post_init__(self) -> None:
        """Reject internally inconsistent samples at their construction boundary."""
        if self.indices.ndim != 1 or self.detector_weights.ndim != 1:
            msg = "Epoch sample indices and weights must be one-dimensional"
            raise ValueError(msg)
        if len(self.indices) != len(self.detector_weights):
            msg = "Epoch sample indices and detector weights must be aligned"
            raise ValueError(msg)
        expected_rows = self.positive_rows + self.negative_rows
        if len(self.indices) != expected_rows:
            msg = "Epoch sample row counts do not match its arrays"
            raise ValueError(msg)
        if self.hard_negative_rows + self.easy_negative_rows != self.negative_rows:
            msg = "Hard and easy row counts must partition the sampled negatives"
            raise ValueError(msg)
        if not np.isfinite(self.detector_weights).all() or np.any(self.detector_weights <= 0.0):
            msg = "Detector weights must be finite and strictly positive"
            raise ValueError(msg)

    def metadata(self) -> dict[str, int | float | str | None]:
        """Return compact, JSON-compatible metadata for this epoch."""
        return {
            **self.state.to_dict(),
            "sampled_rows": len(self.indices),
            "positive_rows": self.positive_rows,
            "negative_rows": self.negative_rows,
            "hard_negative_rows": self.hard_negative_rows,
            "easy_negative_rows": self.easy_negative_rows,
            "detector_weight_sum": float(self.detector_weights.sum(dtype=np.float64)),
        }


def _canonical_indices(values: npt.ArrayLike, name: str) -> IndexArray:
    """Return a sorted, unique int64 copy of a nonempty row-index pool."""
    array = np.asarray(values)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional"
        raise ValueError(msg)
    if not np.issubdtype(array.dtype, np.integer):
        msg = f"{name} must contain integer row indices"
        raise TypeError(msg)
    if not len(array):
        msg = f"{name} must not be empty"
        raise ValueError(msg)

    canonical = np.array(array, dtype=np.int64, copy=True)
    if np.any(canonical < 0):
        msg = f"{name} must contain non-negative row indices"
        raise ValueError(msg)
    strictly_increasing = True
    for start in range(1, len(canonical), _MONOTONICITY_CHECK_ROWS):
        stop = min(len(canonical), start + _MONOTONICITY_CHECK_ROWS)
        if np.any(canonical[start:stop] <= canonical[start - 1 : stop - 1]):
            strictly_increasing = False
            break
    if not strictly_increasing:
        canonical.sort()
        if np.any(canonical[1:] == canonical[:-1]):
            msg = f"{name} must not contain duplicate row indices"
            raise ValueError(msg)
    return canonical


def _validate_disjoint(left: IndexArray, right: IndexArray, left_name: str, right_name: str) -> None:
    """Require two sorted unique pools to be disjoint."""
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
    locations = np.searchsorted(larger, smaller)
    in_bounds = locations < len(larger)
    if np.any(larger[locations[in_bounds]] == smaller[in_bounds]):
        msg = f"{left_name} and {right_name} must be disjoint"
        raise ValueError(msg)


def _validate_subset(subset: IndexArray, population: IndexArray, subset_name: str) -> npt.NDArray[np.int64]:
    """Return positions proving that ``subset`` belongs to ``population``."""
    locations = np.searchsorted(population, subset)
    in_bounds = locations < len(population)
    if not np.all(in_bounds):
        msg = f"{subset_name} must be a subset of negative_indices"
        raise ValueError(msg)
    if np.any(population[locations] != subset):
        msg = f"{subset_name} must be a subset of negative_indices"
        raise ValueError(msg)
    return locations


def _positive_integer(value: int, name: str) -> int:
    """Validate a positive integer argument without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return int(value)


def _nonnegative_integer(value: int, name: str) -> int:
    """Validate a non-negative integer argument without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        msg = f"{name} must be a non-negative integer"
        raise ValueError(msg)
    return int(value)


def _stream_seed(seed: int, stream: int, epoch: int = 0) -> int:
    """Derive an independent deterministic uint64 seed for one sampler stream."""
    sequence = np.random.SeedSequence((seed, stream, epoch))
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _shuffled_order(indices: IndexArray, seed: int, stream: int) -> IndexArray:
    """Shuffle a private, canonicalized index pool once in place."""
    np.random.default_rng(_stream_seed(seed, stream)).shuffle(indices)
    return indices


def _cyclic_take(order: IndexArray, size: int, epoch: int) -> tuple[IndexArray, int]:
    """Take a deterministic circular slice and return its starting offset."""
    offset = ((epoch - 1) * size) % len(order)
    stop = offset + size
    if stop <= len(order):
        return order[offset:stop], offset
    return np.concatenate((order[offset:], order[: stop - len(order)])), offset


def _epoch_number(epoch: int) -> int:
    """Validate the one-indexed epoch used for random-access sampling."""
    return _positive_integer(epoch, "epoch")


def _shuffle_aligned(
    indices: IndexArray,
    weights: WeightArray,
    seed: int,
    epoch: int,
) -> tuple[IndexArray, WeightArray, int]:
    """Shuffle indices and weights with the same deterministic permutation."""
    shuffle_seed = _stream_seed(seed, _EPOCH_SHUFFLE_STREAM, epoch)
    permutation = np.random.default_rng(shuffle_seed).permutation(len(indices))
    return indices[permutation], weights[permutation], shuffle_seed


def _base_protocol(
    policy: str,
    seed: int,
    n_positive: int,
    n_negative: int,
    negative_budget: int,
) -> dict[str, int | float | str]:
    """Build metadata shared by both population-corrected samplers."""
    return {
        "policy": policy,
        "version": 1,
        "seed": seed,
        "positive_population_rows": n_positive,
        "negative_population_rows": n_negative,
        "positive_rows_per_epoch": n_positive,
        "negative_rows_per_epoch": negative_budget,
        "sampled_rows_per_epoch": n_positive + negative_budget,
        "negative_sampling_fraction": negative_budget / n_negative,
        "population_prior_correction_delta_s": log(n_negative / negative_budget),
    }


class CyclicUniformNegativeSampler:
    """Keep every positive and cycle uniformly through a shuffled negative pool.

    Sampling is random-access by one-indexed epoch.  Consequently a resumed run
    needs to record only the run seed, configuration, and next epoch number.
    """

    policy: Final = "cyclic-uniform-negative-v1"

    def __init__(
        self,
        positive_indices: npt.ArrayLike,
        negative_indices: npt.ArrayLike,
        negative_budget: int,
        seed: int,
    ) -> None:
        """Initialize a deterministic uniform sampler.

        Parameters
        ----------
        positive_indices:
            Global row indices for all positive examples.  Every epoch retains
            all of them.
        negative_indices:
            Global row indices for the complete negative population.
        negative_budget:
            Exact number of negatives selected in every epoch.
        seed:
            Non-negative run seed controlling pool order and epoch shuffling.
        """
        self.seed = _nonnegative_integer(seed, "seed")
        self.positive_indices = _canonical_indices(positive_indices, "positive_indices")
        negative = _canonical_indices(negative_indices, "negative_indices")
        _validate_disjoint(self.positive_indices, negative, "positive_indices", "negative_indices")
        self.negative_budget = _positive_integer(negative_budget, "negative_budget")
        if self.negative_budget > len(negative):
            msg = "negative_budget must not exceed the negative population"
            raise ValueError(msg)
        self.negative_order = _shuffled_order(negative, self.seed, _ORDER_STREAM)

    def sample_epoch(self, epoch: int) -> EpochSample:
        """Return the complete, deterministically shuffled sample for ``epoch``."""
        epoch = _epoch_number(epoch)
        selected_negative, offset = _cyclic_take(self.negative_order, self.negative_budget, epoch)
        indices = np.concatenate((self.positive_indices, selected_negative))
        weights = np.ones(len(indices), dtype=np.float32)
        indices, weights, shuffle_seed = _shuffle_aligned(indices, weights, self.seed, epoch)
        state = SamplingState(
            policy=self.policy,
            epoch=epoch,
            shuffle_seed=shuffle_seed,
            negative_offset=offset,
        )
        return EpochSample(
            indices=indices,
            detector_weights=weights,
            state=state,
            positive_rows=len(self.positive_indices),
            negative_rows=self.negative_budget,
            hard_negative_rows=0,
            easy_negative_rows=self.negative_budget,
        )

    def metadata(self) -> dict[str, int | float | str]:
        """Return the fixed protocol metadata needed for provenance."""
        n_negative = len(self.negative_order)
        return {
            **_base_protocol(
                self.policy,
                self.seed,
                len(self.positive_indices),
                n_negative,
                self.negative_budget,
            ),
            "negative_order_seed": _stream_seed(self.seed, _ORDER_STREAM),
            "full_negative_coverage_by_epoch": ceil(n_negative / self.negative_budget),
            "negative_sequence_period_epochs": n_negative // gcd(n_negative, self.negative_budget),
            "detector_weight": 1.0,
        }


class StratifiedHardNegativeSampler:
    """Cycle through fixed hard/easy strata with exact importance correction.

    The hard pool must be a nonempty proper subset of the complete negative
    population.  The complement is constructed internally, ensuring that every
    negative has a known, strictly positive sampling probability.
    """

    policy: Final = "stratified-hard-easy-cyclic-negative-v1"

    def __init__(
        self,
        positive_indices: npt.ArrayLike,
        negative_indices: npt.ArrayLike,
        hard_negative_indices: npt.ArrayLike,
        negative_budget: int,
        hard_fraction: float,
        seed: int,
    ) -> None:
        """Initialize a deterministic, population-corrected hard sampler.

        Parameters
        ----------
        positive_indices:
            Global row indices for all positive examples.
        negative_indices:
            Global row indices for the complete negative population.
        hard_negative_indices:
            A fixed, training-only subset of ``negative_indices`` produced by a
            separate mining step.
        negative_budget:
            Exact total number of hard plus easy negatives per epoch.
        hard_fraction:
            Fraction of the epoch negative budget allocated to the hard stratum.
            Both strata must receive at least one row.
        seed:
            Non-negative run seed controlling stratum order and epoch shuffling.
        """
        self.seed = _nonnegative_integer(seed, "seed")
        self.positive_indices = _canonical_indices(positive_indices, "positive_indices")
        negative = _canonical_indices(negative_indices, "negative_indices")
        hard = _canonical_indices(hard_negative_indices, "hard_negative_indices")
        _validate_disjoint(self.positive_indices, negative, "positive_indices", "negative_indices")
        hard_locations = _validate_subset(hard, negative, "hard_negative_indices")
        if len(hard) == len(negative):
            msg = "hard_negative_indices must leave a nonempty easy-negative stratum"
            raise ValueError(msg)

        self.negative_budget = _positive_integer(negative_budget, "negative_budget")
        if self.negative_budget > len(negative):
            msg = "negative_budget must not exceed the negative population"
            raise ValueError(msg)
        if not np.isfinite(hard_fraction) or not 0.0 < hard_fraction < 1.0:
            msg = "hard_fraction must be finite and lie strictly between zero and one"
            raise ValueError(msg)

        self.requested_hard_fraction = float(hard_fraction)
        self.hard_budget = int(np.floor(self.negative_budget * hard_fraction + 0.5))
        self.easy_budget = self.negative_budget - self.hard_budget
        if self.hard_budget < 1 or self.easy_budget < 1:
            msg = "hard_fraction and negative_budget must allocate at least one row to each stratum"
            raise ValueError(msg)

        hard_membership = np.zeros(len(negative), dtype=np.bool_)
        hard_membership[hard_locations] = True
        easy = negative[~hard_membership]
        if self.hard_budget > len(hard):
            msg = "Hard-negative epoch budget must not exceed the fixed hard pool"
            raise ValueError(msg)
        if self.easy_budget > len(easy):
            msg = "Easy-negative epoch budget must not exceed the easy pool"
            raise ValueError(msg)

        self.hard_order = _shuffled_order(hard, self.seed, _HARD_ORDER_STREAM)
        self.easy_order = _shuffled_order(easy, self.seed, _EASY_ORDER_STREAM)
        self.negative_sampling_fraction = self.negative_budget / len(negative)
        self.hard_weight = self.negative_sampling_fraction * len(hard) / self.hard_budget
        self.easy_weight = self.negative_sampling_fraction * len(easy) / self.easy_budget

    def sample_epoch(self, epoch: int) -> EpochSample:
        """Return one aligned hard/easy importance-weighted epoch sample."""
        epoch = _epoch_number(epoch)
        selected_hard, hard_offset = _cyclic_take(self.hard_order, self.hard_budget, epoch)
        selected_easy, easy_offset = _cyclic_take(self.easy_order, self.easy_budget, epoch)
        indices = np.concatenate((self.positive_indices, selected_hard, selected_easy))
        weights = np.concatenate(
            (
                np.ones(len(self.positive_indices), dtype=np.float32),
                np.full(self.hard_budget, self.hard_weight, dtype=np.float32),
                np.full(self.easy_budget, self.easy_weight, dtype=np.float32),
            )
        )
        indices, weights, shuffle_seed = _shuffle_aligned(indices, weights, self.seed, epoch)
        state = SamplingState(
            policy=self.policy,
            epoch=epoch,
            shuffle_seed=shuffle_seed,
            hard_offset=hard_offset,
            easy_offset=easy_offset,
        )
        return EpochSample(
            indices=indices,
            detector_weights=weights,
            state=state,
            positive_rows=len(self.positive_indices),
            negative_rows=self.negative_budget,
            hard_negative_rows=self.hard_budget,
            easy_negative_rows=self.easy_budget,
        )

    def metadata(self) -> dict[str, int | float | str]:
        """Return fixed protocol, inclusion-rate, and weight metadata."""
        n_hard = len(self.hard_order)
        n_easy = len(self.easy_order)
        n_negative = n_hard + n_easy
        sampled_rows = len(self.positive_indices) + self.negative_budget
        sum_squared_weights = (
            len(self.positive_indices) + self.hard_budget * self.hard_weight**2 + self.easy_budget * self.easy_weight**2
        )
        return {
            **_base_protocol(
                self.policy,
                self.seed,
                len(self.positive_indices),
                n_negative,
                self.negative_budget,
            ),
            "hard_order_seed": _stream_seed(self.seed, _HARD_ORDER_STREAM),
            "easy_order_seed": _stream_seed(self.seed, _EASY_ORDER_STREAM),
            "hard_population_rows": n_hard,
            "easy_population_rows": n_easy,
            "hard_rows_per_epoch": self.hard_budget,
            "easy_rows_per_epoch": self.easy_budget,
            "requested_hard_fraction": self.requested_hard_fraction,
            "actual_hard_fraction": self.hard_budget / self.negative_budget,
            "hard_inclusion_rate_per_epoch": self.hard_budget / n_hard,
            "easy_inclusion_rate_per_epoch": self.easy_budget / n_easy,
            "hard_detector_weight": self.hard_weight,
            "easy_detector_weight": self.easy_weight,
            "positive_detector_weight": 1.0,
            "detector_weight_sum_per_epoch": float(sampled_rows),
            "importance_weight_effective_sample_size": sampled_rows**2 / sum_squared_weights,
            "hard_full_coverage_by_epoch": ceil(n_hard / self.hard_budget),
            "easy_full_coverage_by_epoch": ceil(n_easy / self.easy_budget),
            "hard_sequence_period_epochs": n_hard // gcd(n_hard, self.hard_budget),
            "easy_sequence_period_epochs": n_easy // gcd(n_easy, self.easy_budget),
        }
