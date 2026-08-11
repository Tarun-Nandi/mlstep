"""Load UKCA data and prepare the train/validation/test split."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

# looks at absolute path for the data folder
DATA = Path(__file__).resolve().parent / "24hr_data"
N_CLASSES = 5  # hardcoded so the model doesnt resort to only predicting 3 halvings
SPIN_UP_STEPS = 1  # We ignore the first timestep due to a spin up effect
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.15
CHUNK_SIZE = 65_536  # rows per block when reducing over the design matrix
QUANTILE_FIT_SAMPLE_ROWS = 1_000_000
QUANTILE_FIT_SEED = 0
MATRIX_NDIM = 2
MIN_STRETCH_BINS = 2
GRID_SHAPE = (38, 72, 96)


@dataclass(frozen=True)
class Feature:
    """Define our input features and any transformations that need to be applied."""

    name: str
    channels: int  # for vector inputs like tracer[]
    transform: str


# Available preprocessing methods for experiments
PREPROCESSING_METHODS = ("standard", "robust", "stretch32", "stretch128", "ple64")

FEATURES = (
    Feature("temp", 1, "standard"),
    Feature("pres", 1, "standard"),
    # Feature("water_vapour", 1, "log1p"),
    Feature("cloud_frac", 1, "standard"),
    Feature("qcl", 1, "log1p"),
    Feature("cell_volume", 1, "standard"),
    Feature("stratflag", 1, "standard"),
    Feature("tracer", 83, "log1p"),
    Feature("rhs", 83, "arcsinh"),
    Feature("photol_rates", 60, "log1p"),
    Feature("wetrt", 34, "log1p"),
)
FEATURE_GROUPS = tuple(feature.name for feature in FEATURES)


@dataclass
class Preprocesser:
    """Convert atmospheric variables into inputs that a FCNN can train on reliably."""

    scale: np.ndarray
    # Since we standardise everything we check if mean=1 and srd = 0
    mean: np.ndarray
    std: np.ndarray
    features: tuple[Feature, ...]

    @classmethod
    def fit_transform(cls, x: np.ndarray, features) -> tuple["Preprocesser", np.ndarray]:
        """Apply transformations to training data and calculate statistics."""
        x = np.asarray(x, dtype=np.float32, order="c")  # matrix get stored in contigous memory blocks
        nonzero = np.zeros(x.shape[1], dtype=np.int64)
        absolute_sum = np.zeros(x.shape[1])

        for chunk in chunks(x):
            nonzero += np.count_nonzero(chunk, axis=0)
            absolute_sum += np.abs(chunk).sum(axis=0, dtype=np.float64)

        scale = np.where(nonzero > 0, absolute_sum / np.maximum(nonzero, 1), 1.0).astype(np.float32)
        # This applies the non linear transformations before everything gets standardised
        transform_in_place(x, scale, features)

        total = np.zeros(x.shape[1], dtype=np.float64)
        for chunk in chunks(x):
            total += chunk.sum(axis=0, dtype=np.float64)
        mean = (total / len(x)).astype(np.float32)  # This should be 0

        # calculating the transformed standard deviations (this should be 1)
        squared_error = np.zeros(x.shape[1], dtype=np.float64)
        for chunk in chunks(x):
            squared_error += np.square(chunk - mean).sum(axis=0, dtype=np.float64)
        std = np.sqrt(squared_error / len(x)).astype(np.float32)
        std[std == 0] = 1.0  # a constant feature has 0 std so we make it 1 to allow for safe divisions

        # actually standardising the training matrix
        x -= mean
        x /= std
        return cls(scale, mean, std, features), x

    def transform(self, x: np.ndarray, *, copy: bool = True) -> np.ndarray:
        """Transform new data like our validation/test sets.

        Uses the scale learned from the training data to prevent leakage.
        """
        out = np.array(x, dtype=np.float32, order="c", copy=copy)  # once again a contigious array
        transform_in_place(out, self.scale, self.features)
        # new data is standardised using the training mean and std
        out -= self.mean
        out /= self.std
        return out

    def state(self) -> dict[str, np.ndarray]:
        """Retruns the learned numerical state(data statistics) to reproduce preprocessing."""
        return {
            "scale": self.scale,
            "mean": self.mean,
            "std": self.std,
        }


def _physical_transform_fit_in_place(x: np.ndarray, features: tuple[Feature, ...]) -> np.ndarray:
    """Fit the existing physical scales and transform one training matrix."""
    nonzero = np.zeros(x.shape[1], dtype=np.int64)
    absolute_sum = np.zeros(x.shape[1], dtype=np.float64)
    for chunk in chunks(x):
        nonzero += np.count_nonzero(chunk, axis=0)
        absolute_sum += np.abs(chunk).sum(axis=0, dtype=np.float64)

    scale = np.where(nonzero > 0, absolute_sum / np.maximum(nonzero, 1), 1.0).astype(np.float32)
    transform_in_place(x, scale, features)
    return scale


def _deterministic_quantile_sample(
    x: np.ndarray,
    *,
    max_rows: int = QUANTILE_FIT_SAMPLE_ROWS,
    seed: int = QUANTILE_FIT_SEED,
) -> np.ndarray:
    """Return a bounded, reproducible training-row sample for quantile fitting.

    One row is selected from each of ``max_rows`` contiguous strata. The sorted
    indices preserve mostly sequential access when ``x`` is backed by a memmap,
    while the fixed random offset within each stratum avoids periodic sampling
    artefacts in spatially ordered fields.
    """
    if x.ndim != MATRIX_NDIM or not len(x):
        msg = "quantile fitting requires a non-empty two-dimensional matrix"
        raise ValueError(msg)
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        msg = "max_rows must be a positive integer"
        raise ValueError(msg)
    if len(x) <= max_rows:
        return np.asarray(x)

    edges = np.floor_divide(
        np.arange(max_rows + 1, dtype=np.int64) * len(x),
        max_rows,
    )
    widths = np.diff(edges)
    generator = np.random.default_rng(seed)
    offsets = np.floor(generator.random(max_rows) * widths).astype(np.int64)
    indices = edges[:-1] + offsets
    return np.array(x[indices], dtype=np.float32, order="c", copy=True)


def _column_extrema(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact per-column extrema with bounded working memory."""
    minimum = np.full(x.shape[1], np.inf, dtype=np.float32)
    maximum = np.full(x.shape[1], -np.inf, dtype=np.float32)
    for chunk in chunks(x):
        np.minimum(minimum, np.min(chunk, axis=0), out=minimum)
        np.maximum(maximum, np.max(chunk, axis=0), out=maximum)
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        msg = "preprocessing inputs must contain only finite values"
        raise ValueError(msg)
    return minimum, maximum


def _smooth_clip_in_place(x: np.ndarray, threshold: float) -> None:
    """Apply RealMLP's smooth clipping ``x / sqrt(1 + (x/t)^2)``."""
    if not math.isfinite(threshold) or threshold < 0.0:
        msg = "clip_threshold must be finite and non-negative"
        raise ValueError(msg)
    if threshold == 0.0:
        return

    for chunk in chunks(x):
        denominator = np.empty_like(chunk)
        np.divide(chunk, threshold, out=denominator)
        np.hypot(denominator, 1.0, out=denominator)
        np.divide(chunk, denominator, out=chunk)


def _stretch_column(values: np.ndarray, boundaries: np.ndarray, n_bins: int) -> np.ndarray:
    """Map one column monotonically through quantile-bin interpolation."""
    bin_indices = np.searchsorted(boundaries[1:-1], values, side="right")
    bin_left = boundaries[bin_indices]
    bin_right = boundaries[bin_indices + 1]
    bin_width = bin_right - bin_left

    position = np.zeros_like(values, dtype=np.float32)
    np.subtract(values, bin_left, out=position)
    np.divide(position, bin_width, out=position, where=bin_width > 0.0)
    # ``side='right'`` skips interior duplicate knots. Remaining zero-width
    # intervals come from constant features or repeated endpoints; retain the
    # expected lower/upper tail ordering for values outside the training range.
    zero_width = bin_width <= 0.0
    position[zero_width] = values[zero_width] >= bin_right[zero_width]
    np.clip(position, 0.0, 1.0, out=position)
    np.add(position, bin_indices, out=position, casting="unsafe")
    position /= n_bins
    return position


@dataclass
class RobustPreprocesser:
    """Convert atmospheric variables using median/IQR robust scaling with smooth clipping.

    This preprocessing is less sensitive to extreme outliers than mean/std scaling.
    It follows RealMLP: median centering, IQR scaling with a range fallback,
    and smooth clipping to a bounded interval.
    """

    scale: np.ndarray
    median: np.ndarray
    iqr: np.ndarray
    clip_threshold: float
    features: tuple[Feature, ...]
    fit_sample_rows: int

    @classmethod
    def fit_transform(
        cls, x: np.ndarray, features, clip_threshold: float = 3.0
    ) -> tuple["RobustPreprocesser", np.ndarray]:
        """Apply transformations and calculate robust statistics."""
        x = np.asarray(x, dtype=np.float32, order="c")
        scale = _physical_transform_fit_in_place(x, features)

        minimum, maximum = _column_extrema(x)
        fit_sample = _deterministic_quantile_sample(x)
        fit_sample_rows = len(fit_sample)
        quartiles = np.percentile(
            fit_sample,
            (25.0, 50.0, 75.0),
            axis=0,
            method="linear",
        ).astype(np.float32)
        del fit_sample
        q25, median, q75 = quartiles
        iqr = q75 - q25

        # RealMLP falls back to scaling a non-constant zero-IQR column into a
        # width-two interval. Constant columns remain zero after centering.
        zero_iqr = iqr == 0.0
        data_range = maximum - minimum
        iqr[zero_iqr & (data_range > 0.0)] = data_range[zero_iqr & (data_range > 0.0)] / 2.0
        iqr[iqr == 0.0] = 1.0

        # Robust scaling: subtract median, divide by IQR
        x -= median
        x /= iqr

        _smooth_clip_in_place(x, clip_threshold)

        return cls(scale, median, iqr, clip_threshold, features, fit_sample_rows), x

    def transform(self, x: np.ndarray, *, copy: bool = True) -> np.ndarray:
        """Transform new data using learned robust statistics."""
        out = np.array(x, dtype=np.float32, order="c", copy=copy)
        transform_in_place(out, self.scale, self.features)
        out -= self.median
        out /= self.iqr
        _smooth_clip_in_place(out, self.clip_threshold)
        return out

    def state(self) -> dict[str, np.ndarray]:
        """Return learned robust statistics."""
        return {
            "scale": self.scale,
            "median": self.median,
            "iqr": self.iqr,
            "clip_threshold": np.array([self.clip_threshold]),
            "fit_sample_rows": np.array([self.fit_sample_rows]),
            "fit_sample_max_rows": np.array([QUANTILE_FIT_SAMPLE_ROWS]),
            "fit_sample_seed": np.array([QUANTILE_FIT_SEED]),
            "quantile_method": np.array("linear"),
        }


@dataclass
class StretchPreprocesser:
    """Apply unsupervised piecewise-linear CDF-like stretch transformation.

    Based on the Tabular Numeric Stretch Transformation paper.
    Divides each feature into quantile bins and applies a CDF-like stretch.
    """

    scale: np.ndarray
    quantiles: np.ndarray  # Shape: (n_bins + 1, n_features)
    n_bins: int
    features: tuple[Feature, ...]
    mean: np.ndarray
    std: np.ndarray
    fit_sample_rows: int

    @classmethod
    def fit_transform(cls, x: np.ndarray, features, n_bins: int = 128) -> tuple["StretchPreprocesser", np.ndarray]:
        """Fit train-only quantile bins, stretch, and standardize each column."""
        if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < MIN_STRETCH_BINS:
            msg = "n_bins must be an integer of at least two"
            raise ValueError(msg)
        x = np.asarray(x, dtype=np.float32, order="c")
        scale = _physical_transform_fit_in_place(x, features)

        minimum, maximum = _column_extrema(x)
        fit_sample = _deterministic_quantile_sample(x)
        fit_sample_rows = len(fit_sample)
        quantile_probs = np.linspace(0.0, 100.0, n_bins + 1)
        quantiles = np.percentile(
            fit_sample,
            quantile_probs,
            axis=0,
            method="linear",
        ).astype(np.float32)
        del fit_sample
        # Preserve the complete observed training range even when the interior
        # quantiles are estimated from the bounded row sample.
        quantiles[0] = minimum
        quantiles[-1] = maximum
        np.maximum.accumulate(quantiles, axis=0, out=quantiles)

        # Apply stretch transformation
        stretched = np.empty_like(x)
        n_features = x.shape[1]
        mean = np.empty(n_features, dtype=np.float32)
        std = np.empty(n_features, dtype=np.float32)

        for feat_idx in range(n_features):
            stretched_pos = _stretch_column(x[:, feat_idx], quantiles[:, feat_idx], n_bins)
            mean[feat_idx] = np.mean(stretched_pos, dtype=np.float64)
            std[feat_idx] = np.std(stretched_pos, dtype=np.float64)
            if std[feat_idx] == 0.0:
                std[feat_idx] = 1.0
            stretched_pos -= mean[feat_idx]
            stretched_pos /= std[feat_idx]
            stretched[:, feat_idx] = stretched_pos

        return cls(scale, quantiles, n_bins, features, mean, std, fit_sample_rows), stretched

    def transform(self, x: np.ndarray, *, copy: bool = True) -> np.ndarray:
        """Transform new data using learned quantiles."""
        out = np.array(x, dtype=np.float32, order="c", copy=copy)
        transform_in_place(out, self.scale, self.features)

        stretched = np.empty_like(out)
        n_features = out.shape[1]

        for feat_idx in range(n_features):
            stretched_pos = _stretch_column(out[:, feat_idx], self.quantiles[:, feat_idx], self.n_bins)
            stretched_pos -= self.mean[feat_idx]
            stretched_pos /= self.std[feat_idx]
            stretched[:, feat_idx] = stretched_pos

        return stretched

    def state(self) -> dict[str, np.ndarray]:
        """Return learned quantile boundaries."""
        return {
            "scale": self.scale,
            "quantiles": self.quantiles,
            "n_bins": np.array([self.n_bins]),
            "mean": self.mean,
            "std": self.std,
            "fit_sample_rows": np.array([self.fit_sample_rows]),
            "fit_sample_max_rows": np.array([QUANTILE_FIT_SAMPLE_ROWS]),
            "fit_sample_seed": np.array([QUANTILE_FIT_SEED]),
            "quantile_method": np.array("linear"),
        }


FittedPreprocesser = Preprocesser | RobustPreprocesser | StretchPreprocesser
_PHYSICAL_STATE_KEYS = frozenset({"scale", "mean", "std", "features"})
_ROBUST_STATE_KEYS = frozenset(
    {
        "scale",
        "median",
        "iqr",
        "clip_threshold",
        "features",
        "fit_sample_rows",
        "fit_sample_max_rows",
        "fit_sample_seed",
        "quantile_method",
    }
)
_STRETCH_STATE_KEYS = frozenset(
    {
        "scale",
        "quantiles",
        "n_bins",
        "mean",
        "std",
        "features",
        "fit_sample_rows",
        "fit_sample_max_rows",
        "fit_sample_seed",
        "quantile_method",
    }
)


def _validate_feature_state(raw_features: np.ndarray, features: tuple[Feature, ...]) -> int:
    """Validate and count the exact feature layout recorded in a state file."""
    if not features or any(not isinstance(feature, Feature) for feature in features):
        msg = "features must be a non-empty tuple of Feature objects"
        raise ValueError(msg)
    if any(
        isinstance(feature.channels, bool)
        or not isinstance(feature.channels, int)
        or feature.channels < 1
        or feature.transform not in {"standard", "log1p", "arcsinh"}
        for feature in features
    ):
        msg = "features contain an invalid channel count or transform"
        raise ValueError(msg)

    if raw_features.shape != (len(features),) or raw_features.dtype != np.dtype(object):
        msg = "preprocessor feature metadata has the wrong shape or dtype"
        raise ValueError(msg)
    saved_features = raw_features.tolist()
    expected_features = [
        {"name": feature.name, "channels": feature.channels, "transform": feature.transform} for feature in features
    ]
    if any(not isinstance(feature, dict) for feature in saved_features) or saved_features != expected_features:
        msg = "preprocessor feature metadata does not match the requested feature layout"
        raise ValueError(msg)
    return sum(feature.channels for feature in features)


def _load_float_state_array(
    state: np.lib.npyio.NpzFile,
    name: str,
    shape: tuple[int, ...],
    *,
    positive: bool = False,
) -> np.ndarray:
    """Load one finite floating-point state array with its required shape."""
    value = np.asarray(state[name])
    if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
        msg = f"preprocessor state {name!r} must be a floating-point array with shape {shape}"
        raise ValueError(msg)
    value = np.array(value, dtype=np.float32, copy=True)
    if not np.isfinite(value).all():
        msg = f"preprocessor state {name!r} must contain only finite values"
        raise ValueError(msg)
    if positive and np.any(value <= 0.0):
        msg = f"preprocessor state {name!r} must contain only positive values"
        raise ValueError(msg)
    return value


def _load_integer_state_scalar(state: np.lib.npyio.NpzFile, name: str) -> int:
    """Load one integer saved using the state-file scalar convention."""
    value = np.asarray(state[name])
    if value.shape != (1,) or not np.issubdtype(value.dtype, np.integer):
        msg = f"preprocessor state {name!r} must be a one-element integer array"
        raise ValueError(msg)
    return int(value[0])


def _validate_quantile_fit_state(state: np.lib.npyio.NpzFile) -> int:
    """Validate shared provenance for sampled training-only quantile fits."""
    sample_rows = _load_integer_state_scalar(state, "fit_sample_rows")
    max_rows = _load_integer_state_scalar(state, "fit_sample_max_rows")
    seed = _load_integer_state_scalar(state, "fit_sample_seed")
    method = np.asarray(state["quantile_method"])
    if method.shape != () or not np.issubdtype(method.dtype, np.str_) or str(method) != "linear":
        msg = "preprocessor state 'quantile_method' must be the scalar string 'linear'"
        raise ValueError(msg)
    if max_rows != QUANTILE_FIT_SAMPLE_ROWS or seed != QUANTILE_FIT_SEED:
        msg = "preprocessor quantile-fit cap or seed does not match the supported training protocol"
        raise ValueError(msg)
    if not 1 <= sample_rows <= max_rows:
        msg = "preprocessor state 'fit_sample_rows' must be in the supported sample range"
        raise ValueError(msg)
    return sample_rows


def load_fitted_preprocessor(
    state_path: Path,
    method: str,
    *,
    features: tuple[Feature, ...] = FEATURES,
) -> FittedPreprocesser:
    """Load a saved train-fitted transform for held-out raw feature matrices.

    Parameters
    ----------
    state_path
        Path to the ``preprocessor_state.npz`` artifact written by the cache builder.
    method
        The cache preprocessing method: ``physical``, ``ple64``, ``robust``,
        ``stretch32`` or ``stretch128``.
    features
        Expected ordered raw-feature specification. The saved specification must
        match exactly, preventing a transform from being applied to reordered data.

    Returns
    -------
    FittedPreprocesser
        A transform reconstructed solely from training-fitted state. Calling
        ``transform`` does not fit or update any statistics.
    """
    supported_methods = {"physical", "ple64", "robust", "stretch32", "stretch128"}
    if method not in supported_methods:
        msg = f"cannot load fitted preprocessing state for method {method!r}"
        raise ValueError(msg)

    path = Path(state_path)
    with np.load(path, allow_pickle=True) as state:
        expected_keys = _PHYSICAL_STATE_KEYS
        if method == "robust":
            expected_keys = _ROBUST_STATE_KEYS
        elif method.startswith("stretch"):
            expected_keys = _STRETCH_STATE_KEYS
        actual_keys = frozenset(state.files)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            msg = f"preprocessor state schema mismatch: missing={missing}, unexpected={unexpected}"
            raise ValueError(msg)

        feature_tuple = tuple(features)
        n_features = _validate_feature_state(state["features"], feature_tuple)
        vector_shape = (n_features,)
        scale = _load_float_state_array(state, "scale", vector_shape, positive=True)

        if method in {"physical", "ple64"}:
            mean = _load_float_state_array(state, "mean", vector_shape)
            std = _load_float_state_array(state, "std", vector_shape, positive=True)
            return Preprocesser(scale, mean, std, feature_tuple)

        sample_rows = _validate_quantile_fit_state(state)
        if method == "robust":
            median = _load_float_state_array(state, "median", vector_shape)
            iqr = _load_float_state_array(state, "iqr", vector_shape, positive=True)
            clip = _load_float_state_array(state, "clip_threshold", (1,))
            if clip[0] < 0.0:
                msg = "preprocessor state 'clip_threshold' must be non-negative"
                raise ValueError(msg)
            return RobustPreprocesser(scale, median, iqr, float(clip[0]), feature_tuple, sample_rows)

        n_bins = _load_integer_state_scalar(state, "n_bins")
        expected_n_bins = int(method.removeprefix("stretch"))
        if n_bins != expected_n_bins or n_bins < MIN_STRETCH_BINS:
            msg = f"preprocessor state n_bins={n_bins} is inconsistent with method {method!r}"
            raise ValueError(msg)
        quantiles = _load_float_state_array(state, "quantiles", (n_bins + 1, n_features))
        if np.any(np.diff(quantiles, axis=0) < 0.0):
            msg = "preprocessor state quantiles must be nondecreasing in every feature"
            raise ValueError(msg)
        mean = _load_float_state_array(state, "mean", vector_shape)
        std = _load_float_state_array(state, "std", vector_shape, positive=True)
        return StretchPreprocesser(scale, quantiles, n_bins, feature_tuple, mean, std, sample_rows)


def chunks(x: np.ndarray):
    """Split the data into chunks for faster preprocessing."""
    for start in range(0, len(x), CHUNK_SIZE):
        yield x[start : start + CHUNK_SIZE]


def feature_names(features: tuple[Feature, ...]) -> list[str]:
    """Return a stable name for each column so we can easily identify each feature."""
    names = []
    for feature in features:
        if feature.channels == 1:  # all scalar inputs use their ordinary name
            names.append(feature.name)
        else:
            names.extend(
                f"{feature.name}[{index}]" for index in range(feature.channels)
            )  # all vector inputs have their appropiate channel like rhs[20]
    return names


def feature_group_names(features: tuple[Feature, ...]) -> list[str]:
    """Return the parent group for each column.

    This lets xgboost determine the overall importance of vector inputs rather
    than looking at the individual channels only.
    """
    return [feature.name for feature in features for _ in range(feature.channels)]


def halving_labels(ncsteps: np.ndarray) -> np.ndarray:
    """Convert the ncsteps data into the no of halvings labels."""
    values = np.asarray(ncsteps)
    if not values.size:
        msg = "ncsteps must contain at least one value"
        raise ValueError(msg)
    if not np.isfinite(values).all() or np.any(values <= 0):
        msg = "ncsteps values must be finite and positive"
        raise ValueError(msg)
    # we take the log of ncsteps to find the number of halvings
    halvings = np.log2(values)
    rounded = np.rint(halvings)  # np.log2 is a floating point calculations and we want integers
    if not np.allclose(halvings, rounded, rtol=0, atol=1e-12):
        msg = "ncsteps values must be powers of two"
        raise ValueError(msg)
    labels = rounded.astype(np.int64).ravel()
    if labels.min() < 0 or labels.max() >= N_CLASSES:
        msg = f"halving labels must be in between 0 and {N_CLASSES - 1}"
        raise ValueError(msg)
    return labels


def discover_timesteps(data_dir: Path) -> tuple[int, ...]:
    """Read the available timesteps from the data rather than a hardcoded range.

    The split then follows whatever is on disk, so the same code works on the 24
    hour set and on a longer run. The first SPIN_UP_STEPS are dropped because the
    solver has not settled yet.
    """
    steps = sorted(int(path.stem.rsplit("_", 1)[-1]) for path in data_dir.glob("ncsteps_*.nc"))[SPIN_UP_STEPS:]
    if not steps:
        msg = f"No ncsteps_*.nc files found in {data_dir}."
        raise ValueError(msg)
    return tuple(steps)


def split_timesteps(
    timesteps: tuple[int, ...],
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split the steps we find into train/val/test sets.

    The split is chronological so validation and test come from times the model
    has never seen. A random split would put neighbouring timesteps on either
    side of the boundary and leak the same data into both.
    """
    test_fraction = 1.0 - train_fraction - validation_fraction
    fractions = (train_fraction, validation_fraction, test_fraction)
    if not all(math.isfinite(fraction) and fraction > 0 for fraction in fractions):
        msg = "train, validation and test fractions must all be positive"
        raise ValueError(msg)

    # Allocate whole timesteps using the largest-remainder method. This is
    # closer to the requested fractions than independently rounding or taking
    # every ceiling, while ensuring the three counts sum exactly to the input.
    exact_counts = tuple(len(timesteps) * fraction for fraction in fractions)
    counts = [math.floor(count) for count in exact_counts]
    remaining = len(timesteps) - sum(counts)
    remainders = tuple(exact - count for exact, count in zip(exact_counts, counts, strict=True))
    # On an exact tie, protect the later held-out split first: test, then
    # validation, then training.
    allocation_order = sorted(range(len(counts)), key=lambda index: (remainders[index], index), reverse=True)
    for index in allocation_order[:remaining]:
        counts[index] += 1

    n_train, n_validation, n_test = counts
    if min(counts) == 0:
        msg = "not enough timesteps to create non-empty train, validation and test splits"
        raise ValueError(msg)
    train = timesteps[:n_train]
    validation = timesteps[n_train : n_train + n_validation]
    test = timesteps[n_train + n_validation : n_train + n_validation + n_test]
    return train, validation, test


def load_train_validation(
    data_dir: Path,
    features: tuple[Feature, ...],
    train_steps: tuple[int, ...],
    validation_steps: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the derived train and validation splits."""
    train_x, train_y = load_timesteps(data_dir, train_steps, features)
    val_x, val_y = load_timesteps(data_dir, validation_steps, features)
    return train_x, train_y, val_x, val_y


def load_labels(data_dir: Path, timesteps: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Load labels without allocating the corresponding feature matrix."""
    targets = [halving_labels(load_variable(data_dir, "ncsteps", timestep)) for timestep in timesteps]
    return np.concatenate(targets), tuple(len(target) for target in targets)


def _selected_row_layout(
    timesteps: tuple[int, ...],
    selected_indices: np.ndarray,
    rows_per_timestep: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group selected rows by their source timestep."""
    row_counts = np.asarray(rows_per_timestep)
    indices = np.asarray(selected_indices)
    if row_counts.ndim != 1 or len(row_counts) != len(timesteps) or np.any(row_counts < 0):
        msg = "rows_per_timestep must contain one non-negative count per timestep"
        raise ValueError(msg)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        msg = "selected_indices must be a one-dimensional integer array"
        raise ValueError(msg)
    if len(np.unique(indices)) != len(indices):
        msg = "selected_indices must not contain duplicates"
        raise ValueError(msg)
    total_rows = int(row_counts.sum())
    if np.any(indices < 0) or np.any(indices >= total_rows):
        msg = f"selected_indices must be in range [0, {total_rows})"
        raise IndexError(msg)
    # Sorting groups disk reads by timestep. destination_rows maps each sorted
    # source row back to its original, possibly shuffled output position.
    destination_rows = np.argsort(indices, kind="stable")
    sorted_indices = indices[destination_rows]
    offsets = np.empty(len(row_counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(row_counts, out=offsets[1:])
    return row_counts, destination_rows, sorted_indices, offsets


def load_selected_rows(
    data_dir: Path,
    timesteps: tuple[int, ...],
    features: tuple[Feature, ...],
    selected_indices: np.ndarray,
    rows_per_timestep: tuple[int, ...],
) -> np.ndarray:
    """Load selected feature rows while preserving their requested order."""
    row_counts, destination_rows, sorted_indices, offsets = _selected_row_layout(
        timesteps, selected_indices, rows_per_timestep
    )
    n_columns = sum(feature.channels for feature in features)
    out = np.empty((len(sorted_indices), n_columns), dtype=np.float32)
    if not len(sorted_indices):
        return out

    loaded_rows = 0
    for position, timestep in enumerate(timesteps):
        row_start, row_stop = offsets[position : position + 2]
        selected_start = np.searchsorted(sorted_indices, row_start, side="left")
        selected_stop = np.searchsorted(sorted_indices, row_stop, side="left")
        if selected_start == selected_stop:
            continue

        destinations = destination_rows[selected_start:selected_stop]
        local_rows = sorted_indices[selected_start:selected_stop] - row_start
        loaded_rows += len(destinations)
        row_count = int(row_counts[position])
        column = 0
        for feature in features:
            values = load_variable(data_dir, feature.name, timestep, channels=feature.channels)
            expected_size = feature.channels * row_count
            if values.size != expected_size:
                msg = f"{feature.name}_{timestep}.nc has {values.size:,} values; expected {expected_size:,}"
                raise ValueError(msg)
            next_column = column + feature.channels
            source = values.reshape(feature.channels, row_count).T
            out[destinations, column:next_column] = source[local_rows]
            column = next_column

    if loaded_rows != len(sorted_indices):
        msg = f"loaded {loaded_rows:,} selected rows; expected {len(sorted_indices):,}"
        raise RuntimeError(msg)
    return out


def load_timesteps(
    data_dir: Path, timesteps: tuple[int, ...], features: tuple[Feature, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Read the appropriate NetCDF files and load the data."""
    y, rows_per_timestep = load_labels(data_dir, timesteps)
    n_columns = sum(feature.channels for feature in features)  # calculating number of model input columns
    x = np.empty((len(y), n_columns), dtype=np.float32)  # Allocate the complete feature matrix once
    # Build the input feature matrix and load all the features
    row_start = 0
    for timestep, row_count in zip(timesteps, rows_per_timestep, strict=True):
        row_stop = row_start + row_count
        block = x[row_start:row_stop]
        column = 0

        for feature in features:
            values = load_variable(data_dir, feature.name, timestep, channels=feature.channels)
            expected_size = feature.channels * row_count
            if values.size != expected_size:
                msg = f"{feature.name}_{timestep}.nc has {values.size:,} values; expected {expected_size:,}"
                raise ValueError(msg)
            next_column = column + feature.channels
            block[:, column:next_column] = values.reshape(feature.channels, -1).T  # transpose to the correct format
            column = next_column

        row_start = row_stop

    return x, y


# perhaps make negative ratio a cli argument
def random_undersampling(positive: np.ndarray, negative: np.ndarray, negative_ratio: int, seed: int) -> np.ndarray:
    """Keep every positive and sample a fixed ratio of negatives.

    Instead of processing the entire dataset (containing mostly 0 halvings) we use
    undersampling where a new negative sample is detected each epoch using a
    different seed so that in each epoch, positive examples appear often enough to
    influence the loss and so the network can encounter different negatives over
    training.
    - also reduces memory access --> faster epochs
    **BUT the model sees a higher positive proportion of positives that exists in
    reality so its output should not automatically be interpreted as the true
    probability of halving.
    """
    n_negative = min(len(negative), negative_ratio * len(positive))
    generator = np.random.default_rng(seed)
    sampled_negative = generator.choice(
        negative,
        size=n_negative,
        replace=False,  # ensure the same row cannot be selected more than once
    )
    indices = np.concatenate([positive, sampled_negative])
    generator.shuffle(indices)
    return indices


def training_index_pools(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute row-index pools once for repeated negative sampling."""
    labels = np.asarray(labels)
    return np.flatnonzero(labels > 0), np.flatnonzero(labels == 0)


def class_counts(labels: np.ndarray) -> list[int]:
    """Count all five halving classes."""
    return np.bincount(
        np.asarray(labels, dtype=np.int64),
        minlength=N_CLASSES,
    ).tolist()


def load_variable(data_dir: Path, name: str, timestep: int, channels: int = 1) -> np.ndarray:
    """Load the variables data into memory and return them as a NumPy array."""
    path = data_dir / f"{name}_{timestep}.nc"
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_dims = ("z", "y", "x") if channels == 1 else ("s", "z", "y", "x")
    expected_shape = GRID_SHAPE if channels == 1 else (channels, *GRID_SHAPE)
    with xr.open_dataset(path) as dataset:
        variable = dataset[name]
        variable = variable.transpose(*expected_dims)
        if variable.shape != expected_shape:
            msg = f"{path} has shape {variable.shape}; expected {expected_shape}"
            raise ValueError(msg)
        return np.asarray(variable.values)


def transform_in_place(
    x: np.ndarray,
    scale: np.ndarray,
    features: tuple[Feature, ...],
) -> None:
    """Define the transformations that can be applied."""
    column = 0
    # Process one feature group at a time
    for feature in features:
        next_column = column + feature.channels
        block = x[:, column:next_column]
        feature_scale = scale[column:next_column]

        if feature.transform == "log1p":
            np.maximum(block, 0, out=block)
            block /= feature_scale
            np.log1p(block, out=block)  # compresses large strongly positive skewed values while handling 0's safely
        elif feature.transform == "arcsinh":
            block /= feature_scale
            np.arcsinh(block, out=block)  # similiar to logarithm but supports negative values and preserves their sign
        elif feature.transform != "standard":
            msg = f"unknown transform: {feature.transform}"
            raise ValueError(msg)

        column = next_column
