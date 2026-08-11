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


@dataclass
class RobustPreprocesser:
    """Convert atmospheric variables using median/IQR robust scaling with smooth clipping.

    This preprocessing is less sensitive to extreme outliers than mean/std scaling.
    Uses the RealMLP approach: median centering, IQR scaling, and smooth tanh clipping.
    """

    scale: np.ndarray
    median: np.ndarray
    iqr: np.ndarray
    clip_threshold: float
    features: tuple[Feature, ...]

    @classmethod
    def fit_transform(
        cls, x: np.ndarray, features, clip_threshold: float = 3.0
    ) -> tuple["RobustPreprocesser", np.ndarray]:
        """Apply transformations and calculate robust statistics."""
        x = np.asarray(x, dtype=np.float32, order="c")
        nonzero = np.zeros(x.shape[1], dtype=np.int64)
        absolute_sum = np.zeros(x.shape[1])

        for chunk in chunks(x):
            nonzero += np.count_nonzero(chunk, axis=0)
            absolute_sum += np.abs(chunk).sum(axis=0, dtype=np.float64)

        scale = np.where(nonzero > 0, absolute_sum / np.maximum(nonzero, 1), 1.0).astype(np.float32)
        transform_in_place(x, scale, features)

        # Compute median (50th percentile)
        median = np.zeros(x.shape[1], dtype=np.float64)
        for chunk in chunks(x):
            median += np.median(chunk, axis=0)
        median = (median / (len(x) / CHUNK_SIZE)).astype(np.float32)

        # Compute IQR (75th - 25th percentile)
        q75 = np.zeros(x.shape[1], dtype=np.float64)
        q25 = np.zeros(x.shape[1], dtype=np.float64)
        for chunk in chunks(x):
            q75 += np.percentile(chunk, 75, axis=0)
            q25 += np.percentile(chunk, 25, axis=0)
        q75 = (q75 / (len(x) / CHUNK_SIZE)).astype(np.float32)
        q25 = (q25 / (len(x) / CHUNK_SIZE)).astype(np.float32)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0

        # Robust scaling: subtract median, divide by IQR
        x -= median
        x /= iqr

        # Smooth clipping using tanh
        if clip_threshold > 0:
            np.tanh(np.clip(x, -clip_threshold, clip_threshold) / clip_threshold, out=x)
            x *= clip_threshold

        return cls(scale, median, iqr, clip_threshold, features), x

    def transform(self, x: np.ndarray, *, copy: bool = True) -> np.ndarray:
        """Transform new data using learned robust statistics."""
        out = np.array(x, dtype=np.float32, order="c", copy=copy)
        transform_in_place(out, self.scale, self.features)
        out -= self.median
        out /= self.iqr
        if self.clip_threshold > 0:
            np.tanh(np.clip(out, -self.clip_threshold, self.clip_threshold) / self.clip_threshold, out=out)
            out *= self.clip_threshold
        return out

    def state(self) -> dict[str, np.ndarray]:
        """Return learned robust statistics."""
        return {
            "scale": self.scale,
            "median": self.median,
            "iqr": self.iqr,
            "clip_threshold": np.array([self.clip_threshold]),
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

    @classmethod
    def fit_transform(cls, x: np.ndarray, features, n_bins: int = 128) -> tuple["StretchPreprocesser", np.ndarray]:
        """Apply transformations and fit quantile bins."""
        x = np.asarray(x, dtype=np.float32, order="c")
        nonzero = np.zeros(x.shape[1], dtype=np.int64)
        absolute_sum = np.zeros(x.shape[1])

        for chunk in chunks(x):
            nonzero += np.count_nonzero(chunk, axis=0)
            absolute_sum += np.abs(chunk).sum(axis=0, dtype=np.float64)

        scale = np.where(nonzero > 0, absolute_sum / np.maximum(nonzero, 1), 1.0).astype(np.float32)
        transform_in_place(x, scale, features)

        # Compute quantiles for each feature
        quantile_probs = np.linspace(0, 1, n_bins + 1)
        quantiles = np.zeros((n_bins + 1, x.shape[1]), dtype=np.float32)

        for i, q in enumerate(quantile_probs):
            q_values = np.zeros(x.shape[1], dtype=np.float64)
            for chunk in chunks(x):
                q_values += np.percentile(chunk, q * 100, axis=0)
            quantiles[i] = (q_values / (len(x) / CHUNK_SIZE)).astype(np.float32)

        # Apply stretch transformation
        stretched = np.empty_like(x)
        n_features = x.shape[1]

        for feat_idx in range(n_features):
            col = x[:, feat_idx]
            feat_quantiles = quantiles[:, feat_idx]

            # For each value, find its bin and apply stretch
            # Digitize returns 1-based bin indices
            bin_indices = np.digitize(col, feat_quantiles[1:-1], right=False)
            bin_indices = np.clip(bin_indices - 1, 0, n_bins - 1)

            # Linear interpolation within bin
            bin_left = feat_quantiles[bin_indices]
            bin_right = feat_quantiles[bin_indices + 1]
            bin_width = bin_right - bin_left
            bin_width[bin_width == 0] = 1.0  # Avoid division by zero

            # Normalized position within bin [0, 1]
            position = (col - bin_left) / bin_width
            position = np.clip(position, 0, 1)

            # Apply CDF-like stretch
            stretched_pos = (bin_indices + position) / n_bins
            stretched[:, feat_idx] = stretched_pos.astype(np.float32)

        return cls(scale, quantiles, n_bins, features), stretched

    def transform(self, x: np.ndarray, *, copy: bool = True) -> np.ndarray:
        """Transform new data using learned quantiles."""
        out = np.array(x, dtype=np.float32, order="c", copy=copy)
        transform_in_place(out, self.scale, self.features)

        stretched = np.empty_like(out)
        n_features = out.shape[1]

        for feat_idx in range(n_features):
            col = out[:, feat_idx]
            feat_quantiles = self.quantiles[:, feat_idx]

            bin_indices = np.digitize(col, feat_quantiles[1:-1], right=False)
            bin_indices = np.clip(bin_indices - 1, 0, self.n_bins - 1)

            bin_left = feat_quantiles[bin_indices]
            bin_right = feat_quantiles[bin_indices + 1]
            bin_width = bin_right - bin_left
            bin_width[bin_width == 0] = 1.0

            position = (col - bin_left) / bin_width
            position = np.clip(position, 0, 1)

            stretched_pos = (bin_indices + position) / self.n_bins
            stretched[:, feat_idx] = stretched_pos.astype(np.float32)

        return stretched

    def state(self) -> dict[str, np.ndarray]:
        """Return learned quantile boundaries."""
        return {
            "scale": self.scale,
            "quantiles": self.quantiles,
            "n_bins": np.array([self.n_bins]),
        }


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
    _,
    selected_indices: np.ndarray,
    rows_per_timestep: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group selected rows by their source timestep."""
    row_counts = np.asarray(rows_per_timestep)
    indices = np.asarray(selected_indices)
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
