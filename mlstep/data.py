"""UKCA data loading and train-fitted preprocessing.

The input contract has 267 columns in the residual-plus-QCF layout used by
the selected research models. Physical scaling is used alone or before numerical
embeddings; Stretch128 is an alternative transform.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr

N_CLASSES = 5
CHUNK_SIZE = 65_536
QUANTILE_FIT_SAMPLE_ROWS = 1_000_000
QUANTILE_FIT_SEED = 0
STRETCH128_N_BINS = 128
PLE_N_BINS = 48
GRID_SHAPE = (38, 72, 96)
MATRIX_NDIM = 2
MIN_BINS = 2
PREPROCESSING_METHODS = ("physical", "ple", "stretch128")


@dataclass(frozen=True)
class Feature:
    """One ordered input group and the associated physical transformation.

    Stores the NetCDF name, number of scalar/species columns and transform.
    """

    name: str
    channels: int
    transform: str


FEATURES = (
    Feature("temp", 1, "standard"),
    Feature("pres", 1, "standard"),
    Feature("cloud_frac", 1, "standard"),
    Feature("qcl", 1, "log1p"),
    Feature("cell_volume", 1, "standard"),
    Feature("stratflag", 1, "standard"),
    Feature("tracer", 83, "log1p"),
    Feature("residual", 83, "arcsinh"),
    Feature("photol_rates", 60, "log1p"),
    Feature("wetrt", 34, "log1p"),
    Feature("qcf", 1, "log1p"),
)

N_FEATURES = sum(feature.channels for feature in FEATURES)
CATEGORICAL_FEATURES = frozenset({"stratflag"})
STRICT_NONNEGATIVE_FEATURES = frozenset({"qcf"})
_TRANSFORMS = frozenset({"standard", "log1p", "arcsinh"})


def feature_names(features: tuple[Feature, ...] = FEATURES) -> list[str]:
    """Return ordered model-column names, such as temp and tracer[0].

    Used to match cache and teacher columns to the current input schema.
    """
    names: list[str] = []
    for feature in features:
        if feature.channels == 1:
            names.append(feature.name)
        else:
            names.extend(f"{feature.name}[{index}]" for index in range(feature.channels))
    return names


def validate_feature_schema(features: tuple[Feature, ...]) -> int:
    """Check feature names, channel counts and transforms; return total width.

    Reject invalid settings and duplicate column names before loading or
    fitting. Callers supply Feature records; Python handles wrong object types.
    """
    if not features:
        raise ValueError("feature schema must not be empty")
    for feature in features:
        invalid_channels = (
            isinstance(feature.channels, bool) or not isinstance(feature.channels, int) or feature.channels < 1
        )
        if not feature.name or invalid_channels or feature.transform not in _TRANSFORMS:
            msg = "features contain an invalid name, channel count or transform"
            raise ValueError(msg)
    names = feature_names(features)
    if len(names) != len(set(names)):
        msg = "feature schema contains duplicate column names"
        raise ValueError(msg)
    return len(names)


def feature_group_names(features: tuple[Feature, ...] = FEATURES) -> list[str]:
    """Return the parent group for each model column.

    PLE uses this list to select groups such as tracer or residual.
    """
    return [feature.name for feature in features for _ in range(feature.channels)]


def feature_schema(features: tuple[Feature, ...] = FEATURES) -> list[dict[str, object]]:
    """Convert ordered Feature records to dictionaries for saved preprocessing.

    Recording transforms as well as names lets loading detect a changed schema.
    """
    return [
        {"name": feature.name, "channels": feature.channels, "transform": feature.transform} for feature in features
    ]


def chunks(x: np.ndarray) -> Iterator[np.ndarray]:
    """Yield row views in blocks of CHUNK_SIZE.

    Used by reductions to limit temporary memory; the full input stays in RAM.
    """
    for start in range(0, len(x), CHUNK_SIZE):
        yield x[start : start + CHUNK_SIZE]


def _feature_matrix(
    x: np.ndarray,
    n_features: int,
    *,
    copy: bool | None,
    context: str,
) -> np.ndarray:
    """Prepare a finite, writable float32 matrix with the expected column count.

    copy=True always copies; False requires a compatible input and reuses it.
    None copies only when dtype, layout or writeability requires it. Fitting
    uses None; held-out transformation defaults to True to preserve raw rows.
    """
    values = np.asarray(x)
    if values.ndim != MATRIX_NDIM or not len(values) or values.shape[1] != n_features:
        msg = f"{context} must have shape (n_rows, {n_features}) with at least one row"
        raise ValueError(msg)
    if copy is None:
        copy = values.dtype != np.float32 or not values.flags.c_contiguous or not values.flags.writeable
    if copy:
        out = np.array(values, dtype=np.float32, order="C", copy=True)
    else:
        if values.dtype != np.float32 or not values.flags.c_contiguous or not values.flags.writeable:
            msg = f"{context} must be a writable C-contiguous float32 matrix when copy=False"
            raise ValueError(msg)
        out = values
    for block in chunks(out):
        if not np.isfinite(block).all():
            msg = f"{context} must contain only finite values"
            raise ValueError(msg)
    return out


def column_extrema(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find each column's exact minimum and maximum in row blocks.

    Stretching and PLE use full-training endpoints alongside sampled quantiles.
    """
    minimum = np.full(x.shape[1], np.inf, dtype=np.float32)
    maximum = np.full(x.shape[1], -np.inf, dtype=np.float32)
    for block in chunks(x):
        np.minimum(minimum, np.min(block, axis=0), out=minimum)
        np.maximum(maximum, np.max(block, axis=0), out=maximum)
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        msg = "preprocessing inputs must contain only finite values"
        raise ValueError(msg)
    return minimum, maximum


def _fit_physical_scale(x: np.ndarray) -> np.ndarray:
    """Calculate mean absolute nonzero value per raw input column.

    Log/arcsinh transforms divide by this scale; all-zero columns use one.
    Float64 sums reduce accumulation error while saved scales use float32.
    """
    nonzero = np.zeros(x.shape[1], dtype=np.int64)
    absolute_sum = np.zeros(x.shape[1], dtype=np.float64)
    for block in chunks(x):
        nonzero += np.count_nonzero(block, axis=0)
        absolute_sum += np.abs(block).sum(axis=0, dtype=np.float64)
    return np.where(nonzero > 0, absolute_sum / np.maximum(nonzero, 1), 1.0).astype(np.float32)


def transform_in_place(x: np.ndarray, scale: np.ndarray, features: tuple[Feature, ...]) -> None:
    """Apply each group's physical transform to the supplied matrix in place.

    Log groups clip negatives and use log1p(x/scale); signed residuals use
    arcsinh(x/scale). Standard groups are unchanged. Callers provide matching
    checked columns/scales; mean/std standardisation is a separate step.
    """
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        msg = "physical scale must contain only finite positive values"
        raise ValueError(msg)

    column = 0
    for feature in features:
        next_column = column + feature.channels
        block = x[:, column:next_column]
        feature_scale = scale[column:next_column]
        if feature.transform == "log1p":
            # Preserve the established zero-clipping policy for logarithmic groups.
            np.maximum(block, 0, out=block)
            block /= feature_scale
            np.log1p(block, out=block)
        elif feature.transform == "arcsinh":
            # Signed residuals need compression without losing their sign.
            block /= feature_scale
            np.arcsinh(block, out=block)
        elif feature.transform != "standard":
            msg = f"unknown transform: {feature.transform}"
            raise ValueError(msg)
        column = next_column


@dataclass
class Preprocesser:
    """Store training-fitted physical scales and standardisation statistics.

    Fit once on training rows, then reuse transform for held-out rows.
    """

    scale: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    features: tuple[Feature, ...]

    @classmethod
    def fit_transform(
        cls,
        x: np.ndarray,
        features: tuple[Feature, ...] = FEATURES,
    ) -> tuple["Preprocesser", np.ndarray]:
        """Fit physical scales and mean/std on training rows; return state and rows.

        Reuses a writable contiguous float32 input, otherwise copies. Statistics
        are fitted after physical transformation; constant columns use std one.
        """
        feature_tuple = tuple(features)
        width = validate_feature_schema(feature_tuple)
        out = _feature_matrix(x, width, copy=None, context="training features")
        scale = _fit_physical_scale(out)
        transform_in_place(out, scale, feature_tuple)

        total = np.zeros(out.shape[1], dtype=np.float64)
        for block in chunks(out):
            total += block.sum(axis=0, dtype=np.float64)
        mean = (total / len(out)).astype(np.float32)
        squared_error = np.zeros(out.shape[1], dtype=np.float64)
        for block in chunks(out):
            # Subtraction owns this scratch array; square it without another copy.
            deviations = block - mean
            np.square(deviations, out=deviations)
            squared_error += deviations.sum(axis=0, dtype=np.float64)
            del deviations
        std = np.sqrt(squared_error / len(out)).astype(np.float32)
        std[std == 0] = 1.0
        out -= mean
        out /= std
        return cls(scale, mean, std, feature_tuple), out

    def transform(self, x: np.ndarray, *, copy: bool = True) -> np.ndarray:
        """Transform raw rows with the stored scales and mean/std, without refitting.

        Copies by default; copy=False modifies a compatible input in place.
        """
        out = _feature_matrix(x, len(self.mean), copy=copy, context="features")
        transform_in_place(out, self.scale, self.features)
        out -= self.mean
        out /= self.std
        return out


def quantile_sample_indices(n_rows: int, max_rows: int, seed: int = 0) -> np.ndarray:
    """Select one row per positional stratum, or every row for small datasets.

    Both PLE and stretching use this reproducible sampling rule to spread a
    bounded quantile-fit sample across ordered timesteps. It does not use labels.
    Callers retain their own sample-size limits; sharing the rule does not make
    the two preprocessing methods use identical fitted quantiles.
    """
    if n_rows < 1 or max_rows < 1:
        raise ValueError("row count and sampling limit must be positive")
    if n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    edges = np.arange(max_rows + 1, dtype=np.int64) * n_rows // max_rows
    generator = np.random.default_rng(seed)
    offsets = np.floor(generator.random(max_rows) * np.diff(edges)).astype(np.int64)
    return edges[:-1] + offsets


def _stretch_column(values: np.ndarray, boundaries: np.ndarray, n_bins: int) -> np.ndarray:
    """Map scalar values to fractional positions along fitted quantile bins.

    Returns one float32 value per input. Repeated boundaries use a step;
    values outside the fitted range saturate at the ends.
    """
    # Interior boundaries choose the interval; ties move to the interval on the right.
    bin_indices = np.searchsorted(boundaries[1:-1], values, side="right")
    bin_left = boundaries[bin_indices]
    bin_right = boundaries[bin_indices + 1]
    bin_width = bin_right - bin_left
    position = np.empty_like(values, dtype=np.float32)
    np.subtract(values, bin_left, out=position)
    np.divide(position, bin_width, out=position, where=bin_width > 0)
    # Repeated quantiles form zero-width intervals: use a step, never divide by zero.
    zero_width = bin_width <= 0
    position[zero_width] = values[zero_width] >= bin_right[zero_width]
    np.clip(position, 0, 1, out=position)
    np.add(position, bin_indices, out=position, casting="unsafe")
    position /= n_bins
    return position


@dataclass
class StretchPreprocesser:
    """Store physical scales, quantile bins and stretched-value mean/std.

    Applies physical transformation, stretching and standardisation in order.
    """

    scale: np.ndarray
    quantiles: np.ndarray
    n_bins: int
    features: tuple[Feature, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit_transform(
        cls,
        x: np.ndarray,
        features: tuple[Feature, ...] = FEATURES,
        n_bins: int = STRETCH128_N_BINS,
    ) -> tuple["StretchPreprocesser", np.ndarray]:
        """Fit physical scaling, quantile stretching and final mean/std on training.

        Returns fitted state and transformed rows, reusing a compatible input.
        Quantiles use a deterministic sample with exact training endpoints.
        The cache uses 128 bins; this fitting method accepts other bin counts.
        """
        if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < MIN_BINS:
            msg = "n_bins must be an integer of at least two"
            raise ValueError(msg)
        feature_tuple = tuple(features)
        width = validate_feature_schema(feature_tuple)
        out = _feature_matrix(x, width, copy=None, context="training features")
        scale = _fit_physical_scale(out)
        transform_in_place(out, scale, feature_tuple)
        minimum, maximum = column_extrema(out)
        # Reuse the whole matrix for small fits; advanced indexing otherwise
        # allocates only the bounded sample, using the same row rule as PLE.
        fit_sample = out
        if len(out) > QUANTILE_FIT_SAMPLE_ROWS:
            fit_sample = out[quantile_sample_indices(len(out), QUANTILE_FIT_SAMPLE_ROWS, QUANTILE_FIT_SEED)]
        quantiles = np.percentile(
            fit_sample,
            np.linspace(0, 100, n_bins + 1),
            axis=0,
            method="linear",
        ).astype(np.float32)
        del fit_sample
        # A sampled quantile fit may miss extremes; keep the full training endpoints.
        quantiles[0] = minimum
        quantiles[-1] = maximum
        np.maximum.accumulate(quantiles, axis=0, out=quantiles)

        mean = np.empty(out.shape[1], dtype=np.float32)
        std = np.empty(out.shape[1], dtype=np.float32)
        for index in range(out.shape[1]):
            stretched = _stretch_column(out[:, index], quantiles[:, index], n_bins)
            mean[index] = np.mean(stretched, dtype=np.float64)
            std[index] = np.std(stretched, dtype=np.float64)
            if std[index] == 0:
                std[index] = 1.0
            stretched -= mean[index]
            stretched /= std[index]
            out[:, index] = stretched
        return cls(scale, quantiles, n_bins, feature_tuple, mean, std), out

    def transform(self, x: np.ndarray, *, copy: bool = True) -> np.ndarray:
        """Apply saved physical scales, quantile bins and mean/std to raw rows.

        Does not refit bins or statistics. Copies unless copy=False is supplied.
        """
        out = _feature_matrix(x, len(self.mean), copy=copy, context="features")
        transform_in_place(out, self.scale, self.features)
        for index in range(out.shape[1]):
            stretched = _stretch_column(out[:, index], self.quantiles[:, index], self.n_bins)
            stretched -= self.mean[index]
            stretched /= self.std[index]
            out[:, index] = stretched
        return out


FittedPreprocesser = Preprocesser | StretchPreprocesser


def preprocessor_state(preprocessor: FittedPreprocesser) -> dict[str, object]:
    """Build the NPZ payload for either fitted preprocessor.

    Includes scales, mean/std and ordered feature records. Stretching adds
    quantiles and a one-element bin-count array. Numerical arrays are not copied.
    """
    state = {
        "scale": preprocessor.scale,
        "mean": preprocessor.mean,
        "std": preprocessor.std,
        "features": feature_schema(preprocessor.features),
    }
    if isinstance(preprocessor, StretchPreprocesser):
        state["quantiles"] = preprocessor.quantiles
        state["n_bins"] = np.array([preprocessor.n_bins], dtype=np.int64)
    return state


def _load_float_array(
    state: np.lib.npyio.NpzFile,
    name: str,
    shape: tuple[int, ...],
    *,
    positive: bool = False,
) -> np.ndarray:
    """Read a saved array with the required shape as an independent float32 copy.

    Reject nonfinite values; require positive entries for scales and stds.
    Shared by the four numerical fields loaded from saved preprocessing.
    """
    value = np.asarray(state[name])
    if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
        msg = f"preprocessor state {name!r} must be a floating-point array with shape {shape}"
        raise ValueError(msg)
    value = np.array(value, dtype=np.float32, copy=True)
    if not np.isfinite(value).all() or (positive and np.any(value <= 0)):
        qualifier = "finite positive" if positive else "finite"
        msg = f"preprocessor state {name!r} must contain only {qualifier} values"
        raise ValueError(msg)
    return value


def load_fitted_preprocessor(
    state_path: Path,
    method: str,
    *,
    features: tuple[Feature, ...] = FEATURES,
) -> FittedPreprocesser:
    """Restore fitted physical/PLE or Stretch128 preprocessing from an NPZ file.

    Checks schema, array shapes, finite statistics and valid stretch bins.
    Missing fields raise KeyError; unrelated extra fields are ignored. No data
    is refitted. The object-array schema requires pickle, so use trusted caches.
    """
    if method not in PREPROCESSING_METHODS:
        msg = f"unsupported preprocessing method {method!r}; expected physical, ple or stretch128"
        raise ValueError(msg)
    feature_tuple = tuple(features)
    width = validate_feature_schema(feature_tuple)
    with np.load(Path(state_path), allow_pickle=True) as state:
        if state["features"].tolist() != feature_schema(feature_tuple):
            raise ValueError("saved preprocessing does not match the requested feature layout")
        shape = (width,)
        scale = _load_float_array(state, "scale", shape, positive=True)
        mean = _load_float_array(state, "mean", shape)
        std = _load_float_array(state, "std", shape, positive=True)
        if method in {"physical", "ple"}:
            return Preprocesser(scale, mean, std, feature_tuple)

        saved_bins = state["n_bins"]
        if saved_bins.shape != (1,) or not np.issubdtype(saved_bins.dtype, np.integer):
            raise ValueError("saved n_bins must be a one-element integer array")
        n_bins = int(saved_bins[0])
        if n_bins != STRETCH128_N_BINS:
            raise ValueError("stretch128 preprocessing requires 128 bins")
        quantiles = _load_float_array(state, "quantiles", (n_bins + 1, width))
        if np.any(np.diff(quantiles, axis=0) < 0):
            msg = "preprocessor state quantiles must be nondecreasing in every feature"
            raise ValueError(msg)
        return StretchPreprocesser(scale, quantiles, n_bins, feature_tuple, mean, std)


def halving_labels(ncsteps: np.ndarray) -> np.ndarray:
    """Convert ncsteps values 1, 2, 4, 8, 16 to a flat int64 vector of counts 0–4.

    Require exact matches so unexpected solver values are not rounded or clipped.
    """
    values = np.asarray(ncsteps)
    if not values.size:
        msg = "ncsteps must contain at least one value"
        raise ValueError(msg)
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values) or not np.isfinite(values).all():
        msg = "ncsteps values must be finite real numbers"
        raise ValueError(msg)
    allowed = np.array([1, 2, 4, 8, 16])
    matches = values.ravel()[:, None] == allowed
    if not matches.any(axis=1).all():
        msg = "ncsteps values must be exactly 1, 2, 4, 8, or 16"
        raise ValueError(msg)
    return matches.argmax(axis=1).astype(np.int64)


def discover_timesteps(data_dir: Path) -> tuple[int, ...]:
    """Find labelled snapshot IDs from ncsteps_<timestep>.nc and sort them.

    Does not check feature-file completeness or exclude a spin-up timestep.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    try:
        steps = sorted(int(path.stem.removeprefix("ncsteps_")) for path in directory.glob("ncsteps_*.nc"))
    except ValueError as error:
        msg = f"target filenames in {directory} must end in an integer timestep"
        raise ValueError(msg) from error
    if not steps:
        msg = f"No usable ncsteps_*.nc files found in {directory}."
        raise ValueError(msg)
    return tuple(steps)


def _validate_timesteps(timesteps: tuple[int, ...]) -> tuple[int, ...]:
    """Require nonempty, increasing, unique, nonnegative snapshot IDs.

    Loaders use this check to preserve consistent row identity. Returns a tuple
    without silently sorting or removing duplicates.
    """
    steps = tuple(timesteps)
    invalid = (
        not steps
        or any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in steps)
        or any(left >= right for left, right in pairwise(steps))
    )
    if invalid:
        msg = "timesteps must be unique, strictly increasing non-negative integers"
        raise ValueError(msg)
    return steps


def load_variable(data_dir: Path, name: str, timestep: int, channels: int = 1) -> np.ndarray:
    """Read one NetCDF variable into canonical spatial/species order.

    Uses the schema name/channel count to open the file, transpose named axes
    and check the fixed grid shape. Rejects nonfinite values and negative QCF.
    Materialises the array before closing the dataset.
    """
    path = Path(data_dir) / f"{name}_{timestep}.nc"
    expected_dims = ("z", "y", "x") if channels == 1 else ("s", "z", "y", "x")
    expected_shape = GRID_SHAPE if channels == 1 else (channels, *GRID_SHAPE)
    with xr.open_dataset(path) as dataset:
        if name not in dataset.data_vars:
            msg = f"{path} does not contain data variable {name!r}"
            raise ValueError(msg)
        variable = dataset[name]
        if set(variable.dims) != set(expected_dims) or len(variable.dims) != len(expected_dims):
            msg = f"{path} has dimensions {variable.dims}; expected {expected_dims}"
            raise ValueError(msg)
        variable = variable.transpose(*expected_dims)
        if variable.shape != expected_shape:
            msg = f"{path} has shape {variable.shape}; expected {expected_shape}"
            raise ValueError(msg)
        values = np.asarray(variable.values)
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        msg = f"{path} must contain only finite numeric values"
        raise ValueError(msg)
    if name in STRICT_NONNEGATIVE_FEATURES and np.any(values < 0):
        msg = f"{path} must contain only non-negative values"
        raise ValueError(msg)
    return values


def load_labels(data_dir: Path, timesteps: tuple[int, ...]) -> np.ndarray:
    """Load only ncsteps files and fill one preallocated int64 label vector.

    Rows follow timestep, z, y, x order. Used before XGBoost samples its inputs.
    """
    steps = _validate_timesteps(timesteps)
    rows = math.prod(GRID_SHAPE)
    labels = np.empty(len(steps) * rows, dtype=np.int64)
    for position, timestep in enumerate(steps):
        # Fill the final array directly instead of keeping every timestep plus a copy.
        labels[position * rows : (position + 1) * rows] = halving_labels(
            load_variable(data_dir, "ncsteps", timestep)
        )
    return labels


def load_timesteps(
    data_dir: Path,
    timesteps: tuple[int, ...],
    features: tuple[Feature, ...] = FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    """Load matching raw feature rows and halving labels for the same timesteps.

    Combines the two loaders used separately by XGBoost and cache preparation.
    """
    # Both loaders validate timestep order and enforce the same fixed grid shape.
    matrix = load_features(data_dir, timesteps, features)
    labels = load_labels(data_dir, timesteps)
    return matrix, labels


def load_features(
    data_dir: Path,
    timesteps: tuple[int, ...],
    features: tuple[Feature, ...] = FEATURES,
) -> np.ndarray:
    """Load NetCDF features into the ordered float32 model-input matrix.

    Rows follow timestep, z, y, x order; species become columns. Loads one
    variable at a time, but retains the complete matrix. No labels are needed.
    """
    feature_tuple = tuple(features)
    width = validate_feature_schema(feature_tuple)
    steps = _validate_timesteps(timesteps)
    rows_per_timestep = math.prod(GRID_SHAPE)
    matrix = np.empty((len(steps) * rows_per_timestep, width), dtype=np.float32)
    row_start = 0
    for timestep in steps:
        row_stop = row_start + rows_per_timestep
        column = 0
        for feature in feature_tuple:
            values = load_variable(data_dir, feature.name, timestep, feature.channels)
            next_column = column + feature.channels
            # NetCDF is (species, z, y, x); rows are grid boxes, columns are species.
            matrix[row_start:row_stop, column:next_column] = values.reshape(feature.channels, rows_per_timestep).T
            del values
            column = next_column
        row_start = row_stop
    return matrix


def training_index_pools(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return indices of positive and zero labels for repeated row sampling.

    Expects class-validated labels. Reusing the pools avoids rescanning each epoch.
    """
    values = np.asarray(labels)
    if values.ndim != 1:
        msg = "labels must be one-dimensional"
        raise ValueError(msg)
    return np.flatnonzero(values > 0), np.flatnonzero(values == 0)


def random_undersampling(
    positive: np.ndarray,
    negative: np.ndarray,
    negative_ratio: int,
    seed: int,
) -> np.ndarray:
    """Keep all positive indices and sample up to negative_ratio zeros per positive.

    Takes integer index pools from training_index_pools and an integer ratio.
    Sampling is without replacement and seed controls the final shuffled order.
    Use these indices for features, labels and teacher targets together.
    """
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")
    n_negative = min(len(negative), negative_ratio * len(positive))
    generator = np.random.default_rng(seed)
    # Keep the same sampled indices for features, labels and any teacher responses.
    sampled_negative = generator.choice(negative, size=n_negative, replace=False)
    indices = np.concatenate([positive, sampled_negative])
    generator.shuffle(indices)
    return indices


def class_counts(labels: np.ndarray) -> list[int]:
    """Validate labels in 0–4 and return a five-element list of class counts.

    Includes zeros for absent classes; used when reading caches and reporting.
    """
    values = np.asarray(labels)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        msg = "labels must be a one-dimensional integer array"
        raise ValueError(msg)
    if len(values) and (values.min() < 0 or values.max() >= N_CLASSES):
        msg = f"labels must be in [0, {N_CLASSES - 1}]"
        raise ValueError(msg)
    return np.bincount(values.astype(np.int64, copy=False), minlength=N_CLASSES).tolist()


def parse_timestep_range(value: str) -> tuple[int, ...]:
    """Convert an inclusive CLI range such as 2:252 into timestep integers.

    Includes the stop value and reports malformed ranges through argparse.
    """
    try:
        start_text, stop_text = value.split(":", maxsplit=1)
        start, stop = int(start_text), int(stop_text)
    except (TypeError, ValueError) as error:
        msg = "timestep ranges must use inclusive START:STOP integers"
        raise argparse.ArgumentTypeError(msg) from error
    if start < 0 or stop < start:
        msg = "timestep ranges require 0 <= START <= STOP"
        raise argparse.ArgumentTypeError(msg)
    return tuple(range(start, stop + 1))


def resolve_timestep_split(
    available: tuple[int, ...],
    train_timesteps: tuple[int, ...],
    validation_timesteps: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Require available, nonempty training/validation ranges with training first.

    Returns the requested tuples unchanged. Does not create a random split
    or exclude spin-up data; CLI parsing/loaders check within-split order.
    """
    train = tuple(train_timesteps)
    validation = tuple(validation_timesteps)
    missing = sorted({*train, *validation} - set(available))
    if missing:
        msg = f"requested timesteps are unavailable: {missing[:5]}"
        raise ValueError(msg)
    if not train or not validation or set(train) & set(validation) or max(train) >= min(validation):
        msg = "training and validation ranges must be non-overlapping and chronological"
        raise ValueError(msg)
    return train, validation


def load_selected_rows(data_dir: Path, timesteps: tuple[int, ...], indices: np.ndarray) -> np.ndarray:
    """Read selected global feature rows, loading one timestep at a time.

    Groups indices for file access, then restores their original order and
    any repeated requests. Skips unused snapshots and retains only selected
    rows, reducing XGBoost training memory without changing its sample order.
    """
    steps = _validate_timesteps(timesteps)
    indices = np.asarray(indices)
    rows = math.prod(GRID_SHAPE)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= rows * len(steps))
    ):
        raise ValueError("sampled row indices are out of range")
    result = np.empty((len(indices), N_FEATURES), dtype=np.float32)
    # Sort for timestep lookup, but scatter back into the caller's original order.
    order = np.argsort(indices)
    ordered = indices[order]
    for position, timestep in enumerate(steps):
        left, right = np.searchsorted(ordered, [position * rows, (position + 1) * rows])
        if left == right:
            continue
        raw = load_features(data_dir, (timestep,))
        result[order[left:right]] = raw[ordered[left:right] - position * rows]
        del raw  # Release this snapshot before allocating the next one.
    return result
