"""Build reusable neural network inputs, fitting preprocessing on training rows only."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from mlstep.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    N_FEATURES,
    PLE_N_BINS,
    PREPROCESSING_METHODS,
    Feature,
    FittedPreprocesser,
    Preprocesser,
    StretchPreprocesser,
    column_extrema,
    discover_timesteps,
    feature_group_names,
    feature_names,
    load_fitted_preprocessor,
    load_timesteps,
    parse_timestep_range,
    preprocessor_state,
    quantile_sample_indices,
    resolve_timestep_split,
)

PLE_QUANTILE_SAMPLE_ROWS = 1_048_576


def _apply_preprocessing(
    x: np.ndarray,
    method: str,
    features: tuple[Feature, ...] = FEATURES,
) -> tuple[FittedPreprocesser, np.ndarray]:
    """Fit training preprocessing and return its state and transformed rows.

    Physical and PLE share physical scaling/standardisation. Stretch128 inserts
    quantile stretching before standardisation. Compatible input is modified.
    """
    if method in {"physical", "ple"}:
        return Preprocesser.fit_transform(x, features)
    if method == "stretch128":
        return StretchPreprocesser.fit_transform(x, features, n_bins=128)
    msg = f"unsupported preprocessing method {method!r}"
    raise ValueError(msg)


def _compute_ple_bin_boundaries(
    x: np.ndarray,
    feature_indices: list[int] | np.ndarray,
    n_bins: int,
    *,
    minimum: np.ndarray,
    maximum: np.ndarray,
    max_sample_rows: int = PLE_QUANTILE_SAMPLE_ROWS,
    seed: int = 0,
) -> np.ndarray:
    """Fit quantile boundaries for selected columns without copying the full matrix.

    Gather one sampled column at a time. Use full-training extrema as endpoints;
    repeated values can produce repeated boundaries rather than equal-size bins.
    """
    sample = quantile_sample_indices(len(x), max_sample_rows, seed)
    probabilities = np.linspace(0, 1, n_bins + 1)
    boundaries = np.empty((len(feature_indices), n_bins + 1), dtype=np.float32)
    for output_index, feature_index in enumerate(feature_indices):
        column = x[sample, feature_index]
        fitted = np.quantile(column, probabilities, method="linear").astype(np.float32)
        # Enclose all fitted training rows, including extrema missed by sampling.
        # Future data can still lie outside these boundaries and will saturate.
        fitted[0] = minimum[feature_index]
        fitted[-1] = maximum[feature_index]
        boundaries[output_index] = np.maximum.accumulate(fitted)
    return boundaries


def _ple_metadata(train_x: np.ndarray, groups: list[str]) -> dict[str, np.ndarray]:
    """Choose requested nonconstant continuous columns and fit their PLE boundaries.

    Preserve schema order. Categorical and constant columns bypass embedding;
    selection uses training values only, without a teacher or feature ranking.
    """
    available = {feature.name for feature in FEATURES} - set(CATEGORICAL_FEATURES)
    requested = available if groups == ["all"] else set(groups)
    if not requested or not requested <= available:
        raise ValueError(f"embedding groups must be 'all' or drawn from {sorted(available)}")
    minimum, maximum = column_extrema(train_x)
    selected = [
        index
        for index, group in enumerate(feature_group_names(FEATURES))
        if group in requested and maximum[index] > minimum[index]
    ]
    if not selected:
        raise ValueError("no nonconstant continuous columns in the requested embedding groups")
    return {
        "ple_indices": np.asarray(selected, dtype=np.int64),
        "bin_boundaries": _compute_ple_bin_boundaries(
            train_x,
            selected,
            PLE_N_BINS,
            minimum=minimum,
            maximum=maximum,
        ),
    }


def load_cache_state(
    cache_dir: Path,
    *,
    expected_preprocess: str | None = None,
) -> tuple[dict[str, Any], FittedPreprocesser, dict[str, np.ndarray] | None]:
    """Return metadata, fitted preprocessing and optional PLE state from one cache.

    Check schema/split compatibility and fitted numerical state. Large feature
    and label arrays are opened separately by each consumer. Nothing is refitted.
    """
    directory = Path(cache_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    method = metadata["preprocess"]
    if method not in PREPROCESSING_METHODS or (expected_preprocess is not None and method != expected_preprocess):
        raise ValueError("cache preprocessing does not match the requested method")
    if metadata["n_features"] != N_FEATURES or metadata["features"] != feature_names(FEATURES):
        raise ValueError("cache feature names/order do not match the 267-column schema")

    train = metadata["train"]["timesteps"]
    validation = metadata["val"]["timesteps"]
    unused = metadata["split"]["unused_timesteps"]
    if set(train) & set(validation) or set(unused) & (set(train) | set(validation)):
        raise ValueError("cache timestep splits must be disjoint")
    if max(train) >= min(validation):
        raise ValueError("cache training timesteps must precede validation")

    preprocessor = load_fitted_preprocessor(directory / "preprocessor_state.npz", method, features=FEATURES)
    ple = None
    if method == "ple":
        with np.load(directory / "ple_metadata.npz", allow_pickle=False) as state:
            indices = state["ple_indices"]
            boundaries = state["bin_boundaries"]
        if (
            indices.ndim != 1
            or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= N_FEATURES)
        ):
            raise ValueError("PLE indices must be unique in-range integers")
        if boundaries.shape != (len(indices), PLE_N_BINS + 1):
            raise ValueError("PLE boundaries do not match the selected columns and bin count")
        if not np.issubdtype(boundaries.dtype, np.floating) or not np.isfinite(boundaries).all():
            raise ValueError("PLE boundaries must contain finite floating-point values")
        if np.any(np.diff(boundaries, axis=1) < 0):
            raise ValueError("PLE boundaries must be nondecreasing")
        ple = {
            "ple_indices": indices.astype(np.int64, copy=False),
            "bin_boundaries": boundaries.astype(np.float32, copy=False),
        }
    return metadata, preprocessor, ple


def _write_split(directory: Path, split: str, x: np.ndarray, y: np.ndarray) -> None:
    """Save one preprocessed split's features and labels into staging."""
    np.save(directory / f"{split}_x.npy", x, allow_pickle=False)
    np.save(directory / f"{split}_y.npy", y, allow_pickle=False)


def _reuse_physical_cache(
    staging: Path, physical_cache: Path, groups: list[str]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Link physical inputs into staging and fit PLE bins on its training rows.

    Physical and PLE use identical scaling. Read the existing training array via
    a memory map and inherit its data provenance and split without refitting.
    Hard links require one filesystem; never fall back to large duplicate copies.
    Linked arrays and fitted preprocessing state must be treated as immutable.
    """
    metadata, _, _ = load_cache_state(physical_cache, expected_preprocess="physical")
    for name in ("train_x.npy", "train_y.npy", "val_x.npy", "val_y.npy", "preprocessor_state.npz"):
        os.link(physical_cache / name, staging / name)
    train_x = np.load(staging / "train_x.npy", mmap_mode="r", allow_pickle=False)
    ple = _ple_metadata(train_x, groups)
    metadata.update(preprocess="ple", embedding_groups=groups)
    return metadata, ple


def run(args: argparse.Namespace) -> Path:
    """Build a new cache, fitting on training rows and reusing state for validation.

    Build raw splits sequentially, or share physical inputs when fitting PLE.
    Both paths publish only after state validation and remove only staging on
    failure. Sharing inherits the source cache's split and data location.
    """
    if args.preprocess != "ple" and args.embedding_groups != ["all"]:
        raise ValueError("--embedding-groups requires --preprocess ple")
    physical_cache = getattr(args, "physical_cache", None)
    if physical_cache is not None:
        if args.preprocess != "ple":
            raise ValueError("--physical-cache requires --preprocess ple")
        if any(
            getattr(args, name, None) is not None for name in ("data_dir", "train_timesteps", "validation_timesteps")
        ):
            raise ValueError("--physical-cache inherits data and splits; omit raw data and timestep options")
    elif args.data_dir is None or args.train_timesteps is None or args.validation_timesteps is None:
        raise ValueError("raw cache building requires data, training and validation timestep options")
    output = args.cache_dir.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        if physical_cache is not None:
            metadata, ple = _reuse_physical_cache(staging, physical_cache, args.embedding_groups)
        else:
            available = discover_timesteps(args.data_dir)
            train_steps, validation_steps = resolve_timestep_split(
                available,
                args.train_timesteps,
                args.validation_timesteps,
            )
            used = set(train_steps) | set(validation_steps)
            unused_steps = tuple(step for step in available if step not in used)
            train_x, train_y = load_timesteps(args.data_dir, train_steps, FEATURES)
            preprocessor, train_x = _apply_preprocessing(train_x, args.preprocess)
            ple = _ple_metadata(train_x, args.embedding_groups) if args.preprocess == "ple" else None
            _write_split(staging, "train", train_x, train_y)
            del train_x, train_y

            val_x, val_y = load_timesteps(args.data_dir, validation_steps, FEATURES)
            val_x = preprocessor.transform(val_x, copy=False)
            _write_split(staging, "val", val_x, val_y)
            del val_x, val_y

            np.savez(staging / "preprocessor_state.npz", **preprocessor_state(preprocessor))
            metadata = {
                "data_dir": str(args.data_dir.expanduser().resolve()),
                "embedding_groups": args.embedding_groups if ple is not None else [],
                "preprocess": args.preprocess,
                "n_features": N_FEATURES,
                "features": feature_names(FEATURES),
                "train": {
                    "timesteps": list(train_steps),
                },
                "val": {
                    "timesteps": list(validation_steps),
                },
                "split": {"unused_timesteps": list(unused_steps)},
            }
        if ple is not None:
            np.savez(staging / "ple_metadata.npz", **ple)
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        load_cache_state(staging, expected_preprocess=args.preprocess)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Saved {args.preprocess} cache to {output}")
    return output


def parse_args() -> argparse.Namespace:
    """Choose raw data with explicit splits, or a physical cache to reuse for PLE."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path)
    source.add_argument(
        "--physical-cache", type=Path, help="PLE only: share this cache's inputs and inherit its splits"
    )
    parser.add_argument("--cache-dir", type=Path, required=True, help="Exact output directory for this cache")
    parser.add_argument("--preprocess", choices=PREPROCESSING_METHODS, required=True)
    parser.add_argument("--train-timesteps", type=parse_timestep_range)
    parser.add_argument("--validation-timesteps", type=parse_timestep_range)
    parser.add_argument(
        "--embedding-groups",
        nargs="+",
        default=["all"],
        help="PLE only: all continuous groups (default), or e.g. tracer residual",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
