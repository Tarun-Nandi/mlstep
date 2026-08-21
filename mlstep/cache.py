"""Cache preprocessed features and optional teacher targets for fast iteration."""

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import QuantileTransformer

from mlstep.data import (
    DATA,
    FEATURES,
    N_CLASSES,
    Feature,
    Preprocesser,
    RobustPreprocesser,
    StretchPreprocesser,
    chunks,
    discover_timesteps,
    feature_names,
    load_timesteps,
    split_timesteps,
)

# PLE64 configuration
PLE64_N_FEATURES = 64
PLE64_N_BINS = 48
PLE64_EMBEDDING_DIM = 12
PLE64_QUANTILE_SAMPLE_ROWS = 1_048_576
PLE_IMPORTANCE_TYPE = "total_gain"
MIN_PLE_BINS = 2
TIMESTEP_RANGE_PARTS = 2
MISSING_TIMESTEP_PREVIEW = 8

DEFAULT_OUTPUT_DIR: Final = Path(__file__).resolve().parent / "runs"
DEFAULT_CACHE_DIR: Final = Path("/rds/user/rc-nand1/hpc-work/mlstep/cache")
STRETCH_METHODS: Final = frozenset({"stretch32", "stretch128", "raw-stretch128"})


def parse_timestep_range(value: str) -> tuple[int, ...]:
    """Parse an inclusive ``START:STOP`` timestep range."""
    parts = value.split(":")
    if len(parts) != TIMESTEP_RANGE_PARTS:
        msg = "timestep ranges must use inclusive START:STOP syntax"
        raise argparse.ArgumentTypeError(msg)
    try:
        start, stop = (int(part) for part in parts)
    except ValueError as error:
        msg = "timestep range bounds must be integers"
        raise argparse.ArgumentTypeError(msg) from error
    if start < 0 or stop < start:
        msg = "timestep ranges must be non-negative and STOP must be at least START"
        raise argparse.ArgumentTypeError(msg)
    return tuple(range(start, stop + 1))


def resolve_timestep_split(
    discovered: tuple[int, ...],
    train_steps: tuple[int, ...] | None,
    validation_steps: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """Resolve an explicit chronological split or retain the legacy default."""
    if train_steps is None and validation_steps is None:
        train, validation, _ = split_timesteps(discovered)
        return train, validation, "chronological-70-15-15-v1"
    if train_steps is None or validation_steps is None:
        msg = "--train-timesteps and --validation-timesteps must be supplied together"
        raise ValueError(msg)
    if not train_steps or not validation_steps:
        msg = "training and validation timestep ranges must both be non-empty"
        raise ValueError(msg)

    requested = (*train_steps, *validation_steps)
    missing = sorted(set(requested).difference(discovered))
    if missing:
        preview = ", ".join(str(step) for step in missing[:MISSING_TIMESTEP_PREVIEW])
        suffix = "..." if len(missing) > MISSING_TIMESTEP_PREVIEW else ""
        msg = f"requested timesteps are not present after spin-up exclusion: {preview}{suffix}"
        raise ValueError(msg)
    overlap = sorted(set(train_steps).intersection(validation_steps))
    if overlap:
        msg = f"training and validation timesteps overlap: {overlap[0]}"
        raise ValueError(msg)
    if max(train_steps) >= min(validation_steps):
        msg = "training timesteps must all precede validation timesteps"
        raise ValueError(msg)
    return train_steps, validation_steps, "explicit-inclusive-ranges-v1"


def resolve_cache_path(args: argparse.Namespace) -> Path:
    """Return an explicit cache leaf or the backward-compatible default."""
    cache_path = getattr(args, "cache_path", None)
    if cache_path is not None:
        return Path(cache_path)
    return Path(args.cache_dir) / "week" / args.preprocess


def _prepare_cache_directory(cache_dir: Path, teacher_path: Path | None) -> None:
    """Create the leaf and reject targets stale from a teacherful cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if teacher_path is not None:
        return
    stale_teacher_paths = sorted(cache_dir.glob("*_teacher_*.npy"))
    if stale_teacher_paths:
        names = ", ".join(path.name for path in stale_teacher_paths)
        msg = f"Teacherless cache output contains stale teacher targets: {names}"
        raise FileExistsError(msg)


def detector_margin(logits: np.ndarray, delta: float) -> np.ndarray:
    """Re-calibrate the probabilities of halving to counter the effect of undersampling negatives."""
    logsumexp_1to4 = np.logaddexp.reduce(logits[:, 1:5], axis=1)
    return logsumexp_1to4 - logits[:, 0] - delta


def severity_distribution(logits: np.ndarray) -> np.ndarray:
    """Undersampling affects P(positive) but not P(class | positive)."""
    logits_1to4 = logits[:, 1:5]
    logsumexp_1to4 = np.logaddexp.reduce(logits_1to4, axis=1, keepdims=True)
    return np.exp(logits_1to4 - logsumexp_1to4)


def _apply_preprocessing(
    x: np.ndarray,
    method: str,
    features: tuple[Feature, ...] | None = None,
) -> tuple[Preprocesser | QuantileTransformer | RobustPreprocesser | StretchPreprocesser | None, np.ndarray]:
    """Apply preprocessing method and return fitted preprocessor + transformed data."""
    if method == "physical":
        if features is None:
            msg = "features required for physical preprocessing"
            raise ValueError(msg)
        return Preprocesser.fit_transform(x, features)

    if method == "quantile":
        n_quantiles = min(100_000, len(x))
        transformer = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            subsample=10_000_000,
            random_state=0,
        )
        # Avoid full sized copy
        x_transformed = transformer.fit_transform(x.astype(np.float32, copy=False))
        return transformer, x_transformed

    if method == "robust":
        if features is None:
            msg = "features required for robust preprocessing"
            raise ValueError(msg)
        return RobustPreprocesser.fit_transform(x, features, clip_threshold=3.0)

    if method in STRETCH_METHODS:
        if features is None:
            msg = "features required for stretch preprocessing"
            raise ValueError(msg)
        n_bins = 128 if method == "raw-stretch128" else int(method.removeprefix("stretch"))
        return StretchPreprocesser.fit_transform(
            x,
            features,
            n_bins=n_bins,
            apply_physical_transforms=method != "raw-stretch128",
        )

    if method == "ple64":
        # PLE64 uses physical preprocessing for features
        # The PLE embedding is applied in the model, not during preprocessing
        if features is None:
            msg = "features required for PLE64 preprocessing"
            raise ValueError(msg)
        return Preprocesser.fit_transform(x, features)

    if method == "raw":
        return None, x

    msg = f"Unknown preprocessing method: {method}"
    raise ValueError(msg)


def _load_features(
    data_dir: Path, train_steps: tuple[int, ...], val_steps: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the raw training and validation features."""
    print("\nLoading training data: ")
    train_x, train_y = load_timesteps(data_dir, train_steps, FEATURES)
    print(f"Loaded train_x: {train_x.shape}, {train_x.nbytes // 2**20:.1f} MB")
    print("Loading validation data: ")
    val_x, val_y = load_timesteps(data_dir, val_steps, FEATURES)
    print(f"Loaded val_x: {val_x.shape}, {val_x.nbytes // 2**20:.1f} MB")
    return train_x, train_y, val_x, val_y


def _preprocess_features(
    train_x: np.ndarray, val_x: np.ndarray, preprocess_method: str
) -> tuple[
    Preprocesser | QuantileTransformer | RobustPreprocesser | StretchPreprocesser | None, np.ndarray, np.ndarray
]:
    """Fit preprocessing on training data and transform both of the splits."""
    print(f"Preprocessing training ({preprocess_method})...")
    feature_methods = {"physical", "robust", "ple64", *STRETCH_METHODS}
    features_arg = FEATURES if preprocess_method in feature_methods else None
    preprocessor, train_x_processed = _apply_preprocessing(
        train_x,
        preprocess_method,
        features_arg,
    )
    print(f"Preprocessing validation ({preprocess_method})...")
    if preprocess_method in ("physical", "ple64"):
        val_x_processed = preprocessor.transform(val_x, copy=False)
    elif preprocess_method == "quantile":
        val_x_processed = preprocessor.transform(val_x.astype(np.float32, copy=False))
    elif preprocess_method == "robust" or preprocess_method in STRETCH_METHODS:
        val_x_processed = preprocessor.transform(val_x, copy=False)
    else:
        val_x_processed = val_x

    return preprocessor, train_x_processed, val_x_processed


def _save_features_and_labels(
    cache_dir: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    preprocessor: Preprocesser | QuantileTransformer | RobustPreprocesser | StretchPreprocesser | None,
    preprocess_method: str,
    ple_metadata: dict | None = None,
) -> None:
    """Save features, labels, and preprocessor state."""
    print("Saving features and labels...")
    np.save(cache_dir / "train_x.npy", train_x)
    np.save(cache_dir / "train_y.npy", train_y)
    np.save(cache_dir / "val_x.npy", val_x)
    np.save(cache_dir / "val_y.npy", val_y)

    if preprocess_method == "physical":
        preprocessor_state = {
            **preprocessor.state(),
            "features": [{"name": f.name, "channels": f.channels, "transform": f.transform} for f in FEATURES],
        }
        np.savez(cache_dir / "preprocessor_state.npz", **preprocessor_state)
    elif preprocess_method == "quantile":
        np.savez(
            cache_dir / "preprocessor_state.npz",
            n_quantiles_in=preprocessor.n_quantiles_,
            quantiles=preprocessor.quantiles_,
        )
    elif preprocess_method == "robust" or preprocess_method in STRETCH_METHODS:
        preprocessor_state = {
            **preprocessor.state(),
            "features": [{"name": f.name, "channels": f.channels, "transform": f.transform} for f in FEATURES],
        }
        np.savez(cache_dir / "preprocessor_state.npz", **preprocessor_state)
    elif preprocess_method == "ple64":
        preprocessor_state = {
            **preprocessor.state(),
            "features": [{"name": f.name, "channels": f.channels, "transform": f.transform} for f in FEATURES],
        }
        np.savez(cache_dir / "preprocessor_state.npz", **preprocessor_state)
        # Save PLE-specific metadata
        if ple_metadata is not None:
            np.savez(
                cache_dir / "ple_metadata.npz",
                **{k: np.asarray(v) if isinstance(v, (list, np.ndarray)) else v for k, v in ple_metadata.items()},
            )
    else:
        np.savez(cache_dir / "preprocessor_state.npz", method=preprocess_method)


def _load_teacher(teacher_path: Path) -> tuple[xgb.XGBClassifier, list[int], list[int], float]:
    """Load teacher model and extract class counts for prior correction."""
    print(f"\nLoading teacher from {teacher_path}...")
    teacher = xgb.XGBClassifier()
    teacher.load_model(str(teacher_path))
    teacher.set_params(
        tree_method="hist",
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        max_delta_step=1,
        max_bin=128,
    )

    teacher_json = Path(teacher_path).with_suffix(".json")
    with open(teacher_json) as f:
        teacher_meta = json.load(f)
    train_class_counts = teacher_meta["data"]["train_class_counts"]
    train_class_counts_used = teacher_meta["data"]["training_class_counts_used"]

    delta_t = np.log(train_class_counts[0] / train_class_counts_used[0])
    print(f"Teacher delta_T = log({train_class_counts[0]} / {train_class_counts_used[0]}) = {delta_t:.4f}")

    return teacher, train_class_counts, train_class_counts_used, delta_t


def _stratified_sample_indices(n_rows: int, max_rows: int, seed: int = 0) -> np.ndarray:
    """Return a deterministic sample spread across an ordered training matrix."""
    if n_rows < 1:
        msg = "cannot sample an empty feature matrix"
        raise ValueError(msg)
    if max_rows < 1:
        msg = "max_rows must be positive"
        raise ValueError(msg)
    if n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)

    # One pseudo-random row per equally sized stratum avoids both a large
    # permutation allocation and bias toward the earliest chronological rows.
    starts = (np.arange(max_rows, dtype=np.int64) * n_rows) // max_rows
    stops = (np.arange(1, max_rows + 1, dtype=np.int64) * n_rows) // max_rows
    widths = stops - starts
    rng = np.random.default_rng(seed)
    offsets = np.floor(rng.random(max_rows) * widths).astype(np.int64)
    return starts + offsets


def _feature_ranges(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact per-column ranges with one sequential pass."""
    minimum = np.full(x.shape[1], np.inf, dtype=np.float64)
    maximum = np.full(x.shape[1], -np.inf, dtype=np.float64)
    for block in chunks(x):
        minimum = np.minimum(minimum, block.min(axis=0))
        maximum = np.maximum(maximum, block.max(axis=0))
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        msg = "PLE inputs must be finite"
        raise ValueError(msg)
    return minimum.astype(np.float32), maximum.astype(np.float32)


def _compute_ple_bin_boundaries(
    x: np.ndarray,
    feature_indices: list[int],
    n_bins: int,
    *,
    max_sample_rows: int = PLE64_QUANTILE_SAMPLE_ROWS,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Compute quantile bin boundaries for PLE.

    Args:
        x: Training data of shape (n_samples, n_features)
        feature_indices: Input columns to compute bins for
        n_bins: Number of bins per feature
        max_sample_rows: Maximum number of deterministic training rows used
        seed: Seed controlling the within-stratum sample position

    Returns
    -------
        Bin boundaries of shape ``(len(feature_indices), n_bins + 1)`` and
        the number of sampled rows.
    """
    if n_bins < MIN_PLE_BINS:
        msg = "n_bins must be at least two"
        raise ValueError(msg)
    if not feature_indices:
        msg = "at least one PLE feature is required"
        raise ValueError(msg)

    sample_indices = _stratified_sample_indices(len(x), max_sample_rows, seed)
    quantile_probs = np.linspace(0.0, 1.0, n_bins + 1)
    boundaries = np.empty((len(feature_indices), n_bins + 1), dtype=np.float32)
    exact_minimum, exact_maximum = _feature_ranges(x)

    for output_index, feature_index in enumerate(feature_indices):
        sampled_column = np.asarray(x[sample_indices, feature_index], dtype=np.float32)
        feature_boundaries = np.quantile(sampled_column, quantile_probs, method="linear").astype(np.float32)
        # Preserve the actual train-domain endpoints even when the bounded
        # quantile sample misses an extreme observation.
        feature_boundaries[0] = exact_minimum[feature_index]
        feature_boundaries[-1] = exact_maximum[feature_index]
        boundaries[output_index] = np.maximum.accumulate(feature_boundaries)

    return boundaries, len(sample_indices)


def _teacher_total_gain(teacher: xgb.XGBClassifier, n_features: int) -> np.ndarray:
    """Return an explicit dense XGBoost total-gain vector."""
    scores = np.zeros(n_features, dtype=np.float64)
    raw_scores = teacher.get_booster().get_score(importance_type=PLE_IMPORTANCE_TYPE)
    names = feature_names(FEATURES)
    name_to_index = {name: index for index, name in enumerate(names)}
    unrecognized = []
    for key, value in raw_scores.items():
        if key.startswith("f") and key[1:].isdigit():
            index = int(key[1:])
        else:
            index = name_to_index.get(key, -1)
        if 0 <= index < n_features:
            scores[index] = float(value)
        else:
            unrecognized.append(key)
    if unrecognized:
        msg = f"Teacher importance contains unrecognized features: {', '.join(sorted(unrecognized)[:5])}"
        raise ValueError(msg)
    return scores


def _select_top_features(
    teacher: xgb.XGBClassifier,
    x: np.ndarray,
    n_top: int,
) -> tuple[list[int], np.ndarray, list[int]]:
    """Select top continuous features by XGBoost importance.

    Args:
        teacher: Fitted XGBoost teacher model
        x: Physically transformed training feature matrix
        n_top: Number of eligible top features to select

    Returns
    -------
        Tuple of selected indices, total-gain scores, and excluded indices.
    """
    if n_top < 1:
        msg = "n_top must be positive"
        raise ValueError(msg)
    names = feature_names(FEATURES)
    if len(names) != x.shape[1]:
        msg = "feature metadata does not match the training matrix"
        raise ValueError(msg)

    minimum, maximum = _feature_ranges(x)
    # PLE is a numerical representation. Bypass the known binary flag and all
    # exact constants; their raw scalar representation is lossless and cheaper.
    excluded = [index for index, name in enumerate(names) if name == "stratflag" or minimum[index] == maximum[index]]
    excluded_set = set(excluded)
    eligible = [index for index in range(x.shape[1]) if index not in excluded_set]
    if len(eligible) < n_top:
        msg = f"Only {len(eligible)} eligible continuous features are available for PLE; requested {n_top}"
        raise ValueError(msg)

    importance = _teacher_total_gain(teacher, x.shape[1])
    selected = sorted(eligible, key=lambda index: (-importance[index], index))[:n_top]
    return selected, importance[selected].astype(np.float32), excluded


def _compute_and_save_teacher_targets(
    cache_dir: Path,
    teacher: xgb.XGBClassifier,
    train_x: np.ndarray,
    val_x: np.ndarray,
    delta_t: float,
) -> None:
    """Compute teacher logits, apply prior correction, and save targets."""
    print("Computing teacher logits on training set...")
    train_logits = teacher.predict(train_x, output_margin=True)

    print("Computing teacher logits on validation set...")
    val_logits = teacher.predict(val_x, output_margin=True)

    print("Computing corrected targets...")
    train_teacher_margin = detector_margin(train_logits, delta_t)
    val_teacher_margin = detector_margin(val_logits, delta_t)

    train_teacher_severity = severity_distribution(train_logits)
    val_teacher_severity = severity_distribution(val_logits)

    print(f"Detector margin range: [{val_teacher_margin.min():.2f}, {val_teacher_margin.max():.2f}]")

    np.save(cache_dir / "train_teacher_margin.npy", train_teacher_margin)
    np.save(cache_dir / "val_teacher_margin.npy", val_teacher_margin)
    np.save(cache_dir / "train_teacher_severity.npy", train_teacher_severity)
    np.save(cache_dir / "val_teacher_severity.npy", val_teacher_severity)


def _save_metadata(
    cache_dir: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    teacher_path: Path | None,
    delta_t: float | None,
    train_class_counts: list[int] | None,
    train_class_counts_used: list[int] | None,
    preprocess_method: str,
    preprocessor: Preprocesser | QuantileTransformer | RobustPreprocesser | StretchPreprocesser | None,
    ple_metadata: dict | None = None,
    *,
    data_dir: Path | None = None,
    discovered_steps: tuple[int, ...] = (),
    train_steps: tuple[int, ...] = (),
    validation_steps: tuple[int, ...] = (),
    split_protocol: str | None = None,
) -> None:
    """Save metadata for reproducibility."""
    metadata = {
        "cache_dir": str(cache_dir),
        "preprocess": preprocess_method,
        "preprocessing": _preprocessing_metadata(preprocess_method, preprocessor),
        "features": feature_names(FEATURES),
        "n_features": sum(f.channels for f in FEATURES),
        "train": {
            "shape": list(train_x.shape),
            "positives": int((train_y > 0).sum()),
            "class_counts": np.bincount(train_y, minlength=N_CLASSES).tolist(),
            "timesteps": list(train_steps),
        },
        "val": {
            "shape": list(val_x.shape),
            "positives": int((val_y > 0).sum()),
            "class_counts": np.bincount(val_y, minlength=N_CLASSES).tolist(),
            "timesteps": list(validation_steps),
        },
        "split": {
            "protocol": split_protocol,
            "data_dir": str(data_dir) if data_dir is not None else None,
            "discovered_timesteps": list(discovered_steps),
            "unused_timesteps": sorted(set(discovered_steps).difference((*train_steps, *validation_steps))),
        },
        "teacher": None,
    }
    if teacher_path is not None:
        if delta_t is None or train_class_counts is None or train_class_counts_used is None:
            msg = "complete teacher metadata is required when teacher_path is set"
            raise ValueError(msg)
        metadata["teacher"] = {
            "path": str(teacher_path),
            "delta_t": float(delta_t),
            "train_class_counts": train_class_counts,
            "train_class_counts_used": train_class_counts_used,
        }
    if ple_metadata is not None:
        metadata["piecewise_linear_embedding"] = {
            "version": str(ple_metadata["version"]),
            "n_ple_features": int(ple_metadata["n_ple_features"]),
            "n_bins": int(ple_metadata["n_bins"]),
            "embedding_dim": int(ple_metadata["embedding_dim"]),
            "output_dim": int(ple_metadata["output_dim"]),
            "importance_type": str(ple_metadata["importance_type"]),
            "quantile_fit_rows": int(ple_metadata["quantile_fit_rows"]),
            "quantile_fit_seed": int(ple_metadata["quantile_fit_seed"]),
            "selected_feature_indices": [int(index) for index in ple_metadata["ple_indices"]],
            "excluded_feature_indices": [int(index) for index in ple_metadata["excluded_indices"]],
            "unique_boundary_counts": [int(value) for value in ple_metadata["unique_boundary_counts"]],
        }

    metadata_path = cache_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_path}")


def _preprocessing_metadata(
    method: str,
    preprocessor: Preprocesser | QuantileTransformer | RobustPreprocesser | StretchPreprocesser | None,
) -> dict:
    """Describe the fitted input transform without serializing large arrays."""
    common = {
        "version": "student-input-preprocessing-v1",
        "method": method,
        "fit_split": "chronological training timesteps only",
        "output_dtype": "float32",
    }
    if method == "physical":
        return {
            **common,
            "pipeline": "feature-specific log1p/arcsinh transforms followed by mean/std standardization",
        }
    if method == "ple64":
        return {
            **common,
            "pipeline": (
                "feature-specific log1p/arcsinh transforms and mean/std standardization; "
                "selected continuous features receive online PLE in the student"
            ),
        }
    if method == "robust" and isinstance(preprocessor, RobustPreprocesser):
        return {
            **common,
            "pipeline": (
                "feature-specific log1p/arcsinh transforms, sampled train median/IQR scaling "
                "with exact-range fallback, then RealMLP smooth clipping"
            ),
            "quantile_fit_rows": preprocessor.fit_sample_rows,
            "quantile_fit_max_rows": int(preprocessor.state()["fit_sample_max_rows"][0]),
            "quantile_fit_seed": int(preprocessor.state()["fit_sample_seed"][0]),
            "quantile_method": "linear",
            "clip_threshold": preprocessor.clip_threshold,
        }
    if method in STRETCH_METHODS and isinstance(preprocessor, StretchPreprocesser):
        physical_prefix = (
            "feature-specific log1p/arcsinh transforms, "
            if preprocessor.apply_physical_transforms
            else "raw inputs without feature-specific physical transforms, "
        )
        metadata = {
            **common,
            "pipeline": (f"{physical_prefix}sampled train-quantile CDF stretch, then train mean/std standardization"),
            "n_bins": preprocessor.n_bins,
            "quantile_fit_rows": preprocessor.fit_sample_rows,
            "quantile_fit_max_rows": int(preprocessor.state()["fit_sample_max_rows"][0]),
            "quantile_fit_seed": int(preprocessor.state()["fit_sample_seed"][0]),
            "quantile_method": "linear",
        }
        if not preprocessor.apply_physical_transforms:
            metadata["input_transform"] = "identity"
        return metadata
    if method == "quantile" and isinstance(preprocessor, QuantileTransformer):
        return {
            **common,
            "pipeline": "sklearn QuantileTransformer on raw inputs with normal output distribution",
            "n_quantiles": int(preprocessor.n_quantiles_),
            "subsample": int(preprocessor.subsample),
            "random_state": 0,
        }
    if method == "raw":
        return {**common, "pipeline": "identity"}
    msg = f"Preprocessor object is inconsistent with method {method!r}"
    raise TypeError(msg)


def run(args: argparse.Namespace) -> None:
    """Execute the caching pipeline."""
    teacher_path = getattr(args, "teacher_path", None)
    if args.preprocess == "ple64" and teacher_path is None:
        msg = "PLE64 cache generation requires --teacher-path for feature selection"
        raise ValueError(msg)

    # Discover and split timesteps. Explicit ranges ensure the HPO cache never
    # reads the later locked-test period.
    timesteps = discover_timesteps(args.data_dir)
    train_steps, val_steps, split_protocol = resolve_timestep_split(
        timesteps,
        getattr(args, "train_timesteps", None),
        getattr(args, "validation_timesteps", None),
    )
    print(f"Train: t{train_steps[0]}-t{train_steps[-1]} ({len(train_steps)} steps)")
    print(f"Val: t{val_steps[0]}-t{val_steps[-1]} ({len(val_steps)} steps)")

    # Resolve and validate the destination before allocating the raw matrices.
    # In particular, a teacherless rebuild must not silently retain targets
    # from an older teacherful cache at the same path.
    cache_dir = resolve_cache_path(args)
    _prepare_cache_directory(cache_dir, teacher_path)

    # Load the raw features
    train_x, train_y, val_x, val_y = _load_features(
        args.data_dir,
        train_steps,
        val_steps,
    )

    teacher = None
    train_cc = None
    train_cc_used = None
    delta_t = None
    if teacher_path is not None:
        # Teacher targets remain available for the existing distillation and
        # PLE workflows, but a supervised Stretch128 cache does not need them.
        teacher, train_cc, train_cc_used, delta_t = _load_teacher(teacher_path)
        _compute_and_save_teacher_targets(cache_dir, teacher, train_x, val_x, delta_t)
    else:
        print("No teacher supplied; writing a supervised-only cache.")

    # Teacher inference is finished, so the arrays may now be transformed.
    preprocessor, train_x_processed, val_x_processed = _preprocess_features(
        train_x,
        val_x,
        args.preprocess,
    )

    # PLE boundaries must use the exact coordinate system consumed by the
    # network. For ple64 this is the physical log/arcsinh + standardised cache,
    # not the raw coordinate system used by the XGBoost teacher.
    ple_metadata = None
    if args.preprocess == "ple64":
        if teacher is None:
            msg = "PLE64 feature selection requires a loaded teacher"
            raise RuntimeError(msg)
        print("\nComputing PLE64 metadata on physically transformed training inputs...")
        ple_indices, ple_importance, excluded_indices = _select_top_features(
            teacher,
            train_x_processed,
            PLE64_N_FEATURES,
        )
        selected = set(ple_indices)
        bypass_indices = [index for index in range(train_x_processed.shape[1]) if index not in selected]
        ple_bin_boundaries, quantile_fit_rows = _compute_ple_bin_boundaries(
            train_x_processed,
            ple_indices,
            PLE64_N_BINS,
        )
        output_dim = PLE64_N_FEATURES * PLE64_EMBEDDING_DIM + len(bypass_indices)
        unique_boundary_counts = np.array(
            [len(np.unique(boundaries)) for boundaries in ple_bin_boundaries],
            dtype=np.int64,
        )

        print(f"PLE64: Selected top {PLE64_N_FEATURES} eligible continuous features by total gain")
        print(f"PLE64: Excluded {len(excluded_indices)} binary/constant features from selection")
        print(f"PLE64: Fitted boundaries from {quantile_fit_rows:,} deterministic training rows")
        print(f"PLE64: Bin boundaries shape: {ple_bin_boundaries.shape}")
        print(f"PLE64: Output dimension will be {output_dim} ({train_x_processed.shape[1]} -> {output_dim})")

        ple_metadata = {
            "version": "piecewise-linear-encoding-linear-projection-v1",
            "n_ple_features": PLE64_N_FEATURES,
            "n_bins": PLE64_N_BINS,
            "embedding_dim": PLE64_EMBEDDING_DIM,
            "output_dim": output_dim,
            "ple_indices": np.asarray(ple_indices, dtype=np.int64),
            "bypass_indices": np.asarray(bypass_indices, dtype=np.int64),
            "excluded_indices": np.asarray(excluded_indices, dtype=np.int64),
            "bin_boundaries": ple_bin_boundaries,
            "feature_importance": ple_importance,
            "importance_type": PLE_IMPORTANCE_TYPE,
            "quantile_fit_rows": quantile_fit_rows,
            "quantile_fit_max_rows": PLE64_QUANTILE_SAMPLE_ROWS,
            "quantile_fit_seed": 0,
            "unique_boundary_counts": unique_boundary_counts,
            "input_coordinate_system": "physical-log-arcsinh-standardized-v1",
        }

    # Save features and labels
    _save_features_and_labels(
        cache_dir,
        train_x_processed,
        train_y,
        val_x_processed,
        val_y,
        preprocessor,
        args.preprocess,
        ple_metadata,
    )

    # Save metadata
    _save_metadata(
        cache_dir,
        train_x_processed,
        train_y,
        val_x_processed,
        val_y,
        teacher_path,
        delta_t,
        train_cc,
        train_cc_used,
        args.preprocess,
        preprocessor,
        ple_metadata,
        data_dir=args.data_dir,
        discovered_steps=timesteps,
        train_steps=train_steps,
        validation_steps=val_steps,
        split_protocol=split_protocol,
    )

    # Summary
    print("\n=== Summary ===")
    print(f"Saved train_x: {train_x_processed.shape}, {train_x_processed.nbytes // 2**20:.1f} MB")
    print(f"Saved val_x: {val_x_processed.shape}, {val_x_processed.nbytes // 2**20:.1f} MB")
    total_size = sum(f.stat().st_size for f in cache_dir.glob("*"))
    print(f"Total cache size: {total_size // 2**20:.1f} MB")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cache preprocessed features and teacher targets for fast iteration.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
        help="Directory containing ncsteps_*.nc files",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory to save cached arrays",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        help="Exact cache output directory; overrides --cache-dir/week/PREPROCESS",
    )
    parser.add_argument(
        "--teacher-path",
        type=Path,
        help="Optional XGBoost .ubj teacher model; required only for PLE64 or distillation targets",
    )
    parser.add_argument(
        "--train-timesteps",
        type=parse_timestep_range,
        help="Inclusive training range as START:STOP; requires --validation-timesteps",
    )
    parser.add_argument(
        "--validation-timesteps",
        type=parse_timestep_range,
        help="Inclusive validation range as START:STOP; requires --train-timesteps",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="physical",
        choices=["physical", "quantile", "raw", "robust", "stretch32", "stretch128", "raw-stretch128", "ple64"],
        help="Preprocessing method: physical (log/arcsinh+standardize),"
        " quantile (force N(0,1)), raw (no transform),"
        " robust (physical transforms + robust scale/smooth clip),"
        " stretch32/128 (physical transforms + unsupervised CDF stretch),"
        " raw-stretch128 (raw inputs + unsupervised CDF stretch)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
