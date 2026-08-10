"""Cache preprocessed features and teacher targets for fast iteration."""

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
    discover_timesteps,
    feature_names,
    load_timesteps,
    split_timesteps,
)

DEFAULT_OUTPUT_DIR: Final = Path(__file__).resolve().parent / "runs"
DEFAULT_CACHE_DIR: Final = Path("/rds/user/rc-nand1/hpc-work/mlstep/cache")


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
) -> tuple[Preprocesser | QuantileTransformer | None, np.ndarray]:
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
) -> tuple[Preprocesser | QuantileTransformer | None, np.ndarray, np.ndarray]:
    """Fit preprocessing on training data and transform both of the splits."""
    print(f"Preprocessing training ({preprocess_method})...")
    features_arg = FEATURES if preprocess_method == "physical" else None
    preprocessor, train_x_processed = _apply_preprocessing(
        train_x,
        preprocess_method,
        features_arg,
    )
    print(f"Preprocessing validation ({preprocess_method})...")
    if preprocess_method == "physical":
        val_x_processed = preprocessor.transform(val_x, copy=False)
    elif preprocess_method == "quantile":
        val_x_processed = preprocessor.transform(val_x.astype(np.float32, copy=False))
    else:
        val_x_processed = val_x

    return preprocessor, train_x_processed, val_x_processed


def _save_features_and_labels(
    cache_dir: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    preprocessor: Preprocesser | QuantileTransformer | None,
    preprocess_method: str,
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
    teacher_path: Path,
    delta_t: float,
    train_class_counts: list[int],
    train_class_counts_used: list[int],
    preprocess_method: str,
) -> None:
    """Save metadata for reproducibility."""
    metadata = {
        "cache_dir": str(cache_dir),
        "preprocess": preprocess_method,
        "features": feature_names(FEATURES),
        "n_features": sum(f.channels for f in FEATURES),
        "train": {
            "shape": list(train_x.shape),
            "positives": int((train_y > 0).sum()),
            "class_counts": np.bincount(train_y, minlength=N_CLASSES).tolist(),
        },
        "val": {
            "shape": list(val_x.shape),
            "positives": int((val_y > 0).sum()),
            "class_counts": np.bincount(val_y, minlength=N_CLASSES).tolist(),
        },
        "teacher": {
            "path": str(teacher_path),
            "delta_t": float(delta_t),
            "train_class_counts": train_class_counts,
            "train_class_counts_used": train_class_counts_used,
        },
    }

    metadata_path = cache_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_path}")


def run(args: argparse.Namespace) -> None:
    """Execute the caching pipeline."""
    # Discover and split timesteps
    timesteps = discover_timesteps(args.data_dir)
    train_steps, val_steps, _ = split_timesteps(timesteps)
    print(f"Train: t{train_steps[0]}-t{train_steps[-1]} ({len(train_steps)} steps)")
    print(f"Val: t{val_steps[0]}-t{val_steps[-1]} ({len(val_steps)} steps)")

    # Load the raw features
    train_x, train_y, val_x, val_y = _load_features(
        args.data_dir,
        train_steps,
        val_steps,
    )

    # Create variant specific directory
    cache_dir = args.cache_dir / "week" / args.preprocess
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load the raw-feature teacher
    teacher, train_cc, train_cc_used, delta_t = _load_teacher(args.teacher_path)

    # Compute and save teacher targets
    _compute_and_save_teacher_targets(cache_dir, teacher, train_x, val_x, delta_t)

    # Teacher inference is finished, so the arrays may now be transformed.
    preprocessor, train_x_processed, val_x_processed = _preprocess_features(
        train_x,
        val_x,
        args.preprocess,
    )

    # Save features and labels
    _save_features_and_labels(
        cache_dir,
        train_x_processed,
        train_y,
        val_x_processed,
        val_y,
        preprocessor,
        args.preprocess,
    )

    # Save metadata
    _save_metadata(
        cache_dir,
        train_x_processed,
        train_y,
        val_x_processed,
        val_y,
        args.teacher_path,
        delta_t,
        train_cc,
        train_cc_used,
        args.preprocess,
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
        "--teacher-path",
        type=Path,
        required=True,
        help="Path to XGBoost .ubj teacher model",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="physical",
        choices=["physical", "quantile", "raw"],
        help="Preprocessing method: physical (log/arcsinh+standardize), quantile (force N(0,1)), raw (no transform)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
