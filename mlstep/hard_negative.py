# ruff: noqa: EM101, EM102
"""Build a reproducible hard-negative pool from trained student checkpoints.

Only rows from ``train_x.npy`` whose corresponding ``train_y.npy`` label is
zero are ranking candidates or retained. Dense inference may also forward the
rare intervening positive training rows before discarding their scores.
Validation arrays are deliberately outside this module's input contract, so
validation examples can never enter the mined pool. Source checkpoints may
themselves have used validation for early stopping; their selection metadata
is retained rather than calling the overall procedure independent of
validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mlstep.data import FEATURES, N_CLASSES, feature_names
from mlstep.student import student_from_checkpoint

FORMAT_VERSION = "hard-negative-pool-v1"
HASH_BLOCK_BYTES = 8 * 1024 * 1024
MATRIX_NDIM = 2
DENSE_INDEX_MIN_FRACTION = 0.9

ScoreFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class HardNegativeSelection:
    """The selected global training-row indices and their mining summary."""

    indices: np.ndarray
    scores: np.ndarray
    negative_rows_scored: int
    chunks_scored: int


@dataclass(frozen=True)
class TrainingCache:
    """Validated, memory-mapped training cache and its provenance."""

    path: Path
    metadata: dict[str, Any]
    features: np.ndarray
    labels: np.ndarray
    class_counts: np.ndarray
    artifacts: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CheckpointEnsemble:
    """Compatible student models and immutable checkpoint provenance."""

    models: tuple[nn.Module, ...]
    artifacts: tuple[dict[str, Any], ...]
    comparable_config: dict[str, Any]
    objective: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def artifact_metadata(path: Path) -> dict[str, Any]:
    """Describe a file by resolved path, byte count, and content digest."""
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _chunked_class_counts(labels: np.ndarray, chunk_rows: int) -> np.ndarray:
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for start in range(0, labels.shape[0], chunk_rows):
        chunk = np.asarray(labels[start : start + chunk_rows])
        if np.any((chunk < 0) | (chunk >= N_CLASSES)):
            raise ValueError(f"Training labels must be in [0, {N_CLASSES - 1}]")
        counts += np.bincount(chunk.astype(np.int64, copy=False), minlength=N_CLASSES)
    return counts


def load_training_cache(  # noqa: PLR0912, PLR0915
    cache_path: Path, *, chunk_rows: int
) -> TrainingCache:
    """Validate and memory-map the training portion of a feature cache."""
    cache_path = cache_path.resolve(strict=True)
    required = ("SUCCESS", "metadata.json", "train_x.npy", "train_y.npy")
    for name in required:
        path = cache_path / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty cache artifact: {path}")
    success_path = cache_path / "SUCCESS"
    if success_path.read_text(encoding="utf-8").strip() != "0":
        raise ValueError(f"Invalid cache SUCCESS marker: {success_path}")

    metadata_path = cache_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("Cache metadata must be a JSON object")

    expected_features = feature_names(FEATURES)
    if metadata.get("features") != expected_features:
        raise ValueError("Cache feature names/order differ from the runtime schema")
    if metadata.get("n_features") != len(expected_features):
        raise ValueError("Cache n_features differs from the runtime schema")
    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("Cache metadata does not contain preprocessing provenance")
    if preprocessing.get("method") != metadata.get("preprocess"):
        raise ValueError("Cache preprocess and preprocessing.method disagree")
    if preprocessing.get("fit_split") != "chronological training timesteps only":
        raise ValueError("Cache preprocessor was not fitted on training timesteps only")
    if preprocessing.get("output_dtype") != "float32":
        raise ValueError("Cache preprocessing output must be float32")

    feature_path = cache_path / "train_x.npy"
    label_path = cache_path / "train_y.npy"
    # Copy-on-write keeps the file immutable while exposing writable views to
    # torch.from_numpy, so dense sequential scoring needs no host-side copy.
    features = np.load(feature_path, mmap_mode="c", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    if features.ndim != MATRIX_NDIM or features.shape[0] == 0:
        raise ValueError("train_x.npy must be a non-empty rank-2 array")
    if features.shape[1] != len(expected_features) or features.dtype != np.float32:
        raise ValueError("train_x.npy has an incompatible shape or dtype")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("train_y.npy must contain one label per training row")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("train_y.npy must have an integer dtype")

    train_metadata = metadata.get("train")
    if not isinstance(train_metadata, dict):
        raise ValueError("Cache metadata does not contain a train split")
    if train_metadata.get("shape") != list(features.shape):
        raise ValueError("Cache train shape metadata does not match train_x.npy")

    counts = _chunked_class_counts(labels, chunk_rows)
    if train_metadata.get("class_counts") != counts.tolist():
        raise ValueError("Cache train class counts do not match train_y.npy")
    if train_metadata.get("positives") != int(counts[1:].sum()):
        raise ValueError("Cache train positive count does not match train_y.npy")

    # Hash small logical provenance files, but do not add a second complete
    # 80-GiB cache scan before mining. Shapes, dtypes and class counts above
    # validate the arrays; metadata.json binds the exact cache contract.
    artifacts = {name: artifact_metadata(cache_path / name) for name in ("SUCCESS", "metadata.json")}
    for name in ("train_x.npy", "train_y.npy"):
        path = (cache_path / name).resolve(strict=True)
        artifacts[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": None}
    return TrainingCache(
        path=cache_path,
        metadata=metadata,
        features=features,
        labels=labels,
        class_counts=counts,
        artifacts=artifacts,
    )


def _comparable_checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "architecture",
        "architecture_version",
        "n_features",
        "hidden",
        "k",
        "dropout",
        "preprocess",
        "preprocessing",
        "ple_config",
        "prediction_temperature",
    )
    comparable = {key: config.get(key) for key in keys}
    comparable["prediction_temperature"] = config.get("prediction_temperature", 1.0)
    return comparable


def _validate_checkpoint_against_cache(
    checkpoint: dict[str, Any],
    cache: TrainingCache,
    *,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if checkpoint.get("checkpoint_role") != "canonical":
        raise ValueError(f"Checkpoint is not canonical: {checkpoint_path}")
    config = checkpoint.get("config")
    objective = checkpoint.get("objective")
    if not isinstance(config, dict) or not isinstance(objective, dict):
        raise ValueError(f"Checkpoint lacks config/objective metadata: {checkpoint_path}")
    if config.get("n_features") != cache.features.shape[1]:
        raise ValueError(f"Checkpoint n_features is incompatible with cache: {checkpoint_path}")
    if config.get("preprocess") != cache.metadata.get("preprocess"):
        raise ValueError(f"Checkpoint preprocessing mode is incompatible with cache: {checkpoint_path}")
    if config.get("preprocessing") != cache.metadata.get("preprocessing"):
        raise ValueError(f"Checkpoint preprocessing provenance is incompatible with cache: {checkpoint_path}")

    protocol = checkpoint.get("training_protocol")
    data_split = protocol.get("data_split") if isinstance(protocol, dict) else None
    if not isinstance(data_split, dict):
        raise ValueError(f"Checkpoint lacks training-split provenance: {checkpoint_path}")
    expected_train_timesteps = cache.metadata.get("train", {}).get("timesteps")
    if data_split.get("train_timesteps") != expected_train_timesteps:
        raise ValueError(f"Checkpoint train timesteps are incompatible with cache: {checkpoint_path}")
    if data_split.get("split") != cache.metadata.get("split"):
        raise ValueError(f"Checkpoint split metadata is incompatible with cache: {checkpoint_path}")
    return _comparable_checkpoint_config(config), objective


def load_checkpoint_ensemble(
    checkpoint_paths: Sequence[Path],
    cache: TrainingCache,
    *,
    device: torch.device,
) -> CheckpointEnsemble:
    """Load canonical, mutually compatible student checkpoints."""
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required")
    resolved_paths = [path.resolve(strict=True) for path in checkpoint_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Checkpoint paths must be unique")

    models: list[nn.Module] = []
    artifacts: list[dict[str, Any]] = []
    reference_config: dict[str, Any] | None = None
    reference_objective: dict[str, Any] | None = None
    for path in resolved_paths:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise TypeError(f"Checkpoint is not a mapping: {path}")
        comparable_config, objective = _validate_checkpoint_against_cache(raw, cache, checkpoint_path=path)
        if reference_config is None:
            reference_config = comparable_config
            reference_objective = objective
        elif comparable_config != reference_config or objective != reference_objective:
            raise ValueError("All mining checkpoints must have identical model/objective contracts")

        model = student_from_checkpoint(raw).to(device)
        model.eval()
        models.append(model)
        artifact = artifact_metadata(path)
        artifact["format_version"] = raw.get("format_version")
        artifact["checkpoint_role"] = raw["checkpoint_role"]
        artifact["seed"] = raw.get("seed")
        artifact["config"] = raw["config"]
        artifact["selection"] = raw.get("selection")
        artifacts.append(artifact)

    assert reference_config is not None
    assert reference_objective is not None
    return CheckpointEnsemble(
        models=tuple(models),
        artifacts=tuple(artifacts),
        comparable_config=reference_config,
        objective=reference_objective,
    )


def deterministic_top_k(
    indices: np.ndarray,
    scores: np.ndarray,
    pool_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top scores, breaking a boundary tie by the smaller row index."""
    indices = np.asarray(indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if indices.ndim != 1 or scores.ndim != 1 or indices.shape != scores.shape:
        raise ValueError("indices and scores must be same-length rank-1 arrays")
    if pool_size < 1 or pool_size > indices.size:
        raise ValueError("pool_size must be in [1, number of candidates]")
    if np.any(indices < 0) or np.any(~np.isfinite(scores)):
        raise ValueError("Candidate indices and scores must be valid and finite")

    if pool_size == indices.size:
        return indices.copy(), scores.copy()

    cutoff_position = scores.size - pool_size
    cutoff = np.partition(scores, cutoff_position)[cutoff_position]
    above = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)
    needed = pool_size - above.size
    if needed < 0 or needed > tied.size:
        raise RuntimeError("Internal top-k boundary calculation failed")
    if needed:
        tied_order = np.argsort(indices[tied], kind="stable")
        chosen = np.concatenate((above, tied[tied_order[:needed]]))
    else:
        chosen = above
    if chosen.size != pool_size:
        raise RuntimeError("Top-k selection did not produce the requested pool size")
    return indices[chosen], scores[chosen]


def mine_hard_negative_indices(
    labels: np.ndarray,
    score_function: ScoreFunction,
    *,
    pool_size: int,
    chunk_rows: int,
) -> HardNegativeSelection:
    """Rank class-zero rows sequentially and retain an exact global top-K."""
    if labels.ndim != 1:
        raise ValueError("labels must be rank 1")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")

    retained_indices = np.empty(0, dtype=np.int64)
    retained_scores = np.empty(0, dtype=np.float64)
    negative_rows_scored = 0
    chunks_scored = 0
    for start in range(0, labels.shape[0], chunk_rows):
        stop = min(start + chunk_rows, labels.shape[0])
        local_negative = np.flatnonzero(np.asarray(labels[start:stop]) == 0)
        if not local_negative.size:
            continue
        global_indices = local_negative.astype(np.int64, copy=False) + start
        scores = np.asarray(score_function(global_indices), dtype=np.float64)
        if scores.shape != global_indices.shape:
            raise ValueError("score_function must return one score per requested index")
        if np.any(~np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("Detector scores must be finite probabilities in [0, 1]")

        negative_rows_scored += global_indices.size
        chunks_scored += 1
        candidate_indices = np.concatenate((retained_indices, global_indices))
        candidate_scores = np.concatenate((retained_scores, scores))
        keep = min(pool_size, candidate_indices.size)
        retained_indices, retained_scores = deterministic_top_k(candidate_indices, candidate_scores, keep)

    if negative_rows_scored < pool_size:
        raise ValueError(f"Requested pool_size={pool_size}, but only {negative_rows_scored} negatives exist")
    if retained_indices.size != pool_size:
        raise RuntimeError("Mining did not produce the exact requested pool size")

    output_order = np.argsort(retained_indices, kind="stable")
    return HardNegativeSelection(
        indices=retained_indices[output_order],
        scores=retained_scores[output_order],
        negative_rows_scored=negative_rows_scored,
        chunks_scored=chunks_scored,
    )


class StudentDetectorScorer:
    """Score indexed feature rows with an equal-weight checkpoint ensemble."""

    def __init__(
        self,
        features: np.ndarray,
        models: Sequence[nn.Module],
        *,
        device: torch.device,
        batch_size: int,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.features = features
        self.models = tuple(models)
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def _score_rows(self, rows: np.ndarray) -> np.ndarray:
        """Score an already selected dense row matrix in bounded GPU batches."""
        scores = np.empty(len(rows), dtype=np.float64)
        for start in range(0, len(rows), self.batch_size):
            stop = min(start + self.batch_size, len(rows))
            batch = np.ascontiguousarray(rows[start:stop])
            if not batch.flags.writeable:
                batch = batch.copy()
            inputs = torch.from_numpy(batch).to(self.device)
            probability_sum: torch.Tensor | None = None
            for model in self.models:
                detector_logits, _ = model.forward_logits(inputs)
                temperature = float(getattr(model, "prediction_temperature", 1.0))
                probability = torch.sigmoid(detector_logits / temperature).mean(dim=0)
                probability_sum = probability if probability_sum is None else probability_sum + probability
            assert probability_sum is not None
            mean_probability = probability_sum / len(self.models)
            scores[start:stop] = mean_probability.detach().cpu().numpy()
        return scores

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        """Return detector probabilities for the requested global row indices.

        Mining requests are sorted and almost contiguous because positives are
        only about one row in ten thousand. Score their enclosing sequential
        span and mask afterward, avoiding a fancy-index copy of nearly every
        feature row. Sparse or unordered external requests retain the direct
        gather path.
        """
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("indices must be a one-dimensional integer array")
        if not len(indices):
            return np.empty(0, dtype=np.float64)
        if np.any(indices < 0) or np.any(indices >= len(self.features)):
            raise ValueError("indices are outside the feature matrix")

        sorted_unique = len(indices) == 1 or np.all(indices[1:] > indices[:-1])
        span_start = int(indices[0])
        span_stop = int(indices[-1]) + 1
        span_rows = span_stop - span_start
        if sorted_unique and len(indices) / span_rows >= DENSE_INDEX_MIN_FRACTION:
            span_scores = self._score_rows(self.features[span_start:span_stop])
            return span_scores[indices.astype(np.int64, copy=False) - span_start]

        rows = np.ascontiguousarray(self.features[indices.astype(np.int64, copy=False)])
        return self._score_rows(rows)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def _prepare_fresh_output(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    return output_dir


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_hard_negative_pool(
    *,
    cache_path: Path,
    checkpoint_paths: Sequence[Path],
    output_dir: Path,
    pool_size: int,
    chunk_rows: int,
    inference_batch_size: int,
    device_name: str,
    score_function: ScoreFunction | None = None,
) -> Path:
    """Validate inputs, mine the pool, and return the new output directory."""
    if pool_size < 1 or chunk_rows < 1 or inference_batch_size < 1:
        raise ValueError("pool_size and batch sizes must be positive")
    device = _resolve_device(device_name)
    cache = load_training_cache(cache_path, chunk_rows=chunk_rows)
    negative_count = int(cache.class_counts[0])
    if pool_size >= negative_count:
        raise ValueError(
            "pool_size must be smaller than the number of training negatives so a non-hard negative stratum remains"
        )
    ensemble = load_checkpoint_ensemble(checkpoint_paths, cache, device=device)

    output_dir = _prepare_fresh_output(output_dir)
    scorer = score_function
    if scorer is None:
        scorer = StudentDetectorScorer(
            cache.features,
            ensemble.models,
            device=device,
            batch_size=inference_batch_size,
        )
    selection = mine_hard_negative_indices(
        cache.labels,
        scorer,
        pool_size=pool_size,
        chunk_rows=chunk_rows,
    )
    if selection.negative_rows_scored != negative_count:
        raise RuntimeError("Mining did not score every class-zero training row exactly once")
    if not np.all(np.asarray(cache.labels[selection.indices]) == 0):
        raise RuntimeError("Selected pool unexpectedly contains a non-zero label")

    index_path = output_dir / "hard_negative_indices.npy"
    _atomic_save_npy(index_path, selection.indices.astype(np.int64, copy=False))
    index_artifact = artifact_metadata(index_path)
    report = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "source_split": "train",
            "source_label": 0,
            "pool_size": int(selection.indices.size),
            "negative_rows_scored": selection.negative_rows_scored,
            "chunks_scored": selection.chunks_scored,
            "chunk_rows": chunk_rows,
            "inference_batch_size": inference_batch_size,
            "checkpoint_aggregation": ("arithmetic mean of member-mean sigmoid detector probabilities"),
            "ranking": "higher detector probability, then lower global train row index",
            "output_order": "ascending global train row index",
            "minimum_selected_score": float(selection.scores.min()),
            "maximum_selected_score": float(selection.scores.max()),
        },
        "cache": {
            "path": str(cache.path),
            "preprocess": cache.metadata.get("preprocess"),
            "preprocessing": cache.metadata.get("preprocessing"),
            "features": cache.metadata.get("features"),
            "n_features": cache.features.shape[1],
            "train_shape": list(cache.features.shape),
            "train_class_counts": cache.class_counts.tolist(),
            "train_timesteps": cache.metadata.get("train", {}).get("timesteps"),
            "split": cache.metadata.get("split"),
            "artifacts": cache.artifacts,
            "validation_arrays_accessed": False,
        },
        "ensemble": {
            "checkpoint_count": len(ensemble.models),
            "model_config": ensemble.comparable_config,
            "objective": ensemble.objective,
            "checkpoints": list(ensemble.artifacts),
        },
        "output_artifacts": {"hard_negative_indices.npy": index_artifact},
    }
    _atomic_write_text(
        output_dir / "metadata.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(output_dir / "SUCCESS", "0\n")
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_paths",
        type=Path,
        nargs="+",
        required=True,
        help="One or more canonical student checkpoints",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--chunk-rows", type=int, default=262_144)
    parser.add_argument("--inference-batch-size", type=int, default=65_536)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the hard-negative mining command."""
    args = parse_args(argv)
    output_dir = build_hard_negative_pool(
        cache_path=args.cache_path,
        checkpoint_paths=args.checkpoint_paths,
        output_dir=args.output_dir,
        pool_size=args.pool_size,
        chunk_rows=args.chunk_rows,
        inference_batch_size=args.inference_batch_size,
        device_name=args.device,
    )
    print(f"Hard-negative pool written to {output_dir}")


if __name__ == "__main__":
    main()
