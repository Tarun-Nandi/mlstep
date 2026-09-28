"""Train TabM-mini K4 with direct supervision or optional XGBoost distillation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mlstep.cache import load_cache_state
from mlstep.data import N_FEATURES, class_counts, random_undersampling, training_index_pools
from mlstep.evaluation import FeatureMatrix, evaluate_probabilities, predict_joint_probabilities
from mlstep.model import (
    N_SEVERITY_CLASSES,
    Student,
    build_model,
    save_model,
)
from mlstep.precision import PRECISIONS, autocast_dtype, resolve_device

SEVERITY_LOSS_WEIGHT: Final = 0.1
LEARNING_RATE: Final = 1e-3
WEIGHT_DECAY: Final = 1e-4
GRADIENT_CLIP_NORM: Final = 5.0

DEFAULT_EPOCHS: Final = 120
DEFAULT_PATIENCE: Final = 12
DEFAULT_BATCH_SIZE: Final = 1024
DEFAULT_EVAL_BATCH_SIZE: Final = 4096
NEGATIVE_RATIO: Final = 256
DEFAULT_CACHE_RESIDENCY: Final = "auto"
CACHE_RESIDENCIES: Final = ("auto", "host", "cuda")

GPU_DATA_RESERVE_BYTES: Final = 16 * 2**30
LOAD_CHUNK_BYTES: Final = 512 * 2**20
PROGRESS_UPDATES_PER_PHASE: Final = 10
EPOCH_SEED_STRIDE: Final = 10_000
MATRIX_NDIM: Final = 2
DETECTOR_LOGIT_NDIM: Final = 2
SEVERITY_LOGIT_NDIM: Final = 3


def compute_loss(
    detector_logits: torch.Tensor,
    severity_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    delta_s: float,
    has_positive: bool | None = None,
    teacher: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.05,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return the FP32 supervised loss, plus optional teacher-response losses.

    Inputs are member logits (K, B)/(K, B, 4) and integer labels (B,) in 0–4.
    ``delta_s`` corrects negative undersampling in supervised detector BCE;
    severity CE uses only true-positive rows. Teacher rows contain a population
    detector margin and four conditional severity probabilities. KD adds detector
    BCE and teacher-positive-weighted severity KL, both scaled by temperature².

    Run setup validates scalar settings and teacher distributions. A supplied
    ``has_positive`` must describe this batch; precomputing it on CPU avoids
    checking a CUDA tensor just to choose the positive-only loss branch.
    """
    # Keep rare-event margins, softmax and loss reductions in FP32 under AMP.
    detector_logits = detector_logits.float()
    severity_logits = severity_logits.float()
    valid_shapes = (
        detector_logits.ndim == DETECTOR_LOGIT_NDIM
        and severity_logits.ndim == SEVERITY_LOGIT_NDIM
        and labels.ndim == 1
        and severity_logits.shape[:2] == detector_logits.shape
        and detector_logits.shape[1] == len(labels)
        and severity_logits.shape[2] == N_SEVERITY_CLASSES
    )
    if not valid_shapes:
        msg = "detector, severity, and label shapes are inconsistent"
        raise ValueError(msg)
    if labels.is_floating_point() or labels.is_complex():
        msg = "labels must use an integer tensor dtype"
        raise TypeError(msg)
    positive = labels > 0
    binary_target = positive.float().unsqueeze(0).expand_as(detector_logits)
    # The sampling correction belongs only in supervised BCE. Unshifted logits
    # represent population odds for prediction and for matching teacher margins.
    loss = F.binary_cross_entropy_with_logits(detector_logits + delta_s, binary_target)
    if has_positive is None:
        has_positive = bool(positive.any())
    if has_positive:
        positive_logits = severity_logits[:, positive, :]
        positive_targets = (labels[positive] - 1).unsqueeze(0).expand(detector_logits.shape[0], -1)
        loss = loss + SEVERITY_LOSS_WEIGHT * F.cross_entropy(
            positive_logits.reshape(-1, N_SEVERITY_CLASSES),
            positive_targets.reshape(-1),
        )
    if teacher is not None:
        if teacher.shape != (len(labels), N_SEVERITY_CLASSES + 1):
            raise ValueError("teacher targets must contain a margin and four severity probabilities per row")
        teacher = teacher.float()
        margin = teacher[:, 0]
        detector_target = torch.sigmoid(margin / temperature).unsqueeze(0).expand_as(detector_logits)
        detector_kd = F.binary_cross_entropy_with_logits(detector_logits / temperature, detector_target)
        # Temperature acts on log probabilities, since the teacher stores no severity logits.
        conditional = teacher[:, 1:].clamp_min(torch.finfo(torch.float32).tiny)
        conditional = torch.softmax(conditional.log() / temperature, dim=-1)
        log_severity = F.log_softmax(severity_logits / temperature, dim=-1)
        conditional = conditional.unsqueeze(0).expand_as(log_severity)
        severity_kl = F.kl_div(log_severity, conditional, reduction="none").sum(dim=-1).mean(dim=0)
        # Severity matters most where the teacher predicts a positive event.
        gate = torch.sigmoid(margin)
        severity_kd = (gate * severity_kl).sum() / gate.sum().clamp_min(torch.finfo(torch.float32).tiny)
        loss = loss + temperature**2 * (alpha * detector_kd + beta * severity_kd)
    return loss


def _early_stopping_key(metrics: dict) -> tuple[float, float, float]:
    """Prefer lower RPS, then lower NLL, then higher detector AP."""
    probability = metrics["probability"]
    values = (
        float(probability["ranked_probability_score"]),
        float(probability["negative_log_likelihood"]),
        float(metrics["ap"]),
    )
    if not all(np.isfinite(value) for value in values):
        msg = "early-stopping metrics must be finite"
        raise ValueError(msg)
    rps, nll, ap = values
    return -rps, -nll, ap


def _load_array_resident(path: Path) -> np.ndarray:
    """Read a cache array into host RAM in chunks, checking floats while copying.

    The memory map avoids a second complete source allocation. The returned
    array is fully resident, so epoch sampling does not trigger scattered disk IO.
    """
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    destination = np.empty(source.shape, dtype=source.dtype)
    bytes_per_row = max(1, source[0:1].nbytes)
    rows_per_chunk = max(1, LOAD_CHUNK_BYTES // bytes_per_row)
    n_chunks = (len(source) + rows_per_chunk - 1) // rows_per_chunk
    progress_every = max(1, n_chunks // PROGRESS_UPDATES_PER_PHASE)
    started = time.perf_counter()
    print(f"Loading {path.name}: {source.nbytes / 2**30:.2f} GiB", flush=True)
    for chunk_number, start in enumerate(range(0, len(source), rows_per_chunk), start=1):
        stop = min(start + rows_per_chunk, len(source))
        block = source[start:stop]
        if np.issubdtype(block.dtype, np.floating) and not np.isfinite(block).all():
            msg = f"{path.name} contains non-finite values"
            raise ValueError(msg)
        destination[start:stop] = block
        if chunk_number % progress_every == 0 or chunk_number == n_chunks:
            print(
                f"  {stop:,}/{len(source):,} rows ({stop / len(source):.0%}) in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return destination


def _place_feature_matrices(
    train_x: np.ndarray,
    val_x: np.ndarray,
    device: torch.device,
    requested_residency: str,
) -> tuple[FeatureMatrix, FeatureMatrix, str]:
    """Choose host or CUDA storage for both full feature caches and return that choice.

    Auto uses CUDA only when both caches fit with a 16 GiB working reserve.
    Explicit CUDA requests must also fit. The reserve is a heuristic: sampled
    epochs, embeddings and optimiser state still need additional device memory.
    """
    if requested_residency == "host" or device.type != "cuda":
        return train_x, val_x, "host"

    feature_bytes = train_x.nbytes + val_x.nbytes
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if feature_bytes + GPU_DATA_RESERVE_BYTES > free_bytes:
        if requested_residency == "auto":
            return train_x, val_x, "host"
        raise MemoryError(
            f"CUDA feature residency needs {feature_bytes / 2**30:.2f} GiB plus "
            f"a {GPU_DATA_RESERVE_BYTES / 2**30:.0f} GiB working reserve"
        )
    print(f"Staging {feature_bytes / 2**30:.2f} GiB of features on {device}", flush=True)
    train_device = torch.from_numpy(train_x).to(device)
    val_device = torch.from_numpy(val_x).to(device)
    return train_device, val_device, "cuda"


def _prepare_epoch(
    features: FeatureMatrix,
    labels: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    negative_ratio: int,
    seed: int,
    batch_size: int,
    device: torch.device,
    teacher_targets: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[bool, ...], torch.Tensor | None]:
    """Sample one epoch and put aligned features, labels and teacher rows on device.

    Keep every positive and sample negatives without replacement using the seed.
    CPU batch-positive flags avoid a device sync in compute_loss. The complete
    sampled epoch must fit on device, independently of the minibatch size.
    """
    indices = random_undersampling(positive_indices, negative_indices, negative_ratio, seed)
    if not len(indices):
        msg = "a sampled epoch must contain at least one training row"
        raise ValueError(msg)
    selected_labels = np.ascontiguousarray(labels[indices])
    batch_has_positive = tuple(
        bool(np.any(selected_labels[start : start + batch_size] > 0))
        for start in range(0, len(selected_labels), batch_size)
    )
    if isinstance(features, torch.Tensor):
        index_tensor = torch.from_numpy(indices).to(device)
        epoch_x = torch.index_select(features, 0, index_tensor)
    else:
        selected = np.ascontiguousarray(features[indices])
        epoch_x = torch.from_numpy(selected).to(device)
    epoch_y = torch.from_numpy(selected_labels).to(device)
    epoch_teacher = None
    if teacher_targets is not None:
        epoch_teacher = torch.from_numpy(np.ascontiguousarray(teacher_targets[indices])).to(device)
    return epoch_x, epoch_y, batch_has_positive, epoch_teacher


def train_epoch(
    model: Student,
    optimizer: torch.optim.Optimizer,
    features: FeatureMatrix,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    negative_ratio: int,
    seed: int,
    delta_s: float,
    *,
    positive_indices: np.ndarray | None = None,
    negative_indices: np.ndarray | None = None,
    teacher_targets: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 0.05,
    temperature: float = 1.0,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run optimiser steps over a newly sampled epoch and return mean batch loss.

    The model supplies logits; this module supplies supervised/KD objectives.
    Autocast applies to the forward pass, losses stay FP32, and FP16 uses gradient
    scaling. Unscale before clipping so the clipping bound applies to real
    gradients. The reported mean weights each batch equally, including the last.
    """
    model.train()
    if positive_indices is None or negative_indices is None:
        positive_indices, negative_indices = training_index_pools(labels)
    epoch_x, epoch_y, batch_has_positive, epoch_teacher = _prepare_epoch(
        features,
        labels,
        positive_indices,
        negative_indices,
        negative_ratio,
        seed,
        batch_size,
        device,
        teacher_targets,
    )

    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    total_loss = torch.zeros((), device=device)
    batches = 0
    for batch_number, start in enumerate(range(0, len(epoch_y), batch_size)):
        batch_x = epoch_x[start : start + batch_size]
        batch_y = epoch_y[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            detector_logits, severity_logits = model.forward_logits(batch_x)
        loss = compute_loss(
            detector_logits,
            severity_logits,
            batch_y,
            delta_s=delta_s,
            has_positive=batch_has_positive[batch_number],
            teacher=None if epoch_teacher is None else epoch_teacher[start : start + batch_size],
            alpha=alpha,
            beta=beta,
            temperature=temperature,
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach()
        batches += 1
    return float((total_loss / batches).cpu())


def _validate_args(args: argparse.Namespace) -> torch.device:
    """Check CLI settings once before allocating caches; return the resolved device.

    Argparse supplies integer counts. Check their ranges, supported precision
    and residency, and finite nonnegative KD weights with positive temperature.
    """
    for name in ("epochs", "patience", "batch_size", "eval_batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.seed < 0:
        msg = "seed must be a non-negative integer"
        raise ValueError(msg)
    if args.cache_residency not in CACHE_RESIDENCIES:
        msg = f"cache-residency must be one of {CACHE_RESIDENCIES}"
        raise ValueError(msg)
    device = resolve_device(args.device)
    autocast_dtype(device, args.precision)
    if (
        not all(np.isfinite(v) for v in (args.alpha, args.beta, args.kd_temperature))
        or min(args.alpha, args.beta) < 0
        or args.kd_temperature <= 0
    ):
        raise ValueError("KD weights must be finite and non-negative, and temperature positive")
    if args.cache_residency == "cuda" and device.type != "cuda":
        msg = "--cache-residency cuda requires a CUDA device"
        raise ValueError(msg)
    return device


def _check_loaded_arrays(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> None:
    """Check cache array shapes, dtypes and label classes before sampling.

    Metadata checks cannot establish the actual array layout. Floating-point
    finiteness was already checked during the sequential load.
    """
    for split, features, labels in (("train", train_x, train_y), ("val", val_x, val_y)):
        if features.ndim != MATRIX_NDIM or not len(features) or features.shape[1] != N_FEATURES:
            raise ValueError(f"{split} features must be a nonempty {N_FEATURES}-column matrix")
        if labels.shape != (len(features),):
            raise ValueError(f"{split} labels must match feature rows")
        if features.dtype != np.float32 or labels.dtype != np.int64:
            raise TypeError(f"{split} features must be float32 and labels int64")
        class_counts(labels)


def _train_until_stopped(
    model: Student,
    optimizer: torch.optim.Optimizer,
    train_x: FeatureMatrix,
    train_y: np.ndarray,
    val_x: FeatureMatrix,
    val_y: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    delta_s: float,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    teacher_targets: np.ndarray | None = None,
) -> tuple[int, dict[str, torch.Tensor]]:
    """Select the best epoch by validation scores and return its independent CPU state.

    Reuse one gradient scaler across epochs. Each epoch resamples negatives with
    a deterministic seed. Stop after patience consecutive non-improvements;
    nothing is saved to disk until run restores and saves the selected state.
    """
    best_key: tuple[float, float, float] | None = None
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    amp_dtype = autocast_dtype(device, args.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            optimizer,
            train_x,
            train_y,
            device,
            args.batch_size,
            NEGATIVE_RATIO,
            args.seed * EPOCH_SEED_STRIDE + epoch,
            delta_s,
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            teacher_targets=teacher_targets,
            alpha=args.alpha,
            beta=args.beta,
            temperature=args.kd_temperature,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        if not np.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        # Validation needs parameters but no gradients; release the last batch's
        # gradients before allocating evaluation activations and output buffers.
        optimizer.zero_grad(set_to_none=True)
        probabilities, detector = predict_joint_probabilities(
            model,
            val_x,
            device,
            batch_size=args.eval_batch_size,
            precision=args.precision,
        )
        metrics = evaluate_probabilities(probabilities, val_y, detector)
        # Only scalar metrics are needed during the next epoch.
        del probabilities, detector
        key = _early_stopping_key(metrics)
        print(
            f"Epoch {epoch:3d} | loss {loss:.5f} | "
            f"validation RPS {metrics['probability']['ranked_probability_score']:.7g}",
            flush=True,
        )

        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            # One independent CPU copy, including when training already runs on CPU.
            best_state = {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    if best_epoch is None or best_state is None:
        msg = "training completed without a valid model"
        raise RuntimeError(msg)
    return best_epoch, best_state


def run(args: argparse.Namespace) -> Path:
    """Load caches, train the shared model, and save its selected weights and settings.

    Optional teacher responses must match the cache's row order and provenance.
    This function computes sampling/prior statistics; model.py owns architecture,
    parameter initialisation and weights serialization. Existing outputs are
    preserved, and the matching preprocessing cache remains necessary for use.
    """
    device = _validate_args(args)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.manual_seed(args.seed)
    cache_dir = args.cache_dir.expanduser().resolve()
    metadata, _, ple = load_cache_state(cache_dir)
    preprocess = metadata["preprocess"]

    train_x_host = _load_array_resident(cache_dir / "train_x.npy")
    train_y = _load_array_resident(cache_dir / "train_y.npy")
    val_x_host = _load_array_resident(cache_dir / "val_x.npy")
    val_y = _load_array_resident(cache_dir / "val_y.npy")
    _check_loaded_arrays(train_x_host, train_y, val_x_host, val_y)

    teacher_targets = None
    if args.teacher_dir is not None:
        teacher_metadata = json.loads((args.teacher_dir / "metadata.json").read_text())
        if any(teacher_metadata.get(key) != metadata.get(key) for key in ("features", "data_dir", "train", "val")):
            raise ValueError("teacher targets do not match the neural cache features, data and splits")
        teacher_targets = _load_array_resident(args.teacher_dir / "train_teacher.npy")
        if teacher_targets.shape != (len(train_y), N_SEVERITY_CLASSES + 1) or teacher_targets.dtype != np.float32:
            raise ValueError("teacher targets must be a matching float32 five-column matrix")
        for start in range(0, len(train_y), 65_536):
            conditional = teacher_targets[start : start + 65_536, 1:]
            if np.any(conditional < 0) or not np.allclose(conditional.sum(axis=1), 1, atol=1e-5):
                raise ValueError("teacher severity targets must be probability distributions")

    positive_indices, negative_indices = training_index_pools(train_y)
    n_positives = len(positive_indices)
    n_negatives = len(negative_indices)
    if not n_positives or not n_negatives:
        msg = "training data must contain positive and negative examples"
        raise ValueError(msg)
    n_sampled_negatives = min(n_negatives, NEGATIVE_RATIO * n_positives)
    # All positives survive sampling; only negative odds change.
    delta_s = float(np.log(n_negatives / n_sampled_negatives))
    population_log_odds = float(np.log(n_positives / n_negatives))

    train_x, val_x, residency = _place_feature_matrices(
        train_x_host,
        val_x_host,
        device,
        args.cache_residency,
    )
    if residency == "cuda":
        del train_x_host, val_x_host

    model = build_model(preprocess, None if ple is None else ple["ple_indices"])
    model.initialize_detector_bias(population_log_odds)
    if ple is not None:
        model.set_ple_bin_boundaries(ple["bin_boundaries"])
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(
        f"Training TabM-mini K={model.k} {list(model.hidden_layers)} on {device} "
        f"with {len(train_y):,} rows ({n_positives:,} positives)",
        flush=True,
    )
    best_epoch, best_state = _train_until_stopped(
        model,
        optimizer,
        train_x,
        train_y,
        val_x,
        val_y,
        device,
        args,
        delta_s,
        positive_indices,
        negative_indices,
        teacher_targets,
    )
    model.load_state_dict(best_state, strict=True)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_path = args.output_dir / f"model_{preprocess}_seed{args.seed}.pt"
    save_model(model_path, model, preprocess)
    summary = {
        "mode": "distilled" if teacher_targets is not None else "supervised",
        "seed": args.seed,
        "selected_epoch": best_epoch,
        "preprocess": preprocess,
        "embedded_features": model.ple_indices,
        "precision": str(autocast_dtype(device, args.precision) or "float32"),
        "architecture": {"members": model.k, "hidden": model.hidden_layers, "dropout": model.dropout},
        "training": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": args.batch_size,
            "negative_ratio": NEGATIVE_RATIO,
            "epochs": args.epochs,
            "patience": args.patience,
            "severity_weight": SEVERITY_LOSS_WEIGHT,
        },
        "distillation": None
        if teacher_targets is None
        else {
            "teacher_dir": str(args.teacher_dir.resolve()),
            "alpha": args.alpha,
            "beta": args.beta,
            "temperature": args.kd_temperature,
        },
        "data": metadata,
    }
    (args.output_dir / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved model from epoch {best_epoch}: {model_path}", flush=True)
    return model_path


def parse_args() -> argparse.Namespace:
    """Parse locations and resource controls for training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Prepared physical, ple or stretch128 cache")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the trained model")
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu or cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=PRECISIONS, default="auto")
    parser.add_argument("--teacher-dir", type=Path, help="Optional prepared XGBoost targets; omit for supervision only")
    parser.add_argument("--alpha", type=float, default=1.0, help="Detector distillation weight")
    parser.add_argument("--beta", type=float, default=0.05, help="Conditional-severity distillation weight")
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument(
        "--cache-residency",
        choices=CACHE_RESIDENCIES,
        default=DEFAULT_CACHE_RESIDENCY,
        help="Keep feature matrices in host RAM, CUDA memory, or choose automatically",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
