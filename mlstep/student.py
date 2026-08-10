"""Unified student model with optional distillation from XGBoost teacher.

The model uses a shared trunk with two heads:
- Detection head: predicts probability of any halving (population-scale)
- Severity head: predicts distribution over 4 halving classes (conditional)

Supports TabM-inspired parameter-efficient ensembling via input adapters.
https://arxiv.org/pdf/2410.24210
"""

import argparse
import json
import time
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mlstep.data import N_CLASSES, random_undersampling, training_index_pools
from mlstep.evaluation import evaluate, output_paths, write_json

# Default paths
DEFAULT_OUTPUT_DIR: Final = Path(__file__).resolve().parent / "runs"
# NOTE: change the directory (update to make all three directories accessible)
DEFAULT_CACHE_DIR: Final = Path("/rds/user/rc-nand1/hpc-work/mlstep/cache")

# Architecture defaults
HIDDEN_LAYERS: Final = [512, 256]
DROPOUT: Final = 0.05
GRADIENT_CLIP_NORM: Final = 5.0
MATRIX_NDIM: Final = 2
N_SEVERITY_CLASSES: Final = N_CLASSES - 1

# Training defaults
DEFAULT_EPOCHS: Final = 80
DEFAULT_PATIENCE: Final = 12
DEFAULT_BATCH_SIZE: Final = 1024
DEFAULT_NEGATIVE_RATIO: Final = 256
DEFAULT_TARGET_RECALL: Final = 0.97

# Interim week-experiment policy. At each epoch, use the highest detector
# threshold that achieves the requested recall. Then compare checkpoints
# lexicographically using these criteria, rather than by detector AP alone.
SELECTION_POLICY: Final = "recall-floor-exact-over-v1"
THRESHOLD_RULE: Final = "highest validation threshold achieving target recall"
THRESHOLD_SCORE: Final = "ensemble any-halving detector probability"
SELECTION_CRITERIA: Final = (
    "maximize severity.exact_on_positive_count",
    "minimize severity.overprediction_count_all",
    "minimize severity.overprediction_level_sum_all",
    "minimize severity.underprediction_level_sum_all",
    "maximize ap",
)
SEVERITY_DECISION: Final = "argmax conditional severity probability"


class Student(nn.Module):
    """Unified two-head student with optional parameter-efficient ensembling.

    Args:
        n_features: Number of input features
        hidden_layers: List of hidden layer sizes
        k: Number of ensemble members (1 = single model)
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_features: int,
        hidden_layers: list[int] | None = None,
        k: int = 1,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if k < 1:
            msg = "The number of ensemble members must be at least one"
            raise ValueError(msg)

        hidden_layers = hidden_layers or HIDDEN_LAYERS
        self.k = k
        self.n_features = n_features

        # Each member gets its own multiplicative scaling of the input features.
        # Shape: (k, n_features) — learnable per-member scaling
        self.adapters = nn.Parameter(torch.ones(k, n_features))

        # Build shared trunk layers
        layers: list[nn.Module] = []
        input_dim = n_features
        for hidden_dim in hidden_layers:
            layers.extend(
                (
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )
            input_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        # Two prediction heads
        self.detector_head = nn.Linear(hidden_layers[-1], 1)  # Output: margin m
        self.severity_head = nn.Linear(hidden_layers[-1], N_SEVERITY_CLASSES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the model and combine ensemble probabilities."""
        # (B, F) -> (K, B, F)
        x = x.unsqueeze(0) * self.adapters.unsqueeze(1)
        n_members, n_batch = x.shape[:2]
        h = self.trunk(x.reshape(n_members * n_batch, -1))
        h = h.reshape(n_members, n_batch, -1)

        d_logit = self.detector_head(h).squeeze(-1)  # (K, B)
        s_logits = self.severity_head(h)  # (K, B, 4)

        detector_probability = torch.sigmoid(d_logit)
        severity_probability = torch.softmax(s_logits, dim=-1)

        # Average each member's joint distribution, rather than multiplying
        # independently averaged detector and severity probabilities.
        d = detector_probability.mean(dim=0)
        positive_mass = (detector_probability.unsqueeze(-1) * severity_probability).mean(dim=0)
        denominator = d.unsqueeze(-1).clamp_min(torch.finfo(d.dtype).tiny)
        s = positive_mass / denominator
        return d_logit, d, s_logits, s

    def joint_distribution(self, d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Construct full 5-class distribution from detection and severity."""
        return torch.cat(
            ((1.0 - d).unsqueeze(-1), d.unsqueeze(-1) * s),
            dim=-1,
        )


def compute_loss(
    d_logit: torch.Tensor,
    s_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_margin: torch.Tensor | None = None,
    teacher_severity: torch.Tensor | None = None,
    delta_s: float = 0.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    lambda_sev: float = 0.1,
) -> torch.Tensor:
    """Compute combined loss with optional distillation."""
    n_members = d_logit.shape[0]
    if (teacher_margin is None) != (teacher_severity is None):
        msg = "Teacher margin and severity targets must either both be present or both absent"
        raise ValueError(msg)

    # Detector ground truth
    binary_target = (labels > 0).to(d_logit.dtype)
    binary_target = binary_target.unsqueeze(0).expand_as(d_logit)
    # d_logit represents the population margin. Negative undersampling makes
    # the sampled-data log odds larger by delta_s, so this shift is used only
    # when comparing against sampled hard labels.
    sampled_margin = d_logit + delta_s
    detector_loss = F.binary_cross_entropy_with_logits(
        sampled_margin,
        binary_target,
    )
    # Severity ground truth, evaluated for every member and positive row
    severity_loss = d_logit.new_zeros(())
    positive_mask = labels > 0

    if positive_mask.any():
        positive_logits = s_logits[:, positive_mask, :]
        severity_target = labels[positive_mask] - 1
        severity_target = severity_target.unsqueeze(0).expand(n_members, -1)

        severity_loss = F.cross_entropy(
            positive_logits.reshape(-1, positive_logits.shape[-1]),
            severity_target.reshape(-1),
        )

    gt_loss = detector_loss + lambda_sev * severity_loss

    # Optional distillation
    distill_loss = d_logit.new_zeros(())

    if teacher_margin is not None and teacher_severity is not None:
        teacher_margin_members = teacher_margin.unsqueeze(0).expand_as(d_logit)

        # Both are population-scale margins.
        margin_loss = F.smooth_l1_loss(
            d_logit,
            teacher_margin_members,
        )

        # Use a valid T=1 probability distribution for the initial baseline.
        teacher_probability = teacher_severity.clamp_min(torch.finfo(s_logits.dtype).tiny)
        teacher_probability = teacher_probability / teacher_probability.sum(dim=-1, keepdim=True)
        teacher_probability = teacher_probability.unsqueeze(0).expand(n_members, -1, -1)

        student_log_probability = F.log_softmax(s_logits, dim=-1)
        severity_kl = F.kl_div(
            student_log_probability,
            teacher_probability,
            reduction="none",
        ).sum(dim=-1)

        teacher_detection_probability = torch.sigmoid(teacher_margin).unsqueeze(0)
        severity_kl = (teacher_detection_probability * severity_kl).mean()

        distill_loss = alpha * margin_loss + beta * severity_kl

    total_loss = gt_loss + distill_loss

    return total_loss


def checkpoint_selection_key(metrics: dict) -> tuple[int, int, int, int, float]:
    """Return the lexicographic key used to select a student checkpoint.

    Every checkpoint is evaluated at the highest detector threshold satisfying
    the requested validation recall. Larger returned tuples are better.
    """
    severity = metrics.get("severity")
    if severity is None:
        msg = "Checkpoint selection requires multiclass severity metrics"
        raise ValueError(msg)

    return (
        int(severity["exact_on_positive_count"]),
        -int(severity["overprediction_count_all"]),
        -int(severity["overprediction_level_sum_all"]),
        -int(severity["underprediction_level_sum_all"]),
        float(metrics["ap"]),
    )


def train_epoch(
    model: Student,
    optimizer: torch.optim.Optimizer,
    features: np.ndarray,
    labels: np.ndarray,
    teacher_margin: np.ndarray | None,
    teacher_severity: np.ndarray | None,
    device: torch.device,
    batch_size: int,
    negative_ratio: int,
    seed: int,
    delta_s: float,
    alpha: float,
    beta: float,
) -> float:
    """Train for one epoch with random undersampling."""
    model.train()

    positives, negatives = training_index_pools(labels)

    # Sample negatives at the specified ratio
    # Use epoch seed so each epoch gets different samples
    indices = random_undersampling(positives, negatives, negative_ratio, seed)
    if not len(indices):
        msg = "An undersampled epoch must contain at least one training row"
        raise ValueError(msg)

    n_batches = int(np.ceil(len(indices) / batch_size))
    total_loss = 0.0

    for start_idx in range(0, len(indices), batch_size):
        batch_indices = indices[start_idx : start_idx + batch_size]

        # Load batch data
        batch_x = torch.from_numpy(features[batch_indices]).to(device)
        batch_y = torch.from_numpy(labels[batch_indices]).to(device)

        # Get teacher targets for this batch (if distilling)
        batch_teacher_margin = None
        batch_teacher_severity = None
        if teacher_margin is not None and teacher_severity is not None:
            batch_teacher_margin = torch.from_numpy(teacher_margin[batch_indices]).to(
                device=device, dtype=torch.float32
            )
            batch_teacher_severity = torch.from_numpy(teacher_severity[batch_indices]).to(
                device=device, dtype=torch.float32
            )

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        d_logit, _, s_logits, _ = model(batch_x)

        # Compute loss
        loss = compute_loss(
            d_logit,
            s_logits,
            batch_y,
            batch_teacher_margin,
            batch_teacher_severity,
            delta_s=delta_s,
            alpha=alpha,
            beta=beta,
        )

        # Backward pass with gradient clipping
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / n_batches


@torch.no_grad()
def evaluate_model(
    model: Student,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int = 65536,
    *,
    target_recall: float = DEFAULT_TARGET_RECALL,
    threshold: float | None = None,
    threshold_source: str | None = None,
) -> dict:
    """Evaluate the joint prediction at a selected or frozen threshold."""
    model.eval()

    # Collect predictions in batches
    all_d = []
    all_s = []

    for start_idx in range(0, len(features), batch_size):
        batch_x = torch.from_numpy(features[start_idx : start_idx + batch_size]).to(device)

        _, d, _, s = model(batch_x)

        all_d.append(d.cpu().numpy())
        all_s.append(s.cpu().numpy())

    # Concatenate all batches
    d = np.concatenate(all_d)
    s = np.concatenate(all_s)

    # Construct full 5-class distribution
    q = np.zeros((len(d), N_CLASSES), dtype=d.dtype)
    q[:, 0] = 1.0 - d
    q[:, 1:] = d[:, None] * s

    # Evaluate using existing evaluation function
    metrics = evaluate(
        q,
        labels,
        threshold=threshold,
        target_recall=target_recall,
        threshold_source=threshold_source,
        detection_outputs=d,
    )

    return metrics


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid training settings before loading a large cache."""
    positive_integer_arguments = {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "negative_ratio": args.negative_ratio,
        "members": args.members,
    }
    for name, value in positive_integer_arguments.items():
        if value < 1:
            msg = f"{name.replace('_', '-')} must be at least one"
            raise ValueError(msg)

    if not 0.0 <= args.dropout < 1.0:
        msg = "dropout must be in [0, 1)"
        raise ValueError(msg)
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        msg = "lr must be positive and weight-decay must be non-negative"
        raise ValueError(msg)
    if args.alpha < 0.0 or args.beta < 0.0:
        msg = "distillation weights must be non-negative"
        raise ValueError(msg)
    if not 0.0 < args.target_recall <= 1.0:
        msg = "target-recall must be in (0, 1]"
        raise ValueError(msg)
    if args.device == "cuda" and not torch.cuda.is_available():
        msg = "CUDA was requested but is not available"
        raise RuntimeError(msg)


def _validate_cache_shapes(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    teacher_margin: np.ndarray | None,
    teacher_severity: np.ndarray | None,
) -> None:
    """Check the inexpensive cache-shape invariants needed for training."""
    if train_x.ndim != MATRIX_NDIM or val_x.ndim != MATRIX_NDIM:
        msg = "Feature caches must be two-dimensional"
        raise ValueError(msg)
    if train_y.ndim != 1 or val_y.ndim != 1:
        msg = "Label caches must be one-dimensional"
        raise ValueError(msg)
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        msg = "Feature and label cache row counts do not match"
        raise ValueError(msg)
    if not len(train_y) or not len(val_y):
        msg = "Training and validation caches must not be empty"
        raise ValueError(msg)
    if not np.any(val_y > 0):
        msg = "Validation labels must contain at least one positive example"
        raise ValueError(msg)
    if train_x.shape[1] != val_x.shape[1]:
        msg = "Training and validation caches have different feature counts"
        raise ValueError(msg)
    if (teacher_margin is None) != (teacher_severity is None):
        msg = "Teacher margin and severity caches must both be present"
        raise ValueError(msg)
    if teacher_margin is not None and teacher_severity is not None:
        if teacher_margin.shape != (len(train_y),):
            msg = "Teacher margin cache has the wrong shape"
            raise ValueError(msg)
        if teacher_severity.shape != (len(train_y), N_SEVERITY_CLASSES):
            msg = "Teacher severity cache has the wrong shape"
            raise ValueError(msg)


def run(args: argparse.Namespace) -> dict:  # noqa: PLR0915
    """Execute training loop with early stopping."""
    _validate_args(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Determine cache directory based on preprocessing method
    cache_dir = args.cache_dir / "week" / args.preprocess

    print(f"Loading cache from: {cache_dir}")

    # Load cached data (memory-mapped for large arrays)
    # Copy-on-write keeps the arrays memory-mapped while presenting writable
    # views to torch.from_numpy; writes would not alter the cache on disk.
    train_x = np.load(cache_dir / "train_x.npy", mmap_mode="c")
    train_y = np.load(cache_dir / "train_y.npy")
    val_x = np.load(cache_dir / "val_x.npy", mmap_mode="c")
    val_y = np.load(cache_dir / "val_y.npy")

    print(f"Train: {train_x.shape}, {train_x.nbytes // 2**20:.1f} MB (mmap)")
    print(f"Val: {val_x.shape}, {val_x.nbytes // 2**20:.1f} MB (mmap)")
    print(f"Train positives: {(train_y > 0).sum()}")
    print(f"Val positives: {(val_y > 0).sum()}")

    # Load teacher targets if distilling
    teacher_margin = None
    teacher_severity = None
    if args.distill:
        print("Loading teacher targets...")
        teacher_margin = np.load(
            cache_dir / "train_teacher_margin.npy",
            mmap_mode="r",
        )
        teacher_severity = np.load(
            cache_dir / "train_teacher_severity.npy",
            mmap_mode="r",
        )

    _validate_cache_shapes(
        train_x,
        train_y,
        val_x,
        val_y,
        teacher_margin,
        teacher_severity,
    )

    metadata_path = cache_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    delta_t = metadata["teacher"]["delta_t"]
    print(f"Teacher delta_T: {delta_t:.4f}")

    # The model emits a population margin. Adding delta_s maps it to the
    # sampled-data prior used by the detector's hard-label BCE.
    n_positives = int((train_y > 0).sum())
    n_negatives_full = int((train_y == 0).sum())
    if not n_positives or not n_negatives_full:
        msg = "Training data must contain both positive and negative examples"
        raise ValueError(msg)
    n_negatives_sampled = min(
        n_negatives_full,
        args.negative_ratio * n_positives,
    )
    delta_s = float(np.log(n_negatives_full / n_negatives_sampled))
    print(f"Student delta_S: {delta_s:.4f}")

    # Create model
    model = Student(
        n_features=train_x.shape[1],
        hidden_layers=args.hidden,
        k=args.members,
        dropout=args.dropout,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    best_selection_key = None
    best_epoch = None
    best_threshold = None
    patience_counter = 0
    best_state = None
    history = []

    print(f"\nTraining for {args.epochs} epochs with patience {args.patience}")
    print(f"Distillation: {args.distill}")
    print(f"Members: {args.members}")
    print(f"Batch size: {args.batch_size}")
    print(f"Negative ratio: {args.negative_ratio}")
    print(f"Target detector recall: {args.target_recall:.1%}")
    print(f"Checkpoint policy: {SELECTION_POLICY}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        # Use different seed each epoch for negative sampling
        epoch_seed = args.seed + epoch

        train_loss = train_epoch(
            model,
            optimizer,
            train_x,
            train_y,
            teacher_margin,
            teacher_severity,
            device,
            args.batch_size,
            args.negative_ratio,
            epoch_seed,
            delta_s,
            args.alpha,
            args.beta,
        )

        # Evaluate on validation set
        val_metrics = evaluate_model(
            model,
            val_x,
            val_y,
            device,
            target_recall=args.target_recall,
        )
        val_ap = float(val_metrics["ap"])
        severity_metrics = val_metrics["severity"]
        val_exact = float(severity_metrics["exact_on_positive"])
        val_overprediction_rate = float(severity_metrics["overprediction_rate_all"])
        achieved_recall = float(val_metrics["recall"])
        selection_key = checkpoint_selection_key(val_metrics)
        selected_threshold = float(val_metrics["threshold"])

        if not all(
            np.isfinite(value)
            for value in (
                train_loss,
                val_ap,
                val_exact,
                val_overprediction_rate,
                achieved_recall,
                selected_threshold,
            )
        ):
            msg = (
                f"Non-finite value at epoch {epoch}: loss={train_loss}, "
                f"val_ap={val_ap}, exact={val_exact}, "
                f"overprediction_rate={val_overprediction_rate}, "
                f"recall={achieved_recall}, "
                f"threshold={selected_threshold}"
            )
            raise FloatingPointError(msg)

        epoch_time = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(train_loss),
                "validation_ap": val_ap,
                "achieved_recall": achieved_recall,
                "exact_on_positive": val_exact,
                "exact_on_positive_count": int(severity_metrics["exact_on_positive_count"]),
                "overprediction_count_all": int(severity_metrics["overprediction_count_all"]),
                "overprediction_rate_all": val_overprediction_rate,
                "overprediction_level_sum_all": int(severity_metrics["overprediction_level_sum_all"]),
                "underprediction_level_sum_all": int(severity_metrics["underprediction_level_sum_all"]),
                "threshold": selected_threshold,
                "selection_key": list(selection_key),
                "seconds": epoch_time,
            }
        )

        print(
            f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"exact+ {val_exact:.4f} | over(all) {val_overprediction_rate:.6f} | "
            f"recall {achieved_recall:.4f} | AP {val_ap:.4f} | "
            f"threshold {selected_threshold:.6g} | "
            f"time {epoch_time:.1f}s"
        )

        # Early stopping
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_epoch = epoch
            best_threshold = selected_threshold
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_state is None or best_selection_key is None or best_epoch is None or best_threshold is None:
        msg = "Training completed without producing a valid checkpoint"
        raise RuntimeError(msg)
    model.load_state_dict(best_state)
    threshold_source = "selected with checkpoint on validation set"
    final_metrics = evaluate_model(
        model,
        val_x,
        val_y,
        device,
        target_recall=args.target_recall,
        threshold=best_threshold,
        threshold_source=threshold_source,
    )
    final_selection_key = checkpoint_selection_key(final_metrics)
    integer_key_matches = final_selection_key[:-1] == best_selection_key[:-1]
    ap_matches = np.isclose(final_selection_key[-1], best_selection_key[-1])
    if not integer_key_matches or not ap_matches:
        msg = "Restored checkpoint does not reproduce its validation selection key"
        raise RuntimeError(msg)
    if final_metrics["recall"] + np.finfo(float).eps < args.target_recall:
        msg = "Restored checkpoint does not satisfy the requested detector recall"
        raise RuntimeError(msg)

    selection = {
        "policy": SELECTION_POLICY,
        "threshold_rule": THRESHOLD_RULE,
        "threshold_score": THRESHOLD_SCORE,
        "target_recall": args.target_recall,
        "criteria": list(SELECTION_CRITERIA),
        "best_epoch": best_epoch,
        "best_key": list(best_selection_key),
        "threshold": best_threshold,
        "achieved_recall": float(final_metrics["recall"]),
        "selected_checkpoint_ap": float(final_metrics["ap"]),
        "max_ap_observed": max(row["validation_ap"] for row in history),
    }

    # Save results
    task = "distilled" if args.distill else "supervised"
    model_path, result_path = output_paths(args.output_dir, f"student_{task}", ".pt")

    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": {
                "n_features": train_x.shape[1],
                "hidden": args.hidden,
                "k": args.members,
                "dropout": args.dropout,
                "delta_s": delta_s,
            },
            "threshold": best_threshold,
            "threshold_source": threshold_source,
            "target_recall": args.target_recall,
            "severity_decision": SEVERITY_DECISION,
            "selection": selection,
        },
        model_path,
    )

    result = {
        "task": task,
        "seed": args.seed,
        "config": vars(args),
        "metrics": final_metrics,
        "threshold": best_threshold,
        "threshold_source": threshold_source,
        "target_recall": args.target_recall,
        "severity_decision": SEVERITY_DECISION,
        "selection": selection,
        "history": history,
    }
    write_json(result_path, result)

    print(f"\nSaved model: {model_path}")
    print(f"Saved results: {result_path}")
    print(f"Final AP: {final_metrics['ap']:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Frozen threshold: {best_threshold:.6g}")
    print(f"Exact on positives: {final_metrics['severity']['exact_on_positive']:.4f}")
    print(f"All-row overprediction rate: {final_metrics['severity']['overprediction_rate_all']:.6f}")

    return result


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train student model with optional distillation")

    # Data paths
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Cache directory")
    parser.add_argument(
        "--preprocess",
        default="physical",
        choices=("physical", "quantile", "raw"),
        help="Preprocessing variant",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")

    # Architecture
    parser.add_argument("--hidden", type=int, nargs="+", default=HIDDEN_LAYERS, help="Hidden layer sizes")
    parser.add_argument("--members", type=int, default=1, help="Number of ensemble members (k)")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout probability")

    # Training
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=DEFAULT_NEGATIVE_RATIO,
        help="Negative:positive sampling ratio",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay")
    parser.add_argument(
        "--target-recall",
        type=float,
        default=DEFAULT_TARGET_RECALL,
        help="Minimum detector recall used to select the validation threshold",
    )

    # Distillation
    parser.add_argument("--distill", action="store_true", help="Use knowledge distillation")
    parser.add_argument("--alpha", type=float, default=1.0, help="Detector distillation weight")
    parser.add_argument("--beta", type=float, default=0.5, help="Severity distillation weight")

    # Misc
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
