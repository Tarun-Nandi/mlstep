"""Standalone five-class XGBoost baseline and neural distillation targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

from mlstep.cache import load_cache_state
from mlstep.data import (
    N_CLASSES,
    class_counts,
    discover_timesteps,
    feature_names,
    load_labels,
    load_selected_rows,
    load_timesteps,
    parse_timestep_range,
    random_undersampling,
    resolve_timestep_split,
    training_index_pools,
)
from mlstep.evaluation import evaluate_probabilities, recall_action_frontier

PARAMS = {
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_delta_step": 1,
    "max_bin": 128,
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "disable_default_eval_metric": True,
}


def _detection_ap(predictions: np.ndarray, matrix: xgb.DMatrix) -> tuple[str, float]:
    """Score any-halving detection for XGBoost's early-stopping callback.

    With the built-in softprob objective, predictions are class probabilities.
    Sum classes 1–4; AP selects boosting rounds, not exact-severity accuracy.
    Training setup requires validation positives before this callback runs.
    """
    probabilities = np.asarray(predictions).reshape(-1, N_CLASSES)
    return "detection_ap", float(average_precision_score(matrix.get_label() > 0, probabilities[:, 1:].sum(axis=1)))


def load_teacher(path: Path | str, device: str = "cpu") -> tuple[xgb.Booster, dict]:
    """Load a saved booster and its data contract, then select the prediction device.

    Require the current feature order and a valid sampling correction so teacher
    margins have the population interpretation expected by neural distillation.
    Historical boosters without this metadata cannot be used automatically.
    """
    booster = xgb.Booster()
    booster.load_model(path)
    metadata = json.loads(booster.attr("mlstep_metadata") or "{}")
    if metadata.get("format") != "mlstep-xgb-v1" or metadata.get("features") != feature_names():
        raise ValueError("teacher must be an mlstep-xgb-v1 model with the current feature order")
    correction = metadata.get("negative_log_correction")
    if not isinstance(correction, (int, float)) or not np.isfinite(correction) or correction < 0:
        raise ValueError("teacher sampling correction is missing or invalid")
    booster.set_param({"device": device})
    return booster, metadata


def teacher_responses(logits: np.ndarray, negative_log_correction: float) -> np.ndarray:
    """Convert raw margins (N, 5) to float32 [detector margin, four severity probabilities].

    The detector margin is logsumexp(positive logits) minus the zero-class logit
    and the sampling correction. Severity is softmax over positive logits only.
    Negative undersampling inflates positive odds by exp(correction). Subtract
    that shift from the detector margin; conditional positive probabilities
    are unchanged because every positive training row was retained.
    """
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_CLASSES or not np.isfinite(values).all():
        raise ValueError("XGBoost margins must be a finite five-column matrix")
    positive = values[:, 1:]
    # Subtract the maximum before exponentiating to keep large logits finite.
    maximum = positive.max(axis=1, keepdims=True)
    exp_positive = np.exp(positive - maximum)
    total = exp_positive.sum(axis=1, keepdims=True)
    result = np.empty(values.shape, dtype=np.float32)
    result[:, 0] = (maximum + np.log(total))[:, 0] - values[:, 0] - negative_log_correction
    result[:, 1:] = exp_positive / total
    return result


def responses_to_probabilities(responses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return joint probabilities (N, 5) and detector scores (N,) from teacher responses.

    Use a stable sigmoid for the margin, then [1-d, d*q1, ..., d*q4]. Keep d
    separately so tiny detector scores are not lost when subtracting from P(0).
    """
    detector = np.exp(-np.logaddexp(0.0, -responses[:, 0]))
    probabilities = np.empty_like(responses)
    probabilities[:, 0] = 1.0 - detector
    probabilities[:, 1:] = detector[:, None] * responses[:, 1:]
    return probabilities, detector


def predict_responses(
    booster: xgb.Booster,
    features: np.ndarray,
    correction: float,
    batch_size: int = 65_536,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Convert raw feature batches into corrected teacher responses.

    Return a float32 (N, 5) array, or write into a supplied writable ``out`` array
    of that shape and dtype. Target generation passes a memory-map slice to avoid
    allocating a full timestep of responses. The saved booster already contains
    only the selected trees; no boosting-round selection is performed here.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if out is None:
        out = np.empty((len(features), N_CLASSES), dtype=np.float32)
    elif out.shape != (len(features), N_CLASSES) or out.dtype != np.float32:
        raise ValueError("response output must be a matching float32 five-column matrix")
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        matrix = xgb.DMatrix(np.asarray(features[start:stop]))
        logits = booster.predict(matrix, output_margin=True)
        out[start:stop] = teacher_responses(logits, correction)
        del matrix, logits
    return out


def train(args: argparse.Namespace) -> Path:
    """Train on one fixed negative sample, select by validation AP, and save results.

    Read raw features, not the neural cache's transformed values. Keep every
    positive training row and sample negatives once. Save the selected booster
    with feature/split/prior metadata, plus population-corrected validation scores.
    Both round selection and recall thresholds use validation labels.
    """
    if min(args.negative_ratio, args.rounds, args.patience, args.threads) < 1 or args.seed < 0:
        raise ValueError("ratios, rounds, patience and threads must be positive; seed must be non-negative")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.recalls or any(not np.isfinite(r) or not 0 < r <= 1 for r in args.recalls):
        raise ValueError("recalls must be nonempty and in (0, 1]")
    if any(left >= right for left, right in zip(args.recalls, args.recalls[1:])):
        raise ValueError("recalls must be strictly increasing")
    train_steps, val_steps = resolve_timestep_split(
        discover_timesteps(args.data_dir),
        args.train_timesteps,
        args.validation_timesteps,
    )
    train_y = load_labels(args.data_dir, train_steps)
    positive, negative = training_index_pools(train_y)
    if not len(positive) or not len(negative):
        raise ValueError("training requires positive and negative examples")
    indices = random_undersampling(positive, negative, args.negative_ratio, args.seed)
    fit_y = train_y[indices]
    # Compute population statistics before releasing full labels and index pools.
    train_counts = class_counts(train_y)
    sample_counts = class_counts(fit_y)
    correction = float(np.log(len(negative) / sample_counts[0]))
    del train_y, positive, negative
    fit_x = load_selected_rows(args.data_dir, train_steps, indices)
    del indices
    val_x, val_y = load_timesteps(args.data_dir, val_steps)
    if not np.any(val_y > 0):
        raise ValueError("validation AP selection requires positive examples")
    parameters = {**PARAMS, "device": args.device, "seed": args.seed, "nthread": args.threads}
    matrix_options = {"max_bin": PARAMS["max_bin"], "nthread": args.threads}
    fit = xgb.QuantileDMatrix(fit_x, label=fit_y, **matrix_options)
    # Use training cut points for validation rather than fitting a separate binning.
    validation = xgb.QuantileDMatrix(val_x, label=val_y, ref=fit, **matrix_options)
    del fit_x, fit_y, val_x
    booster = xgb.train(
        parameters,
        fit,
        num_boost_round=args.rounds,
        evals=[(validation, "validation")],
        custom_metric=_detection_ap,
        callbacks=[
            xgb.callback.EarlyStopping(
                rounds=args.patience,
                metric_name="detection_ap",
                data_name="validation",
                maximize=True,
                save_best=True,
            )
        ],
        verbose_eval=25,
    )
    del fit
    metadata = {
        "format": "mlstep-xgb-v1",
        "features": feature_names(),
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "train_timesteps": list(train_steps),
        "validation_timesteps": list(val_steps),
        "negative_log_correction": correction,
        "seed": args.seed,
        "train_class_counts": train_counts,
        "sample_class_counts": sample_counts,
        "parameters": parameters,
        "best_iteration": booster.best_iteration,
    }
    booster.set_attr(mlstep_metadata=json.dumps(metadata))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_path = args.output_dir / "model.ubj"
    booster.save_model(model_path)
    logits = booster.predict(validation, output_margin=True)
    probabilities, detector = responses_to_probabilities(teacher_responses(logits, correction))
    result = {
        "configuration": metadata,
        "validation_class_counts": class_counts(val_y),
        "probability_metrics": evaluate_probabilities(probabilities, val_y, detector),
        "operating_points": recall_action_frontier(probabilities, detector, val_y, args.recalls),
        "note": "Population-corrected probabilities. Thresholds and boosting round selected on validation.",
    }
    (args.output_dir / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {model_path}")
    return model_path


def prepare_targets(args: argparse.Namespace) -> Path:
    """Write teacher responses in exactly the neural training cache's row order.

    Check recorded data/splits and each raw label slice before writing responses.
    The teacher sees raw inputs even when the student uses transformed inputs.
    Write completion metadata last, so a partial target file is not mistaken for
    a completed cache. Existing output directories are never overwritten.
    """

    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cache, _, _ = load_cache_state(args.cache_dir)
    booster, teacher = load_teacher(args.model, args.device)
    if (
        teacher["train_timesteps"] != cache["train"]["timesteps"]
        or teacher["validation_timesteps"] != cache["val"]["timesteps"]
    ):
        raise ValueError("teacher and neural cache must use identical training/validation timesteps")
    if teacher["data_dir"] != cache["data_dir"]:
        raise ValueError("teacher and neural cache record different raw data directories")
    labels = np.load(args.cache_dir / "train_y.npy", mmap_mode="r", allow_pickle=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = np.lib.format.open_memmap(
        args.output_dir / "train_teacher.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(labels), N_CLASSES),
    )
    offset = 0
    for timestep in cache["train"]["timesteps"]:
        raw, raw_y = load_timesteps(Path(cache["data_dir"]), (timestep,))
        stop = offset + len(raw)
        if not np.array_equal(raw_y, labels[offset:stop]):
            raise ValueError(f"raw labels at t{timestep} do not match the neural cache")
        predict_responses(
            booster, raw, teacher["negative_log_correction"], args.batch_size, out=output[offset:stop],
        )
        offset = stop
        # Release this timestep before allocating the next raw feature matrix.
        del raw, raw_y
    if offset != len(labels):
        raise ValueError("teacher targets do not match the neural cache row count")
    output.flush()
    del output
    # Written last: incomplete target generation cannot be consumed by training.
    metadata = {
        "teacher": str(args.model.resolve()),
        "features": feature_names(),
        "data_dir": cache["data_dir"],
        "train": cache["train"],
        "val": cache["val"],
        "rows": len(labels),
        "columns": ["population_detector_margin", "severity_1", "severity_2", "severity_3", "severity_4"],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved teacher targets to {args.output_dir}")
    return args.output_dir


def main() -> None:
    """Parse and dispatch standalone baseline training or aligned teacher-target generation.

    XGBoost remains an optional dependency: neural training reads saved responses
    and never imports this module to run the teacher itself.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="train a standalone five-class baseline on raw inputs")
    fit.add_argument("--data-dir", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--train-timesteps", type=parse_timestep_range, required=True)
    fit.add_argument("--validation-timesteps", type=parse_timestep_range, required=True)
    fit.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--negative-ratio", type=int, default=64)
    fit.add_argument("--rounds", type=int, default=500)
    fit.add_argument("--patience", type=int, default=25)
    fit.add_argument("--threads", type=int, default=8)
    fit.add_argument("--recalls", type=float, nargs="+", default=[0.95, 0.97, 0.99])
    targets = commands.add_parser("targets", help="prepare population-corrected training targets")
    targets.add_argument("--model", type=Path, required=True)
    targets.add_argument("--cache-dir", type=Path, required=True)
    targets.add_argument("--output-dir", type=Path, required=True)
    targets.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    targets.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        prepare_targets(args)


if __name__ == "__main__":
    main()
