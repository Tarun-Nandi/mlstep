"""Binary and multiclass XGBoost training and evaluation."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

from mlstep.data import (
    DATA,
    FEATURES,
    N_CLASSES,
    Feature,
    discover_timesteps,
    feature_group_names,
    feature_names,
    load_labels,
    load_selected_rows,
    load_timesteps,
    random_undersampling,
    split_timesteps,
    training_index_pools,
)
from mlstep.evaluation import (
    benchmark,
    detection_ap,
    evaluate,
    output_paths,
    run_metadata,
    write_json,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"

PARAMS = {
    "tree_method": "hist",  # Using hist is much faster and more memory efficient
    "learning_rate": 0.05,  # Each tree contributes 5% of its correctness
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,  # Each round randomly uses 80% of the available training rows
    "colsample_bytree": 0.8,  # Each tree random;y uses 80% of the feature columns
    "max_delta_step": 1,
    "max_bin": 128,
}


def xgb_detection_ap(predictions: np.ndarray, matrix: xgb.DMatrix):
    """Provide exact detection average precision for optional multiclass early stopping."""
    return "detection_ap", detection_ap(predictions, matrix.get_label())


class XGBoostDashboardCallback(xgb.callback.TrainingCallback):
    """Forward validation metrics to the optional live dashboard."""

    def __init__(self, dashboard, metric_name: str) -> None:
        self.dashboard = dashboard
        self.metric_name = metric_name

    def after_iteration(self, _model, epoch: int, evals_log: dict) -> bool:
        """Record one completed boosting round."""
        value = evals_log["validation"][self.metric_name][-1]
        if isinstance(value, tuple):
            value = value[0]
        self.dashboard.update_training(epoch + 1, float(value), self.metric_name)
        return False


def load_training_data(
    data_dir: Path,
    timesteps: tuple[int, ...],
    features: tuple[Feature, ...],
    *,
    multiclass: bool,
    negative_ratio: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the training representation required by an XGBoost task."""
    if multiclass:
        train_y, rows_per_timestep = load_labels(data_dir, timesteps)
        if not np.any(train_y > 0):
            msg = "the training split contains no positive halving examples"
            raise ValueError(msg)
        indices = random_undersampling(*training_index_pools(train_y), negative_ratio, seed)
        fit_x = load_selected_rows(data_dir, timesteps, features, indices, rows_per_timestep)
        fit_y = train_y[indices].astype(np.int32)
        return fit_x, train_y, fit_y

    fit_x, train_y = load_timesteps(data_dir, timesteps, features)
    fit_y = (train_y > 0).astype(np.int32)
    return fit_x, train_y, fit_y


def run(args: argparse.Namespace) -> dict:  # noqa: PLR0915
    """Conduct the complete XGBoost experiment."""
    features = FEATURES
    names = feature_names(features)
    groups = [feature.name for feature in features]
    task = "multiclass" if args.multiclass else "binary"
    timesteps = discover_timesteps(args.data_dir)
    splits = split_timesteps(timesteps)
    train_steps, validation_steps, _ = splits

    # QuantileDMatrix stores a compact quantized representation for histogram
    # training, so raw NumPy arrays can be released as soon as each matrix exists.
    matrix_options = {"max_bin": PARAMS["max_bin"]}
    if args.threads:
        matrix_options["nthread"] = args.threads
    started = time.perf_counter()
    fit_x, train_y, fit_y = load_training_data(
        args.data_dir,
        train_steps,
        features,
        multiclass=args.multiclass,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    training_load_seconds = time.perf_counter() - started
    training_rows_used = len(fit_y)
    training_class_counts_used = np.bincount(
        fit_y.astype(np.int64, copy=False),
        minlength=N_CLASSES if args.multiclass else 2,
    ).tolist()

    # XGBoost does not need the nonlinear preprocessing used by the FCNN.
    print(
        f"{task} XGBoost | train {(len(train_y), len(names))} on t{train_steps[0]}-t{train_steps[-1]} "
        f"({int((train_y > 0).sum())} positives; {training_rows_used:,} rows used)"
    )

    started = time.perf_counter()
    train_matrix = xgb.QuantileDMatrix(fit_x, label=fit_y, **matrix_options)
    training_matrix_seconds = time.perf_counter() - started
    del fit_x, fit_y

    # Load validation only after the raw training matrix has been released. The
    # complete validation array remains available for final metrics and timing.
    started = time.perf_counter()
    val_x, val_y = load_timesteps(args.data_dir, validation_steps, features)
    validation_load_seconds = time.perf_counter() - started
    print(
        f"validation {val_x.shape} on t{validation_steps[0]}-t{validation_steps[-1]} "
        f"({int((val_y > 0).sum())} positives)"
    )

    validation_labels = val_y if args.multiclass else (val_y > 0).astype(np.int32)
    started = time.perf_counter()
    val_matrix = xgb.QuantileDMatrix(
        val_x,
        label=validation_labels,
        ref=train_matrix,
        **matrix_options,
    )
    validation_matrix_seconds = time.perf_counter() - started
    del validation_labels

    load_seconds = training_load_seconds + validation_load_seconds
    matrix_seconds = training_matrix_seconds + validation_matrix_seconds
    # Copy ths fixed params and add seed + device
    params = {**PARAMS, "seed": args.seed, "device": args.device}
    if args.threads:
        params["nthread"] = args.threads
    # if multiclass we want to train one 5 class model that outputs a prob distribution
    if args.multiclass:
        params.update(
            {
                "objective": "multi:softprob",
                "num_class": N_CLASSES,
                "disable_default_eval_metric": True,
            }
        )
        metric_name = "detection_ap"
        custom_metric = xgb_detection_ap
    else:  # Otherwise we just train one binary logictic classifier (we dont use a 2 stage approach here)
        params.update({"objective": "binary:logistic", "eval_metric": "aucpr"})
        metric_name = "aucpr"
        custom_metric = None

    dashboard = None
    callbacks = []
    if args.dashboard:
        from mlstep.dashboard import XGBoostTrainingDashboard  # noqa: PLC0415

        dashboard = XGBoostTrainingDashboard(task)
        # This callback must run before early stopping so it also receives the final round.
        callbacks.append(XGBoostDashboardCallback(dashboard, metric_name))
    callbacks.append(
        xgb.callback.EarlyStopping(
            rounds=args.patience,
            metric_name=metric_name,
            data_name="validation",
            maximize=True,
            save_best=True,
        )
    )
    evals_result = {}
    started = time.perf_counter()
    model = xgb.train(
        params,
        train_matrix,
        num_boost_round=args.rounds,  # maximum number of boosting rounds, defaults to 500
        # Evaluate model on entire validation set after each boosting rounf
        evals=[(val_matrix, "validation")],
        custom_metric=custom_metric,
        evals_result=evals_result,
        # Implementing early stopping (if validation metric doesnt improve for (args.patience) rounds we stop)
        callbacks=callbacks,
        verbose_eval=25,  # Only print progress every 25 rounds
    )
    # Measure time to train the model
    training_seconds = time.perf_counter() - started
    training_history = [
        {"round": round_number, "validation_metric": float(value)}
        for round_number, value in enumerate(evals_result["validation"][metric_name], start=1)
    ]
    if dashboard is not None:
        dashboard.set_training_history(training_history, metric_name)
    # calculate metrics
    metrics = evaluate(
        model.inplace_predict(val_x),
        val_y,
        target_recall=args.target_recall,
    )
    # Measure model only predicition time
    inference = benchmark(lambda: model.inplace_predict(val_x))
    inference.update(
        {
            "rows": len(val_x),
            "rows_per_second": len(val_x) / inference["median_seconds"],
            "device": args.device,
            "scope": ("model prediction on a preassembled raw matrix. Doesnt include feature assembly"),
        }
    )
    timing = {
        "data_loading_seconds": load_seconds,
        "training_data_loading_seconds": training_load_seconds,
        "validation_data_loading_seconds": validation_load_seconds,
        "matrix_construction_seconds": matrix_seconds,
        "training_matrix_construction_seconds": training_matrix_seconds,
        "validation_matrix_construction_seconds": validation_matrix_seconds,
        "training_seconds": training_seconds,
        "inference": inference,
    }
    # Saving the model itself and the results(JSON)
    model_path, result_path = output_paths(args.output_dir, f"xgb_{task}", ".ubj")
    dashboard_path = None
    if dashboard is not None:
        dashboard_path = args.output_dir.parent / "plots" / f"{model_path.stem}.png"
    score_context = "ranking score; probability calibration not established"
    model.set_attr(
        mlstep_metadata=json.dumps(
            {
                "task": task,
                "decision_threshold": metrics["threshold"],
                "target_recall_requested": args.target_recall,
                "feature_groups": groups,
                "feature_names": names,
                "score_context": score_context,
            }
        )
    )
    model.save_model(str(model_path))
    importance, group_importance = feature_importance(
        model,
        names,
        feature_group_names(features),
    )
    if dashboard is not None:
        dashboard.update_final_metrics(
            metrics,
            importance,
            group_importance,
            int(model.best_iteration),
            timing,
        )
        dashboard.save(dashboard_path)
    # Constructing the final results
    result = run_metadata(args, features, train_y, val_y, {"xgboost": xgb.__version__}, splits)
    result["config"].update(
        {
            "parameters": params,
            "score_context": score_context,
        }
    )
    result["data"].update(
        {
            "training_loader": "selected_rows" if args.multiclass else "full_rows",
            "training_rows_used": training_rows_used,
            "training_class_counts_used": training_class_counts_used,
        }
    )
    result.update(
        {
            "task": task,
            "model_file": model_path.name,
            "model_size_bytes": model_path.stat().st_size,
            "dashboard_file": str(dashboard_path) if dashboard_path is not None else None,
            "history": {
                "metric": metric_name,
                "validation": training_history,
            },
            # Best boosting round and validation metrics
            "selection": {
                "best_iteration": int(model.best_iteration),
                "early_stopping_metric": (
                    "exact sklearn detection AP"
                    if args.multiclass
                    else "XGBoost aucpr (headline AP is computed after selection)"
                ),
                "best_early_stopping_score": float(model.best_score),
                "selected_rounds": model.num_boosted_rounds(),
                "metrics": metrics,
            },
            # Record the importance of features
            "importance": {
                "normalized_total_gain": importance,
                "group_normalized_total_gain": group_importance,
            },
            # Record the loading, training and inference times
            "timing": timing,
        }
    )
    write_json(result_path, result)
    print(
        f"\nselected {model.num_boosted_rounds()} rounds: "
        f"AP {metrics['ap']:.4f}, precision {metrics['precision']:.4f}, "
        f"recall {metrics['recall']:.4f}"
    )
    print(f"inference {inference['median_seconds']:.4f}s for {len(val_x):,} rows")
    print("top features:")
    for name, gain in list(importance.items())[:10]:
        print(f"  {name:<24} {gain:.4f}")
    print(f"saved {model_path}")
    print(f"saved {result_path}")
    if dashboard is not None:
        print(f"saved {dashboard_path}")
        dashboard.close()
    return result


def feature_importance(
    model: xgb.Booster, names: list[str], groups: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Find the importance of scalar and vector inputs in the decision trees."""
    gain = np.zeros(len(names), dtype=np.float64)
    # Create one importance value per model input column
    for feature, value in model.get_score(importance_type="total_gain").items():
        gain[int(feature.removeprefix("f"))] = value
    if gain.sum():
        # Normalise the values so all feature importances sum to one
        gain /= gain.sum()

    importance = {names[index]: float(gain[index]) for index in np.argsort(-gain) if gain[index] > 0}
    totals = defaultdict(float)
    # Looking at importance of the vector inputs combines (so e.g. all of tracer importance)
    for group, value in zip(groups, gain, strict=True):
        totals[group] += float(value)
    group_importance = dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))
    return importance, group_importance


def parse_args() -> argparse.Namespace:
    """Adding CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--multiclass", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--negative-ratio", type=int, default=64)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--target-recall", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
