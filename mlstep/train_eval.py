"""Train and evaluate the UKCA timestep-halving network."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

if __package__:
    from . import data_utils
    from .dashboard import LiveTrainingDashboard
    from .net import FCNN
else:
    import data_utils
    from dashboard import LiveTrainingDashboard
    from net import FCNN

# Edit these values directly when trying a different training setup.
EPOCHS = 30
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_HIDDEN = 64
TOP_FRACTION = 0.001
FIRST_SEED = 0
RUNS_DIR = Path(__file__).resolve().parent / "runs"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def confusion_matrix(y_true, y_pred, n_classes):
    """Return counts with true classes in rows and predictions in columns."""
    flat = y_true * n_classes + y_pred
    return np.bincount(flat, minlength=n_classes**2).reshape(n_classes, n_classes)


def compute_metrics(probabilities, targets):
    """Compute the small set of metrics used to compare experiments."""
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)

    positive = targets > 0
    scores = 1 - probabilities[:, 0]
    predictions = probabilities.argmax(axis=1)
    n_positive = int(positive.sum())

    ap = float(average_precision_score(positive, scores)) if n_positive else 0.0
    roc_auc = (
        float(roc_auc_score(positive, scores))
        if n_positive and (~positive).any()
        else None
    )

    top_n = max(1, int(np.ceil(TOP_FRACTION * len(targets))))
    top = np.argsort(-scores)[:top_n]
    metrics = {
        "ap": ap,
        "roc_auc": roc_auc,
        "prevalence": float(positive.mean()),
        "n_pos": n_positive,
        "top_fraction": TOP_FRACTION,
        "precision_at_top": float(positive[top].mean()),
        "recall_at_top": (
            float(positive[top].sum() / n_positive) if n_positive else None
        ),
        "false_positives": int(((predictions > 0) & ~positive).sum()),
        "false_negatives": int(((predictions == 0) & positive).sum()),
        "confusion": confusion_matrix(
            targets, predictions, data_utils.N_CLASSES
        ).tolist(),
    }

    if n_positive:
        errors = predictions[positive] - targets[positive]
        metrics["mae_on_pos"] = float(np.abs(errors).mean())
        metrics["exact_on_pos"] = float((errors == 0).mean())
        metrics["mean_signed_error"] = float(errors.mean())
        metrics["underprediction_rate"] = float((errors < 0).mean())
        metrics["overprediction_rate"] = float((errors > 0).mean())
    else:
        metrics["mae_on_pos"] = None
        metrics["exact_on_pos"] = None
        metrics["mean_signed_error"] = None
        metrics["underprediction_rate"] = None
        metrics["overprediction_rate"] = None

    per_class = {}
    for class_index in range(data_utils.N_CLASSES):
        true_class = targets == class_index
        predicted_class = predictions == class_index
        support = int(true_class.sum())
        n_predicted = int(predicted_class.sum())
        per_class[class_index] = {
            "support": support,
            "predicted": n_predicted,
            "recall": (float(predicted_class[true_class].mean()) if support else None),
            "precision": (
                float(true_class[predicted_class].mean()) if n_predicted else None
            ),
        }
    metrics["per_class"] = per_class
    return metrics


def class_statistics(targets):
    """Return empirical priors and balanced class weights."""
    counts = torch.bincount(targets, minlength=data_utils.N_CLASSES).float()
    priors = counts / counts.sum()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return counts, priors, weights


def initial_logits(priors, weights):
    """Return the best constant prediction for the weighted loss."""
    probabilities = priors * weights
    probabilities /= probabilities.sum()
    return probabilities.clamp_min(1e-12).log()


def evaluate(model, x, y, class_weights, batch_size=65536):
    """Evaluate a model over a full array in manageable batches."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat(
            [
                model(x[start : start + batch_size])
                for start in range(0, len(x), batch_size)
            ]
        )

    probabilities = torch.softmax(logits, dim=1)
    metrics = compute_metrics(probabilities.numpy(), y.numpy())
    metrics["weighted_ce"] = nn.functional.cross_entropy(
        logits, y, weight=class_weights
    ).item()
    metrics["unweighted_ce"] = nn.functional.cross_entropy(logits, y).item()
    return metrics


def train(
    train_x,
    train_y,
    val_x,
    val_y,
    seed,
    epochs=EPOCHS,
    verbose=True,
    epoch_callback=None,
):
    """Train one seeded network and return its history and model."""
    torch.manual_seed(seed)
    _, priors, weights = class_statistics(train_y)
    model = FCNN(
        train_x.shape[1],
        data_utils.N_CLASSES,
        n_hidden=N_HIDDEN,
        init_logits=initial_logits(priors, weights),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(train_x), generator=generator)
        loss_sum = 0.0
        weight_sum = 0.0

        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            batch_x = train_x[indices]
            batch_y = train_y[indices]

            optimizer.zero_grad()
            logits = model(batch_x)
            batch_loss = nn.functional.cross_entropy(
                logits,
                batch_y,
                weight=weights,
                reduction="sum",
            )
            batch_weight = weights[batch_y].sum()
            (batch_loss / batch_weight).backward()
            optimizer.step()

            loss_sum += batch_loss.item()
            weight_sum += batch_weight.item()

        record = {
            "epoch": epoch + 1,
            "train_weighted_ce": loss_sum / weight_sum,
            **evaluate(model, val_x, val_y, weights),
        }
        history.append(record)
        if epoch_callback is not None:
            epoch_callback(history)
        if verbose:
            print(
                f"seed {seed} epoch {epoch + 1:>2}: "
                f"train CE {record['train_weighted_ce']:.4f}, "
                f"val CE {record['weighted_ce']:.4f}, "
                f"AP {record['ap']:.4f}"
            )

    return history, model


def run_experiment(
    split="time",
    include_t1=False,
    seeds=5,
    epochs=EPOCHS,
    live_plots=True,
    baseline_metrics=None,
):
    """Train several seeds, update the dashboard, and save one JSON file."""
    splits = data_utils.prepare_splits(split=split, include_t1=include_t1)
    train_x, train_y = (
        torch.from_numpy(np.ascontiguousarray(array)) for array in splits["train"]
    )
    val_x, val_y = (
        torch.from_numpy(np.ascontiguousarray(array)) for array in splits["val"]
    )
    print(
        f"train {tuple(train_x.shape)} ({int((train_y > 0).sum())} pos), "
        f"val {tuple(val_x.shape)} ({int((val_y > 0).sum())} pos)"
    )

    dashboard = None
    if live_plots:
        try:
            dashboard = LiveTrainingDashboard(
                data_utils.N_CLASSES,
                baseline_metrics=baseline_metrics,
            )
        except Exception as error:
            print(f"Could not open live dashboard: {error}")

    histories = []
    seed_results = []
    completed_histories = []
    for seed in range(FIRST_SEED, FIRST_SEED + seeds):
        callback = None
        if dashboard is not None:

            def callback(current_history, current_seed=seed):
                dashboard.update(
                    current_seed,
                    current_history,
                    completed_histories,
                )

        history, _ = train(
            train_x,
            train_y,
            val_x,
            val_y,
            seed=seed,
            epochs=epochs,
            epoch_callback=callback,
        )
        histories.append(history)
        completed_histories.append((seed, history))
        seed_results.append(
            {
                "seed": seed,
                "final": history[-1],
                "best": max(history, key=lambda record: record["ap"]),
            }
        )

    final_aps = np.array([result["final"]["ap"] for result in seed_results])
    best_aps = np.array([result["best"]["ap"] for result in seed_results])
    summary = {
        "final_ap_mean": float(final_aps.mean()),
        "final_ap_std": (float(final_aps.std(ddof=1)) if len(final_aps) > 1 else 0.0),
        "best_ap_mean": float(best_aps.mean()),
        "best_ap_std": (float(best_aps.std(ddof=1)) if len(best_aps) > 1 else 0.0),
        "n_seeds": seeds,
    }
    output = {
        "config": {
            "split": split,
            "include_t1": include_t1,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "n_hidden": N_HIDDEN,
        },
        "seeds": seed_results,
        "histories": histories,
        "summary": summary,
    }

    RUNS_DIR.mkdir(exist_ok=True)
    tag = f"{split}{'_t1' if include_t1 else ''}_{time.strftime('%m%d-%H%M%S')}.json"
    output_path = RUNS_DIR / tag
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"final AP {summary['final_ap_mean']:.4f} "
        f"+/- {summary['final_ap_std']:.4f}\n"
        f"saved {output_path}"
    )
    if dashboard is not None:
        dashboard_path = PLOTS_DIR / (f"{output_path.stem}_dashboard.png")
        dashboard.save(dashboard_path)
        print(f"saved {dashboard_path}")
        dashboard.finish()
    return summary


def main():
    """Run an experiment from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["time", "random"], default="time")
    parser.add_argument("--include-t1", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--live-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--baseline-json", type=Path)
    args = parser.parse_args()

    baseline_metrics = None
    if args.baseline_json:
        baseline_metrics = json.loads(args.baseline_json.read_text(encoding="utf-8"))[
            "metrics"
        ]
    run_experiment(
        split=args.split,
        include_t1=args.include_t1,
        seeds=args.seeds,
        live_plots=args.live_plots,
        baseline_metrics=baseline_metrics,
    )


if __name__ == "__main__":
    main()
