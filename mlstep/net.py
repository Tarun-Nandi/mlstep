"""FCNN hard-box detector and exact-halvings training and evaluation."""

import argparse
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mlstep.data import (
    DATA,
    FEATURES,
    N_CLASSES,
    Preprocesser,
    discover_timesteps,
    feature_names,
    load_train_validation,
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
    threshold_at_recall,
    write_json,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"
INFERENCE_BATCH_SIZE = 65_536
# Limits the size of gradients during training
# number is subject to change through hyper parameter tuning
GRADIENT_CLIP_NORM = 5.0

"""
Hard Boxes Detector Architecture:
input features -> 128 neurons -> 64 neurals -> 1 binary output
"""
HARD_BOXES_DETECTOR_HIDDEN_LAYERS = (128, 64)
HARD_BOXES_DETECTOR_DROPOUT = 0.05  # 5% of hidden activations are randomly zeroed

"""
The data for the exact halving model only uses positive examples so we can use a much
smaller network with more epochs and stronger weight decay.
"""
EXACT_HALVINGS_HIDDEN_LAYERS = (64, 32)
EXACT_HALVINGS_EPOCHS = 400
EXACT_HALVINGS_PATIENCE = 60
EXACT_HALVINGS_BATCH_SIZE = 64
EXACT_HALVINGS_LEARNING_RATE = 1e-3
EXACT_HALVINGS_WEIGHT_DECAY = 1e-3


class FCNN(nn.Module):
    """A configurable MLP shared by both the Hard boxes detector and exact halvings model."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_layers: tuple[int, ...],
        activation: type[nn.Module],
        dropout: float = 0.0,
        initial_bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features  # first linear layer receives n_features values
        for next_width in hidden_layers:
            # adds the next layer with the appropiate activation
            layers.extend((nn.Linear(width, next_width), activation()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = next_width
        layers.append(nn.Linear(width, n_outputs))  # Add the final output layer
        # Now combine the list of layers into one sequention pytorch model
        self.layers = nn.Sequential(*layers)

        # Only the Hard boxes detector supplies an initial bias
        if initial_bias is not None:
            with torch.no_grad():
                self.layers[-1].bias.copy_(initial_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of mlp."""
        return self.layers(x)


def predict_hard_boxes_scores(hard_boxes_detector: FCNN, x: torch.Tensor, device: torch.device) -> np.ndarray:
    """Produce uncalibrated scores for detecting any halving."""
    return batched_forward(hard_boxes_detector, x, device, lambda logits: torch.sigmoid(logits[:, 0]))


def predict_exact_halvings(
    exact_halvings: FCNN,
    x: torch.Tensor,
    device: torch.device,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Predict the exact number of halvings needed."""
    return batched_forward(
        exact_halvings,
        x,
        device,
        lambda logits: torch.softmax(logits, dim=1),
        indices=indices,
    )


def predict_probabilities(
    hard_boxes_detector: FCNN,
    x: torch.Tensor,
    device: torch.device,
    exact_halvings: FCNN | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """Calculate the hard boxes detection scores and the 5 class probabilities.

    We have a threshold where if the hard box detection score is above a certain
    threshold, only then we use our second model to evaluate exact halvings.
    """
    hard_detection = predict_hard_boxes_scores(hard_boxes_detector, x, device).astype(np.float64)
    # Allows binary training
    if exact_halvings is None:
        return hard_detection

    if threshold is None:
        # If we deice against a threshold we run the exact halving networks on every grid box
        conditional = predict_exact_halvings(exact_halvings, x, device)
    else:
        candidates = np.flatnonzero(hard_detection >= threshold)  # find positive rows
        conditional = np.zeros((len(x), N_CLASSES - 1))
        conditional[:, 0] = 1.0
        if len(candidates):
            conditional[candidates] = predict_exact_halvings(exact_halvings, x, device, indices=candidates)
    conditional = conditional.astype(np.float64, copy=False)
    conditional /= conditional.sum(axis=1, keepdims=True)
    # returns a 5 column matrix contain prob for each of the 5 classes
    probabilities = np.empty((len(x), N_CLASSES))
    probabilities[:, 0] = 1 - hard_detection
    probabilities[:, 1:] = hard_detection[:, None] * conditional
    largest = conditional.argmax(axis=1) + 1
    probabilities[np.arange(len(x)), largest] += hard_detection - probabilities[:, 1:].sum(axis=1)
    return probabilities


def hard_boxes_detector_loss(logits: torch.Tensor, targets: torch.Tensor, method: str) -> torch.Tensor:
    """With the hard boxes detector we can experiment with three loss functions for now."""
    targets = targets.float()
    cross_entropy = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if method == "bce":
        return cross_entropy.mean()

    probabilities = torch.sigmoid(logits)
    if method == "focal":
        # gamma = 2, no alpha weighting.
        probability_true = torch.where(targets.bool(), probabilities, 1 - probabilities)
        return ((1 - probability_true).square() * cross_entropy).mean()

    if method == "asymmetric":
        # ASL with gamma_positive=0, gamma_negative=4 and no clipping.
        modulation = torch.where(targets.bool(), torch.ones_like(probabilities), probabilities.pow(4))
        return (modulation * cross_entropy).mean()
    msg = f"unknown detector loss: {method}"
    raise ValueError(msg)


def train_hard_boxes_detector(
    train_x: torch.Tensor,
    train_y: np.ndarray,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    negative_ratio: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    loss_method: str,
    on_epoch=None,
) -> tuple[FCNN, list[dict], int]:
    """Train the binary detector and return it with its history and best epoch."""
    # Split the training row indices into hard boxes and ordinary boxes
    positive_indices, negative_indices = training_index_pools(train_y)
    n_negative = min(len(negative_indices), negative_ratio * len(positive_indices))

    torch.manual_seed(seed)  # Makes network initialisation reproducible
    model = FCNN(
        train_x.shape[1],
        1,
        hidden_layers=HARD_BOXES_DETECTOR_HIDDEN_LAYERS,
        activation=nn.SiLU,
        dropout=HARD_BOXES_DETECTOR_DROPOUT,
        initial_bias=prior_logit(len(positive_indices), n_negative),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    binary_y = torch.from_numpy(train_y > 0)  # convert all physical positive classes into one boolean target
    # If validation AP doesnt improve for several epochs, training ends and restore the best model
    best_ap = -np.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []

    # Training loop
    for epoch in range(1, epochs + 1):
        indices = torch.from_numpy(
            random_undersampling(positive_indices, negative_indices, negative_ratio, seed * 10000 + epoch)
        )
        model.train()
        loss_sum = 0.0  # Track total weighted training loss
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            # take batch from memory and transfer to training device
            batch_x = train_x[batch_indices].to(device)
            batch_y = binary_y[batch_indices].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = hard_boxes_detector_loss(model(batch_x)[:, 0], batch_y, loss_method)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)  # limit the combined gradient norm
            optimizer.step()
            loss_sum += loss.item() * len(batch_indices)

        # After every epoch we run the hard box detector on the complete validation set
        val_ap = detection_ap(predict_hard_boxes_scores(model, val_x, device), val_y)
        history.append(
            {
                "epoch": epoch,
                "training_loss": loss_sum / len(indices),
                "validation_ap": val_ap,
                "training_rows": len(indices),
            }
        )
        if on_epoch is not None:
            on_epoch(history)
        print(
            f"hard box detector epoch {epoch:>2}: loss {history[-1]['training_loss']:.4f}, validation AP {val_ap:.4f}"
        )

        # implementing early stopping
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"hard box detector early stopping after {epoch} epohcs.")
                break
    # retore the model from the best validation epoch rather than the final epoch
    model.load_state_dict(best_state)
    return model, history, best_epoch


def train_exact_halvings(
    train_x: torch.Tensor,
    train_y: np.ndarray,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    device: torch.device,
    seed: int,
    on_epoch=None,
) -> tuple[FCNN, list[dict], int]:
    """We train the exact halvings model only on rows where a halving occured."""
    # Find only rows where atleast 1 halving occured.
    train_positive = np.flatnonzero(train_y > 0)
    val_positive = np.flatnonzero(val_y > 0)
    if not len(train_positive) or not len(val_positive):
        msg = "exact halving training requires positive train and validation rows"
        raise ValueError(msg)
    positive_x = train_x[torch.from_numpy(train_positive)]
    positive_y = torch.from_numpy(train_y[train_positive] - 1)
    val_positive_x = val_x[torch.from_numpy(val_positive)]
    val_positive_y = val_y[val_positive]

    torch.manual_seed(seed)
    model = FCNN(
        train_x.shape[1],
        N_CLASSES - 1,
        hidden_layers=EXACT_HALVINGS_HIDDEN_LAYERS,
        activation=nn.ReLU,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=EXACT_HALVINGS_LEARNING_RATE,
        weight_decay=EXACT_HALVINGS_WEIGHT_DECAY,
    )
    """
    Best model tracking: (uses a three part tuple)
    1. Exact classification rate — higher is better
    2. Mean supported-class recall — higher is better
    3. Negative MAE — higher means lower MAE
    """
    best_key = (-np.inf, -np.inf, -np.inf)
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []

    # Training Loop
    for epoch in range(1, EXACT_HALVINGS_EPOCHS + 1):
        generator = torch.Generator().manual_seed(seed * 10000 + epoch)
        # Create a reproducible random order of all positive training rows
        order = torch.randperm(len(positive_x), generator=generator)
        model.train()
        loss_sum = 0.0

        for start in range(0, len(order), EXACT_HALVINGS_BATCH_SIZE):
            rows = order[start : start + EXACT_HALVINGS_BATCH_SIZE]
            # take batch from memory and transfer to training device
            batch_x = positive_x[rows].to(device)
            batch_y = positive_y[rows].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(rows)

        metrics = exact_halvings_selection_metrics(
            predict_exact_halvings(model, val_positive_x, device), val_positive_y
        )
        history.append({"epoch": epoch, "training_loss": loss_sum / len(order), **metrics})
        if on_epoch is not None:
            on_epoch(history)
        key = (
            metrics["exact"],
            metrics["mean_supported_class_recall"],
            -metrics["mae"],
        )
        # This model has early stopping has well
        improved = key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        # instead of printing every epoch we only print the first and then every 25th epoch or any improved epoch
        if epoch == 1 or epoch % 25 == 0 or improved:
            print(
                f"exact halvings epoch {epoch:>3}: "
                f"loss {history[-1]['training_loss']:.4f}, "
                f"exact {metrics['exact']:.4f}, MAE {metrics['mae']:.4f}"
            )
        if stale_epochs >= EXACT_HALVINGS_PATIENCE:
            print(f"exact halvings early stopping after {epoch} epochs.")
            break
    # Restore the best exact halvings model
    model.load_state_dict(best_state)
    return model, history, best_epoch


def run(args: argparse.Namespace) -> dict:  # noqa: PLR0915
    """Coordinate the complete experiment."""
    features = FEATURES
    groups = [feature.name for feature in features]
    task = "multiclass" if args.multiclass else "binary"
    device = resolve_device(args.device)  # set the appropiate device to train on
    torch.set_num_threads(args.threads)  # set the number of threads pytorch may use
    timesteps = discover_timesteps(args.data_dir)
    splits = split_timesteps(timesteps)
    train_steps, validation_steps, _ = splits
    # Meaure time to load data
    started = time.perf_counter()
    train_x_array, train_y, val_x_array, val_y = load_train_validation(
        args.data_dir, features, train_steps, validation_steps
    )
    load_seconds = time.perf_counter() - started
    print(
        f"{task} FCNN | train {train_x_array.shape} on t{train_steps[0]}-t{train_steps[-1]} "
        f"({int((train_y > 0).sum())} positives)"
    )
    print(
        f"validation {val_x_array.shape} on t{validation_steps[0]}-t{validation_steps[-1]} "
        f"({int((val_y > 0).sum())} positives)"
    )

    # Measure time to preprocess training + validation data
    started = time.perf_counter()
    preprocessor, train_x_array = Preprocesser.fit_transform(train_x_array, features)
    train_preprocessing_seconds = time.perf_counter() - started
    started = time.perf_counter()
    val_x_array = preprocessor.transform(val_x_array, copy=False)
    validation_preprocessing_seconds = time.perf_counter() - started

    train_x = torch.from_numpy(train_x_array)
    val_x = torch.from_numpy(val_x_array)
    dashboard = None
    if args.dashboard:
        from mlstep.dashboard import LiveTrainingDashboard  # noqa: PLC0415

        dashboard = LiveTrainingDashboard()
    # Training both models and measuring time
    started = time.perf_counter()
    hard_boxes_detector, hard_boxes_detector_history, hard_boxes_detector_best_epoch = train_hard_boxes_detector(
        train_x,
        train_y,
        val_x,
        val_y,
        device=device,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        negative_ratio=args.negative_ratio,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_method=args.detector_loss,
        on_epoch=dashboard.update_detector if dashboard is not None else None,
    )
    hard_boxes_detector_training_seconds = time.perf_counter() - started

    exact_halvings = None
    exact_halvings_history = None
    exact_halvings_best_epoch = None
    exact_halvings_training_seconds = 0.0
    if args.multiclass:
        started = time.perf_counter()
        exact_halvings, exact_halvings_history, exact_halvings_best_epoch = train_exact_halvings(
            train_x,
            train_y,
            val_x,
            val_y,
            device=device,
            seed=args.seed,
            on_epoch=dashboard.update_exact_halvings if dashboard is not None else None,
        )
        exact_halvings_training_seconds = time.perf_counter() - started

    hard_box_detector_scores = predict_hard_boxes_scores(hard_boxes_detector, val_x, device)
    # by default we chose a threshold that achieves a target recall of 1.0 so every validation positive gets captured
    threshold = threshold_at_recall(val_y, hard_box_detector_scores, args.target_recall)

    def predict() -> np.ndarray:
        return predict_probabilities(
            hard_boxes_detector,
            val_x,
            device,
            exact_halvings=exact_halvings,
            threshold=threshold,
        )

    metrics = evaluate(
        predict(),
        val_y,
        target_recall=args.target_recall,
        # threshold might not be the final test threshold
        threshold_source="selected on validation set",
        detection_outputs=hard_box_detector_scores,
    )
    if dashboard is not None:
        dashboard.update_final_metrics(metrics)
    inference = benchmark(predict)
    inference.update(
        {
            "rows": len(val_x),
            "rows_per_second": len(val_x) / inference["median_seconds"],
            "device": str(device),
            "exact_halvings_candidate_rows": (
                int(np.count_nonzero(hard_box_detector_scores >= threshold)) if args.multiclass else None
            ),
            "scope": (
                "model prediction on a preprocessed matrix, including batch "
                "transfer and output copy; multiclass severity runs only on "
                "detector candidates; excludes NetCDF feature assembly and "
                "preprocessing"
            ),
        }
    )

    # Constructing the checkpoints including saved weights
    model_path, result_path = output_paths(args.output_dir, f"fcnn_{task}", ".pt")
    dashboard_path = None
    if dashboard is not None:
        dashboard_path = Path(__file__).resolve().parent / "plots" / f"{model_path.stem}.png"
    architecture = {
        "hard_boxes_detector": {
            "hidden_layers": HARD_BOXES_DETECTOR_HIDDEN_LAYERS,
            "activation": "SiLU",
            "dropout": HARD_BOXES_DETECTOR_DROPOUT,
        },
        "exact_halvings": (
            {
                "hidden_layers": EXACT_HALVINGS_HIDDEN_LAYERS,
                "activation": "ReLU",
                "dropout": 0.0,
            }
            if exact_halvings is not None
            else None
        ),
    }
    torch.save(
        {
            "task": task,
            "architecture": architecture,
            "feature_groups": groups,
            "feature_names": feature_names(features),
            "preprocessing": {name: torch.from_numpy(values) for name, values in preprocessor.state().items()},
            # Move saved parameters to cpu so the checkpoint is portable
            "hard_boxes_detector_state_dict": cpu_state(hard_boxes_detector),
            "exact_halvings_state_dict": cpu_state(exact_halvings) if exact_halvings is not None else None,
            "threshold": threshold,
            "threshold_source": "Selected on validation set",
            "n_classes": N_CLASSES,
            "score_context": "uncalibrates ranking scores because we undersampled the training negatives ",
        },
        model_path,
    )
    # Building the shared configuration and data summary as a JSON report
    result = run_metadata(args, features, train_y, val_y, {"torch": torch.__version__}, splits)
    result["config"].update(
        {
            "device": str(device),
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "architecture": architecture,
        }
    )
    result.update(
        {
            "task": task,
            "model_file": model_path.name,
            "model_size_bytes": model_path.stat().st_size,
            "dashboard_file": str(dashboard_path) if dashboard_path is not None else None,
            # Best epoch, and validation metrics
            "selection": {
                "hard_boxes_best_epoch": hard_boxes_detector_best_epoch,
                "exact_halvings_best_epoch": exact_halvings_best_epoch,
                "metrics": metrics,
            },
            "history": {
                "hard_boxes_detector": hard_boxes_detector_history,
                "exact_halvings": exact_halvings_history,
            },
            # Preprocessing, training and inference durations
            "timing": {
                "data_loading_seconds": load_seconds,
                "training_preprocessing_seconds": train_preprocessing_seconds,
                "validation_preprocessing_seconds": (validation_preprocessing_seconds),
                "detector_training_seconds": hard_boxes_detector_training_seconds,
                "severity_training_seconds": exact_halvings_training_seconds,
                "model_prediction_only": inference,
            },
        }
    )

    write_json(result_path, result)
    if dashboard is not None:
        dashboard.save(dashboard_path)
    print(
        f"\nbest detector epoch {hard_boxes_detector_best_epoch}: AP {metrics['ap']:.4f}, "
        f"precision {metrics['precision']:.4f}, recall {metrics['recall']:.4f}"
    )
    if args.multiclass:
        severity_metrics = metrics["severity"]
        print(
            f"best exact_halving epoch {exact_halvings_best_epoch}: "
            f"exact {severity_metrics['exact_on_positive']:.4f}, "
            f"MAE {severity_metrics['mae_on_positive']:.4f}"
        )
    print(f"inference {inference['median_seconds']:.4f}s for {len(val_x):,} rows")
    print(f"saved {model_path}")
    print(f"saved {result_path}")
    if dashboard is not None:
        print(f"saved {dashboard_path}")
        dashboard.close()
    return result


def batched_forward(
    model: FCNN,
    x: torch.Tensor,
    device: torch.device,
    activation,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Run the model over x in batches and concatenate the activated outputs."""
    model.eval()  # Disables training behaviour such as dropout
    outputs = []
    # Whether to process every row or only select candidate rows
    n_rows = len(x) if indices is None else len(indices)
    with torch.inference_mode():  # Disable gradient tracking to reduce inference memory
        for start in range(0, n_rows, INFERENCE_BATCH_SIZE):
            if indices is None:
                batch = x[start : start + INFERENCE_BATCH_SIZE]
            else:
                rows = torch.from_numpy(indices[start : start + INFERENCE_BATCH_SIZE])
                batch = x[rows]
            logits = model(batch.to(device))
            outputs.append(activation(logits).cpu())
    return torch.cat(outputs).numpy()


def prior_logit(n_positive: int, n_negative: int) -> torch.Tensor:
    """Return the output bias matching the positive rate of one undersampled epoch."""
    if not n_positive:
        msg = "training data contains no positives"
        raise ValueError(msg)
    # Calculates the positive fraction in one undersampled epoch
    probability = n_positive / (n_positive + n_negative)
    return torch.tensor([math.log(probability / (1 - probability))], dtype=torch.float32)


def exact_halvings_selection_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Score exact-halvings predictions for validation epoch selection."""
    # Convert physical indices 0-3 into physicsal classes 1-4
    predictions = probabilities.argmax(axis=1) + 1
    # prediction 1, target 3 -> error -2
    # prediction 4, target 2 -> error +2
    errors = predictions - targets
    recalls = [float(np.mean(predictions[targets == label] == label)) for label in np.unique(targets)]
    return {
        "exact": float(np.mean(errors == 0)),
        "mae": float(np.mean(np.abs(errors))),
        "mean_supported_class_recall": float(np.mean(recalls)),
    }


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return every saved weight and bias on the CPU."""
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def resolve_device(requested: str) -> torch.device:
    """Choose CUDA when available otherwise resorts to CPU."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        msg = "CUDA was requested but is not available"
        raise RuntimeError(msg)
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--multiclass", action="store_true")  # Train the second exact halving network
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--patience", type=int, default=12
    )  # Hard box detector maximum training length and early stopping
    parser.add_argument(
        "--negative-ratio", type=int, default=256
    )  # Undersampling rate ( number of negatives per positive in each hard box detector epoch)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--hard-box-detector-loss",
        dest="detector_loss",
        choices=("bce", "focal", "asymmetric"),
        default="asymmetric",
    )
    parser.add_argument("--target-recall", type=float, default=0.97)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
