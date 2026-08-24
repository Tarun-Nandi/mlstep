"""Main training script for all experimental phases."""

import argparse
import json
from pathlib import Path

import torch
from torch import optim

from mlstep.experiments import (
    ArchitectureConfig,
    PreprocessConfig,
    get_config,
)
from mlstep.student import (
    Student,
    evaluate_model,
    train_epoch,
)


def parse_args() -> argparse.Namespace:
    """Parse experiment arguments."""
    parser = argparse.ArgumentParser(description="Train experimental models for Phase 1-3")

    # Experiment selection
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True, help="Experiment phase (1, 2, or 3)")
    parser.add_argument("--preprocess", type=str, help="Preprocessing method name")
    parser.add_argument("--architecture", type=str, help="Architecture configuration name")
    parser.add_argument("--advanced", type=str, help="Advanced technique name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Data configuration
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/tnandi/Adaptive Timestepping/mlstep/mlstep/24hr_data",
        help="Data directory",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="/rds/user/rc-nand1/hpc-work/mlstep/cache", help="Cache directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="/rds/user/rc-nand1/hpc-work/mlstep/experiments", help="Output directory"
    )
    parser.add_argument("--feature-set", type=str, default="baseline-qcf", help="Feature set to use")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--target-recall", type=float, default=0.97, help="Target recall")

    # Device configuration
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    return parser.parse_args()


# Phase constants
PHASE_1 = 1
PHASE_2 = 2
PHASE_3 = 3


def setup_experiment_config(args: argparse.Namespace) -> dict:
    """Set up experiment configuration based on phase."""
    config = {"phase": args.phase, "seed": args.seed}

    if args.phase == PHASE_1:
        # Preprocessing experiments
        preprocess_name = args.preprocess or "supervised_stretch128"
        config["preprocess"] = get_config(preprocess_name, "preprocess")
        config["architecture"] = ArchitectureConfig("default_k4", k=4, hidden_layers=(512, 256))

    elif args.phase == PHASE_2:
        # Architecture experiments
        arch_name = args.architecture or "tabm_k8"
        config["preprocess"] = PreprocessConfig("supervised_stretch128", "stretch128", supervised=True)
        config["architecture"] = get_config(arch_name, "architecture")

    elif args.phase == PHASE_3:
        # Advanced techniques
        advanced_name = args.advanced or "label_smooth_0.1"
        config["preprocess"] = PreprocessConfig("supervised_stretch128", "stretch128", supervised=True)
        config["architecture"] = ArchitectureConfig("default_k4", k=4, hidden_layers=(512, 256))
        config["advanced"] = get_config(advanced_name, "advanced")

    return config


def create_model(config: dict, n_features: int, device: torch.device) -> Student:
    """Create model based on configuration."""
    arch_config = config["architecture"]

    # Handle PLE64 preprocessing
    preprocess_config = config.get("preprocess")
    ple_config = None

    if preprocess_config and preprocess_config.method == "ple64":
        ple_config = {
            "n_ple_features": preprocess_config.n_features,
            "n_bins": preprocess_config.n_bins,
            "embedding_dim": preprocess_config.embedding_dim,
            "ple_indices": list(range(preprocess_config.n_features)),
            "bypass_indices": list(range(preprocess_config.n_features, n_features)),
            "version": "B",
        }

    model = Student(
        n_features=n_features,
        hidden_layers=list(arch_config.hidden_layers),
        k=arch_config.k,
        architecture="tabm-mini" if arch_config.k > 1 else "mlp",
        ple_config=ple_config,
    ).to(device)

    return model


def setup_optimizer(model: Student, args: argparse.Namespace):
    """Set up optimizer and scheduler."""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    return optimizer, scheduler


def train_with_config(config: dict, args: argparse.Namespace, train_x, train_y, val_x, val_y) -> dict:
    """Train model with experimental configuration."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get feature dimensions
    n_features = train_x.shape[1]

    # Create model
    model = create_model(config, n_features, device)

    # Setup optimizer
    optimizer, scheduler = setup_optimizer(model, args)

    # Training loop
    best_metrics = None
    patience_counter = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        # Train one epoch
        train_loss = train_epoch(
            model,
            optimizer,
            train_x,
            train_y,
            None,
            None,  # No teacher for now
            device,
            args.batch_size,
            256,
            args.seed,
            0.0,
            1.0,
            0.1,
            0.1,
            1.0,
            0.0,
            positive_indices=None,
            negative_indices=None,
            return_details=True,
        )

        # Validation
        val_metrics = evaluate_model(model, val_x, val_y, device, batch_size=65536, target_recall=args.target_recall)

        # Check for improvement
        current_loss = train_loss.loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Step scheduler
        scheduler.step(current_loss)

        print(f"Epoch {epoch}: Train Loss={current_loss:.4f}, Val AP={val_metrics['ap']:.4f}")

    return best_metrics


def run_phase1_experiment(config: dict, args: argparse.Namespace) -> dict:
    """Run Phase 1 preprocessing experiment."""
    print(f"Running Phase 1 experiment: {config['preprocess'].name}")

    # Load data (simplified for now - in practice, use cached data)
    print("Loading data...")

    # Create dummy data for testing structure
    # In practice, this would load from cache
    train_x = torch.randn(10000, 267).to(args.device)
    train_y = torch.randint(0, 5, (10000,)).to(args.device)
    val_x = torch.randn(2000, 267).to(args.device)
    val_y = torch.randint(0, 5, (2000,)).to(args.device)

    metrics = train_with_config(config, args, train_x, train_y, val_x, val_y)

    return {
        "phase": 1,
        "experiment": config["preprocess"].name,
        "metrics": metrics,
        "config": {"preprocess": config["preprocess"].name, "seed": args.seed},
    }


def run_phase2_experiment(config: dict, args: argparse.Namespace) -> dict:
    """Run Phase 2 architecture experiment."""
    print(f"Running Phase 2 experiment: {config['architecture'].name}")

    # Similar structure to Phase 1
    train_x = torch.randn(10000, 267).to(args.device)
    train_y = torch.randint(0, 5, (10000,)).to(args.device)
    val_x = torch.randn(2000, 267).to(args.device)
    val_y = torch.randint(0, 5, (2000,)).to(args.device)

    metrics = train_with_config(config, args, train_x, train_y, val_x, val_y)

    return {
        "phase": 2,
        "experiment": config["architecture"].name,
        "metrics": metrics,
        "config": {
            "architecture": config["architecture"].name,
            "k": config["architecture"].k,
            "hidden_layers": config["architecture"].hidden_layers,
            "seed": args.seed,
        },
    }


def run_phase3_experiment(config: dict, args: argparse.Namespace) -> dict:
    """Run Phase 3 advanced techniques experiment."""
    print(f"Running Phase 3 experiment: {config['advanced'].name}")

    # Similar structure with advanced techniques
    train_x = torch.randn(10000, 267).to(args.device)
    train_y = torch.randint(0, 5, (10000,)).to(args.device)
    val_x = torch.randn(2000, 267).to(args.device)
    val_y = torch.randint(0, 5, (2000,)).to(args.device)

    metrics = train_with_config(config, args, train_x, train_y, val_x, val_y)

    return {
        "phase": 3,
        "experiment": config["advanced"].name,
        "metrics": metrics,
        "config": {
            "advanced": config["advanced"].name,
            "label_smoothing": config["advanced"].label_smoothing,
            "temperature": config["advanced"].temperature,
            "seed": args.seed,
        },
    }


def save_results(results: dict, args: argparse.Namespace):
    """Save experiment results."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase = results["phase"]
    experiment = results["experiment"]
    seed = args.seed

    filename = f"phase{phase}_{experiment}_seed{seed}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_path}")


def main():
    """Run the main training function."""
    args = parse_args()

    # Setup configuration
    config = setup_experiment_config(args)

    print(f"Starting Phase {args.phase} experiment")
    print(f"Configuration: {config}")

    # Run experiment based on phase
    if args.phase == PHASE_1:
        results = run_phase1_experiment(config, args)
    elif args.phase == PHASE_2:
        results = run_phase2_experiment(config, args)
    elif args.phase == PHASE_3:
        results = run_phase3_experiment(config, args)
    else:
        error_msg = f"Unknown phase: {args.phase}"
        raise ValueError(error_msg)

    # Save results
    save_results(results, args)

    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()
