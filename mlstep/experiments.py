"""Phase 1-3 experiment configurations and utilities."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PreprocessConfig:
    """Preprocessing experiment configuration."""

    name: str
    method: str
    n_features: int | None = None
    n_bins: int | None = None
    embedding_dim: int | None = None
    supervised: bool = True
    contrastive: bool = False


@dataclass
class ArchitectureConfig:
    """Architecture experiment configuration."""

    name: str
    k: int
    hidden_layers: tuple[int, ...]
    dropout: float = 0.05


@dataclass
class AdvancedConfig:
    """Advanced techniques configuration."""

    name: str
    label_smoothing: float = 0.0
    temperature: float = 1.0
    ensemble: bool = False
    hard_negative: bool = False
    lambda_ordinal: float = 0.0


# Phase 1: Preprocessing Optimization
PREPROCESS_EXPERIMENTS = [
    PreprocessConfig("supervised_stretch128", "stretch128", supervised=True),
    PreprocessConfig("supervised_stretch32", "stretch32", supervised=True),
    PreprocessConfig("physical", "physical", supervised=False),
    PreprocessConfig("ple64_supervised", "ple64", n_features=64, n_bins=48, embedding_dim=12, supervised=True),
    PreprocessConfig("contrastive_stretch128", "stretch128", supervised=True, contrastive=True),
]

# Phase 2: Architecture Scaling
ARCHITECTURE_EXPERIMENTS = [
    ArchitectureConfig("tabm_k8", k=8, hidden_layers=(512, 256)),
    ArchitectureConfig("tabm_k16", k=16, hidden_layers=(512, 256)),
    ArchitectureConfig("deeper_512_512_256", k=4, hidden_layers=(512, 512, 256)),
    ArchitectureConfig("wider_768_384", k=4, hidden_layers=(768, 384)),
]

# Phase 3: Advanced Techniques
ADVANCED_EXPERIMENTS = [
    AdvancedConfig("label_smooth_0.1", label_smoothing=0.1),
    AdvancedConfig("label_smooth_0.2", label_smoothing=0.2),
    AdvancedConfig("temperature_0.8", temperature=0.8),
    AdvancedConfig("temperature_1.5", temperature=1.5),
    AdvancedConfig("ensemble_best", ensemble=True),
    AdvancedConfig("hard_negative_qcf", hard_negative=True),
]

# Combined configs for final experiments
FINAL_EXPERIMENTS = [
    # Best preprocessing + QCF + different architectures
    ("ple64_qcf_k8", "ple64_supervised", "tabm_k8"),
    ("ple64_qcf_k16", "ple64_supervised", "tabm_k16"),
    ("stretch128_qcf_k8", "supervised_stretch128", "tabm_k8"),
    # Advanced techniques with best configs
    ("ple64_qcf_label_smooth", "ple64_supervised", "label_smooth_0.1"),
    ("ple64_qcf_temperature", "ple64_supervised", "temperature_0.8"),
]


def get_config(name: str, config_type: str) -> Any:
    """Get experiment configuration by name and type."""
    if config_type == "preprocess":
        for config in PREPROCESS_EXPERIMENTS:
            if config.name == name:
                return config
    elif config_type == "architecture":
        for config in ARCHITECTURE_EXPERIMENTS:
            if config.name == name:
                return config
    elif config_type == "advanced":
        for config in ADVANCED_EXPERIMENTS:
            if config.name == name:
                return config
    error_msg = f"Unknown {config_type} config: {name}"
    raise ValueError(error_msg)


def validate_experiment_config(
    preprocess_name: str, arch_name: str | None = None, advanced_name: str | None = None
) -> dict:
    """Validate and return experiment configuration."""
    config = {"preprocess": get_config(preprocess_name, "preprocess")}

    if arch_name:
        config["architecture"] = get_config(arch_name, "architecture")
    if advanced_name:
        config["advanced"] = get_config(advanced_name, "advanced")

    return config
