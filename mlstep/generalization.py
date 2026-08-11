"""Freeze validation decisions before one-shot chronological test evaluation.

This module deliberately separates model/threshold selection (``freeze``) from
held-out test access (``test``).  The freeze step can read only cached
validation inputs.  The test step verifies the frozen manifest, reconstructs
the train-fitted preprocessor, and evaluates exactly timesteps 144--168 without
refitting the primary detector threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import torch
import xgboost as xgb

from mlstep.cache import detector_margin, severity_distribution
from mlstep.data import (
    FEATURES,
    GRID_SHAPE,
    class_counts,
    discover_timesteps,
    feature_names,
    load_fitted_preprocessor,
    load_timesteps,
    split_timesteps,
)
from mlstep.evaluation import detection_counts, evaluate, safe_divide, threshold_at_recall
from mlstep.student import fp_recall_diagnostics, predict_joint_probabilities, student_from_checkpoint

MANIFEST_VERSION: Final = "held-out-student-ensemble-v1"
REPORT_VERSION: Final = "held-out-student-test-v1"
ALLOWED_PREPROCESSING: Final = frozenset({"stretch32", "stretch128", "ple64"})
EXPECTED_DISCOVERED_TIMESTEPS: Final = tuple(range(2, 169))
EXPECTED_TRAIN_TIMESTEPS: Final = tuple(range(2, 119))
EXPECTED_VALIDATION_TIMESTEPS: Final = tuple(range(119, 144))
HELD_OUT_TEST_TIMESTEPS: Final = tuple(range(144, 169))
EXPECTED_TEST_ROWS: Final = len(HELD_OUT_TEST_TIMESTEPS) * int(np.prod(GRID_SHAPE))
DEFAULT_TARGET_RECALL: Final = 0.97
DEFAULT_BATCH_SIZE: Final = 65_536
PLE_MAX_EVAL_BATCH_SIZE: Final = 16_384
MATRIX_NDIM: Final = 2
EXPECTED_TEACHER_THRESHOLD: Final = 0.177640569121596
EXPECTED_ENSEMBLE_SEEDS: Final = tuple(range(5))
BOOTSTRAP_REPEATS: Final = 10_000
BOOTSTRAP_SEED: Final = 0
READ_ONLY_MODE: Final = 0o444


def _utc_now() -> str:
    """Return an unambiguous UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 2**20) -> str:
    """Hash one file without loading it completely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str | int]:
    """Describe one immutable input artifact."""
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        msg = f"Expected a file artifact: {resolved}"
        raise ValueError(msg)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _verify_artifact(artifact: dict, label: str) -> Path:
    """Verify an artifact recorded in a frozen manifest."""
    if not isinstance(artifact, dict):
        msg = f"Invalid {label} artifact record"
        raise ValueError(msg)
    path = Path(artifact.get("path", ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != artifact.get("bytes"):
        msg = f"{label} size changed after manifest freeze: {path}"
        raise RuntimeError(msg)
    if sha256_file(path) != artifact.get("sha256"):
        msg = f"{label} SHA-256 changed after manifest freeze: {path}"
        raise RuntimeError(msg)
    return path


def _exclusive_json(path: Path, content: dict) -> None:
    """Create a JSON artifact once and make accidental edits fail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(content, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        msg = f"Refusing to replace existing artifact: {path}"
        raise FileExistsError(msg) from None
    path.chmod(READ_ONLY_MODE)


def _sidecar_path(manifest_path: Path) -> Path:
    """Return the manifest digest sidecar path."""
    return Path(f"{manifest_path}.sha256")


def _receipt_path(test_report_path: Path) -> Path:
    """Return the one-shot held-out-access receipt path."""
    return Path(f"{test_report_path}.ACCESS")


def _success_path(test_report_path: Path) -> Path:
    """Return the successful held-out-evaluation marker path."""
    return Path(f"{test_report_path}.SUCCESS")


def _attempt_receipt_path(test_report_path: Path, attempt: int) -> Path:
    """Return a stable receipt path for one incomplete/retried attempt."""
    base = _receipt_path(test_report_path)
    return base if attempt == 1 else Path(f"{base}.{attempt}")


def _start_access_attempt(test_report_path: Path, manifest_path: Path, manifest_digest: str) -> tuple[Path, int]:
    """Record held-out access, allowing only unchanged retries after failure."""
    attempt = 1
    while _attempt_receipt_path(test_report_path, attempt).exists():
        receipt_path = _attempt_receipt_path(test_report_path, attempt)
        receipt = _load_json(receipt_path, "held-out access receipt")
        if (
            receipt.get("manifest_sha256") != manifest_digest
            or receipt.get("declared_test_report_path") != str(test_report_path)
            or receipt.get("timesteps") != list(HELD_OUT_TEST_TIMESTEPS)
        ):
            msg = f"Existing access receipt is for a different frozen test: {receipt_path}"
            raise RuntimeError(msg)
        attempt += 1

    receipt_path = _attempt_receipt_path(test_report_path, attempt)
    _exclusive_json(
        receipt_path,
        {
            "status": "held-out-test-access-started",
            "attempt": attempt,
            "retry_policy": "retry allowed only before report/SUCCESS and with identical manifest SHA-256",
            "created_utc": _utc_now(),
            "manifest_path": str(manifest_path.expanduser().resolve(strict=True)),
            "manifest_sha256": manifest_digest,
            "declared_test_report_path": str(test_report_path),
            "timesteps": list(HELD_OUT_TEST_TIMESTEPS),
        },
    )
    return receipt_path, attempt


def _write_manifest(path: Path, manifest: dict) -> str:
    """Write a frozen manifest and its independently verifiable digest."""
    sidecar = _sidecar_path(path)
    if path.exists() or sidecar.exists():
        msg = f"Refusing to replace existing manifest or digest: {path}"
        raise FileExistsError(msg)
    _exclusive_json(path, manifest)
    digest = sha256_file(path)
    _exclusive_json(
        sidecar,
        {
            "algorithm": "sha256",
            "manifest": path.name,
            "sha256": digest,
        },
    )
    return digest


def _read_verified_manifest(path: Path) -> tuple[dict, str]:
    """Read a manifest only after validating its frozen digest."""
    path = path.expanduser().resolve(strict=True)
    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    digest_record = json.loads(sidecar.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if digest_record.get("algorithm") != "sha256" or digest_record.get("sha256") != digest:
        msg = "Manifest SHA-256 validation failed"
        raise RuntimeError(msg)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != MANIFEST_VERSION:
        msg = f"Unsupported manifest format: {manifest.get('format_version')!r}"
        raise ValueError(msg)
    return manifest, digest


def _load_json(path: Path, label: str) -> dict:
    """Load a required JSON object."""
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        msg = f"{label} must contain a JSON object"
        raise ValueError(msg)
    return content


def _checkpoint_role(checkpoint: dict) -> str:
    """Return the supported checkpoint selection role."""
    if checkpoint.get("checkpoint_role") == "canonical":
        return "canonical"
    if checkpoint.get("checkpoint_role") == "diagnostic" and checkpoint.get("diagnostic_name") == "fp97_best":
        return "fp97_best"
    msg = "Only canonical and fp97_best checkpoints can be frozen for held-out evaluation"
    raise ValueError(msg)


def _validate_checkpoint_report(checkpoint: dict, report: dict, report_path: Path, role: str) -> None:
    """Verify that a checkpoint and source report describe the same run."""
    if report.get("config", {}).get("test_set_used") is True or report.get("test_set_used") is True:
        msg = "Source reports that already accessed the test set cannot enter a frozen ensemble"
        raise ValueError(msg)
    if checkpoint.get("objective") != report.get("objective"):
        msg = "Checkpoint and source report objective metadata differ"
        raise ValueError(msg)
    if checkpoint.get("training_protocol") != report.get("training_protocol"):
        msg = "Checkpoint and source report training protocols differ"
        raise ValueError(msg)

    if role == "canonical":
        if checkpoint.get("selection") != report.get("selection"):
            msg = "Canonical checkpoint and source report selection metadata differ"
            raise ValueError(msg)
    else:
        source_report = checkpoint.get("source_report")
        if source_report != report_path.name:
            msg = "FP@97 checkpoint refers to a different source report"
            raise ValueError(msg)
        report_selection = report.get("diagnostic_checkpoints", {}).get("fp97_best")
        if checkpoint.get("selection") != report_selection:
            msg = "FP@97 checkpoint and source report diagnostic metadata differ"
            raise ValueError(msg)
        if report_selection.get("saved") is not True:
            msg = "Source report does not record a saved FP@97 checkpoint"
            raise ValueError(msg)


def _member_record(checkpoint_path: Path, report_path: Path, preprocess: str) -> tuple[dict, dict, dict]:
    """Validate and describe one explicitly selected ensemble member."""
    checkpoint_path = checkpoint_path.expanduser().resolve(strict=True)
    report_path = report_path.expanduser().resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        msg = f"Checkpoint is not a dictionary: {checkpoint_path}"
        raise ValueError(msg)
    report = _load_json(report_path, "source report")
    role = _checkpoint_role(checkpoint)
    _validate_checkpoint_report(checkpoint, report, report_path, role)

    report_preprocess = report.get("config", {}).get("preprocess")
    if report_preprocess != preprocess:
        msg = f"Source report preprocessing {report_preprocess!r} does not match cache {preprocess!r}"
        raise ValueError(msg)
    if checkpoint.get("config", {}).get("n_features") is None:
        msg = "Checkpoint config is missing n_features"
        raise ValueError(msg)
    teacher_threshold = report.get("teacher_validation", {}).get("metrics", {}).get("threshold")
    if not isinstance(teacher_threshold, (int, float)) or not np.isfinite(teacher_threshold):
        msg = "Source report is missing its finite cached-teacher validation threshold"
        raise ValueError(msg)
    return (
        {
            "role": role,
            "seed": report.get("seed"),
            "teacher_validation_threshold": float(teacher_threshold),
            "checkpoint": _artifact(checkpoint_path),
            "source_report": _artifact(report_path),
        },
        checkpoint,
        report,
    )


def _predict_equal_ensemble(
    checkpoints: Sequence[dict],
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average full probabilities from predeclared checkpoints with equal weight."""
    if not checkpoints:
        msg = "At least one checkpoint is required"
        raise ValueError(msg)
    probability_sum: np.ndarray | None = None
    detector_sum: np.ndarray | None = None
    for checkpoint in checkpoints:
        model = student_from_checkpoint(checkpoint).to(device)
        probabilities, detector = predict_joint_probabilities(model, features, device, batch_size)
        if probability_sum is None:
            probability_sum = probabilities.astype(np.float64)
            detector_sum = detector.astype(np.float64)
        else:
            probability_sum += probabilities
            if detector_sum is None:
                msg = "Internal detector accumulator was not initialized"
                raise RuntimeError(msg)
            detector_sum += detector
        del model, probabilities, detector
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if probability_sum is None or detector_sum is None:
        msg = "Ensemble prediction did not produce outputs"
        raise RuntimeError(msg)
    probability_sum /= len(checkpoints)
    detector_sum /= len(checkpoints)
    return probability_sum, detector_sum


def _cache_artifacts(
    cache_dir: Path,
    preprocess: str,
    *,
    include_teacher: bool,
) -> dict[str, dict[str, str | int]]:
    """Hash every validation/preprocessing artifact needed after freezing."""
    names = ["val_x.npy", "val_y.npy", "metadata.json", "preprocessor_state.npz"]
    if preprocess == "ple64":
        names.append("ple_metadata.npz")
    if include_teacher:
        names.extend(("val_teacher_margin.npy", "val_teacher_severity.npy"))
    return {name: _artifact(cache_dir / name) for name in names}


def _runtime_provenance() -> dict:
    """Pin the local inference implementation and relevant package versions."""
    package_dir = Path(__file__).resolve().parent
    source_names = (
        "generalization.py",
        "data.py",
        "evaluation.py",
        "student.py",
        "tabm.py",
        "embeddings.py",
        "cache.py",
        "policy.py",
    )
    return {
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "torch", "xgboost", "xarray")
        },
        "source_artifacts": {name: _artifact(package_dir / name) for name in source_names},
    }


def _verify_runtime_provenance(provenance: dict) -> None:
    """Refuse test access if inference code or runtime versions changed."""
    required_packages = {"numpy", "scikit-learn", "torch", "xgboost", "xarray"}
    required_sources = {
        "generalization.py",
        "data.py",
        "evaluation.py",
        "student.py",
        "tabm.py",
        "embeddings.py",
        "cache.py",
        "policy.py",
    }
    if set(provenance.get("packages", {})) != required_packages:
        msg = "Manifest runtime package provenance is incomplete"
        raise ValueError(msg)
    if set(provenance.get("source_artifacts", {})) != required_sources:
        msg = "Manifest inference-source provenance is incomplete"
        raise ValueError(msg)
    if provenance.get("python") != sys.version:
        msg = "Python runtime changed after manifest freeze"
        raise RuntimeError(msg)
    expected_packages = provenance.get("packages", {})
    for name, expected_version in expected_packages.items():
        if importlib.metadata.version(name) != expected_version:
            msg = f"{name} version changed after manifest freeze"
            raise RuntimeError(msg)
    for name, artifact in provenance.get("source_artifacts", {}).items():
        _verify_artifact(artifact, f"inference source {name}")


def _joint_teacher_probabilities(margin: np.ndarray, severity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct the teacher's population-corrected five-class distribution."""
    margin = np.asarray(margin)
    severity = np.asarray(severity)
    if margin.ndim != 1 or severity.shape != (len(margin), 4):
        msg = "Teacher margin/severity arrays have incompatible shapes"
        raise ValueError(msg)
    detector = np.empty(margin.shape, dtype=np.result_type(margin.dtype, np.float32))
    nonnegative = margin >= 0
    detector[nonnegative] = 1.0 / (1.0 + np.exp(-margin[nonnegative]))
    exponential = np.exp(margin[~nonnegative])
    detector[~nonnegative] = exponential / (1.0 + exponential)
    probabilities = np.empty((len(margin), 5), dtype=np.result_type(severity.dtype, np.float32))
    probabilities[:, 0] = 1.0 - detector
    probabilities[:, 1:] = detector[:, None] * severity
    return probabilities, detector


def _teacher_comparator_record(
    teacher_model_path: Path,
    teacher_report_path: Path,
    metadata: dict,
    reports: Sequence[dict],
    validation_cache: Path,
    val_y: np.ndarray,
) -> dict:
    """Freeze the canonical XGBoost comparator and its validation threshold."""
    teacher_model_path = teacher_model_path.expanduser().resolve(strict=True)
    teacher_report_path = teacher_report_path.expanduser().resolve(strict=True)
    recorded_path = Path(metadata.get("teacher", {}).get("path", "")).expanduser().resolve(strict=True)
    if teacher_model_path != recorded_path:
        msg = "Requested teacher comparator is not the model used to generate cached teacher responses"
        raise ValueError(msg)
    teacher_report = _load_json(teacher_report_path, "teacher source report")
    if teacher_report.get("task") != "multiclass" or teacher_report.get("model_file") != teacher_model_path.name:
        msg = "Teacher report does not describe the requested multiclass XGBoost model"
        raise ValueError(msg)
    if teacher_report.get("config", {}).get("test_set_used") is not False:
        msg = "Teacher report must explicitly record that its held-out test set was unused"
        raise ValueError(msg)
    teacher_config = teacher_report["config"]
    if teacher_config.get("device") != "cpu" or teacher_config.get("parameters", {}).get("device") != "cpu":
        msg = "Teacher report must record CPU training and inference"
        raise ValueError(msg)
    expected_config_values = {
        "train_steps": list(EXPECTED_TRAIN_TIMESTEPS),
        "validation_steps": list(EXPECTED_VALIDATION_TIMESTEPS),
        "held_out_test_steps": list(HELD_OUT_TEST_TIMESTEPS),
        "feature_names": feature_names(FEATURES),
        "n_features": sum(feature.channels for feature in FEATURES),
    }
    for name, expected in expected_config_values.items():
        if teacher_config.get(name) != expected:
            msg = f"Teacher report has unexpected {name}; raw feature/split provenance is inconsistent"
            raise ValueError(msg)

    thresholds = [report.get("teacher_validation", {}).get("metrics", {}).get("threshold") for report in reports]
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in thresholds):
        msg = "Every student report must contain a finite teacher validation threshold"
        raise ValueError(msg)
    if any(not np.isclose(value, thresholds[0], rtol=0.0, atol=1e-15) for value in thresholds[1:]):
        msg = "Student source reports disagree about the cached teacher validation threshold"
        raise ValueError(msg)
    if not np.isclose(thresholds[0], EXPECTED_TEACHER_THRESHOLD, rtol=0.0, atol=1e-15):
        msg = "Cached teacher threshold differs from the predeclared canonical value"
        raise ValueError(msg)

    margin = np.load(validation_cache / "val_teacher_margin.npy", mmap_mode="r")
    severity = np.load(validation_cache / "val_teacher_severity.npy", mmap_mode="r")
    probabilities, detector = _joint_teacher_probabilities(margin, severity)
    recomputed_threshold = threshold_at_recall(val_y, detector, DEFAULT_TARGET_RECALL)
    if not np.isclose(recomputed_threshold, thresholds[0], rtol=0.0, atol=1e-15):
        msg = "Cached teacher responses do not reproduce the source reports' frozen threshold"
        raise ValueError(msg)
    threshold_source = "frozen cached-teacher validation threshold before held-out test access"
    metrics = evaluate(
        probabilities,
        val_y,
        threshold=float(thresholds[0]),
        threshold_source=threshold_source,
        detection_outputs=detector,
    )
    return {
        "model": _artifact(teacher_model_path),
        "source_report": _artifact(teacher_report_path),
        "population_prior_correction_delta_t": float(metadata["teacher"]["delta_t"]),
        "inference_device": "cpu",
        "target_recall": DEFAULT_TARGET_RECALL,
        "frozen_threshold": float(thresholds[0]),
        "threshold_source": threshold_source,
        "validation_metrics": metrics,
    }


def freeze_manifest(  # noqa: PLR0912, PLR0915
    member_paths: Sequence[tuple[Path, Path]],
    validation_cache: Path,
    manifest_path: Path,
    test_report_path: Path,
    *,
    teacher_model_path: Path | None = None,
    teacher_report_path: Path | None = None,
    target_recall: float = DEFAULT_TARGET_RECALL,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Freeze an explicit equal-weight ensemble and validation threshold.

    This function has no raw-data path argument by design, preventing held-out
    test access during ensemble and threshold selection.
    """
    if not member_paths:
        msg = "At least one explicit checkpoint/report pair is required"
        raise ValueError(msg)
    if not 0.0 < target_recall <= 1.0:
        msg = "target_recall must be in (0, 1]"
        raise ValueError(msg)
    if batch_size < 1:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if (teacher_model_path is None) != (teacher_report_path is None):
        msg = "teacher_model_path and teacher_report_path must be provided together"
        raise ValueError(msg)

    validation_cache = validation_cache.expanduser().resolve(strict=True)
    manifest_path = manifest_path.expanduser().resolve()
    test_report_path = test_report_path.expanduser().resolve()
    blocked_outputs = [
        manifest_path,
        _sidecar_path(manifest_path),
        test_report_path,
        _success_path(test_report_path),
    ]
    blocked_outputs.extend(test_report_path.parent.glob(f"{test_report_path.name}.ACCESS*"))
    existing = [str(path) for path in blocked_outputs if path.exists()]
    if existing:
        msg = f"Refusing to freeze over existing artifacts: {', '.join(existing)}"
        raise FileExistsError(msg)

    metadata = _load_json(validation_cache / "metadata.json", "cache metadata")
    preprocess = metadata.get("preprocess")
    if preprocess not in ALLOWED_PREPROCESSING:
        msg = f"Held-out evaluation supports only {sorted(ALLOWED_PREPROCESSING)}, found {preprocess!r}"
        raise ValueError(msg)
    if metadata.get("preprocessing", {}).get("fit_split") != "chronological training timesteps only":
        msg = "Cache does not prove that preprocessing was fitted on chronological training timesteps only"
        raise ValueError(msg)
    effective_batch_size = min(batch_size, PLE_MAX_EVAL_BATCH_SIZE) if preprocess == "ple64" else batch_size

    val_x = np.load(validation_cache / "val_x.npy", mmap_mode="r")
    val_y = np.load(validation_cache / "val_y.npy", mmap_mode="r")
    if val_x.ndim != MATRIX_NDIM or val_y.shape != (len(val_x),):
        msg = "Validation cache feature/label shapes are inconsistent"
        raise ValueError(msg)
    if metadata.get("val", {}).get("shape") != list(val_x.shape):
        msg = "Validation cache shape differs from metadata"
        raise ValueError(msg)
    if metadata.get("val", {}).get("class_counts") != class_counts(val_y):
        msg = "Validation cache labels differ from metadata"
        raise ValueError(msg)

    records = []
    checkpoints = []
    reports = []
    seen_checkpoints: set[str] = set()
    expected_config: dict | None = None
    for checkpoint_path, report_path in member_paths:
        record, checkpoint, report = _member_record(checkpoint_path, report_path, preprocess)
        checkpoint_key = record["checkpoint"]["path"]
        if checkpoint_key in seen_checkpoints:
            msg = f"Duplicate ensemble checkpoint: {checkpoint_key}"
            raise ValueError(msg)
        seen_checkpoints.add(checkpoint_key)
        config = checkpoint["config"]
        if config["n_features"] != val_x.shape[1]:
            msg = "Checkpoint feature count does not match validation cache"
            raise ValueError(msg)
        if expected_config is None:
            expected_config = config
        elif config != expected_config:
            msg = "All ensemble checkpoints must have identical model configuration"
            raise ValueError(msg)
        records.append(record)
        checkpoints.append(checkpoint)
        reports.append(report)
    roles = {record["role"] for record in records}
    if len(roles) != 1:
        msg = "A frozen ensemble must not mix canonical and FP@97 checkpoint roles"
        raise ValueError(msg)
    seeds = [record["seed"] for record in records]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        msg = "Every ensemble source report must record an integer seed"
        raise ValueError(msg)
    if sorted(seeds) != list(EXPECTED_ENSEMBLE_SEEDS):
        msg = f"Frozen confirmation ensemble must contain exactly one member for seeds {EXPECTED_ENSEMBLE_SEEDS}"
        raise ValueError(msg)

    torch_device = torch.device(device)
    probabilities, detector = _predict_equal_ensemble(checkpoints, val_x, torch_device, effective_batch_size)
    threshold = threshold_at_recall(val_y, detector, target_recall)
    threshold_source = "frozen on validation equal-probability ensemble before held-out test access"
    validation_metrics = evaluate(
        probabilities,
        val_y,
        threshold=threshold,
        threshold_source=threshold_source,
        detection_outputs=detector,
    )
    if validation_metrics["recall"] + np.finfo(float).eps < target_recall:
        msg = "Frozen validation threshold does not achieve its requested recall"
        raise RuntimeError(msg)

    teacher_comparator = None
    if teacher_model_path is not None and teacher_report_path is not None:
        teacher_comparator = _teacher_comparator_record(
            teacher_model_path,
            teacher_report_path,
            metadata,
            reports,
            validation_cache,
            val_y,
        )

    weight = 1.0 / len(records)
    for record in records:
        record["probability_weight"] = weight
    manifest = {
        "format_version": MANIFEST_VERSION,
        "status": "frozen-before-held-out-test-access",
        "created_utc": _utc_now(),
        "selection_constraints": {
            "held_out_test_used": False,
            "ensemble_predeclared": True,
            "threshold_fitted_on": "validation",
            "test_results_must_not_change_ensemble_or_threshold": True,
        },
        "runtime_provenance": _runtime_provenance(),
        "ensemble": {
            "aggregation": "equal arithmetic mean of full joint class probabilities",
            "member_count": len(records),
            "checkpoint_role": next(iter(roles)),
            "validation_inference_batch_size": effective_batch_size,
            "members": records,
        },
        "validation": {
            "cache_dir": str(validation_cache),
            "preprocess": preprocess,
            "artifacts": _cache_artifacts(
                validation_cache,
                preprocess,
                include_teacher=teacher_comparator is not None,
            ),
            "rows": len(val_y),
            "class_counts": class_counts(val_y),
            "target_recall": target_recall,
            "frozen_threshold": float(threshold),
            "threshold_source": threshold_source,
            "metrics": validation_metrics,
        },
        "teacher_comparator": teacher_comparator,
        "held_out_test_plan": {
            "expected_discovered_timesteps": list(EXPECTED_DISCOVERED_TIMESTEPS),
            "timesteps": list(HELD_OUT_TEST_TIMESTEPS),
            "expected_rows": EXPECTED_TEST_ROWS,
            "test_report_path": str(test_report_path),
            "access_policy": (
                "create an immutable ACCESS receipt before reading held-out files; an interrupted attempt may retry "
                "only with the identical manifest SHA-256 and before report/SUCCESS exists"
            ),
            "primary_threshold_policy": "reuse validation frozen_threshold exactly; never fit on test labels",
            "oracle_diagnostics_policy": "clearly labeled descriptive test-fitted thresholds; never select a model",
        },
    }
    manifest_digest = _write_manifest(manifest_path, manifest)
    print(f"Frozen manifest: {manifest_path}")
    print(f"Manifest SHA-256: {manifest_digest}")
    print(f"Frozen validation threshold: {threshold:.9g}")
    return manifest


def _verify_frozen_inputs(manifest: dict) -> tuple[list[dict], Path, str]:
    """Verify all member and cache hashes before test access."""
    ensemble = manifest.get("ensemble", {})
    members = ensemble.get("members", [])
    if not members or ensemble.get("aggregation") != "equal arithmetic mean of full joint class probabilities":
        msg = "Manifest does not contain a valid equal-probability ensemble"
        raise ValueError(msg)
    roles = {member.get("role") for member in members}
    if roles != {ensemble.get("checkpoint_role")} or len(roles) != 1:
        msg = "Manifest checkpoint roles are inconsistent"
        raise ValueError(msg)
    expected_weight = 1.0 / len(members)
    if any(not np.isclose(member.get("probability_weight"), expected_weight) for member in members):
        msg = "Manifest ensemble weights are not exactly equal"
        raise ValueError(msg)

    checkpoints = []
    for index, member in enumerate(members):
        checkpoint_path = _verify_artifact(member.get("checkpoint"), f"member {index} checkpoint")
        report_path = _verify_artifact(member.get("source_report"), f"member {index} source report")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        report = _load_json(report_path, "source report")
        role = _checkpoint_role(checkpoint)
        if role != member.get("role"):
            msg = f"Member {index} checkpoint role changed"
            raise RuntimeError(msg)
        _validate_checkpoint_report(checkpoint, report, report_path, role)
        checkpoints.append(checkpoint)

    validation = manifest.get("validation", {})
    preprocess = validation.get("preprocess")
    if preprocess not in ALLOWED_PREPROCESSING:
        msg = f"Unsupported frozen preprocessing method: {preprocess!r}"
        raise ValueError(msg)
    cache_dir = Path(validation.get("cache_dir", "")).resolve(strict=True)
    required_artifacts = {"val_x.npy", "val_y.npy", "metadata.json", "preprocessor_state.npz"}
    if preprocess == "ple64":
        required_artifacts.add("ple_metadata.npz")
    if manifest.get("teacher_comparator") is not None:
        required_artifacts.update(("val_teacher_margin.npy", "val_teacher_severity.npy"))
    artifacts = validation.get("artifacts", {})
    if set(artifacts) != required_artifacts:
        msg = "Manifest validation-cache artifact set is incomplete or unexpected"
        raise ValueError(msg)
    for name, artifact in artifacts.items():
        path = _verify_artifact(artifact, f"validation cache {name}")
        if path.parent != cache_dir:
            msg = f"Validation artifact escaped its frozen cache directory: {path}"
            raise ValueError(msg)
    return checkpoints, cache_dir, preprocess


def _verify_teacher_comparator(record: dict | None) -> dict | None:
    """Verify the optional frozen XGBoost comparator artifacts."""
    if record is None:
        return None
    if not isinstance(record, dict):
        msg = "Invalid teacher comparator record"
        raise ValueError(msg)
    model_path = _verify_artifact(record.get("model"), "teacher model")
    report_path = _verify_artifact(record.get("source_report"), "teacher source report")
    teacher_report = _load_json(report_path, "teacher source report")
    if teacher_report.get("task") != "multiclass" or teacher_report.get("model_file") != model_path.name:
        msg = "Frozen teacher report/model association is inconsistent"
        raise ValueError(msg)
    threshold = record.get("frozen_threshold")
    if not np.isclose(threshold, EXPECTED_TEACHER_THRESHOLD, rtol=0.0, atol=1e-15):
        msg = "Frozen teacher comparator threshold changed"
        raise ValueError(msg)
    delta_t = record.get("population_prior_correction_delta_t")
    if not isinstance(delta_t, (int, float)) or not np.isfinite(delta_t):
        msg = "Frozen teacher prior correction must be finite"
        raise ValueError(msg)
    if record.get("inference_device") != "cpu":
        msg = "Frozen teacher comparator must use its source-report CPU inference device"
        raise ValueError(msg)
    return {**record, "verified_model_path": model_path}


def _predict_teacher(
    teacher_record: dict,
    raw_features: np.ndarray,
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the raw-feature XGBoost teacher with population prior correction."""
    teacher = xgb.XGBClassifier()
    teacher.load_model(str(teacher_record["verified_model_path"]))
    teacher.set_params(device=device)
    logits = teacher.predict(raw_features, output_margin=True)
    margin = detector_margin(
        logits,
        float(teacher_record["population_prior_correction_delta_t"]),
    )
    severity = severity_distribution(logits)
    return _joint_teacher_probabilities(margin, severity)


def _per_timestep_counts(
    detector: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    timesteps: Sequence[int],
) -> list[dict[str, float | int]]:
    """Return frozen-threshold detector counts for each chronological timestep."""
    rows_per_timestep = int(np.prod(GRID_SHAPE))
    if len(labels) != len(detector) or len(labels) != rows_per_timestep * len(timesteps):
        msg = "Predictions do not match the fixed UKCA timestep layout"
        raise ValueError(msg)
    result = []
    for index, timestep in enumerate(timesteps):
        start = index * rows_per_timestep
        stop = start + rows_per_timestep
        positive = labels[start:stop] > 0
        counts = detection_counts(detector[start:stop] >= threshold, positive)
        predicted_positive = counts["true_positive"] + counts["false_positive"]
        actual_positive = counts["true_positive"] + counts["false_negative"]
        result.append(
            {
                "timestep": int(timestep),
                "rows": rows_per_timestep,
                "positive_examples": int(positive.sum()),
                "precision": safe_divide(counts["true_positive"], predicted_positive),
                "recall": safe_divide(counts["true_positive"], actual_positive),
                **counts,
            }
        )
    return result


def _block_bootstrap_intervals(
    per_timestep: Sequence[dict],
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap uncertainty by resampling whole chronological timesteps.

    Spatial rows within a timestep are never resampled independently.  This
    avoids pretending that millions of strongly related grid boxes are IID.
    """
    if not per_timestep or repeats < 1:
        msg = "Block bootstrap requires timesteps and at least one repeat"
        raise ValueError(msg)
    names = ("true_positive", "false_positive", "false_negative", "true_negative")
    blocks = np.array([[row[name] for name in names] for row in per_timestep], dtype=np.int64)
    generator = np.random.default_rng(seed)
    sample_indices = generator.integers(0, len(blocks), size=(repeats, len(blocks)))
    totals = blocks[sample_indices].sum(axis=1)
    true_positive, false_positive, false_negative, true_negative = totals.T
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    actual_negative = false_positive + true_negative
    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros(repeats, dtype=np.float64),
        where=predicted_positive > 0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros(repeats, dtype=np.float64),
        where=actual_positive > 0,
    )
    false_positives_per_million_negatives = np.divide(
        false_positive * 1_000_000.0,
        actual_negative,
        out=np.zeros(repeats, dtype=np.float64),
        where=actual_negative > 0,
    )

    observed = blocks.sum(axis=0)
    observed_tp, observed_fp, observed_fn, observed_tn = observed

    def interval(values: np.ndarray, estimate: float) -> dict[str, float]:
        lower, upper = np.quantile(values, (0.025, 0.975))
        return {"estimate": float(estimate), "lower_95": float(lower), "upper_95": float(upper)}

    return {
        "method": "percentile bootstrap of whole chronological timestep blocks",
        "rows_resampled_independently": False,
        "blocks": len(blocks),
        "repeats": repeats,
        "seed": seed,
        "metrics": {
            "true_positive": interval(true_positive, observed_tp),
            "false_positive": interval(false_positive, observed_fp),
            "precision": interval(
                precision,
                safe_divide(int(observed_tp), int(observed_tp + observed_fp)),
            ),
            "recall": interval(
                recall,
                safe_divide(int(observed_tp), int(observed_tp + observed_fn)),
            ),
            "false_positives_per_million_negatives": interval(
                false_positives_per_million_negatives,
                safe_divide(int(observed_fp) * 1_000_000, int(observed_fp + observed_tn)),
            ),
        },
    }


def _raw_test_artifacts(data_dir: Path, timesteps: Sequence[int]) -> dict[str, dict[str, str | int]]:
    """Fingerprint every raw file used for the held-out evaluation."""
    names = ("ncsteps", *(feature.name for feature in FEATURES))
    return {
        f"{name}_{timestep}.nc": _artifact(data_dir / f"{name}_{timestep}.nc")
        for timestep in timesteps
        for name in names
    }


def evaluate_frozen_manifest(  # noqa: PLR0912, PLR0915
    manifest_path: Path,
    data_dir: Path,
    *,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Access the exact held-out period once using a frozen manifest."""
    if batch_size < 1:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    manifest, manifest_digest = _read_verified_manifest(manifest_path)
    plan = manifest.get("held_out_test_plan", {})
    if tuple(plan.get("expected_discovered_timesteps", ())) != EXPECTED_DISCOVERED_TIMESTEPS:
        msg = "Manifest does not predeclare the exact t2--168 source period"
        raise ValueError(msg)
    if tuple(plan.get("timesteps", ())) != HELD_OUT_TEST_TIMESTEPS:
        msg = "Manifest does not predeclare exact held-out timesteps t144--168"
        raise ValueError(msg)
    if plan.get("expected_rows") != EXPECTED_TEST_ROWS:
        msg = "Manifest has the wrong held-out row count"
        raise ValueError(msg)

    test_report_path = Path(plan.get("test_report_path", ""))
    if not test_report_path.is_absolute():
        msg = "Frozen test report path must be absolute"
        raise ValueError(msg)
    test_report_path = test_report_path.resolve()
    success_path = _success_path(test_report_path)
    existing = [str(path) for path in (test_report_path, success_path) if path.exists()]
    if existing:
        msg = f"Held-out test was already completed: {', '.join(existing)}"
        raise FileExistsError(msg)

    _verify_runtime_provenance(manifest.get("runtime_provenance", {}))
    checkpoints, cache_dir, preprocess = _verify_frozen_inputs(manifest)
    teacher_record = _verify_teacher_comparator(manifest.get("teacher_comparator"))
    effective_batch_size = min(batch_size, PLE_MAX_EVAL_BATCH_SIZE) if preprocess == "ple64" else batch_size
    threshold = manifest.get("validation", {}).get("frozen_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        msg = "Manifest frozen threshold must be finite and in [0, 1]"
        raise ValueError(msg)

    # The receipt is intentionally written before discovering, hashing, or
    # loading raw held-out files.  A failed run therefore cannot silently be
    # rerun and treated as a first look at the test set.
    receipt_path, access_attempt = _start_access_attempt(
        test_report_path,
        manifest_path,
        manifest_digest,
    )

    data_dir = data_dir.expanduser().resolve(strict=True)
    discovered = discover_timesteps(data_dir)
    if discovered != EXPECTED_DISCOVERED_TIMESTEPS:
        msg = "Raw source must contain exactly discovered timesteps t2--168 after spin-up removal"
        raise RuntimeError(msg)
    _, _, derived_test = split_timesteps(discovered)
    if derived_test != HELD_OUT_TEST_TIMESTEPS:
        msg = "Chronological split does not reproduce held-out timesteps t144--168"
        raise RuntimeError(msg)

    raw_artifacts = _raw_test_artifacts(data_dir, HELD_OUT_TEST_TIMESTEPS)
    raw_x, test_y = load_timesteps(data_dir, HELD_OUT_TEST_TIMESTEPS, FEATURES)
    if raw_x.shape != (EXPECTED_TEST_ROWS, sum(feature.channels for feature in FEATURES)):
        msg = f"Held-out feature matrix has unexpected shape: {raw_x.shape}"
        raise ValueError(msg)
    if test_y.shape != (EXPECTED_TEST_ROWS,):
        msg = f"Held-out target vector has unexpected shape: {test_y.shape}"
        raise ValueError(msg)

    teacher_evaluation = None
    if teacher_record is not None:
        teacher_probabilities, teacher_detector = _predict_teacher(
            teacher_record,
            raw_x,
            device="cpu",
        )
        teacher_threshold = float(teacher_record["frozen_threshold"])
        teacher_threshold_source = "teacher validation threshold frozen before held-out test access"
        teacher_primary = evaluate(
            teacher_probabilities,
            test_y,
            threshold=teacher_threshold,
            threshold_source=teacher_threshold_source,
            detection_outputs=teacher_detector,
        )
        teacher_per_timestep = _per_timestep_counts(
            teacher_detector,
            test_y,
            teacher_threshold,
            HELD_OUT_TEST_TIMESTEPS,
        )
        teacher_evaluation = {
            "inference_device": "cpu",
            "frozen_threshold": teacher_threshold,
            "threshold_source": teacher_threshold_source,
            "primary_metrics": teacher_primary,
            "per_timestep_counts": teacher_per_timestep,
            "whole_timestep_block_bootstrap_95_ci": _block_bootstrap_intervals(teacher_per_timestep),
            "threshold_free_metrics": {
                "ap": teacher_primary["ap"],
                "ranked_probability_score": teacher_primary["probability"]["ranked_probability_score"],
                "negative_log_likelihood": teacher_primary["probability"]["negative_log_likelihood"],
                "brier_score": teacher_primary["probability"]["brier_score"],
            },
            "oracle_test_fitted_fixed_recall_diagnostics": {
                "warning": "Descriptive test-fitted thresholds only; never used for selection.",
                "used_for_selection": False,
                "metrics": fp_recall_diagnostics(teacher_detector, test_y),
            },
        }
        del teacher_probabilities, teacher_detector

    preprocessor = load_fitted_preprocessor(cache_dir / "preprocessor_state.npz", preprocess, features=FEATURES)
    test_x = preprocessor.transform(raw_x, copy=False)
    if test_x is not raw_x:
        del raw_x
    probabilities, detector = _predict_equal_ensemble(
        checkpoints,
        test_x,
        torch.device(device),
        effective_batch_size,
    )
    threshold_source = "validation threshold frozen in immutable manifest before held-out access"
    primary_metrics = evaluate(
        probabilities,
        test_y,
        threshold=float(threshold),
        threshold_source=threshold_source,
        detection_outputs=detector,
    )
    per_timestep = _per_timestep_counts(
        detector,
        test_y,
        float(threshold),
        HELD_OUT_TEST_TIMESTEPS,
    )
    oracle = fp_recall_diagnostics(detector, test_y)

    report = {
        "format_version": REPORT_VERSION,
        "status": "held-out-test-evaluated-once",
        "held_out": True,
        "test_set_used": True,
        "created_utc": _utc_now(),
        "student_inference_batch_size": effective_batch_size,
        "manifest": {
            "path": str(manifest_path.expanduser().resolve(strict=True)),
            "sha256": manifest_digest,
        },
        "access_attempt": access_attempt,
        "access_receipt": _artifact(receipt_path),
        "selection_constraints": {
            "ensemble_changed_after_test": False,
            "primary_threshold_refitted_on_test": False,
            "test_results_permitted_for_further_model_selection": False,
        },
        "data": {
            "source_dir": str(data_dir),
            "discovered_timesteps": list(discovered),
            "test_timesteps": list(HELD_OUT_TEST_TIMESTEPS),
            "rows": len(test_y),
            "class_counts": class_counts(test_y),
            "raw_artifacts": raw_artifacts,
        },
        "primary_frozen_threshold_evaluation": {
            "threshold": float(threshold),
            "threshold_source": threshold_source,
            "metrics": primary_metrics,
            "per_timestep_counts": per_timestep,
            "whole_timestep_block_bootstrap_95_ci": _block_bootstrap_intervals(per_timestep),
        },
        "threshold_free_test_metrics": {
            "ap": primary_metrics["ap"],
            "ranked_probability_score": primary_metrics["probability"]["ranked_probability_score"],
            "negative_log_likelihood": primary_metrics["probability"]["negative_log_likelihood"],
            "brier_score": primary_metrics["probability"]["brier_score"],
        },
        "oracle_test_fitted_fixed_recall_diagnostics": {
            "warning": (
                "These thresholds were fitted on held-out test labels for description only. "
                "They are not the primary result and must not drive model, checkpoint, or threshold selection."
            ),
            "used_for_primary_evaluation": False,
            "used_for_selection": False,
            "metrics": oracle,
        },
        "teacher_comparator": teacher_evaluation,
    }
    _exclusive_json(test_report_path, report)
    report_digest = sha256_file(test_report_path)
    _exclusive_json(
        success_path,
        {
            "status": "completed",
            "test_report": test_report_path.name,
            "test_report_sha256": report_digest,
            "manifest_sha256": manifest_digest,
        },
    )
    print(f"Held-out report: {test_report_path}")
    print(f"Held-out report SHA-256: {report_digest}")
    print(f"Frozen-threshold precision: {primary_metrics['precision']:.9g}")
    print(f"Frozen-threshold recall: {primary_metrics['recall']:.9g}")
    print(f"Frozen-threshold false positives: {primary_metrics['false_positive']}")
    return report


def _member_pair(values: Sequence[str]) -> tuple[Path, Path]:
    """Convert one CLI checkpoint/report pair to paths."""
    checkpoint, report = values
    return Path(checkpoint), Path(report)


def parse_args() -> argparse.Namespace:
    """Parse freeze/test command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="Freeze an ensemble and validation threshold")
    freeze.add_argument("--validation-cache", type=Path, required=True)
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--test-report", type=Path, required=True)
    freeze.add_argument(
        "--teacher-model",
        type=Path,
        help="Optional canonical multiclass XGBoost comparator model",
    )
    freeze.add_argument(
        "--teacher-report",
        type=Path,
        help="Source JSON report for --teacher-model",
    )
    freeze.add_argument(
        "--member",
        nargs=2,
        action="append",
        required=True,
        metavar=("CHECKPOINT", "SOURCE_REPORT"),
        help="Explicit checkpoint and its source JSON report; repeat for each equal-weight member",
    )
    freeze.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    freeze.add_argument("--device", default="cpu")
    freeze.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    test = subparsers.add_parser("test", help="Evaluate a frozen manifest on held-out t144--168 once")
    test.add_argument("--manifest", type=Path, required=True)
    test.add_argument("--data-dir", type=Path, required=True)
    test.add_argument("--device", default="cpu")
    test.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> dict:
    """Run the selected freeze/test command."""
    args = parse_args() if args is None else args
    if args.command == "freeze":
        members = [_member_pair(member) for member in args.member]
        return freeze_manifest(
            members,
            args.validation_cache,
            args.manifest,
            args.test_report,
            teacher_model_path=args.teacher_model,
            teacher_report_path=args.teacher_report,
            target_recall=args.target_recall,
            device=args.device,
            batch_size=args.batch_size,
        )
    if args.command == "test":
        return evaluate_frozen_manifest(
            args.manifest,
            args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
    msg = f"Unsupported command: {args.command!r}"
    raise ValueError(msg)


if __name__ == "__main__":
    main()
