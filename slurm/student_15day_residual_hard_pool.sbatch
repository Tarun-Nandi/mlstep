#!/bin/bash
#SBATCH --job-name=student-hard-pool
#SBATCH --account=FTORCH-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=36:00:00
#SBATCH --output=/rds/user/rc-nand1/hpc-work/mlstep/logs/%x-%j.out
#SBATCH --error=/rds/user/rc-nand1/hpc-work/mlstep/logs/%x-%j.err
#SBATCH --mail-type=NONE
#SBATCH --no-requeue

# Build one fixed hard-negative pool from the corrected-residual baseline.
# Only class-zero rows from the t2--t252 training cache are ranking candidates
# or retained. Dense inference may forward rare intervening positive training
# rows before discarding their scores. The validation arrays t253--t306 and
# locked rows t307--t360 are never opened.
# Three canonical c00 baseline seeds are averaged to make the ranking less
# dependent on one model initialization.  The validated result is atomically
# published and is then shared by every hard-negative experiment cell.  The
# default 1,753,600-row pool equals one full c00 negative epoch budget, so a
# 25% hard stratum cycles through roughly four epochs before repeating.
# The candidate rows are training-only, but the source baseline checkpoints
# were selected on t253--t306; that period remains development data, not test.

. /etc/profile.d/modules.sh || exit 1
module purge || exit 1
module load rhel8/default-amp || exit 1

set -Eeuo pipefail

readonly conda_setup="${MLSTEP_CONDA_SH:-/rds/user/rc-nand1/hpc-work/conda/etc/profile.d/conda.sh}"
readonly conda_environment="${MLSTEP_CONDA_ENV:-mlstep}"
if [[ ! -r $conda_setup ]]; then
    echo "Cannot read Conda setup: $conda_setup" >&2
    exit 67
fi
source "$conda_setup" || exit 1
conda activate "$conda_environment" || exit 1
if [[ ${CONDA_DEFAULT_ENV:-} != "$conda_environment" ]]; then
    echo "Expected Conda environment $conda_environment, found: ${CONDA_DEFAULT_ENV:-unset}" >&2
    exit 67
fi
readonly conda_libstdcpp="$CONDA_PREFIX/lib/libstdc++.so.6"
if [[ ! -r $conda_libstdcpp ]] || ! grep -aFq GLIBCXX_3.4.29 "$conda_libstdcpp"; then
    echo "Conda C++ runtime is missing GLIBCXX_3.4.29: $conda_libstdcpp" >&2
    exit 69
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

readonly repo_dir="${MLSTEP_REPO_DIR:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
readonly work_root="${MLSTEP_WORK_ROOT:-/rds/user/rc-nand1/hpc-work/mlstep}"
readonly cache_path="${MLSTEP_CACHE_PATH:-$work_root/cache/15day-residual-v1/stretch128}"
readonly baseline_root="${MLSTEP_BASELINE_ROOT:-$work_root/15day-residual-stretch128-k4-baseline-v1/job-33920827/c00}"
readonly pool_size="${MLSTEP_HARD_POOL_SIZE:-1753600}"
readonly pool_path="${MLSTEP_HARD_NEGATIVE_POOL:-$work_root/hard-negative-pools/15day-residual-stretch128-k4-v1/top-$pool_size}"
readonly inference_batch_size="${MLSTEP_HARD_POOL_BATCH_SIZE:-65536}"
readonly chunk_rows="${MLSTEP_HARD_POOL_CHUNK_ROWS:-262144}"
readonly python_bin=python

if [[ ! -f $repo_dir/pyproject.toml || ! -f $repo_dir/mlstep/hard_negative.py ]]; then
    echo "MLSTEP_REPO_DIR is not an mlstep source checkout with hard-negative mining: $repo_dir" >&2
    exit 66
fi
if [[ ! -d $baseline_root ]]; then
    echo "Corrected-residual baseline root does not exist: $baseline_root" >&2
    exit 66
fi
if [[ ! $pool_size =~ ^[1-9][0-9]*$ ]]; then
    echo "MLSTEP_HARD_POOL_SIZE must be a positive integer: $pool_size" >&2
    exit 64
fi
if [[ ! $inference_batch_size =~ ^[1-9][0-9]*$ || ! $chunk_rows =~ ^[1-9][0-9]*$ ]]; then
    echo "Hard-pool batch and chunk sizes must be positive integers" >&2
    exit 64
fi

actual_commit=$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || true)
readonly actual_commit
if [[ -n ${MLSTEP_EXPECTED_COMMIT:-} && $actual_commit != "$MLSTEP_EXPECTED_COMMIT" ]]; then
    echo "Repository commit $actual_commit does not match MLSTEP_EXPECTED_COMMIT=$MLSTEP_EXPECTED_COMMIT" >&2
    exit 65
fi
readonly -a source_contract=(
    mlstep/hard_negative.py
    mlstep/student.py
    mlstep/sampling.py
    mlstep/losses.py
    mlstep/policy.py
    mlstep/evaluation.py
    mlstep/data.py
    mlstep/tabm.py
)
if ! git -C "$repo_dir" ls-files --error-unmatch "${source_contract[@]}" >/dev/null 2>&1; then
    echo "All experiment Python files must be committed before submission" >&2
    exit 65
fi
if ! git -C "$repo_dir" diff --quiet -- "${source_contract[@]}" || \
    ! git -C "$repo_dir" diff --cached --quiet -- "${source_contract[@]}"; then
    echo "Experiment Python files differ from repository commit $actual_commit" >&2
    exit 65
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$repo_dir"

miner_module=$(
    "$python_bin" - <<'PY'
from pathlib import Path
import mlstep.hard_negative

print(Path(mlstep.hard_negative.__file__).resolve(strict=True))
PY
)
readonly miner_module
case "$miner_module" in
    "$repo_dir"/*) ;;
    *)
        echo "Python resolved mlstep.hard_negative outside MLSTEP_REPO_DIR: $miner_module" >&2
        exit 65
        ;;
esac

checkpoints=()
reports=()
for seed in 0 1 2; do
    seed_dir="$baseline_root/seed-$seed"
    if [[ ! -d $seed_dir ]]; then
        echo "Missing corrected-residual baseline seed directory: $seed_dir" >&2
        exit 68
    fi
    mapfile -t seed_success < <(find "$seed_dir" -mindepth 1 -maxdepth 2 -type f -name SUCCESS -print | sort)
    if (( ${#seed_success[@]} != 1 )) || [[ $(<"${seed_success[0]}") != 0 ]]; then
        echo "Expected exactly one successful run below $seed_dir" >&2
        exit 68
    fi
    run_dir=${seed_success[0]%/SUCCESS}
    mapfile -t seed_checkpoints < <(
        find "$run_dir" -maxdepth 1 -type f -name '*.pt' \
            ! -name '*_ap_best.pt' ! -name '*_fp97_best.pt' -print | sort
    )
    if (( ${#seed_checkpoints[@]} != 1 )); then
        echo "Expected one canonical checkpoint in $run_dir" >&2
        exit 70
    fi
    mapfile -t seed_reports < <(find "$run_dir" -maxdepth 1 -type f -name '*.json' -print | sort)
    if (( ${#seed_reports[@]} != 1 )); then
        echo "Expected one canonical JSON report in $run_dir" >&2
        exit 70
    fi
    checkpoints+=("${seed_checkpoints[0]}")
    reports+=("${seed_reports[0]}")
done
readonly -a checkpoints
readonly -a reports

# Check the corrected feature schema, exact chronological training split,
# baseline roles/seeds and GPU allocation before scanning 65.9 million rows.
"$python_bin" - "$cache_path" "${checkpoints[@]}" -- "${reports[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from mlstep.data import FEATURES, feature_names

cache = Path(sys.argv[1]).resolve(strict=True)
separator = sys.argv.index("--")
checkpoint_paths = [Path(value).resolve(strict=True) for value in sys.argv[2:separator]]
report_paths = [Path(value).resolve(strict=True) for value in sys.argv[separator + 1 :]]
if len(checkpoint_paths) != 3 or len(set(checkpoint_paths)) != 3:
    raise RuntimeError("Hard-pool mining requires three distinct canonical checkpoints")
if len(report_paths) != 3 or len(set(report_paths)) != 3:
    raise RuntimeError("Hard-pool mining requires three distinct canonical reports")

# Deliberately do not open val_x.npy or val_y.npy in this job.
for name in ("SUCCESS", "metadata.json", "train_x.npy", "train_y.npy"):
    path = cache / name
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty training-cache artifact: {path}")
if (cache / "SUCCESS").read_text(encoding="utf-8").strip() != "0":
    raise RuntimeError("Cache SUCCESS marker is invalid")
if list(cache.glob("*_teacher_*.npy")):
    raise RuntimeError("Hard-negative experiments require a teacherless cache")

metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
expected_features = feature_names(FEATURES)
if "residual[0]" not in expected_features or any(name.startswith("rhs[") for name in expected_features):
    raise RuntimeError("Runtime feature schema is not the corrected residual schema")
if metadata.get("features") != expected_features or metadata.get("n_features") != 266:
    raise RuntimeError("Cache feature schema differs from the corrected runtime schema")
if metadata.get("preprocess") != "stretch128" or metadata.get("teacher") is not None:
    raise RuntimeError("Mining requires the teacherless corrected-residual Stretch128 cache")
if metadata.get("train", {}).get("timesteps") != list(range(2, 253)):
    raise RuntimeError("Mining cache does not contain exact training timesteps t2--t252")
split = metadata.get("split", {})
if split.get("unused_timesteps") != list(range(307, 361)):
    raise RuntimeError("Mining cache does not keep t307--t360 wholly unused")

train_x = np.load(cache / "train_x.npy", mmap_mode="r", allow_pickle=False)
train_y = np.load(cache / "train_y.npy", mmap_mode="r", allow_pickle=False)
expected_counts = [65_919_806, 5_735, 1_025, 89, 1]
if train_x.shape != (65_926_656, 266) or train_x.dtype != np.float32:
    raise RuntimeError("Training features differ from the 15-day residual experiment contract")
if train_y.shape != (65_926_656,) or not np.issubdtype(train_y.dtype, np.integer):
    raise RuntimeError("Training labels differ from the 15-day residual experiment contract")
if metadata.get("train", {}).get("class_counts") != expected_counts:
    raise RuntimeError("Training class counts differ from the corrected 15-day cache")

for expected_seed, (checkpoint_path, report_path) in enumerate(zip(checkpoint_paths, report_paths, strict=True)):
    if checkpoint_path.parent != report_path.parent:
        raise RuntimeError("Each mining checkpoint must be paired with its run report")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = checkpoint.get("config", {})
    if checkpoint.get("checkpoint_role") != "canonical":
        raise RuntimeError(f"Mining source is not canonical: {checkpoint_path}")
    if checkpoint.get("seed") not in (None, expected_seed):
        raise RuntimeError(f"Mining checkpoint has the wrong seed: {checkpoint_path}")
    if config.get("architecture") != "tabm-mini" or config.get("k") != 4:
        raise RuntimeError(f"Mining source is not TabM-mini K4: {checkpoint_path}")
    if config.get("hidden") != [512, 256] or config.get("preprocess") != "stretch128":
        raise RuntimeError(f"Mining source is not the fixed c00 model: {checkpoint_path}")
    protocol = checkpoint.get("training_protocol", {}).get("data_split", {})
    if protocol.get("train_timesteps") != list(range(2, 253)):
        raise RuntimeError(f"Mining source used a different training split: {checkpoint_path}")
    report_config = report.get("config", {})
    report_model = report.get("model", {})
    if report.get("task") != "supervised" or report.get("seed") != expected_seed:
        raise RuntimeError(f"Mining report has the wrong task or seed: {report_path}")
    if report_config.get("seed") != expected_seed or report_config.get("distill") is not False:
        raise RuntimeError(f"Mining report has the wrong training objective: {report_path}")
    if Path(report_config.get("cache_path", "")).resolve(strict=True) != cache:
        raise RuntimeError(f"Mining report records a different corrected cache: {report_path}")
    if report_model.get("architecture") != "tabm-mini" or report_model.get("k") != 4:
        raise RuntimeError(f"Mining report is not TabM-mini K4: {report_path}")
    if report_model.get("hidden") != [512, 256] or report_model.get("preprocess") != "stretch128":
        raise RuntimeError(f"Mining report is not the fixed c00 model: {report_path}")
    expected_scalars = {
        "dropout": 0.05,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "lambda_sev": 0.1,
    }
    for key, expected in expected_scalars.items():
        if not math.isclose(float(report_config.get(key)), expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Mining report has the wrong c00 {key}: {report_path}")
    if report_config.get("negative_ratio") != 256 or report_config.get("members") != 4:
        raise RuntimeError(f"Mining report has the wrong c00 training budget: {report_path}")
    report_split = report.get("training_protocol", {}).get("data_split", {})
    if report_split != checkpoint.get("training_protocol", {}).get("data_split", {}):
        raise RuntimeError(f"Checkpoint/report split provenance differs: {report_path}")
    if report.get("objective") != checkpoint.get("objective"):
        raise RuntimeError(f"Checkpoint/report objective provenance differs: {report_path}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Hard-pool mining requires exactly one allocated CUDA GPU")
PY

readonly pool_parent=${pool_path%/*}
readonly pool_name=${pool_path##*/}
if [[ -z $pool_parent || -z $pool_name || $pool_parent == "$pool_path" ]]; then
    echo "MLSTEP_HARD_NEGATIVE_POOL must identify an explicit output leaf: $pool_path" >&2
    exit 64
fi
mkdir -p "$pool_parent"
if [[ -e $pool_path ]]; then
    echo "Refusing to overwrite existing hard-negative pool: $pool_path" >&2
    exit 73
fi
readonly stage_path="$pool_parent/.${pool_name}.building-${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
if [[ -e $stage_path ]]; then
    echo "Refusing to reuse hard-pool staging directory: $stage_path" >&2
    exit 73
fi

pool_complete=0
on_exit() {
    status=$?
    if (( status != 0 || pool_complete != 1 )); then
        if (( status == 0 )); then
            status=71
        fi
        if [[ -d $stage_path ]]; then
            printf '%s\n' "$status" >"$stage_path/FAILED"
        fi
    fi
}
trap on_exit EXIT

command=(
    "$python_bin" -u -m mlstep.hard_negative
    --cache-path "$cache_path"
    --checkpoint "${checkpoints[@]}"
    --output-dir "$stage_path"
    --pool-size "$pool_size"
    --chunk-rows "$chunk_rows"
    --inference-batch-size "$inference_batch_size"
    --device cuda
)

srun --ntasks=1 --kill-on-bad-exit=1 "${command[@]}"

{
    printf 'job_script_version=%s\n' 'student-15day-residual-hard-pool-v1'
    printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
    printf 'repository=%s\n' "$repo_dir"
    printf 'git_commit=%s\n' "${actual_commit:-unavailable}"
    printf 'miner_module=%s\n' "$miner_module"
    printf 'cache=%s\n' "$cache_path"
    printf 'baseline_root=%s\n' "$baseline_root"
    printf 'final_pool=%s\n' "$pool_path"
    sha256sum \
        "$repo_dir/mlstep/hard_negative.py" \
        "$repo_dir/mlstep/student.py" \
        "$repo_dir/mlstep/sampling.py" \
        "$repo_dir/mlstep/data.py" \
        "$cache_path/metadata.json" \
        "${checkpoints[@]}" \
        "${reports[@]}"
    printf 'git_status_begin\n'
    git -C "$repo_dir" status --short || true
    printf 'git_status_end\n'
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'
    "$python_bin" - <<'PY'
import platform
import torch

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
} >"$stage_path/launch.txt"

"$python_bin" - "$stage_path" "$cache_path" "$pool_size" "${checkpoints[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

pool = Path(sys.argv[1]).resolve(strict=True)
cache = Path(sys.argv[2]).resolve(strict=True)
pool_size = int(sys.argv[3])
expected_checkpoints = {str(Path(value).resolve(strict=True)) for value in sys.argv[4:]}
if len(expected_checkpoints) != 3:
    raise RuntimeError("Expected exactly three distinct mining checkpoints")
for name in ("SUCCESS", "metadata.json", "hard_negative_indices.npy", "launch.txt"):
    path = pool / name
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty hard-pool artifact: {path}")
if (pool / "SUCCESS").read_text(encoding="utf-8").strip() != "0":
    raise RuntimeError("Hard-pool SUCCESS marker is invalid")

metadata = json.loads((pool / "metadata.json").read_text(encoding="utf-8"))
if metadata.get("format_version") != "hard-negative-pool-v1":
    raise RuntimeError("Unsupported hard-negative pool format")
selection = metadata.get("selection", {})
if selection.get("source_split") != "train" or selection.get("source_label") != 0:
    raise RuntimeError("Hard pool was not selected exclusively from training class zero")
if selection.get("pool_size") != pool_size or selection.get("negative_rows_scored") != 65_919_806:
    raise RuntimeError("Hard-pool selection size or scored population is wrong")
if selection.get("output_order") != "ascending global train row index":
    raise RuntimeError("Hard-pool index ordering is not deterministic")
cache_metadata = metadata.get("cache", {})
if Path(cache_metadata.get("path", "")).resolve(strict=True) != cache:
    raise RuntimeError("Hard pool records a different training cache")
if cache_metadata.get("validation_arrays_accessed") is not False:
    raise RuntimeError("Hard-pool metadata does not prove validation arrays were excluded")
if cache_metadata.get("train_timesteps") != list(range(2, 253)):
    raise RuntimeError("Hard-pool metadata records the wrong training timesteps")
if cache_metadata.get("train_class_counts") != [65_919_806, 5_735, 1_025, 89, 1]:
    raise RuntimeError("Hard-pool metadata records the wrong training labels")

ensemble = metadata.get("ensemble", {})
checkpoints = ensemble.get("checkpoints", [])
recorded_checkpoints = {str(Path(item.get("path", "")).resolve(strict=True)) for item in checkpoints}
if ensemble.get("checkpoint_count") != 3 or recorded_checkpoints != expected_checkpoints:
    raise RuntimeError("Hard pool was not mined with the three resolved baseline checkpoints")
if any(item.get("checkpoint_role") != "canonical" for item in checkpoints):
    raise RuntimeError("Hard pool records a non-canonical mining checkpoint")

indices_path = pool / "hard_negative_indices.npy"
indices = np.load(indices_path, mmap_mode="r", allow_pickle=False)
if indices.shape != (pool_size,) or indices.dtype != np.int64:
    raise RuntimeError("Hard-negative index array has the wrong shape or dtype")
if np.any(indices < 0) or np.any(indices >= 65_926_656) or np.any(indices[1:] <= indices[:-1]):
    raise RuntimeError("Hard-negative indices are out of range, duplicated, or unsorted")
labels = np.load(cache / "train_y.npy", mmap_mode="r", allow_pickle=False)
if np.any(labels[indices] != 0):
    raise RuntimeError("Hard-negative pool contains a non-zero training label")

digest = hashlib.sha256()
with indices_path.open("rb") as handle:
    while block := handle.read(8 * 1024 * 1024):
        digest.update(block)
artifact = metadata.get("output_artifacts", {}).get("hard_negative_indices.npy", {})
if artifact.get("bytes") != indices_path.stat().st_size or artifact.get("sha256") != digest.hexdigest():
    raise RuntimeError("Hard-negative index artifact differs from its recorded digest")
PY

mv -T "$stage_path" "$pool_path"
pool_complete=1
echo "Published validated hard-negative pool: $pool_path"
