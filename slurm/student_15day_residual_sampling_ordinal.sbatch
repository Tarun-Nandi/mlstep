#!/bin/bash
#SBATCH --job-name=student-hard-ordinal
#SBATCH --account=FTORCH-SL2-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=36:00:00
#SBATCH --array=0-8%3
#SBATCH --output=/rds/user/rc-nand1/hpc-work/mlstep/logs/%x-%A_%a.out
#SBATCH --error=/rds/user/rc-nand1/hpc-work/mlstep/logs/%x-%A_%a.err
#SBATCH --mail-type=NONE
#SBATCH --no-requeue

# Controlled corrected-residual experiment: three treatments x three seeds.
# Model architecture, optimizer, negative budget and data split are fixed to
# the previous c00 baseline.  Only sampling and the ordinal RPS term change:
#
#   c00: cyclic traversal of the full class-zero pool, ordinal weight 0;
#   c01: 25% fixed hard + 75% cyclic easy negatives, ordinal weight 0;
#   c02: the same hard sampler, conditional severity ordinal weight 1.
#
# Checkpoint selection uses threshold-free proper probability scores.  Actions
# are joint MAP with a validation-only posterior-risk frontier for lambda
# 0, 0.25, 0.5, 1 and 2; recall never selects or controls an action threshold.
# Training is t2--t252, validation is t253--t306, and t307--t360 stays unused.

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
readonly hard_pool_size="${MLSTEP_HARD_POOL_SIZE:-1753600}"
readonly hard_pool="${MLSTEP_HARD_NEGATIVE_POOL:-$work_root/hard-negative-pools/15day-residual-stretch128-k4-v1/top-$hard_pool_size}"
readonly output_root="${MLSTEP_OUTPUT_ROOT:-$work_root/15day-residual-stretch128-k4-sampling-ordinal-v1}"
readonly python_bin=python

if [[ ! -f $repo_dir/pyproject.toml || ! -f $repo_dir/mlstep/student.py ]]; then
    echo "MLSTEP_REPO_DIR is not an mlstep source checkout: $repo_dir" >&2
    exit 66
fi
if [[ ! $hard_pool_size =~ ^[1-9][0-9]*$ ]]; then
    echo "MLSTEP_HARD_POOL_SIZE must be a positive integer: $hard_pool_size" >&2
    exit 64
fi

actual_commit=$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || true)
readonly actual_commit
if [[ -n ${MLSTEP_EXPECTED_COMMIT:-} && $actual_commit != "$MLSTEP_EXPECTED_COMMIT" ]]; then
    echo "Repository commit $actual_commit does not match MLSTEP_EXPECTED_COMMIT=$MLSTEP_EXPECTED_COMMIT" >&2
    exit 65
fi
readonly -a source_contract=(
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

declare -ar experiment_cells=(
    "c00-cyclic|cyclic|0|0"
    "c01-hard|hard|0.25|0"
    "c02-hard-ordinal|hard|0.25|1"
)
readonly task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
readonly n_cells=${#experiment_cells[@]}
readonly n_seeds=3
readonly n_tasks=$((n_cells * n_seeds))
if (( task_id < 0 || task_id >= n_tasks )); then
    echo "Array task must be in 0..$((n_tasks - 1))" >&2
    exit 64
fi

readonly cell_index=$((task_id % n_cells))
readonly seed=$((task_id / n_cells))
IFS='|' read -r cell_slug negative_sampling hard_fraction lambda_ordinal \
    <<<"${experiment_cells[$cell_index]}"
readonly cell_slug negative_sampling hard_fraction lambda_ordinal seed

# Everything below is deliberately fixed across the controlled cells.
readonly architecture=tabm-mini
readonly members=4
readonly preprocess=stretch128
readonly epochs=120
readonly patience=12
readonly batch_size=1024
readonly eval_batch_size=65536
readonly negative_ratio=256
readonly learning_rate=0.001
readonly weight_decay=0.0001
readonly dropout=0.05
readonly lambda_sev=0.1
readonly -a hidden_layers=(512 256)
readonly -a policy_lambdas=(0 0.25 0.5 1 2)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=$seed
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$repo_dir"

student_module=$(
    "$python_bin" - <<'PY'
from pathlib import Path
import mlstep.student

print(Path(mlstep.student.__file__).resolve(strict=True))
PY
)
readonly student_module
case "$student_module" in
    "$repo_dir"/*) ;;
    *)
        echo "Python resolved mlstep.student outside MLSTEP_REPO_DIR: $student_module" >&2
        exit 65
        ;;
esac

pool_argument=-
if [[ $negative_sampling == hard ]]; then
    pool_argument=$hard_pool
fi
readonly pool_argument

# Validate all cheap provenance and shapes before the trainer reads the 80-GiB
# feature cache into host memory.  Hard cells also independently validate that
# the shared pool contains only training class-zero rows.
"$python_bin" - "$cache_path" "$negative_sampling" "$pool_argument" "$hard_pool_size" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from mlstep.data import FEATURES, feature_names, load_fitted_preprocessor

cache = Path(sys.argv[1]).resolve(strict=True)
sampling = sys.argv[2]
pool_argument = sys.argv[3]
expected_pool_size = int(sys.argv[4])
required = (
    "SUCCESS",
    "metadata.json",
    "preprocessor_state.npz",
    "train_x.npy",
    "train_y.npy",
    "val_x.npy",
    "val_y.npy",
)
for name in required:
    path = cache / name
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty cache artifact: {path}")
if (cache / "SUCCESS").read_text(encoding="utf-8").strip() != "0":
    raise RuntimeError("Cache SUCCESS marker is invalid")
if list(cache.glob("*_teacher_*.npy")):
    raise RuntimeError("The controlled supervised experiment requires a teacherless cache")

metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
expected_features = feature_names(FEATURES)
if "residual[0]" not in expected_features or any(name.startswith("rhs[") for name in expected_features):
    raise RuntimeError("Runtime feature schema is not the corrected residual schema")
if metadata.get("features") != expected_features or metadata.get("n_features") != 266:
    raise RuntimeError("Cache feature schema differs from the corrected residual schema")
if metadata.get("preprocess") != "stretch128" or metadata.get("teacher") is not None:
    raise RuntimeError("Experiment cache is not teacherless corrected-residual Stretch128")
if metadata.get("train", {}).get("timesteps") != list(range(2, 253)):
    raise RuntimeError("Cache does not contain exact training timesteps t2--t252")
if metadata.get("val", {}).get("timesteps") != list(range(253, 307)):
    raise RuntimeError("Cache does not contain exact validation timesteps t253--t306")
split = metadata.get("split", {})
if split.get("protocol") != "explicit-inclusive-ranges-v1":
    raise RuntimeError("Cache does not record the explicit chronological split")
if split.get("unused_timesteps") != list(range(307, 361)):
    raise RuntimeError("Cache does not keep t307--t360 wholly unused")

expected = {
    "train": ((65_926_656, 266), [65_919_806, 5_735, 1_025, 89, 1]),
    "val": ((14_183_424, 266), [14_182_168, 1_045, 191, 20, 0]),
}
arrays = {}
for split_name, (feature_shape, class_counts) in expected.items():
    features = np.load(cache / f"{split_name}_x.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(cache / f"{split_name}_y.npy", mmap_mode="r", allow_pickle=False)
    arrays[split_name] = (features, labels)
    if features.shape != feature_shape or labels.shape != (feature_shape[0],):
        raise RuntimeError(f"{split_name} cache shape differs from the 15-day contract")
    if features.dtype != np.float32 or not np.issubdtype(labels.dtype, np.integer):
        raise RuntimeError(f"{split_name} cache dtype is invalid")
    split_metadata = metadata.get(split_name, {})
    if split_metadata.get("shape") != list(feature_shape) or split_metadata.get("class_counts") != class_counts:
        raise RuntimeError(f"{split_name} metadata differs from the corrected 15-day cache")

fitted = load_fitted_preprocessor(cache / "preprocessor_state.npz", "stretch128", features=FEATURES)
if fitted.n_bins != 128 or fitted.fit_sample_rows != 1_000_000:
    raise RuntimeError("Stretch128 preprocessor state differs from the frozen representation")

if sampling == "hard":
    pool = Path(pool_argument).resolve(strict=True)
    for name in ("SUCCESS", "metadata.json", "hard_negative_indices.npy"):
        path = pool / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty hard-pool artifact: {path}")
    if (pool / "SUCCESS").read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError("Hard-pool SUCCESS marker is invalid")
    pool_metadata = json.loads((pool / "metadata.json").read_text(encoding="utf-8"))
    if pool_metadata.get("format_version") != "hard-negative-pool-v1":
        raise RuntimeError("Unsupported hard-negative pool format")
    if pool_metadata.get("selection", {}).get("pool_size") != expected_pool_size:
        raise RuntimeError("Hard-negative pool size differs from this experiment contract")
    pool_cache = pool_metadata.get("cache", {})
    if Path(pool_cache.get("path", "")).resolve(strict=True) != cache:
        raise RuntimeError("Hard-negative pool was built from a different cache")
    if pool_cache.get("validation_arrays_accessed") is not False:
        raise RuntimeError("Hard-negative pool does not prove validation isolation")
    indices_path = pool / "hard_negative_indices.npy"
    indices = np.load(indices_path, mmap_mode="r", allow_pickle=False)
    hard_per_epoch = round(0.25 * (256 * 6_850))
    if indices.ndim != 1 or indices.dtype != np.int64 or len(indices) < hard_per_epoch:
        raise RuntimeError("Hard-negative pool is too small or has the wrong dtype")
    if np.any(indices < 0) or np.any(indices >= len(arrays["train"][1])):
        raise RuntimeError("Hard-negative indices are outside the training cache")
    if np.any(indices[1:] <= indices[:-1]):
        raise RuntimeError("Hard-negative indices are not sorted and unique")
    if np.any(arrays["train"][1][indices] != 0):
        raise RuntimeError("Hard-negative pool contains a non-zero training label")
    digest = hashlib.sha256()
    with indices_path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    artifact = pool_metadata.get("output_artifacts", {}).get("hard_negative_indices.npy", {})
    if artifact.get("bytes") != indices_path.stat().st_size or artifact.get("sha256") != digest.hexdigest():
        raise RuntimeError("Hard-negative index file differs from its recorded digest")
elif sampling != "cyclic" or pool_argument != "-":
    raise RuntimeError("Unexpected sampling/pool combination")

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Each experiment task requires exactly one allocated CUDA GPU")
PY

mkdir -p "$output_root/job-${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}/$cell_slug/seed-$seed"
readonly run_dir="$output_root/job-$SLURM_ARRAY_JOB_ID/$cell_slug/seed-$seed/task-$task_id"
if [[ -e $run_dir ]]; then
    echo "Refusing to reuse existing experiment run directory: $run_dir" >&2
    exit 73
fi
mkdir "$run_dir"

run_complete=0
on_exit() {
    status=$?
    if (( status == 0 && run_complete == 1 )); then
        printf '0\n' >"$run_dir/SUCCESS"
    else
        if (( status == 0 )); then
            status=71
        fi
        printf '%s\n' "$status" >"$run_dir/FAILED"
    fi
}
trap on_exit EXIT

command=(
    "$python_bin" -u -m mlstep.student
    --cache-path "$cache_path"
    --preprocess "$preprocess"
    --cache-residency host
    --architecture "$architecture"
    --members "$members"
    --hidden "${hidden_layers[@]}"
    --dropout "$dropout"
    --epochs "$epochs"
    --patience "$patience"
    --batch-size "$batch_size"
    --eval-batch-size "$eval_batch_size"
    --negative-ratio "$negative_ratio"
    --negative-sampling "$negative_sampling"
    --lr "$learning_rate"
    --weight-decay "$weight_decay"
    --alpha 0
    --beta 0
    --lambda-sev "$lambda_sev"
    --lambda-ordinal "$lambda_ordinal"
    --action-policy posterior-risk
    --policy-lambdas "${policy_lambdas[@]}"
    --device cuda
    --seed "$seed"
    --output-dir "$run_dir"
)
if [[ $negative_sampling == hard ]]; then
    command+=(
        --hard-negative-pool "$hard_pool"
        --hard-negative-fraction "$hard_fraction"
    )
fi
readonly -a command

{
    printf 'job_script_version=%s\n' 'student-15day-residual-sampling-ordinal-v1'
    printf 'slurm_array_job_id=%s\n' "$SLURM_ARRAY_JOB_ID"
    printf 'slurm_array_task_id=%s\n' "$task_id"
    printf 'cell_index=%s\n' "$cell_index"
    printf 'cell=%s\n' "${experiment_cells[$cell_index]}"
    printf 'seed=%s\n' "$seed"
    printf 'repository=%s\n' "$repo_dir"
    printf 'git_commit=%s\n' "${actual_commit:-unavailable}"
    printf 'student_module=%s\n' "$student_module"
    sha256sum \
        "$repo_dir/mlstep/student.py" \
        "$repo_dir/mlstep/tabm.py" \
        "$repo_dir/mlstep/data.py" \
        "$repo_dir/mlstep/sampling.py" \
        "$repo_dir/mlstep/losses.py" \
        "$repo_dir/mlstep/policy.py" \
        "$repo_dir/mlstep/evaluation.py" \
        "$cache_path/metadata.json" \
        "$cache_path/preprocessor_state.npz"
    if [[ $negative_sampling == hard ]]; then
        sha256sum "$hard_pool/metadata.json" "$hard_pool/hard_negative_indices.npy"
    fi
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
} >"$run_dir/launch.txt"

srun --ntasks=1 --kill-on-bad-exit=1 "${command[@]}"

mapfile -t checkpoint_paths < <(
    find "$run_dir" -maxdepth 1 -type f -name '*.pt' \
        ! -name '*_ap_best.pt' ! -name '*_fp97_best.pt' -print
)
mapfile -t report_paths < <(find "$run_dir" -maxdepth 1 -type f -name '*.json' -print)
if (( ${#checkpoint_paths[@]} != 1 || ${#report_paths[@]} != 1 )); then
    echo "Expected one canonical checkpoint and one JSON report in $run_dir" >&2
    exit 70
fi

"$python_bin" - \
    "${checkpoint_paths[0]}" "${report_paths[0]}" "$seed" "$negative_sampling" \
    "$hard_fraction" "$lambda_ordinal" "$pool_argument" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

from mlstep.policy import policy_from_descriptor
from mlstep.student import student_from_checkpoint

checkpoint_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
seed = int(sys.argv[3])
negative_sampling = sys.argv[4]
hard_fraction = float(sys.argv[5])
lambda_ordinal = float(sys.argv[6])
pool_argument = sys.argv[7]
policy_lambdas = [0.0, 0.25, 0.5, 1.0, 2.0]

report = json.loads(report_path.read_text(encoding="utf-8"))
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
restored = student_from_checkpoint(checkpoint)
if report.get("task") != "supervised" or report.get("seed") != seed:
    raise RuntimeError("Experiment report task or seed is incorrect")
model = report.get("model", {})
if model.get("architecture") != "tabm-mini" or model.get("k") != 4:
    raise RuntimeError("Experiment report is not a TabM-mini K4 model")
if model.get("preprocess") != "stretch128" or model.get("hidden") != [512, 256]:
    raise RuntimeError("Experiment model differs from the fixed c00 representation")
if model.get("cache_residency") != "host":
    raise RuntimeError("The 15-day feature cache must remain host-resident")

config = report.get("config", {})
expected_scalars = {
    "lr": 0.001,
    "weight_decay": 0.0001,
    "dropout": 0.05,
    "lambda_sev": 0.1,
    "lambda_ordinal": lambda_ordinal,
}
for key, expected in expected_scalars.items():
    if not math.isclose(float(config.get(key)), expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Experiment report has the wrong {key}")
if config.get("negative_ratio") != 256 or config.get("batch_size") != 1024:
    raise RuntimeError("Experiment negative budget or batch size differs from c00")
if config.get("negative_sampling") != negative_sampling:
    raise RuntimeError("Experiment report records the wrong negative sampler")
if config.get("action_policy") != "posterior-risk":
    raise RuntimeError("Experiment report does not record recall-independent actions")
if [float(value) for value in config.get("policy_lambdas", [])] != policy_lambdas:
    raise RuntimeError("Experiment report records the wrong policy frontier")
if negative_sampling == "hard":
    if not math.isclose(float(config.get("hard_negative_fraction")), hard_fraction, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Experiment report records the wrong hard-negative fraction")
    if Path(config.get("hard_negative_pool", "")).resolve(strict=True) != Path(pool_argument).resolve(strict=True):
        raise RuntimeError("Experiment report records the wrong hard-negative pool")
else:
    if config.get("hard_negative_pool") is not None or config.get("hard_negative_fraction") is not None:
        raise RuntimeError("Cyclic control unexpectedly records hard-negative settings")

if report.get("objective", {}).get("distill") is not False:
    raise RuntimeError("Controlled experiment unexpectedly used distillation")
if report.get("teacher_validation") is not None or report.get("student_teacher_agreement") is not None:
    raise RuntimeError("Teacher diagnostics unexpectedly appeared in a teacherless run")
protocol = report.get("training_protocol", {})
data_split = protocol.get("data_split", {})
if data_split.get("train_timesteps") != list(range(2, 253)):
    raise RuntimeError("Experiment training provenance has the wrong timesteps")
if data_split.get("validation_timesteps") != list(range(253, 307)):
    raise RuntimeError("Experiment validation provenance has the wrong timesteps")
sampling_protocol = protocol.get("negative_sampling", {})
expected_sampling_policy = {
    "cyclic": "cyclic-uniform-negative-v1",
    "hard": "stratified-hard-easy-cyclic-negative-v1",
}[negative_sampling]
if sampling_protocol.get("policy") != expected_sampling_policy:
    raise RuntimeError("Training protocol records the wrong negative-sampling policy")

# Recall may still be reported later as a descriptive metric, but it cannot
# appear as a requested target or threshold-selection policy in this run.
for forbidden in ("target_recall", "threshold", "threshold_policy", "epsilon_policy_diagnostic"):
    if forbidden in report:
        raise RuntimeError(f"Recall-controlled report field is forbidden: {forbidden}")
action_policy = report.get("action_policy", {})
if action_policy.get("version") != "posterior-risk-frontier-v1":
    raise RuntimeError("Report does not contain the recall-independent policy contract")
if action_policy.get("uses_recall_threshold") is not False:
    raise RuntimeError("Posterior policy unexpectedly uses a recall threshold")
if action_policy.get("validation_labels_required_at_inference") is not False:
    raise RuntimeError("Posterior policy unexpectedly requires validation labels at inference")
if [float(value) for value in action_policy.get("candidate_lambdas", [])] != policy_lambdas:
    raise RuntimeError("Action-policy contract records the wrong lambda frontier")
if not str(action_policy.get("selection", "")).startswith("not selected"):
    raise RuntimeError("Validation frontier was unexpectedly used to select a deployment policy")
default_policy = action_policy.get("default", {})
if default_policy.get("version") != "joint-map-v1" or default_policy.get("prediction_requires_labels") is not False:
    raise RuntimeError("Default action policy is not label-free joint MAP")
policy_from_descriptor(default_policy)
frontier = report.get("policy_frontier")
if not isinstance(frontier, list) or len(frontier) != len(policy_lambdas):
    raise RuntimeError("Posterior-risk policy frontier is missing or incomplete")
frontier_lambdas = [float(entry.get("policy", {}).get("lambda")) for entry in frontier]
if frontier_lambdas != policy_lambdas:
    raise RuntimeError("Posterior-risk policy frontier has the wrong lambdas")
for entry in frontier:
    descriptor = entry.get("policy", {})
    if descriptor.get("version") != "asymmetric-exact-overprediction-v1":
        raise RuntimeError("Policy frontier contains an unsupported policy")
    if descriptor.get("prediction_requires_labels") is not False:
        raise RuntimeError("Policy frontier unexpectedly requires validation labels at prediction time")
    policy_from_descriptor(descriptor)

if checkpoint.get("checkpoint_role") != "canonical" or checkpoint.get("seed") != seed:
    raise RuntimeError("Canonical checkpoint role or seed differs from the experiment")
if checkpoint.get("objective") != report.get("objective"):
    raise RuntimeError("Canonical checkpoint and report contracts differ")
if checkpoint.get("training_protocol") != protocol:
    raise RuntimeError("Checkpoint and report training protocols differ")
checkpoint_config = checkpoint.get("config", {})
if checkpoint_config.get("architecture") != "tabm-mini" or checkpoint_config.get("k") != 4:
    raise RuntimeError("Checkpoint architecture differs from the experiment contract")
if checkpoint_config.get("hidden") != [512, 256] or checkpoint_config.get("preprocess") != "stretch128":
    raise RuntimeError("Checkpoint configuration differs from fixed c00")
if checkpoint_config.get("negative_sampling") != negative_sampling:
    raise RuntimeError("Checkpoint records the wrong negative sampler")
expected_checkpoint_fraction = hard_fraction if negative_sampling == "hard" else None
if checkpoint_config.get("hard_negative_fraction") != expected_checkpoint_fraction:
    raise RuntimeError("Checkpoint records the wrong hard-negative fraction")
if not math.isclose(float(checkpoint_config.get("lambda_ordinal")), lambda_ordinal, rel_tol=0.0, abs_tol=1e-12):
    raise RuntimeError("Checkpoint records the wrong ordinal weight")
if checkpoint.get("action_policy") != action_policy:
    raise RuntimeError("Checkpoint and report action policies differ")
for forbidden in ("target_recall", "threshold", "threshold_policy"):
    if forbidden in checkpoint:
        raise RuntimeError(f"Recall-controlled checkpoint field is forbidden: {forbidden}")
if restored.architecture != "tabm-mini" or restored.k != 4 or list(restored.hidden_layers) != [512, 256]:
    raise RuntimeError("Checkpoint could not be reconstructed as fixed TabM-mini c00")
PY

run_complete=1
echo "Completed $cell_slug, seed $seed: $run_dir"
