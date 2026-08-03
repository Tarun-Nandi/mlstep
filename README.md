# Timestep prediciton using machine learning

This repository is deisgned to estimate the number of times a timestep length should be halved in order for a given solver to attin convergence. For solvers with adaptive timestepping functionality, this allows the user to predict the timesteo length that can be used at each step in the timestepping scheme, If this predeiciton is accurate then it avoids the need for tial and error approaches, whereby sucessively halved timestep lengths are tried until the solver convergence.

## Installation

We strongly advise that users of `mlstep` create a Python virtual environment
before installing it. Doing so avoids polluting the system Python environment.
See https://docs.python.org/3/library/venv.html for details on how to do this.

For a basic install the `mlstep` module, activate your virtual environment,
clone the repository, and then run
```sh
cd mlstep
pip install -e .
```

For a development install, some further steps are recommended:
```sh
cd mlstep
# Install optional dev dependencies
pip install -e .[dev]

# Configure pre-commit hooks
pre-commit install
```

In order to run the models directly, use these:

```sh
python -m mlstep.net
python -m mlstep.net --multiclass
python -m mlstep.net --multiclass --dashboard
python -m mlstep.xgb
python -m mlstep.xgb --multiclass
python -m mlstep.xgb --multiclass --dashboard
```

Both scripts train the binary hard-box detector by default. Add `--multiclass`
to train exact-halving outputs. The optional FCNN dashboard updates during
training and saves its final plot under `plots/`. It includes binary and
multiclass true-versus-predicted counts and confusion matrices.

FCNN preprocessing is fitted only on the training timesteps and produces
`float32` features. XGBoost uses the raw features because trees do not need the
same scaling. The loader currently builds the full design matrix in memory; a
substantially larger dataset will need chunked or memory-mapped loading.

## Development split

Both training scripts use exactly the same chronological split:

```text
training:   t2-t18
validation: t19-t21
test:       t22-t24
excluded:   t1 (spin-up)
```

The validation period is used to choose the FCNN epoch or XGBoost boosting
round and the operating threshold. The test period is not loaded or evaluated
by either training command yet is reserved for later evaluation.



## Metrics

Average precision (AP) for detecting any halving is the primary metric because
positive prevalence is tiny. Reports also include:

- precision and recall in the top 0.1% of scores;
- precision, recall, F0.5, and MCC at the validation-selected threshold;
- multiclass confusion and per-class metrics; and
- exactness, MAE, underprediction, and overprediction on hard boxes.

Threshold-dependent metrics are optimistic because the same validation period
selects the threshold. They remain development diagnostics until the final
frozen model and threshold are evaluated once on t22–t24.

## Outputs and source layout

Runs write JSON summaries to `runs/`, model files alongside them, and optional
dashboards to `plots/`. With the commands above these directories are
`mlstep/runs/` and `mlstep/plots/` relative to the repository root.

```text
data.py              loading, labels, split, and FCNN preprocessing
net.py               FCNN training and evaluation
evaluation.py        shared detection and exact-halving metrics
xgb.py               binary and multiclass XGBoost training
dashboard.py         optional FCNN + XGBoost live plots

