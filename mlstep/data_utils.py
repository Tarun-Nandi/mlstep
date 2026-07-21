"""Loads the data and preprocesses it into input features for the network.

(Data) experiments:
- split = "time" | "random"
- include_t1: train with/without the first (spin-up) timestep
- ablation studies over variables (edit FEATURES)
"""
from pathlib import Path
import numpy as np
import xarray as xr


# Constants (decided in the data-analysis notebook)
N_CLASSES = 5          # hard-coded: t2-9 alone contains no class-4 sample
TRAIN_STEPS = (2, 3, 4, 5, 6, 7)
VAL_STEPS = (8,)
TEST_STEPS = (9,)      # touch once, at the very end
VAL_FRACTION = 0.15    # only used by split="random"

DATA = Path("data")
GRID = (38, 72, 96)    # (z, y, x); z=0 is the surface

# name -> (number of slices, transform); column order = this order
FEATURES = {
    "temp":         (1,  "standardise"),
    "pres":         (1,  "standardise"),
    "water_vapour": (1,  "log1p"),
    "cloud_frac":   (1,  "standardise"),
    "qcl":          (1,  "log1p"),
    "cell_volume":  (1,  "standardise"),
    "stratflag":    (1,  "standardise"),
    "tracer":       (83, "log1p"),
    "rhs":          (83, "arcsinh"),
    "photol_rates": (60, "log1p"),
    "wetrt":        (34, "log1p"),
}

APPLY_TRANSFORM = {
    "standardise": lambda block, s: block,
    "arcsinh":     lambda block, s: np.arcsinh(block / s),
    "log1p":       lambda block, s: np.log1p(np.maximum(block, 0.0) / s),
}


def load_variable(var, timestep):
    """One variable at one timestep, as a plain numpy array."""
    return xr.open_dataset(DATA / f"{var}_{timestep}.nc")[var].values


def build_features(timesteps):
    """Design matrix (n_boxes, n_features) and integer labels 0-4.

    One row per (box, timestep); column order follows FEATURES.
    """
    boxes, targets = [], []
    for t in timesteps:
        cols = []
        for name, (n, _) in FEATURES.items():
            cols.append(load_variable(name, t).reshape(n, -1))
        boxes.append(np.concatenate(cols, axis=0).T)
        targets.append(np.log2(load_variable("ncsteps", t)).astype(np.int64).ravel())

    x = np.concatenate(boxes, axis=0)
    y = np.concatenate(targets)
    assert x.shape[1] == sum(n for n, _ in FEATURES.values())
    assert 0 <= y.min() and y.max() < N_CLASSES
    return x, y


def fit_preprocessing(train_x):
    """All preprocessing statistics, from TRAINING data only:
    per-column transform scales and post-transform mean/std."""
    counts = np.count_nonzero(train_x, axis=0)
    sums = np.abs(train_x).sum(axis=0, dtype=np.float64)
    scales = np.where(counts > 0, sums / np.maximum(counts, 1), 1.0)
    transformed = transform(train_x, scales)
    mu = transformed.mean(axis=0, keepdims=True)
    sd = transformed.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return {"scales": scales, "mu": mu, "sd": sd}


def transform(x, scales):
    """Apply each variable's transform to its column block."""
    out = x.astype(np.float32).copy()
    j = 0
    for name, (n, tf) in FEATURES.items():
        out[:, j:j + n] = APPLY_TRANSFORM[tf](out[:, j:j + n], scales[j:j + n])
        j += n
    return out


def apply_preprocessing(x, stats):
    """Transform then standardise with frozen train-fitted stats."""
    return ((transform(x, stats["scales"]) - stats["mu"])
            / stats["sd"]).astype(np.float32)


def prepare_splits(split="time", include_t1=False, seed=0):
    """Return {"train": (x, y), "val": (x, y), "stats"}.

    split="time" is the honest evaluation; split="random" exists to measure
    the leakage. include_t1 adds t1 to the TRAINING pool only, so every
    experiment arm is validated on the same timestep.
    """
    train_steps = (1,) + TRAIN_STEPS if include_t1 else TRAIN_STEPS

    if split == "time":
        tr_x, tr_y = build_features(train_steps)
        va_x, va_y = build_features(VAL_STEPS)
    elif split == "random":
        pool_x, pool_y = build_features(train_steps + VAL_STEPS)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(pool_y))
        n_val = int(len(idx) * VAL_FRACTION)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        tr_x, tr_y = pool_x[tr_i], pool_y[tr_i]
        va_x, va_y = pool_x[va_i], pool_y[va_i]
    else:
        raise ValueError(f"unknown split: {split!r}")

    stats = fit_preprocessing(tr_x)      # train only — never refit on val/test
    print(stats)
    return {"train": (apply_preprocessing(tr_x, stats), tr_y),
            "val": (apply_preprocessing(va_x, stats), va_y),
            "stats": stats}

prepare_splits()
