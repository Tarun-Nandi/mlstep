import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import data_utils
from net import FCNN

SEED = 0


def pr_auc(scores, y_any):
    """Average precision (exact, full set); the metric of choice at 0.016% positives."""
    order = np.argsort(-scores)
    hits = y_any[order]
    precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float(precision[hits].mean()) if hits.any() else 0.0


def evaluate(model, x, y, class_weights, batch=65536):
    model.eval()
    with torch.no_grad(): # prevent gradient storage to reduce memory usage
        logits = torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])
    
    weighted_ce = nn.functional.cross_entropy(logits, y, weight=class_weights).item()
    probabilities = torch.softmax(logits, dim=1)
    # checking the prob that any halving is required
    score_hard = (1.0 - probabilities[:, 0]).numpy()
    y_np = y.numpy()
    pos = y_np > 0
    out = {"weighted_ce": weighted_ce,
           "pr_auc": pr_auc(score_hard, pos),
           "n_pos": int(pos.sum())}
    # checking among the difficult boxes how far away is the predicted halving count
    if pos.any():
        pred = logits.argmax(dim=1).numpy()
        # This is the mean absolute error on the positive boxes (halving >= 1)
        out["mae_on_pos"] = float(np.abs(pred[pos] - y_np[pos]).mean()) 
    return out

def class_statistics(targets):
    """ Returns the counts, empirical priors, and the inverse frequency class weights (lower freq -> higher weight)"""
    counts = torch.bincount(targets, minlength = data_utils.N_CLASSES).float()
    priors = counts / counts.sum()
    weights = counts.sum() / (data_utils.N_CLASSES * counts.clamp_min(1))

    return counts, priors, weights


def train(tr_x, tr_y, va_x, va_y, seed, epochs=30, batch_size=4096, lr=1e-3,
          weight_decay=1e-4, n_hidden=64, verbose=False):
    # controls model parameter initialisation
    torch.manual_seed(seed)
    # initialising the output bias in the FCNN
    counts, priors, weights = class_statistics(tr_y)
    model = FCNN(tr_x.shape[1], n_hidden=n_hidden, class_priors=priors.tolist())
    # use adamW with weight decay to discourage excessively large weights
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    """
    1)clear gradients from prev batch
    2)perform forward pass
    3)calculate weighted cross entropy
    4)backpropogate
    5)update model parameters
    """
    history, n = [], len(tr_x)
    for epoch in range(epochs):
        model.train()
        # create a reproducible random ordering of training samples
        generator = torch.Generator().manual_seed(seed + epoch)
        permutation  = torch.randperm(n, generator=generator)
        running_loss_sum = 0.0
        running_weight_sum = 0.0

        for i in range(0, n, batch_size):
            idx = permutation[i:i + batch_size]
            batch_x = tr_x[idx]
            batch_y = tr_y[idx]

            opt.zero_grad(set_to_none=True)
            logits = model(batch_x)
            batch_loss_sum = nn.functional.cross_entropy(logits, batch_y, weight=weights, reduction="sum")
            batch_weight_sum = weights[batch_y].sum()

            loss = batch_loss_sum / batch_weight_sum
            loss.backward()
            opt.step()

            running_loss_sum += batch_loss_sum.detach().item()
            running_weight_sum += batch_weight_sum.item()

        train_weighted_ce = (running_loss_sum / running_weight_sum)
        # after evry training epoch, entire validation set is evaluated
        val_metrics = evaluate(model, va_x, va_y, class_weights=weights)

        history.append({
            "epoch": epoch +1,
            "train_weighted_ce": train_weighted_ce,
            **val_metrics,
        })
        if verbose:
            print(
                f"  seed {seed} "
                f"epoch {epoch + 1:>3}  "
                f"train CE {train_weighted_ce:.6f}  "
                f"val CE {val_metrics['weighted_ce']:.6f}  "
                f"val AP {val_metrics['pr_auc']:.6f}"
            )
    return history, model


def run_experiment(config, seeds=5):
    """
    Performs one complete experiments for one configuration over several seeds
    A configuration might be for example: 
    {
        "split": "time",
        "include_t1": False,
    }   
    """
    splits = data_utils.prepare_splits(split=config["split"],include_t1=config["include_t1"])
    tr_x, tr_y = map(torch.from_numpy, map(np.ascontiguousarray, splits["train"]))
    va_x, va_y = map(torch.from_numpy, map(np.ascontiguousarray, splits["val"]))
    print(f"config {config} | train {tuple(tr_x.shape)} "
          f"({int((tr_y > 0).sum())} pos) | val {tuple(va_x.shape)} "
          f"({int((va_y > 0).sum())} pos)")

    results = []
    for s in range(seeds):
        history, _ = train(tr_x, tr_y, va_x, va_y, seed=SEED + s)
        results.append(history[-1])
        print(f"  seed {s}: PR-AUC {history[-1]['pr_auc']:.4f}  "
              f"MAE|pos {history[-1].get('mae_on_pos', float('nan')):.3f}")

    aucs = np.array([r["pr_auc"] for r in results])
    summary = {"pr_auc_mean": float(aucs.mean()), "pr_auc_std": float(aucs.std()),
               "n_seeds": seeds}
    print(f"  => PR-AUC {aucs.mean():.4f} +/- {aucs.std():.4f}")
    # produces experiment tag e.g: "time_t1_0722-104233"
    tag = (f"{config['split']}"
           f"{'_t1' if config['include_t1'] else ''}"
           f"_{time.strftime('%m%d-%H%M%S')}")
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/{tag}.json").write_text(json.dumps(
        {"config": config, "git": _git_commit(), "seeds": results,
         "summary": summary, "time": time.strftime("%Y-%m-%d %H:%M")}, indent=2))
    print(f"  logged runs/{tag}.json")
    return summary


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def run_checks(tr_x, tr_y):
    counts, priors, weights = class_statistics(tr_y)
    torch.manual_seed(SEED)

    # 1. Verify prior-bias initialization and initial weighted loss.
    model = FCNN(tr_x.shape[1],class_priors=priors.tolist(),)
    # Create a small stratified subset so that every class present in the
    # training data participates in the loss check.
    check_indices = []
    for class_index in range(data_utils.N_CLASSES):
        class_indices = torch.nonzero(tr_y == class_index,as_tuple=False,).squeeze(1)
        if class_indices.numel() > 0:
            check_indices.append(class_indices[:128])

    check_indices = torch.cat(check_indices)
    check_x = tr_x[check_indices]
    check_y = tr_y[check_indices]

    # The bias itself should encode the empirical class-prior distribution.
    expected_probabilities = priors.clamp_min(1e-12)
    expected_probabilities = (expected_probabilities/ expected_probabilities.sum())
    bias_probabilities = torch.softmax(model.head.bias,dim=0,)
    assert torch.allclose(bias_probabilities,expected_probabilities,atol=1e-6,)
    
    # We deliberately retain the default random head weights. This allows
    # gradients to reach the hidden layer from the first training batch.
    assert model.head.weight.abs().sum() > 0
    print("[check] prior bias initialization  OK")

    # Verify the weighted cross-entropy calculation for the bias-only
    # prediction. This is the analytically predictable special case.
    bias_only_logits = model.head.bias.unsqueeze(0).expand(len(check_y),-1,)
    measured_bias_only_loss = nn.functional.cross_entropy(bias_only_logits,check_y,weight=weights,)
    sample_weights = weights[check_y]

    expected_bias_only_loss = -(sample_weights* expected_probabilities.log()[check_y]).sum() / sample_weights.sum()
    assert torch.allclose(measured_bias_only_loss,expected_bias_only_loss,atol=1e-6,)
    print(
        f"[check] bias-only weighted loss  "
        f"{measured_bias_only_loss.item():.6f} = "
        f"expected {expected_bias_only_loss.item():.6f}  OK"
    )

    # The real network also contains randomly initialized output weights,
    # so its complete initial loss is input-dependent and need not equal
    # the bias-only analytical loss. It should simply be finite.
    with torch.no_grad():
        actual_initial_logits = model(check_x)
    actual_initial_loss = nn.functional.cross_entropy(actual_initial_logits,check_y,weight=weights,)
    assert torch.isfinite(actual_initial_loss)
    print(
        f"[check] actual initial weighted loss  "
        f"{actual_initial_loss.item():.6f}  finite  OK"
    )
    # 2. batch independence: gradient of output i touches only input i.
    model = FCNN(tr_x.shape[1])
    xi = torch.randn(4, tr_x.shape[1], requires_grad=True)
    model(xi)[2].sum().backward()
    g = xi.grad.abs().sum(dim=1)
    assert g[2] > 0 and g[[0, 1, 3]].max() == 0
    print("[check] batch independence  OK")

    # 3. inputs just before the net: the 'source of truth' view.
    bad = int(np.isnan(tr_x.numpy()).sum() + np.isinf(tr_x.numpy()).sum())
    mu, sd = tr_x.numpy().mean(0), tr_x.numpy().std(0)
    print(f"[check] inputs  NaN/Inf {bad}  |mean|max {np.abs(mu).max():.3f}  "
          f"std range [{sd.min():.2f}, {sd.max():.2f}]  "
          f"range [{tr_x.min():.1f}, {tr_x.max():.1f}]")
    assert bad == 0 and np.abs(mu).max() < 1e-3

    # 4. overfit a tiny set -> loss ~0 AND perfect accuracy, asserted.
    pos = torch.nonzero(tr_y > 0).squeeze(1)
    neg = torch.nonzero(tr_y == 0).squeeze(1)[:512]
    xs, ys = tr_x[torch.cat([pos, neg])], tr_y[torch.cat([pos, neg])]
    model = FCNN(tr_x.shape[1], n_hidden=256)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(2000):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(xs), ys)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final_logits = model(xs)
        final_loss = nn.functional.cross_entropy(final_logits, ys,).item()
        accuracy = (final_logits.argmax(dim=1) == ys).float().mean().item()
    assert final_loss < 0.01, f"tiny-set overfit failed: loss={final_loss:.6f}"
    assert accuracy == 1.0, f"tiny-set overfit failed: accuracy={accuracy:.6f}"
    print(f"[check] overfit tiny  loss {final_loss:.6f}  acc {accuracy:.3f}  OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--split", choices=["time", "random"], default="time")
    ap.add_argument("--include-t1", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    config = {"split": args.split, "include_t1": args.include_t1}

    if args.check:
        splits = data_utils.prepare_splits(**config)
        tr_x, tr_y = map(torch.from_numpy, splits["train"])
        run_checks(tr_x, tr_y)

    if args.train:
        run_experiment(config, seeds=args.seeds)


if __name__ == "__main__":
    main()
