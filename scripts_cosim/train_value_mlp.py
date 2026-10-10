#!/usr/bin/env python3
"""cost_to_go_v1 rung (b): the pointwise MLP V (DeepSets over the post-plan replicas), trained on the paired H=5 advantage.

Run through run_experiment.py (trainer "value_mlp"); --cache-dir holds `states.jsonl` from cost_to_go_mlpv_data.py build.

  V(state, plan) = rho( mean_r phi(x_r)  ++  sum_r phi(x_r) ),   phi: F -> 64 -> 64, rho: 128 -> 64 -> 1 (ReLU)
  loss: per state, mean over the non-policy plans k of Huber_1s( V(k) - V(policy) - (Q5(k) - Q5(policy)) ); mean over states.

Split (fixed by the design, 2026-10-11): sort the distinct topology ids of the cache; positions 0, 5, 10, ... are SELECTION, the
rest TRAIN. Selection: the epoch with the best pooled Spearman of dV vs dQ on SELECTION. lambda: argmin over the plans of
dS + lambda*dV, on the hand V's grid plus lambda = inf (V alone), maximising Q5 saved on SELECTION. --no-inflight zeroes the
in-flight feature (the declared second variant). The checkpoint carries a contract sidecar (features, variant, epoch, lambda).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_to_go_mlpv_data import FEATURES  # noqa: E402

INFLT = FEATURES.index("inflt")


def lam_grid():
    return [0.0] + [10 ** (e / 4.0) for e in range(-16, 17)] + [math.inf]


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return sxy / (sx * sy) if sx and sy else float("nan")


class ValueMLP(nn.Module):
    def __init__(self, f, h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, mask):
        """x: (P, R, F) plans x replicas x features, mask: (P, R) -> (P,) V per plan."""
        z = self.phi(x) * mask.unsqueeze(-1)
        s = z.sum(1)
        m = s / mask.sum(1, keepdim=True).clamp(min=1.0)
        return self.rho(torch.cat([m, s], -1)).squeeze(-1)


def tensors(state, no_inflight):
    plans = state["plans"]
    R = max(len(p["x"]) for p in plans)
    x = torch.zeros(len(plans), R, len(FEATURES))
    mask = torch.zeros(len(plans), R)
    for i, p in enumerate(plans):
        if p["x"]:
            x[i, : len(p["x"])] = torch.tensor(p["x"], dtype=torch.float32)
            mask[i, : len(p["x"])] = 1.0
    if no_inflight:
        x[..., INFLT] = 0.0
    dq = torch.tensor([p["q"] - plans[0]["q"] for p in plans], dtype=torch.float32)
    ds = [p["s"] - plans[0]["s"] for p in plans]
    return x, mask, dq, ds


def predict(model, data):
    model.eval()
    out = []
    with torch.no_grad():
        for x, mask, dq, ds in data:
            v = model(x, mask)
            out.append(((v - v[0]).tolist(), dq.tolist(), ds))
    return out


def pooled_spearman(pred):
    xs, ys = [], []
    for dv, dq, _ds in pred:
        xs += dv[1:]
        ys += dq[1:]
    return spearman(xs, ys)


def choose(dv, ds, lam):
    if math.isinf(lam):
        key = lambda k: (dv[k], k != 0, k)
    else:
        key = lambda k: (ds[k] + lam * dv[k], k != 0, k)
    return min(range(len(dv)), key=key)


def saved(pred, lam):
    got = head = 0.0
    hit = 0
    for dv, dq, ds in pred:
        k = choose(dv, ds, lam)
        b = min(range(len(dq)), key=lambda i: (dq[i], i != 0, i))
        got += -dq[k]
        head += -dq[b]
        hit += k == b
    return got, head, hit / max(1, len(pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-states", type=int, default=64)
    ap.add_argument("--no-inflight", action="store_true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--wandb-project", default="herosim-cost-to-go")
    a = ap.parse_args()

    seed = int(os.environ.get("VALUE_MLP_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    states = [json.loads(l) for l in open(Path(a.cache_dir) / "states.jsonl")]
    topos = sorted({s["topo"] for s in states})
    sel_t = set(topos[0::5])
    train = [s for s in states if s["topo"] not in sel_t]
    sel = [s for s in states if s["topo"] in sel_t]
    tr_data = [tensors(s, a.no_inflight) for s in train]
    sel_data = [tensors(s, a.no_inflight) for s in sel]
    print(f"seed {seed}; {len(topos)} topologies -> SELECTION {sorted(sel_t)} ({len(sel)} states), TRAIN {len(train)} states; "
          f"in-flight feature {'OFF' if a.no_inflight else 'ON (b1)'}")

    import wandb

    run = wandb.init(project=a.wandb_project, name=os.environ.get("WANDB_RUN_NAME"),
                     tags=[t for t in os.environ.get("WANDB_TAGS", "").split(",") if t],
                     config={**vars(a), "seed": seed, "features": FEATURES, "selection_topos": sorted(sel_t)})
    model = ValueMLP(len(FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    huber = nn.HuberLoss(delta=1.0)
    best = (-2.0, -1, None)
    order = list(range(len(tr_data)))
    rng = random.Random(seed)
    for ep in range(a.epochs):
        model.train()
        rng.shuffle(order)
        tot = 0.0
        for b in range(0, len(order), a.batch_states):
            opt.zero_grad()
            loss = 0.0
            for i in order[b: b + a.batch_states]:
                x, mask, dq, _ = tr_data[i]
                v = model(x, mask)
                loss = loss + huber(v[1:] - v[0], dq[1:])
            loss = loss / len(order[b: b + a.batch_states])
            loss.backward()
            opt.step()
            tot += float(loss) * len(order[b: b + a.batch_states])
        sp = pooled_spearman(predict(model, sel_data))
        wandb.log({"epoch": ep, "train_huber": tot / len(order), "sel_pooled_spearman": sp})
        if sp > best[0] + 1e-12:
            best = (sp, ep, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    pred = predict(model, sel_data)
    lam_best, lam_got = None, None
    for lam in lam_grid():
        got, head, _hit = saved(pred, lam)
        if lam_got is None or got > lam_got + 1e-12:
            lam_best, lam_got = lam, got
    got, head, hit = saved(pred, lam_best)
    got0, _h, hit0 = saved(pred, 0.0)
    summary = {"selected_epoch": best[1], "sel_pooled_spearman": best[0], "lambda": lam_best, "sel_saved_s": got, "sel_headroom_s": head,
               "sel_saved_share": got / head if head > 0 else float("nan"), "sel_hit": hit, "sel_hit_lambda0": hit0, "sel_saved_lambda0_s": got0,
               "n_train_states": len(train), "n_sel_states": len(sel)}
    wandb.summary.update({k: (v if not (isinstance(v, float) and math.isinf(v)) else "inf") for k, v in summary.items()})
    print(json.dumps(summary))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    contract = {"model": "value_mlp_deepsets_v1", "features": list(FEATURES), "no_inflight": a.no_inflight, "seed": seed,
                "hidden": 64, "pool": "mean+sum", "target": "Q5(plan) - Q5(cd_exactS plan), paired", "H": 5.0,
                "lambda": "inf" if math.isinf(lam_best) else lam_best, "selection_topos": sorted(sel_t), **{k: v for k, v in summary.items() if k != "lambda"},
                "cache_dir": str(a.cache_dir), "wandb_run": run.name}
    out.with_suffix(".contract.json").write_text(json.dumps(contract, indent=1))
    run.finish()


if __name__ == "__main__":
    main()
