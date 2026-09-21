#!/usr/bin/env python3
"""rollout_imitation_v1 -- train and PERSIST the served scorer for the live gate.

The offline train-screen (`rollout_imitation_v1_train_screen.py`) trains in-process and saves
nothing; per CLAUDE.md a comparison that keeps no weights cannot be gated. This script trains the
SAME small MLP over the rule's own score terms [drain, cold, exec, latency, exchange] on the
rollout label, and writes:

  <out>.pt            the state_dict of nn.Sequential(Linear(5,16),ReLU,Linear(16,16),ReLU,Linear(16,1))
  <out>.contract.json feature_order + feature_mean/sd (the serving normalisation), hidden, corpus
                      provenance -- a checkpoint without a contract is not evidence.

`PeerGreedyLearnedNetworkScheduler` (registry name peer_greedy_learned_network) loads exactly this
pair via HEROSIM_ROLLOUT_SCORER and scores each candidate per arrival, no rollout at serving time.

Every training run logs to W&B (CLAUDE.md rule 5); WANDB_MODE=offline is fine on the cluster.

Usage (datalab, micromamba gnn, PYTHONPATH=.):
  WANDB_MODE=offline python3 scripts_cosim/rollout_imitation_v1_train_scorer.py \
      --labels-dir results/rollout_imitation_v1/labels results/rollout_imitation_v1/labels_f1000 \
      --out models/rollout_imitation_v1/scorer
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics as st
import subprocess
from typing import List, Optional, Sequence

FEATURE_ORDER = ("drain", "cold", "exec", "latency", "exchange")
FEATURE_COLS = (4, 5, 6, 7, 8)  # columns of a candidate row that hold FEATURE_ORDER
HORIZON = "100"
HIDDEN = 16


def _load(dirs) -> List[dict]:
    out = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "s*.json"))):
            rec = json.load(open(f))
            seed = rec["seed"]
            for dec in rec["decisions"]:
                if "candidates" not in dec or dec.get("rule_idx", -1) < 0:
                    continue
                costs = dec["costs"].get(HORIZON) if isinstance(dec["costs"], dict) else None
                if costs is None or len(costs) != len(dec["candidates"]):
                    continue
                label = int(dec["argmins"][HORIZON] if HORIZON in dec["argmins"] else dec["argmins"][100])
                out.append({"seed": seed,
                            "feats": [[float(c[i]) for i in FEATURE_COLS] for c in dec["candidates"]],
                            "label": label, "rule_idx": int(dec["rule_idx"]),
                            "costs": [float(x) for x in costs]})
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="path prefix; writes <out>.pt and <out>.contract.json")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    data = _load(a.labels_dir)
    if not data:
        print("no decisions found"); return 1
    ncol = len(FEATURE_ORDER)

    # serving normalisation: mean/sd over every candidate row in the corpus
    allf = [f for d in data for f in d["feats"]]
    mean = [st.fmean(r[j] for r in allf) for j in range(ncol)]
    sd = [(st.pstdev(r[j] for r in allf) or 1.0) for j in range(ncol)]

    def norm(row):
        return [(row[j] - mean[j]) / sd[j] for j in range(ncol)]

    scorer = nn.Sequential(nn.Linear(ncol, HIDDEN), nn.ReLU(),
                           nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
    opt = torch.optim.Adam(scorer.parameters(), lr=a.lr)

    try:
        import wandb
        run = wandb.init(project="rollout_imitation_v1", name=f"scorer-s{a.seed}",
                         config={"epochs": a.epochs, "lr": a.lr, "hidden": HIDDEN,
                                 "n_decisions": len(data), "features": list(FEATURE_ORDER),
                                 "git_commit": _git_commit()})
    except Exception as exc:  # W&B must be logged, but never block the checkpoint on a transient
        print(f"WARNING: wandb init failed ({exc}); continuing (set WANDB_MODE=offline)")
        run = None

    def full_loss():
        loss = 0.0
        for d in data:
            x = torch.tensor([norm(f) for f in d["feats"]], dtype=torch.float32)
            logits = -scorer(x).squeeze(1)
            loss = loss + nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([d["label"]]))
        return loss / len(data)

    for ep in range(a.epochs):
        opt.zero_grad()
        loss = full_loss()
        loss.backward()
        opt.step()
        if run is not None and (ep % 10 == 0 or ep == a.epochs - 1):
            run.log({"epoch": ep, "train_ce": float(loss.item())})

    # train-fit report (generalisation is the train_screen's held-out job, not this one's)
    def pick(d):
        x = torch.tensor([norm(f) for f in d["feats"]], dtype=torch.float32)
        with torch.no_grad():
            return int(torch.argmin(scorer(x).squeeze(1)).item())
    acc = sum(1 for d in data if pick(d) == d["label"]) / len(data)
    rule_acc = sum(1 for d in data if d["rule_idx"] == d["label"]) / len(data)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pt_path = a.out + ".pt"
    contract_path = a.out + ".contract.json"
    torch.save(scorer.state_dict(), pt_path)
    contract = {
        "lineage": "rollout_imitation_v1",
        "arch": "mlp", "hidden": HIDDEN, "feature_order": list(FEATURE_ORDER),
        "feature_mean": mean, "feature_sd": sd, "horizon": int(HORIZON),
        "label": "argmin realised group-local rollout cost at N=100",
        "n_decisions": len(data), "labels_dir": list(a.labels_dir),
        "train_fit_acc": acc, "rule_fit_acc": rule_acc,
        "git_commit": _git_commit(), "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
    }
    json.dump(contract, open(contract_path, "w"), indent=1)

    print("=== rollout_imitation_v1 scorer trained & persisted ===")
    print(f"  decisions: {len(data)}   train-fit acc: model {acc:.3f}  vs rule {rule_acc:.3f}")
    print(f"  wrote {pt_path}")
    print(f"  wrote {contract_path}")
    if run is not None:
        run.summary["train_fit_acc"] = acc
        run.summary["rule_fit_acc"] = rule_acc
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
