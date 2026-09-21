#!/usr/bin/env python3
"""rollout_imitation_v1 -- offline train-screen: can a learned scorer on the rule's OWN features
learn the rollout label and beat the rule on realised group-local cost, generalising across
held-out topologies? This ORDERS the live gate (CLAUDE.md rule 6: an offline screen never closes a
lineage); a positive screen motivates wiring the scorer into a scheduler for the live gate.

Each decision is (candidate feature rows, rollout label). Candidate features are the rule's own
score terms [drain, cold, exec, latency, exchange]; the rule itself scores drain+cold+exec+lat+exch
and takes the argmin. Label = argmin of the realised N=100 rollout cost. We train a small MLP that
scores each candidate from its features and picks the argmin score, split by topology (no topology
in both train and test), and compare on the held-out decisions:

  - accuracy: fraction where the model's pick == the rollout label (chance = 1/mean_n_candidates)
  - realised cost following the model vs the rule vs the oracle (always the label), from costs[100]
  - the fraction of the oracle-over-rule gain the model captures.

Usage (datalab, micromamba gnn, PYTHONPATH=.):
  python3 scripts_cosim/rollout_imitation_v1_train_screen.py \
      --labels-dir results/rollout_imitation_v1/labels results/rollout_imitation_v1/labels_f1000 \
      --out results/rollout_imitation_v1/train_screen.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from statistics import median
from typing import List, Optional, Sequence

FEATURE_COLS = (4, 5, 6, 7, 8)  # drain, cold, exec, latency, exchange in a candidate row
HORIZON = "100"


def _load(dirs) -> List[dict]:
    out = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "s*.json"))):
            rec = json.load(open(f))
            seed = rec["seed"]
            for dec in rec["decisions"]:
                # need features, a label, and the horizon costs
                if "candidates" not in dec or dec.get("rule_idx", -1) < 0:
                    continue
                costs = dec["costs"].get(HORIZON) if isinstance(dec["costs"], dict) else None
                if costs is None or len(costs) != len(dec["candidates"]):
                    continue
                out.append({"seed": seed,
                            "feats": [[float(c[i]) for i in FEATURE_COLS] for c in dec["candidates"]],
                            "label": int(dec["argmins"][HORIZON] if HORIZON in dec["argmins"] else dec["argmins"][100]),
                            "rule_idx": int(dec["rule_idx"]), "costs": [float(x) for x in costs]})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    a = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    data = _load(a.labels_dir)
    if not data:
        print("no decisions found"); return 1
    seeds = sorted({d["seed"] for d in data})
    random.shuffle(seeds)
    n_ho = max(1, int(len(seeds) * a.holdout_frac))
    ho_seeds = set(seeds[:n_ho])
    train = [d for d in data if d["seed"] not in ho_seeds]
    test = [d for d in data if d["seed"] in ho_seeds]

    # standardise features on train
    import statistics as st
    allf = [f for d in train for f in d["feats"]]
    ncol = len(FEATURE_COLS)
    mean = [st.fmean(r[j] for r in allf) for j in range(ncol)]
    sd = [(st.pstdev(r[j] for r in allf) or 1.0) for j in range(ncol)]

    def norm(row):
        return [(row[j] - mean[j]) / sd[j] for j in range(ncol)]

    scorer = nn.Sequential(nn.Linear(ncol, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(scorer.parameters(), lr=1e-2)

    def batch_loss(rows):
        loss = 0.0
        for d in rows:
            x = torch.tensor([norm(f) for f in d["feats"]], dtype=torch.float32)
            logits = -scorer(x).squeeze(1)  # lower score = better -> higher logit
            loss = loss + nn.functional.cross_entropy(logits.unsqueeze(0),
                                                      torch.tensor([d["label"]]))
        return loss / len(rows)

    for ep in range(a.epochs):
        opt.zero_grad()
        loss = batch_loss(train)
        loss.backward()
        opt.step()

    # evaluate on held-out topologies
    def pick(d):
        x = torch.tensor([norm(f) for f in d["feats"]], dtype=torch.float32)
        with torch.no_grad():
            return int(torch.argmin(scorer(x).squeeze(1)).item())

    n = len(test)
    acc = sum(1 for d in test if pick(d) == d["label"]) / n
    rule_acc = sum(1 for d in test if d["rule_idx"] == d["label"]) / n
    model_cost = sum(d["costs"][pick(d)] for d in test)
    rule_cost = sum(d["costs"][d["rule_idx"]] for d in test)
    oracle_cost = sum(d["costs"][d["label"]] for d in test)
    # per-decision realised %Δ (model vs rule), median
    deltas = [100.0 * (d["costs"][pick(d)] - d["costs"][d["rule_idx"]]) / d["costs"][d["rule_idx"]]
              for d in test if d["costs"][d["rule_idx"]] > 0]
    gain_captured = ((rule_cost - model_cost) / (rule_cost - oracle_cost)) if rule_cost > oracle_cost else None

    R = {"n_train": len(train), "n_test": n, "n_topologies": len(seeds), "n_holdout_topos": len(ho_seeds),
         "model_acc": acc, "rule_acc": rule_acc, "chance_acc": 0.5,
         "model_cost": model_cost, "rule_cost": rule_cost, "oracle_cost": oracle_cost,
         "median_realised_delta_pct": median(deltas) if deltas else None,
         "oracle_gain_captured_frac": gain_captured}

    print("=== rollout_imitation_v1 train-screen (offline; orders the live gate) ===")
    print(f"  decisions: {len(train)} train / {n} test   topologies: {len(seeds)} ({len(ho_seeds)} held out)")
    print(f"  accuracy predicting the rollout label:  model {acc:.3f}  vs rule {rule_acc:.3f}  (chance 0.5)")
    print(f"  realised held-out cost: model {model_cost:.0f}  rule {rule_cost:.0f}  oracle {oracle_cost:.0f}")
    print(f"  median realised delta (model vs rule): "
          f"{R['median_realised_delta_pct'] if R['median_realised_delta_pct'] is None else round(R['median_realised_delta_pct'],2)}%")
    print(f"  oracle gain captured: {gain_captured if gain_captured is None else round(gain_captured,3)}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(R, open(a.out, "w"), indent=1)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
