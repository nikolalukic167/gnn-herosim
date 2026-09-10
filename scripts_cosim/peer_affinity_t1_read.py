#!/usr/bin/env python3
"""peer_affinity_v1 -- the registered T1 reading (docs/lineages/peer_affinity_v1.md, "Training
registration T1"). Everything here is fixed by the registration; this script only executes it.

Per checkpoint: run scripts_cosim/eval_route_b_stage2_arm.py under the arm's environment
(contract v2, alpha 2.5, replica reuse + counted relaxation, peer-mass per arm, MP-off for the
mpoff arm) and keep the per-dataset decode regrets split by the artifact's train/val/test.
Per arm: choose the learning rate whose mean over seeds of the per-seed VAL median regret is
lowest (validation only, never test). Per seed: the TEST median regret at the chosen lr, for
the val-selected checkpoint and (GNN arms) the -final checkpoint. Contrasts paired by seed with
an exact Wilcoxon signed-rank test; readings GNN-NEEDED / TIE / POINTWISE-BETTER / INDETERMINATE
at alpha 0.05 and a 1 pp bar. Convergence: the last-20-epoch slope of the trainer's logged val
regret, from the SLURM .out logs when given.

Usage:
  peer_affinity_t1_read.py --models-dir models --cache-dir simulation_data/graphs_cache_peer_affinity_v1_t1 \
      --split experiments/peer_affinity_v1_t1_split.json --logs-dir logs --out simulation_data/peer_affinity_t1_read.json
  peer_affinity_t1_read.py --smoke   # the four 5-epoch smoke checkpoints, pipeline check only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts_cosim/eval_route_b_stage2_arm.py"
ARMS = ("gnn", "mpoff", "mlp_t1", "mlp_t1x")
LRS = ("lr5e4", "lr1e3", "lr2e3")
ALPHA_KEY = "2.5"
BAR_PP = 1.0
ALPHA = 0.05


def arm_env(arm: str) -> Dict[str, str]:
    env = dict(os.environ)
    env.update({"PARTIAL_STATE_CONTRACT": "partial_state_v2", "EVAL_DECODE_REPLICA_REUSE": "1",
                "EVAL_DECODE_RELAX": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4",
                "PYTHONPATH": str(ROOT), "PARTIAL_STATE_PEER_MASS": "0" if arm == "mlp_t1" else "1"})
    if arm == "mpoff":
        env["GNN_DISABLE_MESSAGE_PASSING"] = "1"
    else:
        env.pop("GNN_DISABLE_MESSAGE_PASSING", None)
    return env


def run_eval(ckpt: Path, arm: str, cache: Path, split: Path, report: Path) -> dict:
    if not report.exists():
        cmd = [sys.executable, str(EVAL), "--checkpoint", str(ckpt), "--cache-dir", str(cache),
               "--split-artifact", str(split), "--alpha-key", ALPHA_KEY, "--report", str(report)]
        r = subprocess.run(cmd, env=arm_env(arm), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"eval failed for {ckpt}:\n{r.stderr[-2000:]}")
    return json.loads(report.read_text())


def split_medians(rep: dict, membership: Dict[str, str]) -> Dict[str, Optional[float]]:
    rows = rep.get("datasets") or rep.get("per_dataset") or []
    by = {"train": [], "val": [], "test": []}
    relaxed = 0; infeasible = 0
    for d in rows:
        s = membership.get(d["dataset_id"])
        if d.get("infeasible"):
            infeasible += 1; continue
        if d.get("decoder", {}).get("relaxed"):
            relaxed += 1
        if s in by and d.get("decode_regret_pct"):
            by[s].append(float(d["decode_regret_pct"]["mean_tied"]))
    out = {k: (st.median(v) if v else None) for k, v in by.items()}
    out["n_test"] = len(by["test"]); out["relaxed"] = relaxed; out["infeasible"] = infeasible
    return out


def wilcoxon_exact(diffs: List[float]) -> Optional[float]:
    """Exact two-sided signed-rank p-value (n <= 20); ties at zero dropped."""
    d = [x for x in diffs if x != 0.0]
    n = len(d)
    if n == 0:
        return None
    ranks = {}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    w_pos = sum(ranks[i] for i in range(n) if d[i] > 0)
    w_neg = sum(ranks[i] for i in range(n) if d[i] < 0)
    w = min(w_pos, w_neg)
    # exact distribution over all sign assignments
    total = 0; count = 0
    from itertools import product
    rk = [ranks[i] for i in range(n)]
    for signs in product((0, 1), repeat=n):
        s = sum(r for r, sg in zip(rk, signs) if sg)
        total += 1
        if min(s, sum(rk) - s) <= w + 1e-12:
            count += 1
    return count / total


def reading(diffs: List[float], p: Optional[float]) -> str:
    med = st.median(diffs) if diffs else 0.0
    if p is not None and p < ALPHA and med >= BAR_PP:
        return "GNN-NEEDED"
    if p is not None and p < ALPHA and med <= -BAR_PP:
        return "POINTWISE-BETTER"
    if p is not None and p >= ALPHA and abs(med) < BAR_PP:
        return "TIE"
    return "INDETERMINATE"


def convergence_from_log(path: Path) -> Optional[dict]:
    """Slope of the trainer's per-epoch val regret over the last 20 epochs (s per epoch),
    relative to its level; 'flat' if |slope*20| < 2 % of the level."""
    if not path.exists():
        return None
    vals = []
    for line in path.read_text(errors="ignore").splitlines():
        m = re.search(r"\[val\].*greedy_regret=([0-9.]+)s", line)
        if m:
            vals.append(float(m.group(1)))
    if len(vals) < 25:
        return {"epochs": len(vals), "flat": None}
    tail = vals[-20:]
    xs = list(range(20)); xm = sum(xs) / 20; ym = sum(tail) / 20
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, tail)) / sum((x - xm) ** 2 for x in xs)
    level = max(ym, 1e-9)
    return {"epochs": len(vals), "tail_slope_per_epoch": slope, "tail_change_rel": slope * 20 / level,
            "flat": abs(slope * 20 / level) < 0.02}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "simulation_data/graphs_cache_peer_affinity_v1_t1")
    ap.add_argument("--split", type=Path, default=ROOT / "experiments/peer_affinity_v1_t1_split.json")
    ap.add_argument("--logs-dir", type=Path, default=None, help="SLURM .out logs for the convergence check")
    ap.add_argument("--reports-dir", type=Path, default=ROOT / "simulation_data/peer_affinity_t1_reports")
    ap.add_argument("--out", type=Path, default=ROOT / "simulation_data/peer_affinity_t1_read.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.cache_dir = ROOT / "simulation_data/graphs_cache_peer_affinity_v1_r2_smoke"
        args.split = ROOT / "experiments/peer_affinity_v1_smoke_split.json"
        args.reports_dir = ROOT / "simulation_data/peer_affinity_t1_reports_smoke"
        args.out = ROOT / "simulation_data/peer_affinity_t1_read_smoke.json"
    split = json.loads(args.split.read_text())
    membership = {i: s for s in ("train", "val", "test") for i in split[s]}
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # discover checkpoints: (arm, lr, seed, variant) -> path
    found: Dict[tuple, Path] = {}
    if args.smoke:
        for arm in ARMS:
            if arm in ("gnn", "mpoff"):
                p = args.models_dir / f"peer-affinity-v1-smoke-{arm}-seed1.pt"
            else:
                p = args.models_dir / "tabular" / f"peer-affinity-v1-smoke-{arm.replace('_', '-')}_seed1.pt"
            if p.exists():
                found[(arm, "lr1e3", 1, "val")] = p
    else:
        for arm in ("gnn", "mpoff"):
            for lr in LRS:
                for seed in range(1, 9):
                    p = args.models_dir / f"peer-affinity-v1-t1-{arm}-{lr}-seed{seed}.pt"
                    if p.exists():
                        found[(arm, lr, seed, "val")] = p
                    pf = args.models_dir / f"peer-affinity-v1-t1-{arm}-{lr}-seed{seed}-final.pt"
                    if pf.exists():
                        found[(arm, lr, seed, "final")] = pf
        for arm in ("mlp_t1", "mlp_t1x"):
            for lr in LRS:
                for seed in range(1, 9):
                    p = args.models_dir / "tabular" / f"peer-affinity-v1-t1-{arm}-{lr}_seed{seed}.pt"
                    if p.exists():
                        found[(arm, lr, seed, "val")] = p
    print(f"[read] {len(found)} checkpoints found", flush=True)

    per: Dict[str, dict] = {}
    for key, ckpt in sorted(found.items()):
        arm, lr, seed, variant = key
        rep = run_eval(ckpt, arm, args.cache_dir, args.split, args.reports_dir / (ckpt.stem + ".json"))
        m = split_medians(rep, membership)
        conv = None
        if args.logs_dir is not None and arm in ("gnn", "mpoff"):
            cfg_index = {("gnn", "lr5e4"): 0, ("gnn", "lr1e3"): 1, ("gnn", "lr2e3"): 2,
                         ("mpoff", "lr5e4"): 3, ("mpoff", "lr1e3"): 4, ("mpoff", "lr2e3"): 5}[(arm, lr)]
            task = cfg_index * 8 + (seed - 1)
            logs = sorted(glob.glob(str(args.logs_dir / f"pa-t1-gnn-*_{task}.out")))
            conv = convergence_from_log(Path(logs[-1])) if logs else None
        per["|".join(map(str, key))] = {"arm": arm, "lr": lr, "seed": seed, "variant": variant,
                                        "checkpoint": str(ckpt), **m, "convergence": conv}
        print(f"[read] {arm:8s} {lr} seed{seed} {variant:5s} val={m['val']} test={m['test']} relaxed={m['relaxed']}", flush=True)

    # lr selection on validation (val-selected checkpoints only)
    chosen: Dict[str, str] = {}
    lr_table: Dict[str, dict] = {}
    for arm in ARMS:
        cands = {}
        for lr in LRS:
            vals = [r["val"] for r in per.values() if r["arm"] == arm and r["lr"] == lr and r["variant"] == "val" and r["val"] is not None]
            if vals:
                cands[lr] = {"mean_val_median": sum(vals) / len(vals), "n_seeds": len(vals)}
        lr_table[arm] = cands
        if cands:
            chosen[arm] = min(cands, key=lambda k: (cands[k]["mean_val_median"], k))

    def seed_test(arm: str, variant: str) -> Dict[int, float]:
        lr = chosen.get(arm)
        return {r["seed"]: r["test"] for r in per.values()
                if r["arm"] == arm and r["lr"] == lr and r["variant"] == variant and r["test"] is not None}

    contrasts = {}
    for a, b in (("gnn", "mlp_t1"), ("gnn", "mpoff"), ("gnn", "mlp_t1x"), ("mpoff", "mlp_t1"), ("mlp_t1x", "mlp_t1")):
        for variant in ("val", "final"):
            ta, tb = seed_test(a, variant if a in ("gnn", "mpoff") else "val"), seed_test(b, variant if b in ("gnn", "mpoff") else "val")
            seeds = sorted(set(ta) & set(tb))
            if not seeds:
                continue
            diffs = [tb[s] - ta[s] for s in seeds]  # positive = a better (lower regret)
            p = wilcoxon_exact(diffs) if len(seeds) <= 20 else None
            contrasts[f"{a}_vs_{b}@{variant}"] = {"n_seeds": len(seeds), "median_delta_pp": st.median(diffs),
                                                  "mean_delta_pp": sum(diffs) / len(diffs), "p_exact_wilcoxon": p,
                                                  "reading": reading(diffs, p), "a_medians": ta, "b_medians": tb}
    out = {"registration": "docs/lineages/peer_affinity_v1.md#T1", "alpha_key": ALPHA_KEY, "bar_pp": BAR_PP, "alpha": ALPHA,
           "chosen_lr": chosen, "lr_table": lr_table, "per_checkpoint": per, "contrasts": contrasts,
           "arm_test_medians": {arm: {v: seed_test(arm, v) for v in ("val", "final")} for arm in ARMS}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"chosen_lr": chosen, "contrasts": {k: {kk: v[kk] for kk in ("n_seeds", "median_delta_pp", "p_exact_wilcoxon", "reading")} for k, v in contrasts.items()}}, indent=1))
    print(f"[read] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
