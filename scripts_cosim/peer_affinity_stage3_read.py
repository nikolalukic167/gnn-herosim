#!/usr/bin/env python3
"""peer_affinity_v1 stage 3 — read the production-trace live gate (registered in the node).

Inputs: one result JSON per arm under --results-dir (written by
scripts_cosim/datalab/peer_affinity_v1_stage3_live_gate.sbatch): knative_network.json,
knative_network_batch.json, gnn_s<N>.json, mpoff_s<N>.json.

Statistic: total_rtt over every task of the trace. Primary contrast gnn vs mpoff paired by
training seed (exact Wilcoxon, alpha 0.05, effect bar 1 % of the mpoff median), readings
GNN-NEEDED / TIE / POINTWISE-BETTER / INDETERMINATE as T1. Secondary: each learned arm's
16 seeds against each reactive arm's single number (sign test over seeds). Also reported:
peer exchange and rendezvous totals, the scheduler counters, and each arm's num_tasks
(a run that did not complete every task is refused, not averaged).

Usage:
  PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=. pipenv run python3 \
    scripts_cosim/peer_affinity_stage3_read.py \
      --results-dir simulation_data/peer_affinity_live_gate/results/workload-150-150-peer_p2_x200__cell_s7901 \
      --output simulation_data/peer_affinity_stage3_read.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_affinity_t1_read import wilcoxon_exact  # noqa: E402

ALPHA = 0.05
EFFECT_BAR = 0.01  # 1 % of the reference arm's median total_rtt


def _stats(path: Path) -> dict:
    d = json.loads(path.read_text())
    st_ = d.get("stats") or d
    prov = d.get("run_provenance") or {}
    return {
        "total_rtt": float(st_["total_rtt"]),
        "num_tasks": int(st_.get("num_tasks") or 0),
        "peer_exchange": float(st_.get("totalPeerExchangeTime") or 0.0),
        "peer_rendezvous": float(st_.get("totalPeerRendezvousWait") or 0.0),
        "avg_elapsed": float(st_.get("averageElapsedTime") or 0.0),
        "end_time": float(st_.get("endTime") or 0.0),
        "counters": st_.get("schedulerCounters") or {},
        "env": {k: v for k, v in (prov.get("env") or {}).items() if v},
        "policy": st_.get("policy"),
    }


def _reading(delta_pct: float, p: Optional[float]) -> str:
    if p is None:
        return "INDETERMINATE"
    if p < ALPHA and delta_pct >= 100 * EFFECT_BAR:
        return "GNN-NEEDED"
    if p < ALPHA and delta_pct <= -100 * EFFECT_BAR:
        return "POINTWISE-BETTER"
    if p >= ALPHA and abs(delta_pct) < 100 * EFFECT_BAR:
        return "TIE"
    return "INDETERMINATE"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    arms: Dict[str, dict] = {}
    missing: List[str] = []
    for name in ["knative_network", "knative_network_batch"] + [
        f"{a}_s{s}" for a in ("gnn", "mpoff") for s in range(1, args.seeds + 1)
    ]:
        p = args.results_dir / f"{name}.json"
        if p.is_file():
            arms[name] = _stats(p)
        else:
            missing.append(name)
    if not arms:
        raise SystemExit(f"no results under {args.results_dir}")
    n_tasks = {a["num_tasks"] for a in arms.values()}
    if len(n_tasks) != 1:
        raise SystemExit(f"arms completed different task counts: {sorted(n_tasks)} — refusing to compare")
    n = n_tasks.pop()

    seeds = [s for s in range(1, args.seeds + 1) if f"gnn_s{s}" in arms and f"mpoff_s{s}" in arms]
    gnn = [arms[f"gnn_s{s}"]["total_rtt"] for s in seeds]
    mpoff = [arms[f"mpoff_s{s}"]["total_rtt"] for s in seeds]
    out: Dict[str, object] = {
        "results_dir": str(args.results_dir), "num_tasks": n, "missing_arms": missing,
        "per_arm": {k: {kk: vv for kk, vv in v.items() if kk != "env"} for k, v in arms.items()},
        "env_by_family": {fam: next((v["env"] for k, v in arms.items() if k.startswith(fam)), {})
                          for fam in ("knative_network", "gnn_s", "mpoff_s")},
    }
    families = {}
    for fam in ("gnn", "mpoff"):
        vals = [arms[f"{fam}_s{s}"]["total_rtt"] for s in range(1, args.seeds + 1) if f"{fam}_s{s}" in arms]
        if vals:
            families[fam] = {"n": len(vals), "median": st.median(vals), "min": min(vals), "max": max(vals),
                             "peer_exchange_median": st.median([arms[f"{fam}_s{s}"]["peer_exchange"]
                                                                 for s in range(1, args.seeds + 1) if f"{fam}_s{s}" in arms])}
    for fam in ("knative_network", "knative_network_batch"):
        if fam in arms:
            families[fam] = {"n": 1, "median": arms[fam]["total_rtt"], "peer_exchange_median": arms[fam]["peer_exchange"]}
    out["families"] = families

    if seeds:
        # paired: positive = gnn better (lower total_rtt)
        diffs = [m - g for g, m in zip(gnn, mpoff)]
        ref = st.median(mpoff)
        delta_pct = 100.0 * st.median(diffs) / ref
        p = wilcoxon_exact(diffs)
        wins = sum(1 for d in diffs if d > 0)
        out["primary_gnn_vs_mpoff"] = {
            "n_pairs": len(seeds), "median_delta_pct_of_mpoff": delta_pct, "p_exact": p,
            "wins": wins, "reading": _reading(delta_pct, p),
        }
    secondary = {}
    for fam, vals in (("gnn", gnn), ("mpoff", mpoff)):
        for ref_name in ("knative_network", "knative_network_batch"):
            if ref_name in arms and vals:
                ref = arms[ref_name]["total_rtt"]
                diffs = [ref - v for v in vals]
                secondary[f"{fam}_vs_{ref_name}"] = {
                    "median_delta_pct_of_reactive": 100.0 * st.median(diffs) / ref,
                    "seeds_better": sum(1 for d in diffs if d > 0), "n": len(vals),
                    "p_sign_exact": wilcoxon_exact(diffs),
                }
    out["secondary_vs_reactive"] = secondary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[stage3-read] {len(arms)} arms, {n} tasks each; missing {missing}")
    for fam, f in families.items():
        print(f"  {fam:22s} n={f['n']:2d} median total_rtt {f['median']:.4e}"
              + (f" [{f['min']:.4e}, {f['max']:.4e}]" if "min" in f else "")
              + f"  peer_exchange {f['peer_exchange_median']:.4e}")
    if "primary_gnn_vs_mpoff" in out:
        pr = out["primary_gnn_vs_mpoff"]
        print(f"  PRIMARY gnn vs mpoff: {pr['median_delta_pct_of_mpoff']:+.2f}% p={pr['p_exact']} "
              f"{pr['wins']}/{pr['n_pairs']} -> {pr['reading']}")
    for k, v in secondary.items():
        print(f"  {k}: {v['median_delta_pct_of_reactive']:+.2f}% {v['seeds_better']}/{v['n']} p={v['p_sign_exact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
