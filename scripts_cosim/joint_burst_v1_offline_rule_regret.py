#!/usr/bin/env python3
"""joint_burst_v1 step 0 (2026-09-20) -- how far is the GREEDY from the sweep optimum, offline?

The burst-trained `gnnedge0` decodes at 24.7 s regret against a 97.4 s optimal group cost on
the val split (`docs/lineages/joint_burst_v1.md`), and live the batched greedy is 16.8 % faster
in the same seat. The snapshots do not record the rule's own plan, so the greedy's OFFLINE
regret -- the number that says whether the gap is model fit or offline/live mismatch -- was
never measured. This script replays hand policies through the real engine on each co-sim
dataset (the path `peer_affinity_live_serve_check.py` proved bit-identical to the sweep) and
reports, per dataset and pooled:

    regret(arm) = total_rtt(arm) - optimal_rtt        (seconds, group-summed elapsed)

for the batched greedy (the learned arms' seat), the immediate rule, reactive Knative, and the
sweep's own random-plan expectation (mean over the enumerated plans).

Usage (datalab, micromamba gnn):
  PYTHONPATH=. python3 scripts_cosim/joint_burst_v1_offline_rule_regret.py \
      --datasets-dir simulation_data/gnn_datasets_joint_burst_v1_heldout \
      --output simulation_data/joint_burst_v1/offline_rule_regret_heldout.json --workers 32
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The env the learned arms and the rule are served with (peer_affinity_live_serve_check.py's
# PHYSICS_ENV + the batch path the gate uses). Set before any src import.
PHYSICS_ENV = {
    "HEROSIM_PEER_EXCHANGE": "1",
    "HEROSIM_COSIM_KEEP_ALIVE": "1000000",
    "COSIM_SUPPRESS_SIM_PRINTS": "1",
    "SIM_FORCE_FULL_STATS": "1",
    "GNN_DECODE_MODE": "masked_topo",
    "GNN_BATCH_BY_PEER_GROUP": "1",
    "GNN_BATCH_SIZE": "10",
    "GNN_BATCH_TIMEOUT": "16",
    "HEROSIM_GNN_DEVICE": "cpu",
    "PYTHONHASHSEED": "0",
}

# arm -> (strategy, extra env). joint_burst_v2 adds the coordinate-descent greedy (the honest
# bar: same score, refined) and the exchange x2 probe (the label's direction: more co-location).
ARMS = {
    "batched_greedy": ("peer_greedy_network_batch_peer_greedy_network_batch", {}),
    "cd_greedy": ("peer_greedy_network_cd_peer_greedy_network_cd", {}),
    "batched_greedy_x2": ("peer_greedy_network_batch_peer_greedy_network_batch", {"HEROSIM_PG_EXCHANGE_SCALE": "2.0"}),
    "immediate_rule": ("peer_greedy_network_peer_greedy_network", {}),
    "drain_only_rule": ("drain_greedy_network_drain_greedy_network", {}),
    "reactive_knative": ("kn_network_batch_kn_network_batch", {}),
}


def _apply_env() -> None:
    for k, v in PHYSICS_ENV.items():
        os.environ[k] = v


def _sweep_stats(ds_dir: Path) -> Dict[str, float]:
    rtts: List[float] = []
    with open(ds_dir / "placements/placements.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rtts.append(float(json.loads(line)["rtt"]))
    if not rtts:
        raise RuntimeError(f"FAIL LOUD: empty sweep in {ds_dir}")
    return {
        "n_plans": len(rtts),
        "optimal_rtt": min(rtts),
        "random_mean_rtt": statistics.fmean(rtts),
        "random_median_rtt": statistics.median(rtts),
        "worst_rtt": max(rtts),
    }


def _one_dataset(job: Dict[str, Any]) -> Dict[str, Any]:
    _apply_env()
    from src.executecosimulation import QUEUE_LENGTH, cosim_keep_alive, execute_simulation

    ds_dir = Path(job["ds_dir"])
    o = json.load(open(ds_dir / "optimal_result.json"))
    best = json.load(open(ds_dir / "best.json"))
    sweep = _sweep_stats(ds_dir)
    if abs(float(best["rtt"]) - sweep["optimal_rtt"]) > 1e-6 * max(1.0, sweep["optimal_rtt"]):
        raise RuntimeError(
            f"FAIL LOUD: best.json {best['rtt']} != sweep min {sweep['optimal_rtt']} in {ds_dir}"
        )
    out: Dict[str, Any] = {"dataset": ds_dir.name, **sweep,
                           "n_tasks": len(o["config"]["workload"]["events"])}
    for arm, (strategy, extra_env) in ARMS.items():
        cfg = copy.deepcopy(o["config"])
        cfg["infrastructure"].pop("forced_placements", None)
        for k in ("HEROSIM_PG_EXCHANGE_SCALE", "HEROSIM_PG_CD_PASSES"):
            os.environ.pop(k, None)
        os.environ.update(extra_env)
        try:
            stats = execute_simulation(
                cfg, o["sim_inputs"], strategy, models=None, cache_policy="fifo",
                task_priority="fifo", keep_alive=cosim_keep_alive(), queue_length=QUEUE_LENGTH,
            )["stats"]
        except Exception as exc:  # recorded, never averaged
            out[arm] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        rtt = float(stats["total_rtt"])
        out[arm] = {
            "total_rtt": rtt,
            "regret": rtt - sweep["optimal_rtt"],
            "num_tasks": int(stats.get("num_tasks") or -1),
            "peer_exchange_time": float(stats.get("totalPeerExchangeTime") or 0.0),
            "counters": {k: v for k, v in (stats.get("schedulerCounters") or {}).items() if k.startswith("pg_cd")},
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    _apply_env()

    ds_dirs = sorted(p for p in args.datasets_dir.iterdir() if p.is_dir() and p.name.startswith("ds_"))
    if args.limit:
        ds_dirs = ds_dirs[: args.limit]
    if not ds_dirs:
        raise SystemExit(f"FAIL LOUD: no ds_* under {args.datasets_dir}")
    jobs = [{"ds_dir": str(p)} for p in ds_dirs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        rows = pool.map(_one_dataset, jobs, chunksize=1)

    summary: Dict[str, Any] = {"n_datasets": len(rows), "arms": {}}
    opt = [r["optimal_rtt"] for r in rows]
    summary["optimal_rtt_mean"] = statistics.fmean(opt)
    summary["random_regret_mean"] = statistics.fmean(r["random_mean_rtt"] - r["optimal_rtt"] for r in rows)
    summary["worst_regret_mean"] = statistics.fmean(r["worst_rtt"] - r["optimal_rtt"] for r in rows)
    for arm in ARMS:
        ok = [r for r in rows if "regret" in r.get(arm, {})]
        regs = [r[arm]["regret"] for r in ok]
        if not regs:
            summary["arms"][arm] = {"n": 0}
            continue
        summary["arms"][arm] = {
            "n": len(regs),
            "errors": len(rows) - len(regs),
            "regret_mean": statistics.fmean(regs),
            "regret_median": statistics.median(regs),
            "regret_frac_of_random": statistics.fmean(regs) / summary["random_regret_mean"]
            if summary["random_regret_mean"] > 0 else None,
            "optimal_hit_frac": sum(1 for x in regs if x <= 1e-6) / len(regs),
            "rtt_over_optimal_pct_median": statistics.median(
                100.0 * r[arm]["regret"] / r["optimal_rtt"] for r in ok),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows, "env": PHYSICS_ENV}, open(args.output, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
