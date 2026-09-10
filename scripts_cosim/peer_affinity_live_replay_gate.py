#!/usr/bin/env python3
"""peer_affinity_v1 live replay gate — reactive schedulers vs decoded plans, one engine.

Why this exists. The T1 read (docs/lineages/peer_affinity_v1.md) scores every arm as decode
regret against the enumerated cap-feasible optimum. Those RTTs come out of the brute-force
sweep, which simulated every plan in the real engine — verified here as the `optimal` arm's
identity check and, for decoded plans, to 0.0e+00 on spot checks. So the T1 numbers are already
real-engine numbers. What they do NOT contain is an arm that is not in the sweep at all: a
**reactive** scheduler that makes its own per-arrival decision, unaware of peer affinity and
unbound by the alpha cap. That is the only genuinely new thing a live gate adds here, and it is
what this script measures.

Arms:
  * `optimal`   — the sweep argmin plan replayed as `forced_placements`. MUST reproduce that
                  row's rtt to 1e-9 or the dataset aborts (engine identity).
  * `<arm>_s<N>`— a learned arm's decoded combo for that dataset, from an
                  `eval_route_b_stage2_arm.py` report, replayed as `forced_placements`.
  * `<reactive>`— a reactive strategy deciding for itself (no forced plan).
  * `<reactive>@replay` — the plan that reactive arm actually chose, replayed through the
                  forced path. This is the arm that pairs against the decoded plans: the
                  determined scheduler charges a `batch_timeout` wait per task that the
                  reactive schedulers do not (gate-tools, 2026-09-07), so a raw reactive rtt is
                  not engine-comparable to a forced one. Both are reported.

Physics: `HEROSIM_PEER_EXCHANGE=1` for every arm — the reactive arms are *charged* peer cost
even though nothing in their scoring knows peers exist, which is the point of the baseline.
Keep-alive is the co-sim value (huge) so the autoscaler never scales a replica down; under
overlapping replicas a scale-down would re-release a slot another task type still holds
(`src/placement/autoscaler.py:342`), which co-sim never exercises. Asserted, not assumed.

What the reading must say: the reactive arms have a strictly LARGER action space (no alpha cap,
no replica-uniqueness) and decide per arrival rather than jointly.

Usage:
  PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=. pipenv run python3 \
    scripts_cosim/peer_affinity_live_replay_gate.py \
      --split-artifact experiments/peer_affinity_v1_t1_split.json --split test \
      --reports-dir simulation_data/peer_affinity_t1_reports \
      --reactive knative_network_batch knative_network \
      --workers 20 --output simulation_data/peer_affinity_live_replay.json
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import statistics as st
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REACTIVE_STRATEGIES = {
    "knative_network": "kn_network_kn_network",
    "knative_network_batch": "kn_network_batch_kn_network_batch",
}
PHYSICS_ENV = {
    "HEROSIM_PEER_EXCHANGE": "1",
    "HEROSIM_COSIM_KEEP_ALIVE": "1000000",
    "COSIM_SUPPRESS_SIM_PRINTS": "1",
    "SIM_FORCE_FULL_STATS": "1",
}
LRS = ("lr5e4", "lr1e3", "lr2e3")


class GateError(RuntimeError):
    pass


def _apply_env() -> None:
    os.environ.update(PHYSICS_ENV)
    os.environ.pop("HEROSIM_DATA_LOCALITY", None)


def _sweep_argmin(ds_dir: Path) -> Tuple[Dict[int, Tuple[int, int]], float]:
    """Lowest-rtt row of placements.jsonl, ties broken by the lexicographically lowest plan."""
    best_rtt: Optional[float] = None
    best_plan: Optional[Dict[int, Tuple[int, int]]] = None
    with open(ds_dir / "placements/placements.jsonl") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rtt = float(row["rtt"])
            plan = {int(k): (int(v[0]), int(v[1])) for k, v in row["placement_plan"].items()}
            key = tuple(plan[i] for i in sorted(plan))
            if best_rtt is None or rtt < best_rtt - 1e-12 or (
                    abs(rtt - best_rtt) <= 1e-12 and key < tuple(best_plan[i] for i in sorted(best_plan))):
                best_rtt, best_plan = rtt, plan
    if best_plan is None:
        raise GateError(f"{ds_dir.name}: empty placements.jsonl")
    return best_plan, best_rtt


def _run(cfg: dict, sim_inputs: dict, strategy: str) -> dict:
    from src.executecosimulation import cosim_keep_alive, QUEUE_LENGTH, execute_simulation
    return execute_simulation(cfg, sim_inputs, strategy, cache_policy="fifo",
                              task_priority="fifo", keep_alive=cosim_keep_alive(),
                              queue_length=QUEUE_LENGTH)


def _node_index(cfg: dict) -> Dict[str, int]:
    nodes = cfg["infrastructure"].get("nodes")
    if isinstance(nodes, list):
        return {n["node_name"]: i for i, n in enumerate(nodes)}
    return {}


def _chosen_plan(stats: dict, cfg: dict, n_tasks: int, tag: str) -> Dict[int, Tuple[int, int]]:
    idx = _node_index(cfg)
    out: Dict[int, Tuple[int, int]] = {}
    for tr in stats.get("taskResults") or []:
        tid = tr.get("taskId")
        if tid is None or int(tid) < 0:
            continue
        name = tr.get("executionNode")
        if name not in idx:
            raise GateError(f"{tag}: task {tid} ran on unknown node {name!r}")
        out[int(tid)] = (idx[name], int(tr.get("executionPlatform")))
    if sorted(out) != list(range(n_tasks)):
        raise GateError(f"{tag}: taskResults cover {sorted(out)}, expected 0..{n_tasks - 1}")
    return out


def _one_dataset(job: Tuple[str, Dict[str, List[List[int]]], List[str]]) -> dict:
    ds_id, decoded, reactive = job
    _apply_env()
    ds_dir = REPO_ROOT / "simulation_data" / ds_id
    o = json.load(open(ds_dir / "optimal_result.json"))
    sim_inputs = o["sim_inputs"]
    n_tasks = len(o["config"]["workload"]["events"])
    out: Dict[str, float] = {}
    peer: Dict[str, float] = {}

    def forced(plan: Dict[int, Tuple[int, int]], tag: str) -> Tuple[float, float]:
        cfg = copy.deepcopy(o["config"])
        cfg["infrastructure"]["forced_placements"] = {int(k): (int(v[0]), int(v[1])) for k, v in plan.items()}
        s = _run(cfg, sim_inputs, "determined_determined")["stats"]
        if int(s.get("num_tasks") or -1) != n_tasks:
            raise GateError(f"{ds_id}/{tag}: {s.get('num_tasks')} tasks completed, expected {n_tasks}")
        return float(s["total_rtt"]), float(s.get("totalPeerExchangeTime") or 0.0)

    # engine identity on the sweep argmin
    plan_opt, rtt_sweep = _sweep_argmin(ds_dir)
    r, p = forced(plan_opt, "optimal")
    if abs(r - rtt_sweep) > 1e-9 * max(1.0, abs(rtt_sweep)):
        raise GateError(f"{ds_id}/optimal: replay {r!r} != sweep argmin {rtt_sweep!r}")
    out["optimal"], peer["optimal"] = r, p

    for arm_seed, combo in decoded.items():
        plan = {i: (int(a), int(b)) for i, (a, b) in enumerate(combo)}
        out[arm_seed], peer[arm_seed] = forced(plan, arm_seed)

    for name in reactive:
        cfg = copy.deepcopy(o["config"])
        cfg["infrastructure"].pop("forced_placements", None)
        s = _run(cfg, sim_inputs, REACTIVE_STRATEGIES[name])["stats"]
        if int(s.get("num_tasks") or -1) != n_tasks:
            raise GateError(f"{ds_id}/{name}: {s.get('num_tasks')} tasks completed, expected {n_tasks}")
        out[name], peer[name] = float(s["total_rtt"]), float(s.get("totalPeerExchangeTime") or 0.0)
        chosen = _chosen_plan(s, cfg, n_tasks, f"{ds_id}/{name}")
        out[f"{name}@replay"], peer[f"{name}@replay"] = forced(chosen, f"{name}@replay")
    return {"dataset_id": ds_id, "rtt": out, "peer": peer, "n_tasks": n_tasks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split-artifact", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--reports-dir", type=Path, required=True)
    ap.add_argument("--read", type=Path, default=REPO_ROOT / "simulation_data/peer_affinity_t1_read.json",
                    help="T1 reading, for the per-arm chosen learning rate")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--reactive", nargs="*", default=sorted(REACTIVE_STRATEGIES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(20, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    split = json.loads(args.split_artifact.read_text())
    ids = list(split[args.split])
    if args.limit:
        ids = ids[:args.limit]
    chosen_lr = json.loads(args.read.read_text())["chosen_lr"]

    # decoded combos per dataset: {"<arm>_s<seed>": combo}
    decoded: Dict[str, Dict[str, List[List[int]]]] = {i: {} for i in ids}
    for arm, lr in chosen_lr.items():
        for seed in range(1, args.seeds + 1):
            stem = (f"peer-affinity-v1-t1-{arm}-{lr}-seed{seed}" if arm in ("gnn", "mpoff")
                    else f"peer-affinity-v1-t1-{arm}-{lr}_seed{seed}")
            path = args.reports_dir / f"{stem}.json"
            if not path.exists():
                raise SystemExit(f"missing eval report {path}")
            for row in json.load(open(path))["per_dataset"]:
                if row["dataset_id"] in decoded and not row.get("infeasible"):
                    decoded[row["dataset_id"]][f"{arm}_s{seed}"] = row["decoded_combo"]

    jobs = [(i, decoded[i], list(args.reactive)) for i in ids]
    print(f"[gate] {len(jobs)} datasets x (1 optimal + {len(decoded[ids[0]])} decoded + "
          f"{2 * len(args.reactive)} reactive) arms, workers={args.workers}", flush=True)
    t0 = time.time()
    rows: List[dict] = []
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for k, row in enumerate(pool.imap_unordered(_one_dataset, jobs), 1):
                rows.append(row)
                print(f"[gate] {k}/{len(jobs)} {row['dataset_id'].split('/')[-1]} ({time.time()-t0:.0f}s)", flush=True)
    else:
        for k, j in enumerate(jobs, 1):
            rows.append(_one_dataset(j))
            print(f"[gate] {k}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    # aggregate: per-arm excess over the enumerated optimum, paired by dataset
    arms = sorted({a for r in rows for a in r["rtt"]})
    agg: Dict[str, dict] = {}
    for a in arms:
        ex = [100.0 * (r["rtt"][a] / r["rtt"]["optimal"] - 1.0) for r in rows if a in r["rtt"]]
        tot = sum(r["rtt"][a] for r in rows if a in r["rtt"])
        agg[a] = {"n": len(ex), "median_excess_pct": st.median(ex), "mean_excess_pct": st.mean(ex),
                  "total_rtt_s": tot,
                  "median_peer_s": st.median([r["peer"][a] for r in rows if a in r["peer"]])}
    # arm families collapsed over seeds
    fam: Dict[str, dict] = {}
    for base in ("gnn", "mpoff", "mlp_t1x", "mlp_t1"):
        per_seed = [agg[k]["median_excess_pct"] for k in agg if k.startswith(base + "_s")]
        if per_seed:
            fam[base] = {"seeds": len(per_seed), "mean_of_seed_medians_pct": st.mean(per_seed),
                         "min": min(per_seed), "max": max(per_seed)}
    payload = {"registration": "docs/lineages/peer_affinity_v1.md#T1-live",
               "split_artifact": str(args.split_artifact), "split": args.split,
               "physics_env": PHYSICS_ENV, "chosen_lr": chosen_lr,
               "caveat": ("reactive arms have a strictly larger action space (no alpha cap, no "
                          "replica uniqueness) and decide per arrival, not jointly"),
               "per_dataset": rows, "per_arm": agg, "per_family": fam}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1))
    print(f"\n[gate] wrote {args.output}  ({time.time()-t0:.0f}s)")
    print(f"\n{'arm':28s} {'median excess vs optimum':>26s} {'total s':>10s}")
    for a in sorted(agg, key=lambda k: agg[k]["median_excess_pct"]):
        if a.startswith(("gnn_s", "mpoff_s", "mlp_t1_s", "mlp_t1x_s")):
            continue
        print(f"{a:28s} {agg[a]['median_excess_pct']:25.2f}% {agg[a]['total_rtt_s']:10.0f}")
    for b, v in fam.items():
        print(f"{b + ' (8 seeds)':28s} {v['mean_of_seed_medians_pct']:25.2f}% "
              f"[{v['min']:.1f}..{v['max']:.1f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
