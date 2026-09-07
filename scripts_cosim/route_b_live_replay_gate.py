#!/usr/bin/env python3
"""route_b live replay gate — reactive schedulers vs decoded plans on the frozen substrate.

Every route_b dataset stores, in `optimal_result.json`, the COMPLETE simulator input the
co-sim sweep ran: the infrastructure block (nodes, fabric, `replica_plan`, deterministic
replica placements and warmup queue depths, `fast_forward_warmup`), the one-event DAG
workload, and the `sim_inputs` with the Arm S 800 MB output override already applied.
The sweep evaluated every `forced_placements` plan through `execute_simulation(...,
'determined_determined')`. This script replays THE SAME input through the same engine
with a *reactive* policy instead of a forced plan, so a Knative scheduler makes its own
per-arrival decisions on the identical warm substrate the offline decode was scored on.

That is the only live comparison that shares the offline object's substrate: same
topology, same replica set, same queue depths, same physics env, same 4-task DAG. What it
does NOT share (and the reading must say so): the reactive arm is not bound by the
alpha=2.0 memory cap or replica uniqueness (a strictly larger action space than the
decoded arms), and it decides each task when the task becomes ready rather than jointly.

Arms:
  * reactive, by simulation.py strategy string — `knative_network`
    (kn_network_kn_network: least-connected among initialized replicas) and
    `knative_network_batch` (kn_network_batch_kn_network_batch).
  * forced, from an eval report of `eval_route_b_stage2_arm.py` — the decoded combo of
    every test parent is replayed as `forced_placements`. Replaying instead of reading the
    sweep RTT is deliberate: every arm's number then comes out of one engine invocation
    path in this process.
  * `optimal` — the sweep's argmin plan (lowest-rtt row of placements.jsonl, ties by the
    lowest plan), replayed as the engine-identity check: it MUST reproduce that row's rtt
    to 1e-9 or the run aborts. NOT optimal_result.json's `forced_placements`: measured
    2026-09-07, in ~20% of datasets of every route_b corpus that file stores a plan that
    is not the sweep argmin (best.json and the sweep agree; the stored replay file does
    not — pre-existing in the `.bak` written before the SSC repair, so it comes from the
    brute-force's best-file bookkeeping, not from the repair).
  * `<reactive>@replay` — the plan a reactive arm actually chose (read off its
    taskResults), replayed through the forced-plan path. This is the arm that pairs
    against the decoded plans: the determined scheduler charges a 0.1 s `batch_timeout`
    wait per task that the reactive schedulers do not (measured 2026-09-07: the same
    4-task plan is 37.306 s under kn_network and 37.606 s under determined), so the raw
    reactive rtt is not engine-comparable to a forced one. Both are reported.

Physics env is set in-process exactly as the Arm S generation block did
(HEROSIM_DATA_LOCALITY=1, keep-alive 1e6); the output size needs no env because it is
baked into the stored sim_inputs (asserted).

Usage:
  PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=. pipenv run python3 \\
    scripts_cosim/route_b_live_replay_gate.py \\
      --split-artifact experiments/route_b_fit_p2_split_r3.json --split test \\
      --reactive knative_network knative_network_batch \\
      --forced gnn_s1=simulation_data/route_b_fit_p2/eval_r3_gnn_seed1.json ... \\
      --workers 24 --output simulation_data/route_b_live_replay/r3_test.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REACTIVE_STRATEGIES = {
    "knative_network": "kn_network_kn_network",
    "knative_network_batch": "kn_network_batch_kn_network_batch",
    "roundrobin": "rr_network_rr_network",
}
ARM_S_ENV = {
    "HEROSIM_DATA_LOCALITY": "1",
    "HEROSIM_COSIM_KEEP_ALIVE": "1000000",
    "COSIM_SUPPRESS_SIM_PRINTS": "1",
}
EXPECTED_OUTPUT_BYTES = 800_000_000


class GateError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _worker_init() -> None:
    for k, v in ARM_S_ENV.items():
        os.environ[k] = v
    # The simulator prints per event; 200 datasets x arms of that is unreadable.
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    import logging

    logging.getLogger("simulation").setLevel(logging.ERROR)


def _load_dataset_input(ds_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    opt = json.loads((ds_dir / "optimal_result.json").read_text())
    cfg = opt["config"]
    sim_inputs = opt["sim_inputs"]
    for tname, tt in sim_inputs["task_types"].items():
        entry = (tt.get("stateSize") or {}).get("nofs-diamond4")
        if not entry or int(entry.get("output", -1)) != EXPECTED_OUTPUT_BYTES:
            raise GateError(
                f"{ds_dir.name}: stored sim_inputs stateSize[nofs-diamond4] for {tname} is "
                f"{entry!r}, expected output={EXPECTED_OUTPUT_BYTES} (Arm S override)"
            )
    return cfg, sim_inputs


def _sweep_argmin(ds_dir: Path) -> Tuple[Dict[int, Tuple[int, int]], float]:
    """(plan, rtt) of the lowest-rtt sweep row; exact ties break on the lowest plan."""
    best: Optional[Tuple[float, Tuple[Tuple[int, int], ...]]] = None
    with open(ds_dir / "placements" / "placements.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            plan = tuple((int(v[0]), int(v[1])) for _k, v in sorted(r["placement_plan"].items(), key=lambda kv: int(kv[0])))
            key = (float(r["rtt"]), plan)
            if best is None or key < best:
                best = key
    if best is None:
        raise GateError(f"{ds_dir}: empty placements.jsonl")
    return {i: p for i, p in enumerate(best[1])}, best[0]


def _node_index(cfg: Dict[str, Any]) -> Dict[str, int]:
    """node_name -> node id. `create_nodes` numbers nodes by their position in the config
    list; the identity check below asserts this against the replayed optimal plan."""
    return {str(nd["node_name"]): i for i, nd in enumerate(cfg["infrastructure"]["nodes"])}


def _run_one(
    job: Tuple[str, str, str, str, Optional[Dict[int, Tuple[int, int]]]]
) -> Tuple[str, str, float, Dict[int, Tuple[int, int]]]:
    """job = (dataset_id, ds_dir, arm, strategy, forced or None)
    -> (dataset_id, arm, rtt, plan actually executed as {task_id: (node_id, platform_id)})."""
    dataset_id, ds_dir, arm, strategy, forced = job
    from src.executecosimulation import (
        cosim_keep_alive,
        execute_simulation,
        rtt_from_stats,
    )
    from src.placement.constants import QUEUE_LENGTH

    cfg, sim_inputs = _load_dataset_input(Path(ds_dir))
    infra = deepcopy(cfg["infrastructure"])
    if forced is None:
        infra.pop("forced_placements", None)
    else:
        infra["forced_placements"] = {int(k): (int(v[0]), int(v[1])) for k, v in forced.items()}
    full_config = {"infrastructure": infra, "workload": deepcopy(cfg["workload"])}
    result = execute_simulation(
        full_config,
        sim_inputs,
        strategy,
        model_locations={},
        models={},
        cache_policy="fifo",
        task_priority="fifo",
        keep_alive=cosim_keep_alive(),
        queue_length=QUEUE_LENGTH,
    )
    stats = result.get("stats") or {}
    n = stats.get("num_tasks")
    if n != 4:
        raise GateError(f"{dataset_id}/{arm}: {n} real tasks completed, expected 4")
    node_ids = _node_index(cfg)
    executed: Dict[int, Tuple[int, int]] = {}
    for tr in stats.get("taskResults") or []:
        tid = tr.get("taskId")
        if tid is None or tid < 0:
            continue
        nname = tr.get("executionNode")
        if nname not in node_ids:
            raise GateError(f"{dataset_id}/{arm}: task {tid} ran on unknown node {nname!r}")
        executed[int(tid)] = (node_ids[nname], int(tr.get("executionPlatform")))
    if sorted(executed) != [0, 1, 2, 3]:
        raise GateError(f"{dataset_id}/{arm}: taskResults cover tasks {sorted(executed)}, expected 0..3")
    if forced is not None:
        want = {int(k): (int(v[0]), int(v[1])) for k, v in forced.items()}
        if executed != want:
            raise GateError(
                f"{dataset_id}/{arm}: forced plan {want} but taskResults show {executed} — "
                "node-id map or forced placement resolution is wrong"
            )
    return dataset_id, arm, float(rtt_from_stats(stats)), executed


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _test_parents(split_artifact: Path, split: str) -> List[str]:
    from src.notebooks.non_unique_lib.training_contract import load_split_artifact

    art, _sha = load_split_artifact(split_artifact)
    ids = sorted({str(x).split("@seq", 1)[0] for x in art[split]})
    if not ids:
        raise GateError(f"{split_artifact}: split {split!r} is empty")
    return ids


def _forced_plans_from_report(path: Path, wanted: List[str]) -> Dict[str, Dict[int, Tuple[int, int]]]:
    rep = json.loads(path.read_text())
    by_id = {p["dataset_id"]: p for p in rep["per_dataset"]}
    out: Dict[str, Dict[int, Tuple[int, int]]] = {}
    for ds in wanted:
        p = by_id.get(ds)
        if p is None:
            raise GateError(f"{path}: no per_dataset entry for {ds}")
        if p.get("infeasible"):
            raise GateError(f"{path}: {ds} decoded infeasible; the gate has no relax path")
        combo = p["decoded_combo"]
        if len(combo) != 4:
            raise GateError(f"{path}: {ds} decoded_combo has {len(combo)} tasks")
        out[ds] = {i: (int(c[0]), int(c[1])) for i, c in enumerate(combo)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split-artifact", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--simulation-data", type=Path, default=REPO_ROOT / "simulation_data")
    ap.add_argument("--reactive", nargs="*", default=["knative_network"],
                    help=f"reactive arms: {sorted(REACTIVE_STRATEGIES)}")
    ap.add_argument("--forced", nargs="*", default=[],
                    help="NAME=eval_report.json (decoded combos replayed as forced plans)")
    ap.add_argument("--limit", type=int, default=None, help="first N test parents (smoke)")
    ap.add_argument("--workers", type=int, default=max(1, min(24, os.cpu_count() or 1)))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.output.exists():
        raise GateError(f"refusing to overwrite {args.output}")
    for name in args.reactive:
        if name not in REACTIVE_STRATEGIES:
            raise GateError(f"unknown reactive arm {name!r}; known {sorted(REACTIVE_STRATEGIES)}")

    parents = _test_parents(args.split_artifact, args.split)
    if args.limit:
        parents = parents[: args.limit]
    ds_dirs = {ds: args.simulation_data / ds for ds in parents}
    for ds, d in ds_dirs.items():
        if not (d / "optimal_result.json").is_file():
            raise GateError(f"{ds}: no optimal_result.json under {d}")

    forced_arms: Dict[str, Dict[str, Dict[int, Tuple[int, int]]]] = {}
    for spec in args.forced:
        if "=" not in spec:
            raise GateError(f"--forced expects NAME=path, got {spec!r}")
        name, path = spec.split("=", 1)
        if name in forced_arms or name in args.reactive or name == "optimal":
            raise GateError(f"duplicate arm name {name!r}")
        forced_arms[name] = _forced_plans_from_report(Path(path), parents)

    # pass 1: the sweep argmin (engine identity), the reactive arms, the forced arms
    jobs: List[Tuple[str, str, str, str, Optional[Dict[int, Tuple[int, int]]]]] = []
    best_by_ds: Dict[str, float] = {}
    for ds in parents:
        _load_dataset_input(ds_dirs[ds])  # asserts the Arm S output override is baked in
        argmin_plan, best = _sweep_argmin(ds_dirs[ds])
        best_json = float(json.loads((ds_dirs[ds] / "best.json").read_text())["rtt"])
        if abs(best_json - best) > 1e-9 * max(1.0, abs(best)):
            raise GateError(f"{ds}: best.json rtt {best_json} != sweep min {best}")
        best_by_ds[ds] = best
        jobs.append((ds, str(ds_dirs[ds]), "optimal", "determined_determined", argmin_plan))
        for name in args.reactive:
            jobs.append((ds, str(ds_dirs[ds]), name, REACTIVE_STRATEGIES[name], None))
        for name, plans in forced_arms.items():
            jobs.append((ds, str(ds_dirs[ds]), name, "determined_determined", plans[ds]))

    n_arms = 1 + 2 * len(args.reactive) + len(forced_arms)
    print(f"[gate] {len(parents)} parents x {n_arms} arms "
          f"= {len(jobs) + len(parents) * len(args.reactive)} simulations on {args.workers} workers",
          flush=True)
    t0 = time.time()
    results: Dict[str, Dict[str, float]] = {ds: {} for ds in parents}
    plans_out: Dict[str, Dict[str, Dict[int, Tuple[int, int]]]] = {ds: {} for ds in parents}
    with mp.get_context("fork").Pool(args.workers, initializer=_worker_init) as pool:
        for k, (ds, arm, rtt, plan) in enumerate(pool.imap_unordered(_run_one, jobs, chunksize=4), 1):
            results[ds][arm] = rtt
            plans_out[ds][arm] = plan
            if k % 500 == 0 or k == len(jobs):
                print(f"[gate] pass 1: {k}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)

        # identity check before spending anything else
        bad = []
        for ds in parents:
            got = results[ds]["optimal"]
            want = best_by_ds[ds]
            if abs(got - want) > 1e-9 * max(1.0, abs(want)):
                bad.append((ds, got, want))
        if bad:
            raise GateError(
                f"engine identity FAILED on {len(bad)} parents — replaying the sweep argmin "
                f"does not reproduce its recorded rtt; first: {bad[:3]}"
            )
        print(f"[gate] engine identity OK: sweep argmin reproduces its rtt on all "
              f"{len(parents)} parents", flush=True)

        # pass 2: the reactive arms' chosen plans through the forced path
        jobs2 = [
            (ds, str(ds_dirs[ds]), f"{name}@replay", "determined_determined", plans_out[ds][name])
            for ds in parents for name in args.reactive
        ]
        for k, (ds, arm, rtt, _plan) in enumerate(pool.imap_unordered(_run_one, jobs2, chunksize=4), 1):
            results[ds][arm] = rtt
            if k % 500 == 0 or k == len(jobs2):
                print(f"[gate] pass 2: {k}/{len(jobs2)} done ({time.time() - t0:.0f}s)", flush=True)

    from src.placement.env_fingerprint import describe_code_provenance, describe_python_env

    out = {
        "split_artifact": str(args.split_artifact),
        "split": args.split,
        "n_parents": len(parents),
        "arms": {"optimal": "sweep argmin replayed (identity check)",
                 **{n: REACTIVE_STRATEGIES[n] for n in args.reactive},
                 **{f"{n}@replay": f"{n}'s chosen plan through determined_determined" for n in args.reactive},
                 **{n: "forced decoded plan" for n in forced_arms}},
        "forced_sources": {n: s.split("=", 1)[1] for n, s in
                           zip(forced_arms, args.forced)},
        "physics_env": ARM_S_ENV,
        "per_dataset": {
            ds: {
                "optimum_rtt": best_by_ds[ds],
                "rtt": results[ds],
                "plans": {arm: [list(plans_out[ds][arm][t]) for t in range(4)] for arm in plans_out[ds]},
            }
            for ds in parents
        },
        "run_provenance": {"code": describe_code_provenance(), "python_env": describe_python_env()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(f"[gate] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
