#!/usr/bin/env python3
"""cd_gap_v1 A -- the CD greedy's plan on every co-sim group, as a training label.

Replays `peer_greedy_network_cd` through the real engine on each dataset (the same call and env as
`joint_burst_v1_offline_rule_regret.py`, which the serve check proved identical to the sweep) and
recovers its plan as the sweep rows whose rtt equals the replay's total_rtt. Several equal rows are
kept as an any-of set (they cost the same to the last digit); no matching row is a hard error.

Output (the trainer's NEAR_RTT_LABEL_OVERRIDE_JSON format), keyed by the cache's dataset id
"<datasets dir name>/<ds_XXXXX>":
  {"<id>": {"plans": [[[node_id, platform_id], ...per task in task-id order], ...],
            "cd_rtt": float, "opt_rtt": float}}

  cd_gap_v1_cd_labels.py --datasets-dir DIR [DIR ...] --output F --workers N
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.joint_burst_v1_offline_rule_regret import ARMS, PHYSICS_ENV  # noqa: E402


def _one(job: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.update(PHYSICS_ENV)
    for k in ("HEROSIM_PG_EXCHANGE_SCALE", "HEROSIM_PG_CD_PASSES", "HEROSIM_PG_BATCH_BLIND"):
        os.environ.pop(k, None)
    from src.executecosimulation import QUEUE_LENGTH, cosim_keep_alive, execute_simulation

    ds_dir = Path(job["ds_dir"])
    o = json.load(open(ds_dir / "optimal_result.json"))
    cfg = copy.deepcopy(o["config"])
    cfg["infrastructure"].pop("forced_placements", None)
    strategy, extra = ARMS["cd_greedy"]
    os.environ.update(extra)
    stats = execute_simulation(cfg, o["sim_inputs"], strategy, models=None, cache_policy="fifo",
                               task_priority="fifo", keep_alive=cosim_keep_alive(),
                               queue_length=QUEUE_LENGTH)["stats"]
    rtt = float(stats["total_rtt"])
    plans, opt = [], float("inf")
    with open(ds_dir / "placements" / "placements.jsonl") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            opt = min(opt, float(r["rtt"]))
            if abs(float(r["rtt"]) - rtt) <= 1e-9 * max(1.0, abs(rtt)):
                pl = r["placement_plan"]
                plans.append([[int(pl[k][0]), int(pl[k][1])] for k in sorted(pl, key=int)])
    if not plans:
        raise RuntimeError(f"FAIL LOUD: CD rtt {rtt} matches no sweep row in {ds_dir}")
    return {"id": f"{ds_dir.parent.name}/{ds_dir.name}", "plans": plans, "cd_rtt": rtt, "opt_rtt": opt}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    jobs = []
    for d in a.datasets_dir:
        for ds in sorted(p for p in d.iterdir() if p.is_dir() and (p / "optimal_result.json").exists()):
            jobs.append({"ds_dir": str(ds)})
    if a.limit:
        jobs = jobs[: a.limit]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(_one, jobs, chunksize=4)
    out = {r.pop("id"): r for r in rows}
    ties = sum(1 for r in out.values() if len(r["plans"]) > 1)
    at_opt = sum(1 for r in out.values() if abs(r["cd_rtt"] - r["opt_rtt"]) <= 1e-9 * max(1.0, r["opt_rtt"]))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w"))
    print(f"{len(out)} groups; tied CD plans on {ties}; CD at the sweep optimum on {at_opt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
