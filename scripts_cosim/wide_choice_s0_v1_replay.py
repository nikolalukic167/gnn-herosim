#!/usr/bin/env python3
"""wide_choice_s0_v1 (docs/lineages/wide_choice_s0_v1.md): replay CD, the 1-pass greedy and the
`xs1load` checkpoints (one-pass, self-refine, GNN-seeded CD) through the real engine on every
wide-slate co-sim group, against the group's exhaustive sweep optimum.

The rule arms are `joint_burst_v1_offline_rule_regret`'s path (it reproduced cd_gap_v1 D0's
8.21 %); the learned arms are `peer_affinity_live_serve_check`'s live-serving path (bit-identical
to the offline decode at P1), under the same physics env as the rules.

Usage (datalab, micromamba gnn):
  PYTHONPATH=. python3 scripts_cosim/wide_choice_s0_v1_replay.py --datasets-dir <dir> \
      --checkpoints models/exchange-seconds-v1-xs1load-lr2e3-seed{1,2,3,4}.pt --output rows.json
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts_cosim.joint_burst_v1_offline_rule_regret as R  # noqa: E402

RULE_ARMS = {k: R.ARMS[k] for k in ("cd_greedy", "batched_greedy")}
# mode -> env; every other mode knob is removed before each run
GNN_MODES = {"onepass": {}, "selfref": {"GNN_PREFIX_SELF_REFINE": "3"}, "cdapply": {"GNN_CD_REFINE": "apply"}}
MODE_KNOBS = ("GNN_PREFIX_SELF_REFINE", "GNN_CD_REFINE")
SERVE_ENV = {"GNN_PREFIX_ALPHA_KEY": "inf", "HEROSIM_INFLIGHT_CAPTURE": "service_end_v1"}
NEAR_FIVE = 0.05

_MODELS: Dict[str, Any] = {}


def _sweep(ds_dir: Path) -> Dict[str, float]:
    rtts: List[float] = []
    with open(ds_dir / "placements/placements.jsonl") as fh:
        for line in fh:
            if line.strip():
                rtts.append(float(json.loads(line)["rtt"]))
    if not rtts:
        raise RuntimeError(f"FAIL LOUD: empty sweep in {ds_dir}")
    opt = min(rtts)
    meta = json.load(open(ds_dir / "placement_metadata.json"))
    if meta.get("sweep_complete") is not True or int(meta["num_placements"]) != len(rtts):
        raise RuntimeError(f"FAIL LOUD: incomplete sweep in {ds_dir}")
    return {"n_plans": len(rtts), "optimal_rtt": opt,
            "near5_share": sum(1 for r in rtts if r <= opt * (1.0 + NEAR_FIVE)) / len(rtts)}


def _run(cfg: dict, sim_inputs: dict, strategy: str, models: Any) -> Dict[str, Any]:
    from src.executecosimulation import QUEUE_LENGTH, cosim_keep_alive, execute_simulation

    return execute_simulation(copy.deepcopy(cfg), sim_inputs, strategy, models=models, cache_policy="fifo",
                              task_priority="fifo", keep_alive=cosim_keep_alive(), queue_length=QUEUE_LENGTH)["stats"]


def _model(ckpt: str):
    if ckpt not in _MODELS:
        from src.executesimulation import load_gnn_model

        _MODELS[ckpt] = load_gnn_model(Path(ckpt), space_config=None)
    return _MODELS[ckpt]


def _one(job: Dict[str, Any]) -> Dict[str, Any]:
    R._apply_env()
    os.environ.update(SERVE_ENV)
    ds_dir = Path(job["ds_dir"])
    o = json.load(open(ds_dir / "optimal_result.json"))
    sweep = _sweep(ds_dir)
    best = float(json.load(open(ds_dir / "best.json"))["rtt"])
    if abs(best - sweep["optimal_rtt"]) > 1e-6 * max(1.0, best):
        raise RuntimeError(f"FAIL LOUD: best.json {best} != sweep min {sweep['optimal_rtt']} in {ds_dir}")
    cfg = copy.deepcopy(o["config"])
    cfg["infrastructure"].pop("forced_placements", None)
    n_tasks = len(cfg["workload"]["events"])
    out: Dict[str, Any] = {"dataset": ds_dir.name, "n_tasks": n_tasks, **sweep}

    def record(arm: str, stats: Dict[str, Any], extra: Dict[str, Any]) -> None:
        rtt = float(stats["total_rtt"])
        if int(stats.get("num_tasks") or -1) != n_tasks:
            raise RuntimeError(f"FAIL LOUD: {arm} on {ds_dir.name} ran {stats.get('num_tasks')} of {n_tasks} tasks")
        if rtt < sweep["optimal_rtt"] * (1.0 - 1e-9):
            raise RuntimeError(f"FAIL LOUD: {arm} on {ds_dir.name} beat the exhaustive optimum "
                               f"({rtt} < {sweep['optimal_rtt']}): a plan outside the slate")
        out[arm] = {"rtt": rtt, "regret_pct": 100.0 * (rtt / sweep["optimal_rtt"] - 1.0), **extra}

    for arm, (strategy, extra_env) in RULE_ARMS.items():
        for k in ("HEROSIM_PG_EXCHANGE_SCALE", "HEROSIM_PG_CD_PASSES") + MODE_KNOBS:
            os.environ.pop(k, None)
        os.environ.update(extra_env)
        record(arm, _run(cfg, o["sim_inputs"], strategy, None), {})

    for s, ckpt in enumerate(job["checkpoints"], start=1):
        model, device = _model(ckpt)
        models = {"gnn_model": model, "device": device, "task_types_data": o["sim_inputs"]["task_types"]}
        for mode, env in GNN_MODES.items():
            for k in MODE_KNOBS:
                os.environ.pop(k, None)
            os.environ.update(env)
            trace = tempfile.NamedTemporaryFile(prefix="wc_trace_", suffix=".pkl", delete=False)
            trace.close()
            os.environ["GNN_PREFIX_TRACE_PATH"] = trace.name
            try:
                stats = _run(cfg, o["sim_inputs"], "gnn_gnn", models)
                n_batches = 0
                with open(trace.name, "rb") as fh:
                    while True:
                        try:
                            pickle.load(fh)
                            n_batches += 1
                        except EOFError:
                            break
            finally:
                os.unlink(trace.name)
                os.environ.pop("GNN_PREFIX_TRACE_PATH", None)
            c = stats.get("schedulerCounters") or {}
            if mode == "selfref" and int(c.get("prefix_self_refine_batches") or 0) == 0:
                raise RuntimeError(f"FAIL LOUD: selfref instrument off on {ds_dir.name}")
            if mode == "cdapply" and int(c.get("cdr_batches") or 0) == 0:
                raise RuntimeError(f"FAIL LOUD: cdapply instrument off on {ds_dir.name}")
            record(f"xs1load_{mode}_s{s}", stats, {"n_batches": n_batches, "cdr_moved": int(c.get("cdr_moved") or 0)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    for c in args.checkpoints:
        if not (Path(c).is_file() and Path(c).with_suffix(".contract.json").is_file()):
            raise SystemExit(f"FAIL LOUD: {c} or its sidecar is missing")
    ds = sorted(p for d in args.datasets_dir for p in d.iterdir() if p.is_dir() and p.name.startswith("ds_"))
    ds = ds[: args.limit] if args.limit else ds
    if not ds:
        raise SystemExit("FAIL LOUD: no ds_* datasets")
    jobs = [{"ds_dir": str(p), "checkpoints": [str(Path(c).resolve()) for c in args.checkpoints]} for p in ds]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "env": {**R.PHYSICS_ENV, **SERVE_ENV}, "checkpoints": args.checkpoints},
              open(args.output, "w"), indent=1)
    print(f"[replay] {len(rows)} groups -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
