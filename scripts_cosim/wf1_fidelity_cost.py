#!/usr/bin/env python3
"""r1_attribution_v1 preparation: what does a co-sim plan cost when the episode runs the autoscaler I11 validated?

For each warm dataset (make_warm_corpus output) take its snapshot (matched by task ids and instant in a capture made
with HEROSIM_SNAPSHOT_FIDELITY=1), and replay plans of its sweep the way physics_audit/i11_replay.py replays a live
decision: the fidelity workload (queued + batch tasks), the live autoscaler settings (`live_run_params`: KPA target 0.7,
1 s x time-scale ticks, keep-alive), the KPA window history and tick phase from the snapshot. Reports per plan the wall
seconds, the peak RSS of the worker (one fresh process per plan), the size of the returned stats, the simulated horizon
(`endTime`), the scale-downs, and any failure; and per dataset the share of plans that fail.

  wf1_fidelity_cost.py --datasets <gnn_datasets dir> --snapshots a.jsonl b.jsonl --sample 60 --out cost.jsonl

This measures cost and completeness. It does not produce labels: a fidelity replay's task set (queued tasks included) is
not the dataset's batch-only workload, and which latency is the label is a decision this script does not take.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import resource
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _read_snapshots(paths: List[str]) -> Dict[Tuple[Tuple[int, ...], float], Dict[str, Any]]:
    out = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    s = json.loads(line)
                    out[(tuple(int(t["task_id"]) for t in s["tasks"]), round(float(s["time"]), 6))] = s
    return out


def _one_plan(job: Dict[str, Any]) -> Dict[str, Any]:
    """One replay in a fresh process (maxtasksperchild=1), so ru_maxrss belongs to this plan."""
    import contextlib
    import io

    t_import = time.perf_counter()
    from src.executecosimulation import rtt_from_stats  # noqa: F401  (logging set-up order, as i11_replay)
    from src.executesimulation import execute_simulation, load_simulation_inputs, prepare_infrastructure_for_real_simulation
    from src.placement import snapshot_fidelity
    from src.placement.live_snapshot_seed import build_live_snapshot_seed

    base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    row: Dict[str, Any] = {"dataset": job["dataset"], "plan_index": job["plan_index"]}
    t0 = time.perf_counter()
    try:
        snap = job["snapshot"]
        fid = snap["fidelity"]
        wl, forced, ids = snapshot_fidelity.replay_workload(fid)
        batch_local = [ids[int(r["gid"])] for r in fid["batch"]]
        forced = dict(forced)
        forced.update({batch_local[i]: tuple(p) for i, p in job["plan"].items()})
        infra = deepcopy(job["base_infra"])
        infra["live_snapshot_seed"] = build_live_snapshot_seed(snap)
        infra["forced_placements"] = forced
        infra["fast_forward_warmup"] = True
        infra["fast_forward_threshold"] = 1
        infra["scheduler"] = {"batch_size": max(len(wl["events"]), 1), "batch_timeout": 0.02}
        kw = snapshot_fidelity.live_run_params()
        sim_inputs = load_simulation_inputs(Path(job["sim_input"]))
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            res = execute_simulation({"infrastructure": infra, "workload": wl}, sim_inputs,
                                     scheduling_strategy="determined_determined", cache_policy="fifo",
                                     task_priority="fifo", **kw)
        stats = res["stats"]
        trs = {tr["taskId"]: tr for tr in stats["taskResults"] if tr.get("taskId", -1) >= 0}
        row["batch_latency"] = sum(trs[i]["doneTime"] - trs[i]["scheduledTime"] for i in batch_local)
        row["end_time"] = float(stats.get("endTime") or 0.0)
        row["n_tasks"] = len(wl["events"])
        row["scale_downs"] = (stats.get("scaleOut") or {}).get("scale_downs")
        row["target"] = (stats.get("scaleOut") or {}).get("target")
        row["stats_mb"] = len(json.dumps(stats, default=str)) / 1e6
        row["captured_output_mb"] = len(sink.getvalue()) / 1e6
    except (Exception, SystemExit) as exc:  # a failed replay is a result, recorded by name
        row["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
    row["seconds"] = time.perf_counter() - t0
    row["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    row["rss_over_import_mb"] = row["rss_mb"] - base_rss
    row["import_seconds"] = t0 - t_import
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=Path, required=True)
    ap.add_argument("--snapshots", nargs="+", required=True, help="captures made with HEROSIM_SNAPSHOT_FIDELITY=1")
    ap.add_argument("--only", default="", help="comma list of dataset ids (default: all)")
    ap.add_argument("--sample", type=int, default=60, help="plans per dataset (0 = every plan)")
    ap.add_argument("--also-all", default="", help="comma list of dataset ids replayed in full regardless of --sample")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sim-input", default=str(REPO / "data" / "nofs-ids"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get("HEROSIM_SNAPSHOT_FIDELITY") != "1":
        raise SystemExit("FAIL LOUD: export HEROSIM_SNAPSHOT_FIDELITY=1 (and the R1 environment) for the replay")

    from src.executesimulation import prepare_infrastructure_for_real_simulation

    caps = _read_snapshots(a.snapshots)
    want = {x for x in a.only.split(",") if x}
    full = {x for x in a.also_all.split(",") if x}
    jobs: List[Dict[str, Any]] = []
    summary_meta: Dict[str, Any] = {}
    rng = random.Random(7)
    for d in sorted(a.datasets.glob("ds_*")):
        if want and d.name not in want:
            continue
        ws = json.loads((d / "warm_snapshot.json").read_text())
        old, prov = ws["snapshot"], ws["provenance"]
        key = (tuple(int(t["task_id"]) for t in old["tasks"]), round(float(old["time"]), 6))
        new = caps.get(key)
        if new is None or new.get("fidelity") is None:
            summary_meta[d.name] = {"error": f"no fidelity snapshot for task ids {key[0]} at t={key[1]}"}
            continue
        snap = dict(old, fidelity=new["fidelity"])
        cfg = json.loads(Path(prov["cell_config"]).read_text())
        base_infra = prepare_infrastructure_for_real_simulation(cfg, seed=None, sim_input_path=Path(a.sim_input))
        rows = [json.loads(l) for l in open(d / "placements" / "placements.jsonl") if l.strip()]
        plans = [{int(k): tuple(int(x) for x in v) for k, v in r["placement_plan"].items()} for r in rows]
        idx = list(range(len(plans)))
        if a.sample and len(idx) > a.sample and d.name not in full:
            idx = sorted(rng.sample(idx, a.sample))
        summary_meta[d.name] = {"plans_in_sweep": len(plans), "plans_replayed": len(idx),
                                "queued_tasks": len(new["fidelity"]["queued"]), "batch_tasks": len(new["fidelity"]["batch"])}
        for i in idx:
            jobs.append({"dataset": d.name, "plan_index": i, "plan": plans[i], "snapshot": snap,
                         "base_infra": base_infra, "sim_input": a.sim_input})
    print(f"{len(jobs)} plan replays over {len(summary_meta)} datasets, {a.workers} workers", flush=True)
    ctx = mp.get_context("spawn")
    results: List[Dict[str, Any]] = []
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, "w") as fh:
        for row in pool.imap_unordered(_one_plan, jobs, chunksize=1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            results.append(row)
    import statistics as st

    print("dataset: replays, failed, seconds median/p95/max, rss MB median/max, stats MB median, horizon s median/max, scale_downs>0")
    for name, meta in summary_meta.items():
        rs = [r for r in results if r["dataset"] == name]
        if not rs:
            print(f"  {name}: {meta}")
            continue
        ok = [r for r in rs if "error" not in r]
        sec = sorted(r["seconds"] for r in rs)
        p95 = sec[min(len(sec) - 1, int(0.95 * len(sec)))]
        line = (f"  {name}: {len(rs)} plans of {meta['plans_in_sweep']}, failed {len(rs) - len(ok)}, "
                f"{st.median(sec):.1f}/{p95:.1f}/{sec[-1]:.1f} s, rss {st.median(r['rss_mb'] for r in rs):.0f}/"
                f"{max(r['rss_mb'] for r in rs):.0f} MB")
        if ok:
            line += (f", stats {st.median(r['stats_mb'] for r in ok):.1f} MB, horizon {st.median(r['end_time'] for r in ok):.0f}/"
                     f"{max(r['end_time'] for r in ok):.0f} s, scale_downs>0 in {sum(1 for r in ok if (r['scale_downs'] or 0) > 0)}")
        print(line + f"  [queued {meta['queued_tasks']}, batch {meta['batch_tasks']}]")
        errs = {}
        for r in rs:
            if "error" in r:
                errs[r["error"][:110]] = errs.get(r["error"][:110], 0) + 1
        for e, n in sorted(errs.items(), key=lambda x: -x[1])[:4]:
            print(f"      {n} x {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
