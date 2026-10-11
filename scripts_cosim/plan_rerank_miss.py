#!/usr/bin/env python3
"""plan_rerank_v1 miss analysis (offline, read-only): on every held-out table where argmin-S misses the label optimum, replay the argmin-S
plan (its best-label tie) and the label-optimal plan through the label's own fidelity replay (tie_components.py's pattern) and diff the
per-task components, grouped. Each replay is timed (wall seconds per plan).

  plan_rerank_miss.py --table table_<ckpt>.jsonl --out miss.jsonl [--workers N] [--all-timing K]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.getcwd())
VIEW = Path("/share/nikola.lukic/accel_prod/20261010_061132/cache_run/view")
FIELDS = ("queueTime", "waitTime", "peerExchangeTime", "peerRendezvousWait", "linkTransferTime", "linkWaitTime", "ingressTransferTime",
          "ingressWaitTime", "coldStartTime", "executionTime", "networkLatency", "pullTime", "initializationTime")
GROUPS = {"queue_fifo": ("queueTime", "waitTime", "lock_wait"), "cold": ("coldStartTime", "pullTime", "initializationTime"),
          "exchange": ("peerExchangeTime", "peerRendezvousWait"),
          "transfer": ("linkTransferTime", "linkWaitTime", "ingressTransferTime", "ingressWaitTime"),
          "exec": ("executionTime",), "latency": ("networkLatency",)}


def replayer(ds: Path):
    os.environ["HEROSIM_SNAPSHOT_FIDELITY"] = "1"
    prov = json.load(open(ds / "generation_provenance.json"))
    for k, v in (prov.get("physics_env") or {}).items():
        if k.startswith("HEROSIM_") or k in ("GNN_DECODE_MODE", "GNN_BATCH_BY_PEER_GROUP", "COSIM_AUTOSCALER_RECONCILE_INTERVAL"):
            os.environ.setdefault(k, str(v))
    from src.placement import fidelity_replay
    seed = json.load(open(ds / "infrastructure.json"))["live_snapshot_seed"]
    spec = dict(seed["fidelity_replay"])
    if not Path(spec["sim_input"]).exists():
        spec["sim_input"] = "data/nofs-ids"
    types = [next(iter(e["application"]["dag"])) for e in json.load(open(ds / "workload.json"))["events"]]
    offered = {t: [(sp["node_name"], sp["platform_id"]) for sp in specs if sp.get("candidate", True)]
               for t, specs in seed["replicas_by_type"].items()}
    return fidelity_replay.FidelityReplay(spec, types, offered)


def replay(fr, plan):
    t0 = time.perf_counter()
    res = fr.cut_to_batch(fr.run({i: tuple(p) for i, p in enumerate(plan)}))
    wall = time.perf_counter() - t0
    st = res["stats"]
    tasks = []
    for tr in st["taskResults"]:
        row = {f: float(tr.get(f) or 0.0) for f in FIELDS}
        row["elapsed"] = float(tr["doneTime"]) - float(tr["scheduledTime"])
        row["lock_wait"] = (float(tr.get("computeStartTime") or 0) - float(tr.get("ioEndTime") or 0)) if tr.get("ioEndTime") is not None else 0.0
        tasks.append(row)
    tot = {g: sum(t[f] for t in tasks for f in fs) for g, fs in GROUPS.items()}
    tot["elapsed"] = sum(t["elapsed"] for t in tasks)
    return float(st["total_rtt"]), tot, wall


def stacking(plan):
    c = Counter(tuple(p) for p in plan)
    return sum(v - 1 for v in c.values())


def one(job):
    r = job
    ds = VIEW / r["ds"]
    try:
        t0 = time.perf_counter()
        fr = replayer(ds)
        setup = time.perf_counter() - t0
        s_tie = min(r["argmin_s"], key=lambda t: t["label"])
        rs, cs, ws = replay(fr, s_tie["plan"])
        ro, co, wo = replay(fr, r["opt_plan"])
        delta = {g: cs[g] - co[g] for g in cs}
        return dict(ds=r["ds"], rung=r["rung"], queued=r["queued"], n_tasks=r["n_tasks"], n_sweep=r["n_sweep"], opt=r["opt"],
                    s_label=s_tie["label"], gap_label=s_tie["label"] - r["opt"], s_replayed=rs, opt_replayed=ro, gap_replayed=rs - ro,
                    delta=delta, s_comp=cs, opt_comp=co, s_stack=stacking(s_tie["plan"]), opt_stack=stacking(r["opt_plan"]),
                    s_platforms=len({tuple(p) for p in s_tie["plan"]}), opt_platforms=len({tuple(p) for p in r["opt_plan"]}),
                    wall_setup=setup, wall_per_plan=[ws, wo])
    except Exception as e:
        return {"ds": r["ds"], "rung": r.get("rung"), "error": f"{type(e).__name__}: {str(e)[:300]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.table)]
    miss = [r for r in rows if r["argmin_s_min"] > r["opt"] * (1 + 1e-9)]
    print(f"{len(rows)} tables, {len(miss)} argmin-S misses", flush=True)
    with Pool(a.workers) as p, open(a.out, "w") as fh:
        for res in p.imap_unordered(one, miss):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
