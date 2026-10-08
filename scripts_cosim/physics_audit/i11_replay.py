#!/usr/bin/env python3
"""physics_audit_v1 I11 -- does co-simulation reproduce live latency from a captured state?

For each live decision with a snapshot: take the plan the live run executed (from the audit trace), replay it from
the snapshot through the co-sim path (`apply_live_snapshot_seed` + `determined_determined` + forced placements),
and compare the batch's post-decision latency (sum of done - scheduled over the batch's tasks) with what the live
run measured for the same tasks. Reports median and p95 absolute relative error overall and split by which
unrepresented state was present at the decision (the trace's `i11` row).

Inputs come from ONE live run made with
  HEROSIM_AUDIT_TRACE=<trace> LIVE_AUDIT_SNAPSHOT_PATH=<snap.jsonl> LIVE_AUDIT_MIN_BATCH_SIZE=1
  LIVE_AUDIT_MIN_CANDIDATES=1 LIVE_AUDIT_MAX_SNAPSHOTS=100000
plus the same environment (R1 flags, HEROSIM_PEER_EXCHANGE=1, ...) exported for this script.

  i11_replay.py --cfg CFG --workload WL --trace T --snapshots S --out OUT.jsonl [--n 40] [--params oracle|live]

--params oracle  passes keep_alive=KEEP_ALIVE and queue_length=QUEUE_LENGTH as live_snapshot_cosim_oracle does.
--params live    passes the values the live driver resolves (target concurrency, scaled keep-alive and tick).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.executecosimulation import rtt_from_stats  # noqa: F401  (import order: sets up logging like the oracle)
from src.executesimulation import (
    _resolve_keep_alive,
    _resolve_queue_length,
    execute_simulation,
    load_simulation_inputs,
    prepare_infrastructure_for_real_simulation,
)
from src.placement.constants import KEEP_ALIVE, QUEUE_LENGTH, RECONCILE_INTERVAL
from src.placement.live_snapshot_seed import build_live_snapshot_seed
from src.placement.scaleout import policy_time_scale


def read_jsonl(path: str, kinds: Optional[set] = None) -> List[Dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            if kinds is not None and not any(line.startswith(f'{{"k":"{k}"') for k in kinds):
                continue
            out.append(json.loads(line))
    return out


def live_states(trace_path: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(trace_path, {"header", "svc", "i11"})
    header = next(r for r in rows if r["k"] == "header")
    node_id = {n["node"]: n["id"] for n in header["nodes"]}
    svc = {r["task"]: r for r in rows if r["k"] == "svc"}
    states = []
    for r in rows:
        if r["k"] != "i11":
            continue
        batch = r["batch"]
        if not all(t in svc for t in batch):
            continue
        sv = [svc[t] for t in batch]
        if any(abs(s["scheduled"] - r["t"]) > 1e-6 for s in sv):
            continue  # a task deferred past this decision: its latency is not this plan's
        plan = {}
        for i, s in enumerate(sv):
            node, plat = s["q"].rsplit(":", 1)
            plan[i] = (int(node_id[node]), int(plat))
        states.append({"t": r["t"], "batch": batch, "plan": plan, "i11": r,
                       "live_latency": [s["done"] - s["scheduled"] for s in sv]})
    return states


def bucket_of(i11: Dict[str, Any]) -> List[str]:
    m = i11["missing"]
    tags = []
    if m["inflight_tasks"] or m["admitted_in_ingress"]:
        tags.append("inflight")
    if m["uninit_replicas"]:
        tags.append("uninit_replica")
    if any(f["panicking"] for f in (m["kpa"] or {"functions": {}})["functions"].values()):
        tags.append("kpa_panic")
    if m["links_busy"]:
        tags.append("links_busy")
    return tags or ["clean"]


def build_workload(batch: List[int], trace: Dict[str, Any]) -> Dict[str, Any]:
    events = []
    for gid in batch:
        ev = trace["events"][gid]
        events.append({"timestamp": 0.0, "application": deepcopy(ev["application"]),
                       "qos": deepcopy(ev.get("qos") or {"name": "medium", "maxDurationDeviation": 15}),
                       "node_name": str(ev["node_name"])})
    local = {gid: i for i, gid in enumerate(batch)}
    pairs, outside = [], 0
    for i, j, payload in trace.get("peer_exchange") or []:
        i, j = int(i), int(j)
        if i in local and j in local:
            pairs.append([local[i], local[j], float(payload)])
        elif i in local or j in local:
            outside += 1
    wl = {"rps": len(events), "duration": 1, "events": events}
    if pairs:
        wl["peer_exchange"] = pairs
    return wl, outside


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--only", default="", help="comma list of batch[0] task ids to replay instead of sampling")
    ap.add_argument("--params", choices=("oracle", "live"), default="oracle")
    ap.add_argument("--sim-input", default="data/nofs-ids")
    args = ap.parse_args()

    space_config = json.load(open(args.cfg))
    sim_inputs = load_simulation_inputs(Path(args.sim_input))
    base_infra = prepare_infrastructure_for_real_simulation(space_config, seed=None, sim_input_path=Path(args.sim_input))
    trace = json.load(open(args.workload))
    snaps = {s["trigger_task_id"]: s for s in read_jsonl(args.snapshots)}
    states = [s for s in live_states(args.trace) if s["batch"][0] in snaps
              and len(snaps[s["batch"][0]]["tasks"]) == len(s["batch"])]
    if args.only:
        want = {int(x) for x in args.only.split(",")}
        chosen = [s for s in states if s["batch"][0] in want]
    else:
        step = max(1, len(states) // args.n)
        chosen = states[::step][: args.n]
    print(f"{len(states)} replayable states, replaying {len(chosen)} (params={args.params})", flush=True)

    ts = policy_time_scale()
    if args.params == "live":
        kw = dict(keep_alive=_resolve_keep_alive(ts), queue_length=_resolve_queue_length(None),
                  reconcile_interval=RECONCILE_INTERVAL if ts == 1.0 else RECONCILE_INTERVAL * ts)
    else:
        kw = dict(keep_alive=KEEP_ALIVE, queue_length=QUEUE_LENGTH)

    results = []
    with open(args.out, "w") as fout:
        for st in chosen:
            snap = snaps[st["batch"][0]]
            wl, outside = build_workload(st["batch"], trace)
            infra = deepcopy(base_infra)
            infra["live_snapshot_seed"] = build_live_snapshot_seed(snap)
            infra["forced_placements"] = {i: tuple(p) for i, p in st["plan"].items()}
            infra["fast_forward_warmup"] = True
            infra["fast_forward_threshold"] = 1
            infra["scheduler"] = {"batch_size": max(len(st["batch"]), 1), "batch_timeout": 0.02}
            row = {"t": st["t"], "batch": st["batch"], "tags": bucket_of(st["i11"]), "outside_pairs": outside,
                   "live": sum(st["live_latency"]), "params": args.params}
            try:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    res = execute_simulation({"infrastructure": infra, "workload": wl}, sim_inputs,
                                             scheduling_strategy="determined_determined", cache_policy="fifo",
                                             task_priority="fifo", **kw)
                stats = res["stats"]
                trs = {tr["taskId"]: tr for tr in stats["taskResults"] if tr.get("taskId", -1) >= 0}
                row["replay"] = sum(trs[i]["doneTime"] - trs[i]["scheduledTime"] for i in range(len(st["batch"])))
                row["scaleout_target"] = (stats.get("scaleOut") or {}).get("target")
                row["replay_cold"] = sum(bool(trs[i]["coldStarted"]) for i in range(len(st["batch"])))
            except Exception as exc:  # a failed replay is a result, recorded by name
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            if "replay" in row:
                row["rel_err"] = (row["replay"] - row["live"]) / row["live"] if row["live"] > 0 else None
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            results.append(row)

    ok = [r for r in results if r.get("rel_err") is not None]
    print(f"replayed {len(results)}: ok {len(ok)}, failed {len(results) - len(ok)}")
    def summ(name: str, rs: List[Dict[str, Any]]) -> None:
        if not rs:
            return
        a = sorted(abs(r["rel_err"]) for r in rs)
        signed = statistics.median(r["rel_err"] for r in rs)
        print(f"{name:16s} n={len(rs):3d} median|err|={statistics.median(a):7.2%} p95|err|={a[min(len(a)-1, int(.95*len(a)))]:7.2%}"
              f" median signed={signed:+7.2%}")
    summ("all", ok)
    for tag in ("clean", "inflight", "uninit_replica", "kpa_panic", "links_busy"):
        summ(tag, [r for r in ok if tag in r["tags"]])
    for r in results:
        if "error" in r:
            print("FAILED", r["batch"][:3], r["error"])
    print("I11 bar: median <= 1 %, p95 <= 5 %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
