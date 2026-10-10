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

  i11_replay.py --cfg CFG --trace T --snapshots S --out OUT.jsonl [--n 40] [--targeted 0] [--params oracle|live]

--params oracle  passes keep_alive=KEEP_ALIVE and queue_length=QUEUE_LENGTH as live_snapshot_cosim_oracle does.
--params live    passes the values the live driver resolves (target concurrency, scaled keep-alive and tick).
"""
from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.executecosimulation import rtt_from_stats  # noqa: F401  (import order: sets up logging like the oracle)
from src.executesimulation import (
    execute_simulation,
    load_simulation_inputs,
    prepare_infrastructure_for_real_simulation,
)
from src.placement import snapshot_fidelity
from src.placement.constants import KEEP_ALIVE, QUEUE_LENGTH
from src.placement.live_snapshot_seed import build_live_snapshot_seed


def read_jsonl(path: str, kinds: Optional[set] = None) -> List[Dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            if kinds is not None and not any(line.startswith(f'{{"k":"{k}"') for k in kinds):
                continue
            out.append(json.loads(line))
    return out


def live_future(trace_path: str) -> Dict[int, Dict[str, Any]]:
    """task id -> {scheduled, node} for every task the live run placed, from its audit trace."""
    rows = read_jsonl(trace_path, {"svc"})
    return {r["task"]: {"scheduled": r["scheduled"], "node": r["q"].rsplit(":", 1)[0]} for r in rows}


def isolated_truth(cfg: str, workload: str, policy: str, batch: List[int], decision_t: float, arrivals: List[float],
                   tmp_dir: Path, workers_env: Dict[str, str], repo: Path) -> Dict[str, Any]:
    """The live run, cut at the decision: the events that arrived by `decision_t` and none after. The simulator is
    deterministic, so it follows the full run exactly up to the decision and then executes the chosen plan with
    nothing else arriving. That is "the same plan executed live from that state". (Cut by arrival time, not by
    task id: peer groups interleave, so a higher id may have arrived, and been placed, before this decision.)
    Returns per-task scheduled/done and where each ran."""
    n = bisect.bisect_right(arrivals, decision_t)
    out = tmp_dir / f"truth_{n}.json"
    if not out.exists():
        env = dict(workers_env, HEROSIM_MAX_EVENTS=str(n))
        for k in ("HEROSIM_AUDIT_TRACE", "LIVE_AUDIT_SNAPSHOT_PATH"):
            env.pop(k, None)
        proc = subprocess.run([sys.executable, str(repo / "src/executesimulation.py"), "--config", cfg,
                               "--workload", workload, "--policy", policy, "--output", str(out)],
                              env=env, cwd=str(repo), capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"isolated live run failed (rc {proc.returncode}): {proc.stderr[-300:]}")
    res = json.load(open(out))
    trs = {tr["taskId"]: tr for tr in res["stats"]["taskResults"] if tr.get("taskId", -1) >= 0}
    return {t: {"scheduled": trs[t]["scheduledTime"], "done": trs[t]["doneTime"],
                "node": trs[t]["executionNode"], "platform": str(trs[t]["executionPlatform"])} for t in batch}


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
        if any(x["t"] == r["t"] and x["batch"] == batch for x in states[-3:]):
            continue  # the same decision probed twice
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
    if i11["carried"]["queued_on_replicas"]:
        tags.append("queued")
    return tags or ["clean"]


# hardest first: the targeted picks cycle through these so each is represented
TARGET_ORDER = ("links_busy", "queued", "uninit_replica", "inflight", "kpa_panic")


def skey(st: Dict[str, Any]) -> Tuple[Tuple[int, ...], float]:
    """A decision: the batch's task ids and the instant (a deferred member makes the scheduler decide again)."""
    return (tuple(st["batch"]), round(float(st["t"]), 6))


def select_states(states: List[Dict[str, Any]], n_uniform: int, n_targeted: int) -> List[Dict[str, Any]]:
    """`n_uniform` evenly spaced over all replayable states, then `n_targeted` more drawn round-robin from the
    states carrying each hard-state tag (evenly spaced within a tag), skipping states already chosen."""
    picked: Dict[Tuple[Tuple[int, ...], float], str] = {}
    if n_uniform:
        step = max(1, len(states) // n_uniform)
        for st in states[::step][:n_uniform]:
            picked[skey(st)] = "uniform"
    pools = {tag: [st for st in states if tag in bucket_of(st["i11"])] for tag in TARGET_ORDER}
    cursor = {tag: 0 for tag in TARGET_ORDER}
    added = 0
    while added < n_targeted and any(pools.values()):
        progressed = False
        for tag in TARGET_ORDER:
            pool = pools[tag]
            while cursor[tag] < len(pool) and skey(pool[cursor[tag]]) in picked:
                cursor[tag] += 1
            if cursor[tag] < len(pool) and added < n_targeted:
                picked[skey(pool[cursor[tag]])] = "targeted"
                # spread within the pool: jump ahead by its stride
                cursor[tag] += max(1, len(pool) // max(1, n_targeted // len(TARGET_ORDER) + 1))
                added += 1
                progressed = True
        if not progressed:
            break
    return [dict(st, stratum=picked[skey(st)]) for st in states if skey(st) in picked]


def summarize(results: List[Dict[str, Any]]) -> None:
    """A replay that raised is a miss with infinite error -- it stays in every denominator."""
    def err(r: Dict[str, Any]) -> float:
        return abs(r["rel_err"]) if r.get("rel_err") is not None else math.inf

    def line(name: str, rs: List[Dict[str, Any]]) -> None:
        if not rs:
            return
        e = sorted(err(r) for r in rs)
        failed = sum(1 for r in rs if r.get("rel_err") is None)
        within1 = sum(1 for x in e if x <= 0.01) / len(e)
        within5 = sum(1 for x in e if x <= 0.05) / len(e)
        p95 = e[min(len(e) - 1, math.ceil(0.95 * len(e)) - 1)]
        print(f"{name:16s} n={len(rs):3d} failed={failed:3d} median|err|={statistics.median(e):8.2%} "
              f"p95|err|={p95:8.2%} within1%={within1:5.1%} within5%={within5:5.1%}")

    line("all", results)
    for stratum in ("uniform", "targeted", "late"):
        line("stratum:" + stratum, [r for r in results if r.get("stratum") == stratum])
    for tag in ("clean", "inflight", "queued", "uninit_replica", "kpa_panic", "links_busy"):
        line(tag, [r for r in results if tag in r["tags"]])
    line("open_peers>0", [r for r in results if r.get("open_peers")])
    line("open_peers=0", [r for r in results if not r.get("open_peers")])
    summarize_continuing(results)


def summarize_continuing(results: List[Dict[str, Any]], tol: float = 0.01) -> None:
    """The second column, reported and not scored: the replay against the live run that kept going (later arrivals
    included). Also the time of the latest state it misses by more than `tol` (labels are cut from twice that)."""
    rs = [r for r in results if r.get("rel_err_continuing") is not None]
    if not rs:
        return
    e = sorted(abs(r["rel_err_continuing"]) for r in rs)
    failed = len(results) - len(rs)
    p95 = e[min(len(e) - 1, math.ceil(0.95 * len(e)) - 1)]
    miss = [r for r in rs if abs(r["rel_err_continuing"]) > tol]
    print(f"continuing-run reference (reported, not scored): n={len(results)} unreplayed={failed} "
          f"median|err|={statistics.median(e):.2%} p95|err|={p95:.2%} max|err|={e[-1]:.2%} "
          f"misses>{tol:.0%}: {len(miss)}; latest miss at t={max((r['t'] for r in miss), default=float('nan')):.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg")
    ap.add_argument("--workload", help="the workload the trace was made on (for the isolated live runs)")
    ap.add_argument("--trace")
    ap.add_argument("--snapshots")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=40, help="uniformly spaced states")
    ap.add_argument("--tmin", type=float, default=0.0, help="uniform states are drawn only from decisions at t >= tmin")
    ap.add_argument("--late-after", type=int, default=0, help="arrival index (batch task id) from which the --late-n states are drawn; "
                    "the uniform states are then drawn from before it")
    ap.add_argument("--late-n", type=int, default=0, help="extra uniformly spaced states with every batch task id >= --late-after")
    ap.add_argument("--targeted", type=int, default=0, help="extra states drawn from the hard-state tags")
    ap.add_argument("--summarize", default="", help="summarize existing result jsonl files (comma list) and exit")
    ap.add_argument("--only", default="", help="comma list of batch[0] task ids to replay instead of sampling")
    ap.add_argument("--params", choices=("oracle", "live"), default="oracle")
    ap.add_argument("--sim-input", default="data/nofs-ids")
    ap.add_argument("--truth", choices=("isolated", "continuing"), default="isolated",
                    help="isolated: the live run truncated after the batch (the label's meaning); continuing: the "
                         "full live run, later arrivals included")
    ap.add_argument("--replay", choices=("fidelity", "original"), default="fidelity",
                    help="original: the replay as live_snapshot_cosim_oracle builds it (batch-only workload, no fidelity "
                         "block, KEEP_ALIVE / QUEUE_LENGTH) -- the before of the before/after")
    ap.add_argument("--policy", default="peer_greedy_network_cd")
    ap.add_argument("--workers", type=int, default=4, help="parallel isolated live runs")
    ap.add_argument("--future", choices=("live", "none"), default="live",
                    help="live: place partners the live run had not scheduled yet at the times it later did")
    ap.add_argument("--cell", default="", help="label recorded on each row")
    args = ap.parse_args()
    if args.summarize:
        rows = [json.loads(l) for f in args.summarize.split(",") for l in open(f)]
        summarize(rows)
        return 0
    missing = [a for a in ("cfg", "trace", "snapshots", "out") if not getattr(args, a)]
    if missing:
        ap.error(f"required: {', '.join('--' + m for m in missing)}")

    space_config = json.load(open(args.cfg))
    node_id_of = {n["node"]: n["id"] for n in read_jsonl(args.trace, {"header"})[0]["nodes"]}
    sim_inputs = load_simulation_inputs(Path(args.sim_input))
    base_infra = prepare_infrastructure_for_real_simulation(space_config, seed=None, sim_input_path=Path(args.sim_input))
    # a batch can be decided more than once (a deferred member is retried), so key a snapshot by decision instant too
    snaps = {(tuple(t["task_id"] for t in s["tasks"]), round(float(s["time"]), 6)): s
             for s in read_jsonl(args.snapshots)}
    future = live_future(args.trace)
    states = [s for s in live_states(args.trace) if skey(s) in snaps and len(snaps[skey(s)]["tasks"]) == len(s["batch"])]
    if args.only:
        want = {int(x) for x in args.only.split(",")}
        chosen = [dict(s, stratum="only") for s in states if s["batch"][0] in want]
    else:
        early = [s for s in states if s["t"] >= args.tmin and (not args.late_after or max(s["batch"]) < args.late_after)]
        chosen = select_states(early, args.n, args.targeted)
        if args.late_n:
            late = [s for s in states if min(s["batch"]) >= args.late_after]
            step = max(1, len(late) // args.late_n)
            chosen += [dict(s, stratum="late") for s in late[::step][:args.late_n]]
            print(f"{len(early)} early and {len(late)} late (task id >= {args.late_after}) replayable states", flush=True)
    print(f"{len(states)} replayable states, replaying {len(chosen)} (params={args.params})", flush=True)

    if args.replay == "original":
        args.params = "oracle"
        from scripts_cosim.live_snapshot_cosim_oracle import build_workload_from_snapshot

        # the oracle context defaults to seed=101, which builds a different topology from the live cell's (a 101-seeded
        # replay of cell 9483 has 316 routes against the live 328, and the simulator exits on a missing connection);
        # the live run's seed (None: the config's own) is used so the comparison isolates what the snapshot lacks
        base_infra = prepare_infrastructure_for_real_simulation(space_config, seed=None, sim_input_path=Path(args.sim_input))
    if args.params == "live":
        kw = snapshot_fidelity.live_run_params()
    else:
        kw = dict(keep_alive=KEEP_ALIVE, queue_length=QUEUE_LENGTH)

    truths: Dict[Any, Dict[str, Any]] = {}
    truth_error: Dict[Any, str] = {}
    if args.truth == "isolated":
        if not args.workload:
            ap.error("--truth isolated needs --workload")
        tmp_dir = Path(args.out).parent / "truth"
        tmp_dir.mkdir(exist_ok=True)
        repo = Path(__file__).resolve().parents[2]
        arrivals = [float(e["timestamp"]) for e in json.load(open(args.workload))["events"]]
        run_env = dict(os.environ, GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1")
        from concurrent.futures import ThreadPoolExecutor

        def one(st: Dict[str, Any]):
            try:
                return skey(st), isolated_truth(args.cfg, args.workload, args.policy, st["batch"], st["t"],
                                                      arrivals, tmp_dir, run_env, repo), None
            except Exception as exc:  # noqa: BLE001 -- recorded on the row, never swallowed
                return skey(st), None, f"{type(exc).__name__}: {str(exc)[:200]}"

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for key, truth, err in pool.map(one, chosen):
                if truth is not None:
                    truths[key] = truth
                else:
                    truth_error[key] = err

    results = []
    with open(args.out, "w") as fout:
        for st in chosen:
            snap = snaps[skey(st)]
            if args.replay == "original":
                # the oracle's own construction: the batch alone, from the snapshot's task list
                snap = {k: v for k, v in snap.items() if k != "fidelity"}
                wl = build_workload_from_snapshot(snap["tasks"])
                batch_local = list(range(len(st["batch"])))
                forced = {i: tuple(p) for i, p in st["plan"].items()}
                fid = {"queued": [], "peers": {}}
                open_peers = 0
            else:
                fid = snap["fidelity"]
                open_peers = sum(1 for rows in fid["peers"].values() for r in rows if r[1] is None)
                if args.future == "live" and open_peers:
                    fid = dict(fid, future={
                        str(r[0]): [future[r[0]]["scheduled"] - st["t"], future[r[0]]["node"]]
                        for rows in fid["peers"].values() for r in rows if r[1] is None and r[0] in future})
                    snap = dict(snap, fidelity=fid)
                wl, forced, ids = snapshot_fidelity.replay_workload(fid)
                batch_local = [ids[g] for g in st["batch"]]
                forced.update({batch_local[i]: tuple(p) for i, p in st["plan"].items()})
            infra = deepcopy(base_infra)
            infra["live_snapshot_seed"] = build_live_snapshot_seed(snap)
            infra["forced_placements"] = forced
            infra["fast_forward_warmup"] = True
            infra["fast_forward_threshold"] = 1
            infra["scheduler"] = {"batch_size": max(len(wl["events"]), 1), "batch_timeout": 0.02,
                               "exact_batch": os.environ.get("HEROSIM_REPLAY_EXACT_BATCH", "1") == "1"}
            row = {"t": st["t"], "batch": st["batch"], "tags": bucket_of(st["i11"]),
                   "stratum": st.get("stratum", "only"), "cell": args.cell, "n_queued": len(fid["queued"]),
                   "open_peers": open_peers,
                   "live_continuing": sum(st["live_latency"]), "params": args.params, "truth": args.truth}
            if args.truth == "isolated":
                truth = truths.get(skey(st))
                if truth is None:
                    row["error"] = "no isolated truth: " + truth_error.get(skey(st), "unknown")
                    fout.write(json.dumps(row) + "\n")
                    results.append(row)
                    continue
                moved = [t for i, t in enumerate(st["batch"])
                         if (node_id_of.get(truth[t]["node"]), int(truth[t]["platform"])) != tuple(st["plan"][i])]
                if moved:
                    row["error"] = f"isolated live run chose a different plan for tasks {moved[:3]}"
                    fout.write(json.dumps(row) + "\n")
                    results.append(row)
                    continue
                late = [t for t in st["batch"] if abs(truth[t]["scheduled"] - st["t"]) > 1e-6]
                if late:
                    row["error"] = f"isolated live run scheduled tasks {late[:3]} at a different instant"
                    fout.write(json.dumps(row) + "\n")
                    results.append(row)
                    continue
                row["live"] = sum(truth[t]["done"] - truth[t]["scheduled"] for t in st["batch"])
                row["truth_each"] = [round(truth[t]["done"] - truth[t]["scheduled"], 4) for t in st["batch"]]
            else:
                row["live"] = row["live_continuing"]
            try:
                quiet = (redirect_stdout(StringIO()), redirect_stderr(StringIO())) if not os.environ.get("I11_VERBOSE") \
                    else (contextlib.nullcontext(), contextlib.nullcontext())
                with quiet[0], quiet[1]:
                    res = execute_simulation({"infrastructure": infra, "workload": wl}, sim_inputs,
                                             scheduling_strategy="determined_determined", cache_policy="fifo",
                                             task_priority="fifo", **kw)
                stats = res["stats"]
                trs = {tr["taskId"]: tr for tr in stats["taskResults"] if tr.get("taskId", -1) >= 0}
                row["replay"] = sum(trs[i]["doneTime"] - trs[i]["scheduledTime"] for i in batch_local)
                row["replay_each"] = [round(trs[i]["doneTime"] - trs[i]["scheduledTime"], 4) for i in batch_local]
                if os.environ.get("I11_DUMP"):
                    keep = ("scheduledTime", "arrivedTime", "startedTime", "doneTime", "coldStartTime", "coldStarted",
                            "peerExchangeTime", "peerRendezvousWait", "linkTransferTime", "linkWaitTime", "executionNode",
                            "executionPlatform", "networkLatency")
                    row["replay_tasks"] = [{k: trs[i].get(k) for k in keep} for i in batch_local]
                if os.environ.get("I11_SCALE"):
                    # accel_replica_v1: the replicas this replay created and the platform mix it ended with (opt-in; the default row is unchanged)
                    row["replay_scale_ups"] = [[round(e["timestamp"], 3), e["name"], e.get("cause")] for e in stats.get("scaleEvents", []) if e.get("action") == "up"]
                    last = {}
                    for e in stats.get("systemEvents", []):
                        last[e["name"]] = {k: v for k, v in e.items() if k in ("xavierGpu", "xavierDla", "xavierCpu", "rpiCpu", "pynqFpga", "count")}
                    row["replay_replicas_end"] = last
                row["scaleout_target"] = (stats.get("scaleOut") or {}).get("target")
                row["replay_cold"] = sum(bool(trs[i]["coldStarted"]) for i in batch_local)
            except (Exception, SystemExit) as exc:  # a failed replay (or a sys.exit inside the simulator) is a result, recorded by name
                row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            if "replay" in row:
                row["rel_err"] = (row["replay"] - row["live"]) / row["live"] if row["live"] > 0 else None
                row["rel_err_continuing"] = ((row["replay"] - row["live_continuing"]) / row["live_continuing"]
                                             if row["live_continuing"] > 0 else None)
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            results.append(row)

    summarize(results)
    print("I11 bar: median <= 1 %, p95 <= 5 %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
