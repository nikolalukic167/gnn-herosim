#!/usr/bin/env python3
"""drainable_debug_v1 D2 -- read a live gate's per-task records and say where queue time came from.

Registered in `docs/lineages/drainable_debug_v1.md`. Bars are the module constants below and
were committed before any run produced a record.

The question this answers. At the drainable rung both learned arms carry ~2x the platform queue
time of reactive Knative, and `queueTime` is `arrivedTime - scheduledTime`
(`src/placement/infrastructure.py:333`) -- the wait behind other tasks on a platform that serves
one at a time (`:1240-1251`). So the excess is either (a) tasks the arm itself put on that
platform, which is a placement fact, or (b) tasks that were already there, which is a load fact.
Those imply different levers, and no gate in this program has been able to tell them apart,
because `taskResults` is empty on every run above 10,000 events unless `SIM_FORCE_FULL_STATS=1`
(the `KEEP_RAW=1` hook in the gate sbatch).

Every statistic here is a function of the retained records alone; nothing is re-simulated.

Definitions, chosen so they are exact rather than indicative:

*   **Serialization overlap.** A platform is busy with a task over `[arrivedTime, doneTime]`.
    A task's queue interval is `[scheduledTime, arrivedTime]`. The overlap of a predecessor's
    busy interval with this task's queue interval is time this task provably spent waiting for
    that predecessor. Summed over predecessors and tasks, divided by total queue time, that is
    the share of queueing this arm's own placements caused. `same_batch` restricts predecessors
    to those scheduled at the same instant -- i.e. put there by the same decode.
*   **`chosen_queue_vs_min`.** `queueSnapshotAtScheduling` holds the depth of each replica the
    task could legally have gone to, so `chosen - min` is the depth the scheduler accepted over
    the shallowest option. Tasks with one candidate had no choice and are excluded from the
    mean and counted separately: a rung where most tasks have one candidate cannot separate
    two schedulers at all, which is a finding about the cell, not the arms.
*   **Peer co-location.** Pairs come from the workload's own `peer_exchange` triples through
    the simulator's `build_peer_exchange_table`, so the pairing is the one the physics charged.
    Co-location on the same platform makes the exchange free.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.placement.orchestrator import build_peer_exchange_table  # noqa: E402

# --- bars (drainable_debug_v1, signed 2026-09-14, before any record was read) ---------------
# D2: share of an arm's EXCESS queue time (over the reactive reference) explained by its own
# same-batch placements. At or above this, the loss is self-inflicted co-location.
D2_SERIALIZATION_MIN_PCT = 50.0
# A read is VOID for an arm with fewer than this many task records.
D2_MIN_TASKS = 1000
# A share of a tiny excess is noise, not a finding, and "tiny" has to be relative: on the landed
# x200 gate the queue is ~1,334 s per task, so a 2.5 s difference is 0.19 % and its "share"
# printed as 167.6 % on the first smoke. An arm must be both this many seconds AND this much of
# the reference's queue time above it before the excess is decomposed. The learned arms at the
# drainable rung carry ~22 s over a 17.8 s reference (122 %), so neither floor touches the case
# the bar is for.
D2_MIN_EXCESS_S = 1.0
D2_MIN_EXCESS_PCT_OF_REF = 5.0
# --------------------------------------------------------------------------------------------


def _platform_key(rec: Dict[str, Any]) -> Optional[str]:
    """The key `queueSnapshotAtScheduling` and `fullQueueSnapshot` use: `<node>:<platform id>`."""
    node, plat = rec.get("executionNode"), rec.get("executionPlatform")
    if not node or plat in (None, ""):
        return None
    return f"{node}:{plat}"


def _task_type_name(rec: Dict[str, Any]) -> Optional[str]:
    """`taskType` is the whole type dict in the live schema, not a name."""
    tt = rec.get("taskType")
    if isinstance(tt, dict):
        return tt.get("name")
    return tt


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _num(rec: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """`float(rec.get(k) or 0.0)` is wrong here and the D2 tests caught it: a legitimate 0.0 is
    falsy, so a batch committed at simulated time 0 read as "no timestamp". Times in this
    schema are genuinely 0-based, so absence has to be tested for explicitly."""
    val = rec.get(key)
    return default if val is None else float(val)


def load_task_results(path: Path) -> List[Dict[str, Any]]:
    doc = json.loads(path.read_text())
    stats = doc.get("stats") or doc
    recs = stats.get("taskResults")
    if not recs:
        raise SystemExit(
            f"FAIL LOUD: {path} carries no taskResults (schema "
            f"{stats.get('statsSchemaVersion')!r}). Re-run the arm with KEEP_RAW=1, which "
            "exports SIM_FORCE_FULL_STATS=1 -- the streaming stats path writes an empty list "
            "above 10,000 events and this read would otherwise report zeros for everything."
        )
    return recs


def decomposition(recs: List[Dict[str, Any]]) -> Dict[str, float]:
    def mean(key: str) -> float:
        vals = [_num(r, key) for r in recs]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "n_tasks": len(recs),
        "elapsed": mean("elapsedTime"),
        "batch_wait": mean("waitTime"),
        "queue": mean("queueTime"),
        "initialization": mean("initializationTime"),
        "compute": mean("computeTime"),
        "peer_exchange": mean("peerExchangeTime"),
        "peer_rendezvous": mean("peerRendezvousWait"),
        "pull": mean("pullTime"),
        "cold_start": mean("coldStartTime"),
    }


def chosen_queue_vs_min(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How much deeper than the shallowest legal replica the scheduler placed."""
    deltas: List[float] = []
    no_choice = 0
    missing = 0
    for rec in recs:
        snap = rec.get("queueSnapshotAtScheduling")
        key = _platform_key(rec)
        if not isinstance(snap, dict) or not snap or key is None:
            missing += 1
            continue
        if key not in snap:
            # The chosen replica was not among the candidates recorded at scheduling: a real
            # inconsistency, not a rounding issue. Never paper over it.
            raise SystemExit(
                f"FAIL LOUD: task {rec.get('taskId')} ran on {key} but that key is absent from "
                f"its own queueSnapshotAtScheduling ({sorted(snap)[:6]}...)"
            )
        if len(snap) == 1:
            no_choice += 1
            continue
        deltas.append(float(snap[key]) - min(float(v) for v in snap.values()))
    n = len(recs)
    out: Dict[str, Any] = {
        "n_with_choice": len(deltas),
        "no_choice_pct": 100.0 * no_choice / n if n else 0.0,
        "missing_snapshot_pct": 100.0 * missing / n if n else 0.0,
    }
    if deltas:
        ordered = sorted(deltas)
        out.update(
            mean=sum(deltas) / len(deltas),
            median=st.median(deltas),
            p95=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            max=ordered[-1],
            pct_above_min=100.0 * sum(1 for d in deltas if d > 0) / len(deltas),
        )
    else:
        out["note"] = (
            f"no delta is computable: {missing} of {n} tasks carry no queueSnapshotAtScheduling "
            "(re-run with KEEP_RAW_QSNAP=1) and the rest had a single candidate replica"
            if missing
            else "no task had more than one candidate replica: at this rung the two schedulers "
                 "cannot differ by placement"
        )
    return out


def serialization(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Queue time provably spent waiting behind another task on the same platform."""
    by_platform: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in recs:
        key = _platform_key(rec)
        if key is not None:
            by_platform[key].append(rec)

    total_queue = sum(_num(r, "queueTime") for r in recs)
    overlap_all = 0.0
    overlap_same_batch = 0.0
    for tasks in by_platform.values():
        tasks.sort(key=lambda r: _num(r, "arrivedTime"))
        for idx, rec in enumerate(tasks):
            q0 = _num(rec, "scheduledTime")
            q1 = _num(rec, "arrivedTime")
            if q1 <= q0:
                continue
            # Predecessors are the tasks this platform served before this one; walking
            # backwards and stopping once a predecessor finished before this queue interval
            # began keeps the scan linear in practice on a serial platform.
            for prev in reversed(tasks[:idx]):
                p0 = _num(prev, "arrivedTime")
                p1 = _num(prev, "doneTime")
                if p1 <= q0:
                    break
                ov = _overlap(p0, p1, q0, q1)
                if ov <= 0.0:
                    continue
                overlap_all += ov
                if prev.get("scheduledTime") is not None and _num(prev, "scheduledTime") == q0:
                    overlap_same_batch += ov
    return {
        "total_queue_time": total_queue,
        "behind_any_predecessor": overlap_all,
        "behind_same_batch": overlap_same_batch,
        "share_any_pct": 100.0 * overlap_all / total_queue if total_queue else 0.0,
        "share_same_batch_pct": 100.0 * overlap_same_batch / total_queue if total_queue else 0.0,
    }


def concentration(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(k for k in (_platform_key(r) for r in recs) if k)
    total = sum(counts.values())
    if not total:
        return {"platforms_used": 0}
    hhi = sum((c / total) ** 2 for c in counts.values())
    node_counts = Counter(k.split(":")[0] for k in counts.elements())
    node_hhi = sum((c / total) ** 2 for c in node_counts.values())
    return {
        "platforms_used": len(counts),
        "nodes_used": len(node_counts),
        "platform_hhi": hhi,
        "effective_platforms": 1.0 / hhi,
        "node_hhi": node_hhi,
        "effective_nodes": 1.0 / node_hhi,
        "busiest_platform_share_pct": 100.0 * max(counts.values()) / total,
    }


def batches(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tasks sharing a `scheduledTime` were committed by one decode."""
    groups: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for rec in recs:
        if rec.get("scheduledTime") is None:
            continue  # never scheduled: it belongs to no decode
        groups[_num(rec, "scheduledTime")].append(rec)
    sizes = [len(g) for g in groups.values()]
    spread = [len({_platform_key(r) for r in g if _platform_key(r)}) for g in groups.values() if len(g) > 1]
    out = {
        "n_batches": len(groups),
        "mean_batch_size": sum(sizes) / len(sizes) if sizes else 0.0,
        "max_batch_size": max(sizes) if sizes else 0,
        "singleton_batch_pct": 100.0 * sum(1 for s in sizes if s == 1) / len(sizes) if sizes else 0.0,
    }
    if spread:
        out["mean_distinct_platforms_per_batch"] = sum(spread) / len(spread)
    return out


def peer_colocation(
    recs: List[Dict[str, Any]], peer_table: Dict[int, Dict[int, float]]
) -> Dict[str, Any]:
    placed = {
        int(r["taskId"]): (r.get("executionNode"), _platform_key(r))
        for r in recs
        if r.get("taskId") is not None and _platform_key(r)
    }
    same_platform = same_node = remote = unplaced = 0
    bytes_free = bytes_remote = 0.0
    for i, peers in peer_table.items():
        for j, payload in peers.items():
            if j <= i:
                continue
            a, b = placed.get(i), placed.get(j)
            if a is None or b is None:
                unplaced += 1
                continue
            if a[1] == b[1]:
                same_platform += 1
                bytes_free += payload
            elif a[0] == b[0]:
                same_node += 1
                bytes_free += payload
            else:
                remote += 1
                bytes_remote += payload
    total = same_platform + same_node + remote
    return {
        "pairs_scored": total,
        "pairs_unplaced": unplaced,
        "same_platform_pct": 100.0 * same_platform / total if total else 0.0,
        "same_node_pct": 100.0 * same_node / total if total else 0.0,
        "remote_pct": 100.0 * remote / total if total else 0.0,
        "bytes_free_pct": 100.0 * bytes_free / (bytes_free + bytes_remote)
        if (bytes_free + bytes_remote)
        else 0.0,
    }


def peer_group_spread(
    recs: List[Dict[str, Any]], group_of: Dict[int, int]
) -> Dict[str, Any]:
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in recs:
        tid = rec.get("taskId")
        if tid is None:
            continue
        gid = group_of.get(int(tid))
        if gid is not None:
            groups[gid].append(rec)
    spread = [len({_platform_key(r) for r in g if _platform_key(r)}) for g in groups.values() if len(g) > 1]
    nodes = [len({r.get("executionNode") for r in g}) for g in groups.values() if len(g) > 1]
    if not spread:
        return {"groups_scored": 0}
    return {
        "groups_scored": len(spread),
        "mean_platforms_per_group": sum(spread) / len(spread),
        "mean_nodes_per_group": sum(nodes) / len(nodes),
        "groups_on_one_node_pct": 100.0 * sum(1 for n in nodes if n == 1) / len(nodes),
    }


def read_arm(
    path: Path,
    peer_table: Dict[int, Dict[int, float]],
    group_of: Dict[int, int],
) -> Dict[str, Any]:
    recs = load_task_results(path)
    out: Dict[str, Any] = {
        "source": str(path),
        "decomposition": decomposition(recs),
        "chosen_queue_vs_min": chosen_queue_vs_min(recs),
        "serialization": serialization(recs),
        "concentration": concentration(recs),
        "batches": batches(recs),
    }
    if peer_table:
        out["peer_colocation"] = peer_colocation(recs, peer_table)
    if group_of:
        out["peer_group_spread"] = peer_group_spread(recs, group_of)
    if len(recs) < D2_MIN_TASKS:
        out["void"] = f"only {len(recs)} task records (< {D2_MIN_TASKS}); D2 is VOID for this arm"
    return out


def d2_verdict(arms: Dict[str, Dict[str, Any]], reference: str) -> Dict[str, Any]:
    """D2: is the learned arms' EXCESS queue time explained by their own same-batch placements?"""
    if reference not in arms:
        return {"verdict": "VOID", "why": f"reference arm {reference!r} not among {sorted(arms)}"}
    ref = arms[reference]
    ref_queue = ref["decomposition"]["queue"]
    ref_overlap_per_task = (
        ref["serialization"]["behind_same_batch"] / ref["decomposition"]["n_tasks"]
    )
    out: Dict[str, Any] = {
        "reference": reference,
        "bar_pct": D2_SERIALIZATION_MIN_PCT,
        "min_excess_s": D2_MIN_EXCESS_S,
        "min_excess_pct_of_reference": D2_MIN_EXCESS_PCT_OF_REF,
        "reference_queue_s_per_task": ref_queue,
        "arms": {},
    }
    for name, arm in arms.items():
        if name == reference:
            continue
        if arm.get("void"):
            out["arms"][name] = {"verdict": "VOID", "why": arm["void"]}
            continue
        excess = arm["decomposition"]["queue"] - ref_queue
        arm_overlap_per_task = (
            arm["serialization"]["behind_same_batch"] / arm["decomposition"]["n_tasks"]
        )
        excess_overlap = arm_overlap_per_task - ref_overlap_per_task
        material = (
            excess >= D2_MIN_EXCESS_S
            and ref_queue > 0
            and 100.0 * excess / ref_queue >= D2_MIN_EXCESS_PCT_OF_REF
        )
        share = 100.0 * excess_overlap / excess if material else float("nan")
        if not material:
            verdict = "NO-EXCESS"
        elif share >= D2_SERIALIZATION_MIN_PCT:
            verdict = "SELF-INFLICTED-COLOCATION"
        else:
            verdict = "NOT-SELF-INFLICTED"
        out["arms"][name] = {
            "excess_queue_s_per_task": excess,
            "excess_pct_of_reference": 100.0 * excess / ref_queue if ref_queue else float("inf"),
            "excess_same_batch_overlap_s_per_task": excess_overlap,
            "share_pct": share,
            "verdict": verdict,
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="append", required=True, metavar="NAME=PATH",
                    help="one retained raw result per arm, e.g. gnn=.../gnn_s1.json")
    ap.add_argument("--workload", type=Path, required=True,
                    help="the trace the arms ran, for peer pairs and peer_group ids")
    ap.add_argument("--reference", default="knative_network",
                    help="the arm the others' excess queue time is measured against")
    ap.add_argument("--out", type=Path, help="write the reading here as JSON")
    args = ap.parse_args(argv)

    workload = json.loads(args.workload.read_text())
    peer_table = build_peer_exchange_table(workload.get("peer_exchange"))
    group_of: Dict[int, int] = {}
    for idx, event in enumerate(workload.get("events") or []):
        gid = event.get("peer_group")
        if gid is not None:
            group_of[idx] = int(gid)

    arms: Dict[str, Dict[str, Any]] = {}
    for spec in args.raw:
        if "=" not in spec:
            raise SystemExit(f"FAIL LOUD: --raw takes NAME=PATH, got {spec!r}")
        name, _, path = spec.partition("=")
        arms[name] = read_arm(Path(path), peer_table, group_of)

    reading = {
        "lineage": "drainable_debug_v1",
        "read": "D2",
        "bars": {
            "d2_serialization_min_pct": D2_SERIALIZATION_MIN_PCT,
            "d2_min_tasks": D2_MIN_TASKS,
        },
        "workload": str(args.workload),
        "peer_pairs": sum(len(v) for v in peer_table.values()) // 2,
        "arms": arms,
        "d2": d2_verdict(arms, args.reference),
    }

    for name, arm in arms.items():
        dec, ser, con = arm["decomposition"], arm["serialization"], arm["concentration"]
        cqm = arm["chosen_queue_vs_min"]
        print(
            f"[{name}] n={dec['n_tasks']} elapsed={dec['elapsed']:.2f} queue={dec['queue']:.2f} "
            f"batch_wait={dec['batch_wait']:.2f} init={dec['initialization']:.2f}"
        )
        print(
            f"    serialization: behind-any {ser['share_any_pct']:.1f}% of queue, "
            f"same-batch {ser['share_same_batch_pct']:.1f}%"
        )
        print(
            f"    concentration: {con.get('platforms_used')} platforms, "
            f"effective {con.get('effective_platforms', float('nan')):.2f}, "
            f"busiest {con.get('busiest_platform_share_pct', float('nan')):.1f}%"
        )
        cqm_mean = (
            f"{cqm['mean']:.3f}" if "mean" in cqm
            else f"n/a ({cqm['missing_snapshot_pct']:.0f}% of tasks carry no snapshot; "
                 "re-run with KEEP_RAW_QSNAP=1)"
        )
        print(
            f"    chosen_queue_vs_min: mean={cqm_mean} "
            f"no-choice {cqm['no_choice_pct']:.1f}% of tasks"
        )
        if "peer_colocation" in arm:
            pc = arm["peer_colocation"]
            print(
                f"    peers: same-platform {pc['same_platform_pct']:.1f}% "
                f"same-node {pc['same_node_pct']:.1f}% remote {pc['remote_pct']:.1f}%"
            )
    for name, verdict in reading["d2"].get("arms", {}).items():
        if verdict.get("verdict") == "VOID":
            print(f"[D2] {name}: VOID -- {verdict.get('why')}")
            continue
        if verdict["verdict"] == "NO-EXCESS":
            print(
                f"[D2] {name}: NO-EXCESS ({verdict['excess_queue_s_per_task']:.2f} s per task = "
                f"{verdict['excess_pct_of_reference']:.2f}% of the reference; below the "
                f"{D2_MIN_EXCESS_PCT_OF_REF:.0f}% floor, not decomposed)"
            )
            continue
        print(
            f"[D2] {name}: {verdict['verdict']} ({verdict['share_pct']:.1f}% of "
            f"{verdict['excess_queue_s_per_task']:.2f} s excess queue per task)"
        )

    if args.out:
        args.out.write_text(json.dumps(reading, indent=1))
        print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
