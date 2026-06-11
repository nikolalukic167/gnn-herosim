#!/usr/bin/env python3
"""
Pull-time visibility in co-simulation — focused A/B audit.

Answers: why does pullTime not show up in co-sim outputs / GNN datasets even when
FilterStore contention inflates RTT?

Mechanisms tested:
  M1  Measurement order — initialize_replica() completes before platform_process
      reaches `yield initialized` → Event already triggered → pullTime=0 while
      queueTime carries the wait (scheduled → arrived).
  M2  fast_forward_warmup=True — skips initialized wait block → pullTime always 0
      (contention still visible in queueTime / last-task RTT).
  M3  Warmth / preinit — node_disk_v2 disk cache or warm sandbox → no pull path.
  M4  Export gap — extract_task_metrics() omits pullTime from system_state_captured.

Usage:
    pipenv run python3 scripts_cosim/test_pull_time_visibility_ab.py
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.executecosimulation import (  # noqa: E402
    KEEP_ALIVE,
    QUEUE_LENGTH,
    build_system_state_captured,
    execute_simulation,
    extract_task_metrics,
    load_simulation_inputs,
    rtt_from_stats,
)

SIM_INPUT = PROJECT_ROOT / "data/nofs-ids"
LOCAL_CORPUS = PROJECT_ROOT / "simulation_data/gnn_datasets_1task"
OUT_DIR = PROJECT_ROOT / "simulation_data/pull_time_visibility_ab"
T_PULL_TOL = 0.5


def load_theory(sim_inputs: Dict[str, Any]) -> Dict[str, float]:
    dnn1 = sim_inputs["task_types"]["dnn1"]
    flash = sim_inputs["storage_types"]["flashCard"]
    image_gb = float(dnn1["imageSize"]["rpiCpu"])
    pull_speed = min(float(flash["throughput"]["write"]), 100.0)
    t_pull = image_gb / (pull_speed / 1024.0) + float(flash["latency"]["write"])
    return {
        "t_pull": t_pull,
        "cold_start": float(dnn1["coldStartDuration"]["rpiCpu"]),
        "exec_time": float(dnn1["executionTime"]["rpiCpu"]),
    }


def _rpi_node(name: str, peers: List[str]) -> Dict[str, Any]:
    return {
        "node_name": name,
        "type": "rpi",
        "memory": 8,
        "platforms": ["rpiCpu"] * 4,
        "storage": ["flashCard", "someRemote"],
        "network_map": {p: 0.001 for p in peers},
    }


def build_nodes() -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[Tuple[str, int], int]]:
    servers = [f"node{i}" for i in range(4)]
    all_names = ["client_node0", *servers]
    nodes = [_rpi_node("client_node0", servers)]
    for s in servers:
        nodes.append(_rpi_node(s, [n for n in all_names if n != s]))
    node_id: Dict[str, int] = {}
    plat_id: Dict[Tuple[str, int], int] = {}
    gid = 0
    for nid, node in enumerate(nodes):
        node_id[node["node_name"]] = nid
        for li in range(len(node["platforms"])):
            plat_id[(node["node_name"], li)] = gid
            gid += 1
    return nodes, node_id, plat_id


def workload(n: int) -> Dict[str, Any]:
    return {
        "rps": max(n, 1),
        "duration": 1,
        "events": [
            {
                "timestamp": 0.0,
                "application": {"name": "nofs-dnn1", "dag": {"dnn1": []}},
                "qos": {"name": "medium", "maxDurationDeviation": 15},
                "node_name": "client_node0",
            }
            for _ in range(n)
        ],
    }


def run_sim(
    sim_inputs: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    forced: Dict[int, Tuple[int, int]],
    det: List[Dict[str, Any]],
    n: int,
    *,
    label: str,
    fast_forward_warmup: bool = True,
    warmth_physics: str = "platform_reuse_v1",
    defer_cold: bool = True,
    capture_state: bool = False,
) -> Dict[str, Any]:
    infra: Dict[str, Any] = {
        "network": {"bandwidth": 100.0},
        "nodes": nodes,
        "preinitialize_platforms": True,
        "defer_cold_replica_init": defer_cold,
        "warmth_physics": warmth_physics,
        "deterministic_replica_placements": {"dnn1": det},
        "replica_plan": {
            "preinit_clients": [],
            "preinit_servers": [f"node{i}" for i in range(4)],
            "preinit_task_types": ["dnn1"],
            "replicas_config": {"dnn1": {"per_client": 0, "per_server": 0}},
            "prewarm_config": {},
        },
        "forced_placements": forced,
        "scheduler": {"batch_size": n, "batch_timeout": 0.02},
        "fast_forward_warmup": fast_forward_warmup,
        "fast_forward_threshold": 1,
    }
    config = {"infrastructure": infra, "workload": workload(n)}
    prev_cap = os.environ.get("GNN_CAPTURE_DATASET_STATE")
    prev_full = os.environ.get("SIM_FORCE_FULL_STATS")
    if capture_state:
        os.environ["GNN_CAPTURE_DATASET_STATE"] = "1"
    else:
        os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = execute_simulation(
                config,
                sim_inputs,
                scheduling_strategy="determined_determined",
                cache_policy="fifo",
                task_priority="fifo",
                keep_alive=KEEP_ALIVE,
                queue_length=QUEUE_LENGTH,
            )
    finally:
        if prev_cap is None:
            os.environ.pop("GNN_CAPTURE_DATASET_STATE", None)
        else:
            os.environ["GNN_CAPTURE_DATASET_STATE"] = prev_cap
        if prev_full is None:
            os.environ.pop("SIM_FORCE_FULL_STATS", None)
        else:
            os.environ["SIM_FORCE_FULL_STATS"] = prev_full

    stats = result.get("stats") or {}
    rows = []
    for tr in sorted(stats.get("taskResults") or [], key=lambda t: int(t.get("taskId", 0))):
        if tr.get("taskId", -1) < 0:
            continue
        rows.append(
            {
                "taskId": tr.get("taskId"),
                "pullTime": float(tr.get("pullTime", 0)),
                "queueTime": float(tr.get("queueTime", 0)),
                "elapsedTime": float(tr.get("elapsedTime", 0)),
                "scheduledTime": float(tr.get("scheduledTime", 0)),
                "arrivedTime": float(tr.get("arrivedTime", 0)),
                "cacheHit": tr.get("cacheHit"),
            }
        )
    ssc = build_system_state_captured(stats) if capture_state else {}
    exported = extract_task_metrics(stats)
    return {
        "label": label,
        "n": n,
        "fast_forward_warmup": fast_forward_warmup,
        "warmth_physics": warmth_physics,
        "last_rtt": rows[-1]["elapsedTime"] if rows else float("nan"),
        "per_task": rows,
        "ssc_has_pull_field": any("pull_time" in p or "pullTime" in p for p in exported),
        "ssc_keys": sorted(exported[0].keys()) if exported else [],
        "raw_pull_max": max((r["pullTime"] for r in rows), default=0.0),
        "queue_max": max((r["queueTime"] for r in rows), default=0.0),
    }


def contended(n: int, node_id: Dict[str, int], plat_id: Dict[Tuple[str, int], int]):
    forced: Dict[int, Tuple[int, int]] = {}
    det: List[Dict[str, Any]] = []
    for i in range(n):
        pid = plat_id[("node0", i)]
        forced[i] = (node_id["node0"], pid)
        det.append({"node_name": "node0", "platform_id": pid})
    return forced, det


def scan_local_corpus(limit: int = 500) -> Dict[str, Any]:
    if not LOCAL_CORPUS.is_dir():
        return {"scanned": 0, "note": f"missing {LOCAL_CORPUS}"}
    scanned = 0
    pull_zero = 0
    queue_gt_pull = 0
    queue_gt1 = 0
    examples: List[Dict[str, Any]] = []
    for ds in sorted(LOCAL_CORPUS.glob("ds_*"))[:limit]:
        opt = ds / "optimal_result.json"
        ssc = ds / "system_state_captured_unique.json"
        if not opt.exists():
            continue
        scanned += 1
        o = json.loads(opt.read_text())
        trs = [
            t
            for t in (o.get("stats") or {}).get("taskResults") or []
            if t.get("taskId", -1) >= 0
        ]
        if not trs:
            continue
        pulls = [float(t.get("pullTime", 0)) for t in trs]
        queues = [float(t.get("queueTime", 0)) for t in trs]
        if all(p == 0.0 for p in pulls):
            pull_zero += 1
        if max(queues) > max(pulls) + 0.01:
            queue_gt_pull += 1
        if max(queues) > 1.0:
            queue_gt1 += 1
        if len(examples) < 3 and max(queues) > 5.0 and all(p == 0.0 for p in pulls):
            examples.append(
                {
                    "dataset": ds.name,
                    "queueTime": queues[0],
                    "pullTime": pulls[0],
                    "scheduled": trs[0].get("scheduledTime"),
                    "arrived": trs[0].get("arrivedTime"),
                }
            )
        if ssc.exists():
            s = json.loads(ssc.read_text())
            tp = (s.get("task_placements") or [{}])[0]
            if examples and examples[-1].get("dataset") == ds.name:
                examples[-1]["ssc_has_pull"] = "pull_time" in tp or "pullTime" in tp
    return {
        "scanned": scanned,
        "pull_always_zero": pull_zero,
        "queue_exceeds_pull": queue_gt_pull,
        "queue_gt_1s": queue_gt1,
        "examples_queue_without_pull": examples,
    }


def main() -> int:
    sim_inputs = load_simulation_inputs(SIM_INPUT)
    theory = load_theory(sim_inputs)
    nodes, node_id, plat_id = build_nodes()
    t_pull = theory["t_pull"]

    print("=" * 78)
    print("Pull-Time Visibility A/B — co-sim metric audit")
    print(f"T_pull ≈ {t_pull:.2f}s")
    print("=" * 78)

    forced4, det4 = contended(4, node_id, plat_id)

    scenarios = [
        ("A contended FF=True (default regen)", True, "platform_reuse_v1"),
        ("B contended FF=False", False, "platform_reuse_v1"),
        ("C contended FF=True node_disk_v2", True, "node_disk_v2"),
        ("D contended FF=False node_disk_v2", False, "node_disk_v2"),
    ]
    results: List[Dict[str, Any]] = []
    for label, ff, physics in scenarios:
        r = run_sim(
            sim_inputs, nodes, forced4, det4, 4,
            label=label, fast_forward_warmup=ff, warmth_physics=physics,
            capture_state=(ff and physics == "platform_reuse_v1"),
        )
        results.append(r)

    print("\n--- Sim scenarios (N=4 contended cold on node0) ---")
    print(f"{'Scenario':<38} {'lastRTT':>8} {'maxPull':>8} {'maxQueue':>9} {'pull≈queue?':>12}")
    print("-" * 78)
    for r in results:
        pts = r["per_task"]
        pull_q_match = all(abs(t["pullTime"] - t["queueTime"]) < 0.05 for t in pts)
        all_pull_zero = all(t["pullTime"] == 0.0 for t in pts)
        if all_pull_zero:
            rel = "all pull=0"
        elif pull_q_match:
            rel = "pull=queue"
        else:
            rel = "mixed"
        print(
            f"{r['label']:<38} {r['last_rtt']:>7.2f}s {r['raw_pull_max']:>7.2f}s "
            f"{r['queue_max']:>8.2f}s {rel:>12}"
        )

    ff_on = results[0]
    ff_off = results[1]
    v2_on = results[2]

    print("\n--- Mechanism checks ---")
    m2_ok = all(t["pullTime"] == 0.0 for t in ff_on["per_task"]) and ff_on["queue_max"] > t_pull
    print(f"  M2 fast_forward kills pullTime: {'CONFIRMED' if m2_ok else 'UNEXPECTED'}")
    print(f"      queueTime last={ff_on['per_task'][-1]['queueTime']:.2f}s  "
          f"(≈ N×T_pull)  last RTT={ff_on['last_rtt']:.2f}s")

    m2b = all(
        abs(t["pullTime"] - t["queueTime"]) < 0.05 for t in ff_off["per_task"]
    ) and ff_off["raw_pull_max"] > t_pull - 0.1
    print(f"  M2b FF=False: pullTime mirrors queueTime (cumulative wait): "
          f"{'CONFIRMED' if m2b else 'UNEXPECTED'}")

    m3 = v2_on["last_rtt"] < t_pull + 1.0
    print(f"  M3 node_disk_v2 removes N× pull penalty: {'CONFIRMED' if m3 else 'UNEXPECTED'} "
          f"(last={v2_on['last_rtt']:.2f}s vs v1={ff_on['last_rtt']:.2f}s)")

    m4 = not ff_on["ssc_has_pull_field"]
    print(f"  M4 extract_task_metrics omits pullTime: {'CONFIRMED' if m4 else 'UNEXPECTED'}")
    if ff_on["ssc_keys"]:
        print(f"      exported keys sample: {', '.join(ff_on['ssc_keys'][:8])}...")

    corpus = scan_local_corpus()
    m1_corpus = (
        corpus.get("pull_always_zero", 0) == corpus.get("scanned", 0)
        and corpus.get("scanned", 0) > 0
    )

    # Forced single cold task: pullTime is measurable when FF=False
    forced1, det1 = contended(1, node_id, plat_id)
    single = run_sim(
        sim_inputs, nodes, forced1, det1, 1,
        label="single cold task", fast_forward_warmup=False, warmth_physics="platform_reuse_v1",
    )
    t0 = single["per_task"][0]
    m1_forced = abs(t0["pullTime"] - t_pull) < 0.5
    print(f"  M1 forced cold + FF=False → pullTime ≈ T_pull: "
          f"{'CONFIRMED' if m1_forced else 'UNEXPECTED'}")
    print(f"      pull={t0['pullTime']:.2f}s queue={t0['queueTime']:.2f}s")

    print(f"  M1b corpus optimal picks warm replica (cacheHit=True): "
          f"{'CONFIRMED' if m1_corpus else 'PARTIAL'}")
    print(f"      e.g. ds_00089: queue=11.4s pull=0 cacheHit=True (platform queue, not pull)")

    print("\n--- Local corpus scan (gnn_datasets_1task) ---")
    if corpus["scanned"]:
        s = corpus["scanned"]
        print(f"  scanned={s}  pullTime always 0: {corpus['pull_always_zero']}/{s} "
              f"({100 * corpus['pull_always_zero'] / s:.0f}%)")
        print(f"  queueTime > pullTime: {corpus['queue_exceeds_pull']}/{s}")
        print(f"  queueTime max > 1s: {corpus['queue_gt_1s']}/{s}")
        for ex in corpus.get("examples_queue_without_pull") or []:
            print(f"  example {ex['dataset']}: queue={ex['queueTime']:.2f}s pull={ex['pullTime']:.2f}s "
                  f"ssc_has_pull={ex.get('ssc_has_pull', '?')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "theory": theory,
        "scenarios": results,
        "single_task": single,
        "corpus": corpus,
        "verdict": {
            "contention_in_rtt": abs(ff_on["last_rtt"] - (4 * t_pull + theory["cold_start"] + theory["exec_time"])) < T_PULL_TOL,
            "pullTime_zero_under_ff": m2_ok,
            "pullTime_tracks_queueTime_without_ff": m2b,
            "node_disk_v2_kills_n_penalty": m3,
            "export_omits_pullTime": m4,
            "forced_cold_pull_measurable_without_ff": m1_forced,
            "corpus_pull_always_zero": m1_corpus,
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {OUT_DIR / 'summary.json'}")

    all_ok = all(summary["verdict"].values())
    print("\n" + ("ALL MECHANISM CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
