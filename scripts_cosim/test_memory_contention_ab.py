#!/usr/bin/env python3
"""
Memory (storage FilterStore) contention A/B test — SimPy mirror + full HeROsim.

Measures how co-located cold image pulls inflate RTT when N tasks share one
node's local storage vs when each task uses a separate node.

  A (Contended): N tasks → N platforms on the SAME server node
                   → image pulls serialize through one FilterStore

  B (Parallel):    N tasks → one platform each on N DIFFERENT server nodes
                   → pulls run concurrently

Uses the determined scheduler with forced placements so routing is identical
across A/B; only the storage topology differs.

Theory (matches knative/autoscaler.initialize_replica):
  T_pull = imageSize_GB / (min(storage_write_mbps, network_mbps) / 1024) + write_latency
  Last-task RTT (contended)  ≈ N × T_pull + cold_start + exec
  Each-task RTT (parallel)   ≈ T_pull + cold_start + exec

Usage:
    pipenv run python3 scripts_cosim/test_memory_contention_ab.py
"""

from __future__ import annotations

import json
import math
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
    execute_simulation,
    load_simulation_inputs,
    rtt_from_stats,
)

SIM_INPUT = PROJECT_ROOT / "data/nofs-ids"
SWEEP_N = [1, 2, 3, 4, 5]
MAX_N = max(SWEEP_N)
TOLERANCE_S = 0.5  # full-sim tolerance (network + I/O comm overhead)
MIRROR_TOLERANCE_S = 0.01

# ---------------------------------------------------------------------------
# Theory constants from data/nofs-ids (rpiCpu + flashCard + 100 Mbps network)
# ---------------------------------------------------------------------------

def load_theory_constants(sim_inputs: Dict[str, Any]) -> Dict[str, float]:
    task_types = sim_inputs["task_types"]
    storage_types = sim_inputs["storage_types"]
    dnn1 = task_types["dnn1"]
    flash = storage_types["flashCard"]
    image_gb = float(dnn1["imageSize"]["rpiCpu"])
    storage_mbps = float(flash["throughput"]["write"])
    storage_lat = float(flash["latency"]["write"])
    network_mbps = 100.0
    pull_speed = min(storage_mbps, network_mbps)
    t_pull = image_gb / (pull_speed / 1024.0) + storage_lat
    cold_start = float(dnn1["coldStartDuration"]["rpiCpu"])
    exec_time = float(dnn1["executionTime"]["rpiCpu"])
    return {
        "image_gb": image_gb,
        "pull_speed_mbps": pull_speed,
        "t_pull": t_pull,
        "cold_start": cold_start,
        "exec_time": exec_time,
        "t_baseline": t_pull + cold_start + exec_time,
    }


def predict_contended_last_rtt(n: int, theory: Dict[str, float]) -> float:
    return n * theory["t_pull"] + theory["cold_start"] + theory["exec_time"]


def predict_parallel_rtt(theory: Dict[str, float]) -> float:
    return theory["t_baseline"]


# ---------------------------------------------------------------------------
# Minimal infrastructure builder
# ---------------------------------------------------------------------------

def _rpi_node(node_name: str, peer_names: List[str], latency: float = 0.001) -> Dict[str, Any]:
    return {
        "node_name": node_name,
        "type": "rpi",
        "memory": 8,
        "platforms": ["rpiCpu"] * 6,
        "storage": ["flashCard", "someRemote"],
        "network_map": {peer: latency for peer in peer_names},
    }


def build_test_nodes(max_n: int) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    """
    1 client + max_n servers. Returns nodes list and lookup maps:
      node_name -> node_id, (node_name, local_platform_index) -> global platform_id
    """
    server_names = [f"node{i}" for i in range(max_n)]
    all_names = ["client_node0", *server_names]
    nodes = [_rpi_node("client_node0", server_names)]
    for sname in server_names:
        peers = [n for n in all_names if n != sname]
        nodes.append(_rpi_node(sname, peers))

    node_id_by_name: Dict[str, int] = {}
    plat_id_by_node_local: Dict[Tuple[str, int], int] = {}
    global_plat = 0
    for node_id, node in enumerate(nodes):
        node_id_by_name[node["node_name"]] = node_id
        for local_idx in range(len(node["platforms"])):
            plat_id_by_node_local[(node["node_name"], local_idx)] = global_plat
            global_plat += 1
    return nodes, node_id_by_name, plat_id_by_node_local


def contended_placements(
    n: int,
    node_id_by_name: Dict[str, int],
    plat_id_by_node_local: Dict[Tuple[str, int], int],
) -> Tuple[Dict[int, Tuple[int, int]], List[Dict[str, Any]]]:
    """All N tasks on node0 platforms 0..N-1."""
    node_name = "node0"
    node_id = node_id_by_name[node_name]
    forced: Dict[int, Tuple[int, int]] = {}
    det: List[Dict[str, Any]] = []
    for i in range(n):
        plat_id = plat_id_by_node_local[(node_name, i)]
        forced[i] = (node_id, plat_id)
        det.append({"node_name": node_name, "platform_id": plat_id})
    return forced, det


def parallel_placements(
    n: int,
    node_id_by_name: Dict[str, int],
    plat_id_by_node_local: Dict[Tuple[str, int], int],
) -> Tuple[Dict[int, Tuple[int, int]], List[Dict[str, Any]]]:
    """Task i on node{i} platform 0."""
    forced: Dict[int, Tuple[int, int]] = {}
    det: List[Dict[str, Any]] = []
    for i in range(n):
        node_name = f"node{i}"
        node_id = node_id_by_name[node_name]
        plat_id = plat_id_by_node_local[(node_name, 0)]
        forced[i] = (node_id, plat_id)
        det.append({"node_name": node_name, "platform_id": plat_id})
    return forced, det


def build_workload(n: int) -> Dict[str, Any]:
    events = []
    for i in range(n):
        events.append(
            {
                "timestamp": 0.0,
                "application": {"name": "nofs-dnn1", "dag": {"dnn1": []}},
                "qos": {"name": "medium", "maxDurationDeviation": 15},
                "node_name": "client_node0",
            }
        )
    return {"rps": max(n, 1), "duration": 1, "events": events}


def run_full_sim_scenario(
    sim_inputs: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    forced: Dict[int, Tuple[int, int]],
    det_placements: List[Dict[str, Any]],
    n: int,
    label: str,
) -> Dict[str, Any]:
    infrastructure: Dict[str, Any] = {
        "network": {"bandwidth": 100.0},
        "nodes": nodes,
        "preinitialize_platforms": True,
        "defer_cold_replica_init": True,
        "deterministic_replica_placements": {"dnn1": det_placements},
        "replica_plan": {
            "preinit_clients": [],
            "preinit_servers": [f"node{i}" for i in range(MAX_N)],
            "preinit_task_types": ["dnn1"],
            "replicas_config": {"dnn1": {"per_client": 0, "per_server": 0}},
            "prewarm_config": {},
        },
        "forced_placements": forced,
        "scheduler": {"batch_size": n, "batch_timeout": 0.02},
        "fast_forward_warmup": True,
        "fast_forward_threshold": 1,
    }
    config = {"infrastructure": infrastructure, "workload": build_workload(n)}

    prev_capture = os.environ.get("GNN_CAPTURE_DATASET_STATE")
    os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
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
        if prev_capture is None:
            os.environ.pop("GNN_CAPTURE_DATASET_STATE", None)
        else:
            os.environ["GNN_CAPTURE_DATASET_STATE"] = prev_capture

    stats = result.get("stats", {})
    total_rtt = rtt_from_stats(stats)
    task_results = stats.get("taskResults") or []
    per_task = [
        {
            "taskId": tr.get("taskId"),
            "elapsedTime": float(tr.get("elapsedTime", 0)),
            "pullTime": float(tr.get("pullTime", 0)),
            "coldStartTime": float(tr.get("coldStartTime", 0)),
            "executionTime": float(tr.get("executionTime", 0)),
            "queueTime": float(tr.get("queueTime", 0)),
            "executionNode": tr.get("executionNode"),
            "executionPlatform": tr.get("executionPlatform"),
        }
        for tr in sorted(task_results, key=lambda t: int(t.get("taskId", 0)))
    ]
    last_rtt = per_task[-1]["elapsedTime"] if per_task else float("nan")
    max_rtt = max((t["elapsedTime"] for t in per_task), default=float("nan"))
    return {
        "label": label,
        "n": n,
        "total_rtt": total_rtt,
        "last_task_rtt": last_rtt,
        "max_task_rtt": max_rtt,
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# SimPy mirror (from test_storage_contention.py — validates FilterStore physics)
# ---------------------------------------------------------------------------

def run_mirror_sweep(theory: Dict[str, float]) -> List[Dict[str, Any]]:
    import simpy
    from simpy.resources.store import FilterStore

    image_gb = theory["image_gb"]
    pull_speed = theory["pull_speed_mbps"]
    t_pull = theory["t_pull"]
    cold_start = theory["cold_start"]
    exec_time = theory["exec_time"]
    storage_lat = 0.00012
    network_mbps = 100.0

    class StorageDevice:
        write_mbps = pull_speed
        latency = storage_lat

    def pull(env, store, task_id, results):
        storage = yield store.get(lambda s: True)
        pull_start = env.now
        spd = min(storage.write_mbps, network_mbps)
        dur = image_gb / (spd / 1024.0) + storage.latency
        yield env.timeout(dur)
        store.put(storage)
        results[task_id] = {"pull_start": pull_start, "pull_end": env.now}

    def task(env, store, task_id, results):
        arrival = env.now
        yield env.process(pull(env, store, task_id, results))
        yield env.timeout(cold_start + exec_time)
        results[task_id]["rtt"] = env.now - arrival

    rows = []
    for n in SWEEP_N:
        # Contended
        env_c = simpy.Environment()
        store_c = FilterStore(env_c)
        store_c.put(StorageDevice())
        res_c: Dict[int, Any] = {}
        for i in range(n):
            env_c.process(task(env_c, store_c, i, res_c))
        env_c.run()

        # Parallel
        env_p = simpy.Environment()
        res_p: Dict[int, Any] = {}
        for i in range(n):
            st = FilterStore(env_p)
            st.put(StorageDevice())
            env_p.process(task(env_p, st, i, res_p))
        env_p.run()

        last_c = res_c[n - 1]["rtt"]
        last_p = res_p[n - 1]["rtt"]
        pred_c = predict_contended_last_rtt(n, theory)
        pred_p = predict_parallel_rtt(theory)
        rows.append(
            {
                "n": n,
                "mirror_contended_last": last_c,
                "mirror_parallel_last": last_p,
                "mirror_penalty_abs": last_c - last_p,
                "mirror_penalty_pct": (last_c - last_p) / last_p * 100 if last_p else float("nan"),
                "predicted_contended": pred_c,
                "predicted_parallel": pred_p,
                "mirror_contended_match": abs(last_c - pred_c) < MIRROR_TOLERANCE_S,
                "mirror_parallel_match": abs(last_p - pred_p) < MIRROR_TOLERANCE_S,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    sim_inputs = load_simulation_inputs(SIM_INPUT)
    theory = load_theory_constants(sim_inputs)
    nodes, node_id_by_name, plat_id_by_node_local = build_test_nodes(MAX_N)

    print("=" * 78)
    print("Memory (FilterStore) Contention A/B — determined scheduler + full HeROsim")
    print(f"T_pull={theory['t_pull']:.2f}s  cold_start={theory['cold_start']:.2f}s  "
          f"exec={theory['exec_time']:.4f}s  baseline={theory['t_baseline']:.2f}s")
    print("=" * 78)

    # Layer 1: SimPy mirror
    print("\n--- Layer 1: SimPy FilterStore mirror (theory cross-check) ---")
    mirror_rows = run_mirror_sweep(theory)
    mirror_ok = True
    print(f"{'N':>3}  {'Contended':>12}  {'Parallel':>12}  {'Penalty':>10}  {'Pred C':>10}  {'Match':>6}")
    for row in mirror_rows:
        match = "OK" if row["mirror_contended_match"] and row["mirror_parallel_match"] else "FAIL"
        if match != "OK":
            mirror_ok = False
        print(
            f"{row['n']:>3}  {row['mirror_contended_last']:>11.2f}s  "
            f"{row['mirror_parallel_last']:>11.2f}s  "
            f"{row['mirror_penalty_pct']:>8.0f}%  {row['predicted_contended']:>9.2f}s  {match:>6}"
        )

    # Layer 2: Full simulation
    print("\n--- Layer 2: Full HeROsim (determined scheduler, forced placements) ---")
    full_rows: List[Dict[str, Any]] = []
    full_ok = True
    print(f"{'N':>3}  {'Last C':>12}  {'Last P':>12}  {'Penalty':>10}  {'Pred C':>10}  {'Match':>6}")
    for n in SWEEP_N:
        forced_c, det_c = contended_placements(n, node_id_by_name, plat_id_by_node_local)
        forced_p, det_p = parallel_placements(n, node_id_by_name, plat_id_by_node_local)

        res_c = run_full_sim_scenario(sim_inputs, nodes, forced_c, det_c, n, "contended")
        res_p = run_full_sim_scenario(sim_inputs, nodes, forced_p, det_p, n, "parallel")

        pred_c = predict_contended_last_rtt(n, theory)
        pred_p = predict_parallel_rtt(theory)
        err_c = abs(res_c["last_task_rtt"] - pred_c)
        err_p = abs(res_p["last_task_rtt"] - pred_p)
        match_c = err_c <= TOLERANCE_S
        match_p = err_p <= TOLERANCE_S
        match = "OK" if match_c and match_p else "FAIL"
        if match != "OK":
            full_ok = False

        penalty_pct = (
            (res_c["last_task_rtt"] - res_p["last_task_rtt"]) / res_p["last_task_rtt"] * 100
            if res_p["last_task_rtt"] > 0
            else float("nan")
        )
        row = {
            "n": n,
            "full_contended_last": res_c["last_task_rtt"],
            "full_parallel_last": res_p["last_task_rtt"],
            "full_contended_total": res_c["total_rtt"],
            "full_parallel_total": res_p["total_rtt"],
            "full_penalty_abs": res_c["last_task_rtt"] - res_p["last_task_rtt"],
            "full_penalty_pct": penalty_pct,
            "predicted_contended": pred_c,
            "predicted_parallel": pred_p,
            "full_contended_error_s": err_c,
            "full_parallel_error_s": err_p,
            "full_match": match == "OK",
            "contended_per_task": res_c["per_task"],
            "parallel_per_task": res_p["per_task"],
        }
        full_rows.append(row)
        print(
            f"{n:>3}  {res_c['last_task_rtt']:>11.2f}s  {res_p['last_task_rtt']:>11.2f}s  "
            f"{penalty_pct:>8.0f}%  {pred_c:>9.2f}s  {match:>6}"
        )

    # Summary table: multiplier growth
    print("\n--- RTT multiplier (contended / parallel) ---")
    print(f"{'N':>3}  {'Mirror':>10}  {'Full sim':>10}  {'Linear N':>10}")
    for i, n in enumerate(SWEEP_N):
        m_mult = mirror_rows[i]["mirror_contended_last"] / mirror_rows[i]["mirror_parallel_last"]
        f_mult = full_rows[i]["full_contended_last"] / full_rows[i]["full_parallel_last"]
        print(f"{n:>3}  {m_mult:>9.2f}x  {f_mult:>9.2f}x  {n:>9}x")

    # N=4 detail
    n_detail = 4
    idx = SWEEP_N.index(n_detail)
    print(f"\n--- Per-task detail N={n_detail} (contended, full sim) ---")
    print(f"{'Task':>6}  {'RTT':>10}  {'pullTime':>10}  {'coldStart':>10}  {'exec':>8}  {'node:plat':>12}")
    for t in full_rows[idx]["contended_per_task"]:
        print(
            f"{t['taskId']:>6}  {t['elapsedTime']:>9.2f}s  {t['pullTime']:>9.2f}s  "
            f"{t['coldStartTime']:>9.2f}s  {t['executionTime']:>7.4f}s  "
            f"{t['executionNode']}:{t['executionPlatform']:>3}"
        )

    out_dir = PROJECT_ROOT / "simulation_data/memory_contention_ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "theory": theory,
        "mirror_rows": mirror_rows,
        "full_rows": [{k: v for k, v in r.items() if k not in ("contended_per_task", "parallel_per_task")} for r in full_rows],
        "mirror_pass": mirror_ok,
        "full_pass": full_ok,
        "tolerance_s": TOLERANCE_S,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {summary_path}")

    all_pass = mirror_ok and full_ok
    print()
    if all_pass:
        print("SWEEP PASSED — memory contention RTT inflation confirmed (mirror + full sim).")
        return 0
    print("SWEEP FAILED — see mismatches above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
