#!/usr/bin/env python3
"""
Cold start, queueTime, and last-task mechanics — deep A/B audit.

Companion to test_memory_contention_ab.py.  Focuses on:

  1. Last-task RTT penalty  ≈ (N−1) × T_pull  under co-located cold pulls
  2. Three different "queue" signals (platform depth vs shared_fate vs task queueTime)
  3. Metric traps: pullTime under fast_forward_warmup, total_rtt vs last-task,
     invisible execution-phase FilterStore wait on early tasks

Scenarios (N-task sweep + N=4 counterfactuals):

  A  Contended cold   — N plats on node0, defer_cold_replica_init=True
  B  Parallel cold    — one plat per node
  C  Contended warm   — same as A but replicas pre-initialized (no pull)
  D  Multi-storage    — A topology but 4× flashCard on node0 (parallel pulls)

Usage:
    pipenv run python3 scripts_cosim/test_cold_start_queue_last_task_ab.py
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
N_DETAIL = 4
TOLERANCE_S = 0.5
MIRROR_TOLERANCE_S = 0.01


# ---------------------------------------------------------------------------
# Theory
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


def predict_warm_rtt(theory: Dict[str, float]) -> float:
    return theory["cold_start"] + theory["exec_time"]


# ---------------------------------------------------------------------------
# Infrastructure builders
# ---------------------------------------------------------------------------

def _rpi_node(
    node_name: str,
    peer_names: List[str],
    storage: Optional[List[str]] = None,
    latency: float = 0.001,
) -> Dict[str, Any]:
    if storage is None:
        storage = ["flashCard", "someRemote"]
    return {
        "node_name": node_name,
        "type": "rpi",
        "memory": 8,
        "platforms": ["rpiCpu"] * 6,
        "storage": storage,
        "network_map": {peer: latency for peer in peer_names},
    }


def build_test_nodes(
    max_n: int,
    *,
    multi_flash_on_node0: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[Tuple[str, int], int]]:
    server_names = [f"node{i}" for i in range(max_n)]
    all_names = ["client_node0", *server_names]
    nodes: List[Dict[str, Any]] = [_rpi_node("client_node0", server_names)]
    for i, sname in enumerate(server_names):
        peers = [n for n in all_names if n != sname]
        if i == 0 and multi_flash_on_node0 > 1:
            storage = ["flashCard"] * multi_flash_on_node0 + ["someRemote"]
        else:
            storage = None
        nodes.append(_rpi_node(sname, peers, storage=storage))

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
    events = [
        {
            "timestamp": 0.0,
            "application": {"name": "nofs-dnn1", "dag": {"dnn1": []}},
            "qos": {"name": "medium", "maxDurationDeviation": 15},
            "node_name": "client_node0",
        }
        for _ in range(n)
    ]
    return {"rps": max(n, 1), "duration": 1, "events": events}


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_scenario(
    sim_inputs: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    forced: Dict[int, Tuple[int, int]],
    det_placements: List[Dict[str, Any]],
    n: int,
    *,
    label: str,
    defer_cold: bool = True,
    fast_forward_warmup: bool = True,
    warmth_physics: str = "platform_reuse_v1",
) -> Dict[str, Any]:
    infrastructure: Dict[str, Any] = {
        "network": {"bandwidth": 100.0},
        "nodes": nodes,
        "preinitialize_platforms": True,
        "defer_cold_replica_init": defer_cold,
        "warmth_physics": warmth_physics,
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
        "fast_forward_warmup": fast_forward_warmup,
        "fast_forward_threshold": 1,
    }
    config = {"infrastructure": infrastructure, "workload": build_workload(n)}

    prev_capture = os.environ.get("GNN_CAPTURE_DATASET_STATE")
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
        if prev_capture is None:
            os.environ.pop("GNN_CAPTURE_DATASET_STATE", None)
        else:
            os.environ["GNN_CAPTURE_DATASET_STATE"] = prev_capture

    stats = result.get("stats", {})
    task_results = stats.get("taskResults") or []
    per_task = [_extract_task_detail(tr) for tr in sorted(task_results, key=lambda t: int(t.get("taskId", 0)))]
    last_rtt = per_task[-1]["elapsedTime"] if per_task else float("nan")
    scheduling_capture = stats.get("schedulingStateCapture") or {}
    init_snapshot = scheduling_capture.get("initialized_snapshot") or {}
    queue_signals = _compute_queue_signals(per_task, init_snapshot, n)
    return {
        "label": label,
        "n": n,
        "defer_cold": defer_cold,
        "fast_forward_warmup": fast_forward_warmup,
        "warmth_physics": warmth_physics,
        "total_rtt": rtt_from_stats(stats),
        "last_task_rtt": last_rtt,
        "max_task_rtt": max((t["elapsedTime"] for t in per_task), default=float("nan")),
        "per_task": per_task,
        "queue_signals": queue_signals,
        "initialized_snapshot": init_snapshot,
    }


def _extract_task_detail(tr: Dict[str, Any]) -> Dict[str, Any]:
    wait = float(tr.get("waitTime", 0))
    queue = float(tr.get("queueTime", 0))
    init_t = float(tr.get("initializationTime", 0))
    pull = float(tr.get("pullTime", 0))
    cold = float(tr.get("coldStartTime", 0))
    exec_t = float(tr.get("executionTime", 0))
    comm = float(tr.get("communicationsTime", 0))
    compute = float(tr.get("computeTime", 0))
    elapsed = float(tr.get("elapsedTime", 0))
    sched = float(tr.get("scheduledTime", 0))
    arr = float(tr.get("arrivedTime", 0))
    start = float(tr.get("startedTime", 0))
    done = float(tr.get("doneTime", 0))
    named_sum = wait + queue + init_t + exec_t + comm
    invisible_gap = elapsed - named_sum
    exec_phase_extra = compute - exec_t - comm
    qsnap = tr.get("queueSnapshotAtScheduling") or {}
    plat_q_at_sched = max((int(v) for v in qsnap.values()), default=0) if qsnap else 0
    exec_key = f"{tr.get('executionNode')}:{tr.get('executionPlatform')}"
    plat_q_self = int(qsnap.get(exec_key, 0)) if qsnap else 0
    return {
        "taskId": tr.get("taskId"),
        "elapsedTime": elapsed,
        "waitTime": wait,
        "queueTime": queue,
        "initializationTime": init_t,
        "pullTime": pull,
        "coldStartTime": cold,
        "executionTime": exec_t,
        "communicationsTime": comm,
        "computeTime": compute,
        "scheduledTime": sched,
        "arrivedTime": arr,
        "startedTime": start,
        "doneTime": done,
        "named_sum": named_sum,
        "invisible_gap": invisible_gap,
        "exec_phase_extra": exec_phase_extra,
        "platQ_at_sched": plat_q_at_sched,
        "platQ_self_at_sched": plat_q_self,
        "executionNode": tr.get("executionNode"),
        "executionPlatform": tr.get("executionPlatform"),
        "queueSnapshotAtScheduling": qsnap,
        "temporalStateAtScheduling": tr.get("temporalStateAtScheduling") or {},
    }


def _compute_queue_signals(
    per_task: List[Dict[str, Any]],
    init_snapshot: Dict[str, bool],
    n: int,
) -> Dict[str, Any]:
    """Summarize the three different 'queue' concepts at scheduling time."""
    if not per_task:
        return {}
    qsnap = per_task[0].get("queueSnapshotAtScheduling") or {}
    max_plat_q = max((int(v) for v in qsnap.values()), default=0) if qsnap else 0
    node0_plats = [k for k in init_snapshot if k.startswith("node0:")]
    total_on_node = len(node0_plats) if node0_plats else 6
    cold_count = sum(1 for k, warm in init_snapshot.items() if k.startswith("node0:") and not warm)
    shared_fate = cold_count / max(total_on_node, 1)
    queue_times = [t["queueTime"] for t in per_task]
    return {
        "gnn_platform_queue_max": max_plat_q,
        "gnn_shared_fate_node0": shared_fate,
        "gnn_cold_count_node0": cold_count,
        "gnn_total_plats_node0": total_on_node,
        "outcome_queue_times": queue_times,
        "outcome_last_queue_time": queue_times[-1] if queue_times else 0.0,
    }


# ---------------------------------------------------------------------------
# SimPy mirror — pull timeline ground truth
# ---------------------------------------------------------------------------

def run_mirror_contended_timeline(n: int, theory: Dict[str, float]) -> List[Dict[str, Any]]:
    import simpy
    from simpy.resources.store import FilterStore

    image_gb = theory["image_gb"]
    pull_speed = theory["pull_speed_mbps"]
    cold_start = theory["cold_start"]
    exec_time = theory["exec_time"]
    storage_lat = 0.00012
    network_mbps = 100.0

    class StorageDevice:
        write_mbps = pull_speed
        latency = storage_lat

    def pull(env, store, task_id, results):
        yield store.get(lambda s: True)
        pull_start = env.now
        spd = min(StorageDevice.write_mbps, network_mbps)
        dur = image_gb / (spd / 1024.0) + storage_lat
        yield env.timeout(dur)
        store.put(StorageDevice())
        results[task_id] = {"pull_start": pull_start, "pull_end": env.now, "pull_dur": dur}

    def task(env, store, task_id, results):
        arrival = env.now
        yield env.process(pull(env, store, task_id, results))
        yield env.timeout(cold_start + exec_time)
        results[task_id]["rtt"] = env.now - arrival

    env = simpy.Environment()
    store = FilterStore(env)
    store.put(StorageDevice())
    results: Dict[int, Any] = {}
    for i in range(n):
        env.process(task(env, store, i, results))
    env.run()
    t_pull = theory["t_pull"]
    rows = []
    for i in range(n):
        r = results[i]
        rows.append(
            {
                "taskId": i,
                "pull_start": r["pull_start"],
                "pull_end": r["pull_end"],
                "expected_pull_start": i * t_pull,
                "queue_time_proxy": r["pull_start"],
                "rtt": r["rtt"],
                "predicted_rtt": (i + 1) * t_pull + cold_start + exec_time,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _fmt_row(cols: List[str], widths: List[int]) -> str:
    return "  ".join(c.rjust(w) for c, w in zip(cols, widths))


def print_sweep_table(theory: Dict[str, float], rows: List[Dict[str, Any]]) -> bool:
    ok = True
    print(f"\n{'N':>3}  {'Last C':>10}  {'Last P':>10}  {'Penalty':>10}  {'Mult':>7}  {'Pred C':>10}  {'Match':>6}")
    for row in rows:
        n = row["n"]
        pred = predict_contended_last_rtt(n, theory)
        err = abs(row["contended_last"] - pred)
        match = err <= TOLERANCE_S
        if not match:
            ok = False
        mult = row["contended_last"] / row["parallel_last"] if row["parallel_last"] else float("nan")
        print(
            f"{n:>3}  {row['contended_last']:>9.2f}s  {row['parallel_last']:>9.2f}s  "
            f"{row['penalty_abs']:>+9.2f}s  {mult:>6.2f}x  {pred:>9.2f}s  {'OK' if match else 'FAIL':>6}"
        )
    return ok


def print_timing_dissection(per_task: List[Dict[str, Any]], theory: Dict[str, float]) -> None:
    print(f"\n--- N={len(per_task)} contended cold — full timing dissection ---")
    hdr = (
        f"{'Task':>4}  {'elapsed':>8}  {'waitT':>6}  {'queueT':>7}  {'initT':>6}  "
        f"{'pullT':>6}  {'coldT':>6}  {'execT':>7}  {'commT':>6}  "
        f"{'namedΣ':>8}  {'invis':>7}  {'platQ':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for t in per_task:
        print(
            f"{t['taskId']:>4}  {t['elapsedTime']:>7.2f}s  {t['waitTime']:>5.2f}s  "
            f"{t['queueTime']:>6.2f}s  {t['initializationTime']:>5.2f}s  "
            f"{t['pullTime']:>5.2f}s  {t['coldStartTime']:>5.2f}s  "
            f"{t['executionTime']:>6.4f}s  {t['communicationsTime']:>5.3f}s  "
            f"{t['named_sum']:>7.2f}s  {t['invisible_gap']:>6.2f}s  {t['platQ_at_sched']:>5}"
        )
    print(f"\n  Timestamp chain (sched → arrived → started → done):")
    for t in per_task:
        print(
            f"  task {t['taskId']}: {t['scheduledTime']:.2f} → {t['arrivedTime']:.2f} → "
            f"{t['startedTime']:.2f} → {t['doneTime']:.2f}"
        )
    t_pull = theory["t_pull"]
    print(f"\n  queueTime steps (expect i×T_pull): ", end="")
    print(", ".join(f"{t['queueTime']:.2f}" for t in per_task))
    print(f"  expected (i+1)×T_pull:                  ", end="")
    print(", ".join(f"{(i + 1) * t_pull:.2f}" for i, _ in enumerate(per_task)))


def print_three_queues(signals: Dict[str, Any], theory: Dict[str, float]) -> None:
    print("\n--- Three different 'queue' signals (N=4 contended cold) ---")
    print(f"  GNN platform queue (max at scheduling):     {signals.get('gnn_platform_queue_max', '?'):>6}  tasks")
    print(f"  GNN shared_fate on node0:                   {signals.get('gnn_shared_fate_node0', 0):>6.2f}  (cold density)")
    print(f"  GNN cold_count / total_plats on node0:      {signals.get('gnn_cold_count_node0', '?')}/{signals.get('gnn_total_plats_node0', '?')}")
    qts = signals.get("outcome_queue_times") or []
    print(f"  Task stat queueTime (arrived−scheduled):   {', '.join(f'{q:.2f}s' for q in qts)}")
    print(f"  Task stat queueTime grows as N×T_pull:      linear in pull ordinal")
    print(f"  T_pull reference:                           {theory['t_pull']:.2f}s")


def print_metric_traps(
    contended_ff: Dict[str, Any],
    contended_no_ff: Dict[str, Any],
    parallel: Dict[str, Any],
    theory: Dict[str, float],
) -> None:
    print("\n--- Metric traps ---")
    n = contended_ff["n"]
    last_ff = contended_ff["last_task_rtt"]
    last_no_ff = contended_no_ff["last_task_rtt"]
    last_par = parallel["last_task_rtt"]
    total_ff = contended_ff["total_rtt"]
    print(f"  1. pullTime with fast_forward_warmup=True → always 0 (broken for contention measurement)")
    for t in contended_ff["per_task"]:
        print(f"       task {t['taskId']}: pullTime={t['pullTime']:.2f}s  queueTime={t['queueTime']:.2f}s")
    print(f"  2. pullTime with fast_forward_warmup=False → equals queueTime (cumulative wait, not per-pull duration)")
    for t in contended_no_ff["per_task"]:
        print(f"       task {t['taskId']}: pullTime={t['pullTime']:.2f}s  queueTime={t['queueTime']:.2f}s")
    print(f"  3. total_rtt (sum elapsed) vs last-task RTT:")
    print(f"       total_rtt={total_ff:.2f}s  last_task={last_ff:.2f}s  ratio={total_ff / last_ff:.2f}x")
    print(f"       → Do NOT use total_rtt as system RTT under contended co-location")
    print(f"  4. Last-task penalty (clean metric):")
    penalty = last_ff - last_par
    predicted = (n - 1) * theory["t_pull"]
    print(f"       contended − parallel = {penalty:.2f}s  (predicted (N−1)×T_pull = {predicted:.2f}s)")
    print(f"  5. Early-task invisible gap (exec-phase FilterStore wait, not in comm/queue):")
    for t in contended_ff["per_task"][:-1]:
        print(
            f"       task {t['taskId']}: elapsed={t['elapsedTime']:.2f}s  named_sum={t['named_sum']:.2f}s  "
            f"invisible={t['invisible_gap']:.2f}s  exec_phase_extra={t['exec_phase_extra']:.2f}s"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    sim_inputs = load_simulation_inputs(SIM_INPUT)
    theory = load_theory_constants(sim_inputs)
    nodes, node_id_by_name, plat_id_by_node_local = build_test_nodes(MAX_N)

    print("=" * 80)
    print("Cold Start / queueTime / Last-Task Mechanics — Deep A/B Audit")
    print(
        f"T_pull={theory['t_pull']:.2f}s  cold={theory['cold_start']:.2f}s  "
        f"exec={theory['exec_time']:.4f}s  baseline={theory['t_baseline']:.2f}s"
    )
    print("=" * 80)

    # --- Layer 1: SimPy pull timeline (N=4) ---
    mirror = run_mirror_contended_timeline(N_DETAIL, theory)
    print(f"\n--- Layer 1: SimPy pull timeline (N={N_DETAIL}, ground truth) ---")
    print(f"{'Task':>4}  {'pull_start':>11}  {'pull_end':>9}  {'queue_proxy':>12}  {'RTT':>8}  {'pred':>8}")
    mirror_ok = True
    for row in mirror:
        match = abs(row["pull_start"] - row["expected_pull_start"]) < MIRROR_TOLERANCE_S
        if not match:
            mirror_ok = False
        print(
            f"{row['taskId']:>4}  {row['pull_start']:>10.2f}s  {row['pull_end']:>8.2f}s  "
            f"{row['queue_time_proxy']:>11.2f}s  {row['rtt']:>7.2f}s  "
            f"{row['predicted_rtt']:>7.2f}s  {'OK' if match else 'FAIL'}"
        )

    # --- Layer 2: N sweep contended vs parallel ---
    print("\n--- Layer 2: Last-task RTT sweep (contended vs parallel) ---")
    sweep_rows: List[Dict[str, Any]] = []
    sweep_ok = True
    for n in SWEEP_N:
        fc, dc = contended_placements(n, node_id_by_name, plat_id_by_node_local)
        fp, dp = parallel_placements(n, node_id_by_name, plat_id_by_node_local)
        res_c = run_scenario(sim_inputs, nodes, fc, dc, n, label="contended", defer_cold=True)
        res_p = run_scenario(sim_inputs, nodes, fp, dp, n, label="parallel", defer_cold=True)
        row = {
            "n": n,
            "contended_last": res_c["last_task_rtt"],
            "parallel_last": res_p["last_task_rtt"],
            "penalty_abs": res_c["last_task_rtt"] - res_p["last_task_rtt"],
            "contended_total": res_c["total_rtt"],
        }
        sweep_rows.append(row)
    if not print_sweep_table(theory, sweep_rows):
        sweep_ok = False

    print("\n--- RTT multiplier (contended_last / parallel_last) ---")
    print(f"{'N':>3}  {'Multiplier':>10}  {'Linear N':>10}")
    for row in sweep_rows:
        mult = row["contended_last"] / row["parallel_last"]
        print(f"{row['n']:>3}  {mult:>9.2f}x  {row['n']:>9}x")

    # --- Layer 3: N=4 counterfactuals ---
    print(f"\n--- Layer 3: Counterfactuals at N={N_DETAIL} ---")
    fc, dc = contended_placements(N_DETAIL, node_id_by_name, plat_id_by_node_local)
    fp, dp = parallel_placements(N_DETAIL, node_id_by_name, plat_id_by_node_local)
    nodes_multi, _, _ = build_test_nodes(MAX_N, multi_flash_on_node0=4)

    scenarios = [
        ("Contended cold (A)", nodes, fc, dc, {"defer_cold": True, "fast_forward_warmup": True}),
        ("Parallel cold (B)", nodes, fp, dp, {"defer_cold": True, "fast_forward_warmup": True}),
        ("Contended warm (C)", nodes, fc, dc, {"defer_cold": False, "fast_forward_warmup": True}),
        ("Multi-storage 4×flash (D)", nodes_multi, fc, dc, {"defer_cold": True, "fast_forward_warmup": True}),
    ]
    cf_results: Dict[str, Dict[str, Any]] = {}
    print(f"{'Scenario':<28}  {'Last RTT':>10}  {'Max RTT':>10}  {'total_rtt':>10}  {'N× gone?':>8}")
    baseline_parallel = None
    for name, nds, forced, det, kwargs in scenarios:
        res = run_scenario(sim_inputs, nds, forced, det, N_DETAIL, label=name, **kwargs)
        cf_results[name] = res
        if "Parallel" in name:
            baseline_parallel = res["last_task_rtt"]
    for name, _, _, _, _ in scenarios:
        res = cf_results[name]
        nx_gone = "—"
        if baseline_parallel is not None and name != "Parallel cold (B)":
            ratio = res["last_task_rtt"] / baseline_parallel if baseline_parallel else float("nan")
            nx_gone = "Yes" if ratio < 1.5 else "No"
        print(
            f"{name:<28}  {res['last_task_rtt']:>9.2f}s  {res['max_task_rtt']:>9.2f}s  "
            f"{res['total_rtt']:>9.2f}s  {nx_gone:>8}"
        )

    contended = cf_results["Contended cold (A)"]
    parallel = cf_results["Parallel cold (B)"]

    # --- Layer 4: Deep dissection ---
    print_timing_dissection(contended["per_task"], theory)
    print_three_queues(contended["queue_signals"], theory)

    # --- Layer 5: Metric traps (FF on vs off) ---
    contended_no_ff = run_scenario(
        sim_inputs, nodes, fc, dc, N_DETAIL,
        label="contended_no_ff", defer_cold=True, fast_forward_warmup=False,
    )
    print_metric_traps(contended, contended_no_ff, parallel, theory)

    # --- Summary verdict ---
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    n = N_DETAIL
    penalty = contended["last_task_rtt"] - parallel["last_task_rtt"]
    pred_penalty = (n - 1) * theory["t_pull"]
    penalty_ok = abs(penalty - pred_penalty) <= TOLERANCE_S
    print(f"  Last-task N× penalty: {penalty:.2f}s  (predicted (N−1)×T_pull = {pred_penalty:.2f}s)  {'OK' if penalty_ok else 'FAIL'}")
    print(f"  Platform queue at scheduling: {contended['queue_signals'].get('gnn_platform_queue_max', 0)}  (not the driver)")
    print(f"  shared_fate saturates at ~{contended['queue_signals'].get('gnn_shared_fate_node0', 0):.1f}  (no N ordinal)")
    print(f"  queueTime grows linearly: last = {contended['queue_signals'].get('outcome_last_queue_time', 0):.2f}s ≈ N×T_pull")
    print(f"  Warm + multi-storage counterfactuals remove N× (see Layer 3)")
    print(f"  Use last-task RTT and queueTime — not pullTime (FF), not total_rtt")

    out_dir = PROJECT_ROOT / "simulation_data/cold_start_queue_last_task_ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "theory": theory,
        "mirror_n4": mirror,
        "sweep_rows": sweep_rows,
        "counterfactuals_n4": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_task"}
            for k, v in cf_results.items()
        },
        "contended_n4_per_task": contended["per_task"],
        "pulltime_ff_comparison": {
            "fast_forward_on": [t["pullTime"] for t in contended["per_task"]],
            "fast_forward_off": [t["pullTime"] for t in contended_no_ff["per_task"]],
        },
        "mirror_pass": mirror_ok,
        "sweep_pass": sweep_ok,
        "penalty_pass": penalty_ok,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {summary_path}")

    # --- Warmth physics matrix (Gate B / F) ---
    print("\n--- Warmth physics matrix (N=4 contended cold) ---")
    warmth_ok = True
    warmth_rows: List[Dict[str, Any]] = []
    for physics, expected_last in (
        ("platform_reuse_v1", 125.57),
        ("node_disk_v2", 31.65),
    ):
        res = run_scenario(
            sim_inputs, nodes, fc, dc, N_DETAIL,
            label=f"contended_{physics}",
            defer_cold=True,
            fast_forward_warmup=True,
            warmth_physics=physics,
        )
        last = res["last_task_rtt"]
        ok = abs(last - expected_last) <= TOLERANCE_S
        warmth_ok = warmth_ok and ok
        tasks_13_cold = all(
            float(t.get("coldStartTime", 0)) > 0
            for t in res["per_task"][1:]
        ) if physics == "node_disk_v2" and len(res["per_task"]) > 1 else True
        if physics == "node_disk_v2" and not tasks_13_cold:
            warmth_ok = False
        warmth_rows.append({
            "warmth_physics": physics,
            "last_task_rtt": last,
            "expected_last": expected_last,
            "pass": ok and tasks_13_cold,
            "tasks_1_3_cold_start": tasks_13_cold,
        })
        print(
            f"  {physics:<20} last={last:>7.2f}s  expected≈{expected_last:.2f}s  "
            f"{'OK' if ok else 'FAIL'}"
            + (f"  tasks1-3 cold={'OK' if tasks_13_cold else 'FAIL'}" if physics == "node_disk_v2" else "")
        )
    summary["warmth_physics_matrix"] = warmth_rows
    summary_path.write_text(json.dumps(summary, indent=2))

    all_pass = mirror_ok and sweep_ok and penalty_ok and warmth_ok
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
