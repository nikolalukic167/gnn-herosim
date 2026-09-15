#!/usr/bin/env python3
"""
Regime B metrics calibration — fail-loud CI for headroom gate.

Modes:
  toy     — N=4 platform_reuse_v1 regression (oracle regret ≈ 3.97×)
  target  — frozen problem spec N=12 (≥10× oracle–greedy on max-burst)

Usage:
    pipenv run python3 scripts_cosim/calibrate_regime_b.py
    pipenv run python3 scripts_cosim/calibrate_regime_b.py --mode target
    pipenv run python3 scripts_cosim/calibrate_regime_b.py --mode both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.regime_b_metrics import (  # noqa: E402
    attach_burst_ids_from_workload,
    burst_regime_summary,
    total_rtt_trap_stats,
)
from scripts_cosim.regime_b_problem_spec import (  # noqa: E402
    GATE_WARMTH_PHYSICS,
    MIN_ORACLE_GREEDY_RATIO,
    PROBLEM_ID,
    SPEC_VERSION,
    TARGET_BURST_ID,
    TARGET_CLIENT,
    TARGET_EXPECTED_RATIO,
    TARGET_N_TASKS,
    TARGET_PLATFORMS_ON_OTHER_NODES,
    TARGET_PLATFORMS_ON_SCARCE_NODE,
    TARGET_SCORE_TOLERANCE_S,
    TARGET_SERVER_COUNT,
    TARGET_TASK_TYPE,
    TOY_MIN_RATIO,
    TOY_N_TASKS,
    as_dict,
    assert_gate_ratio,
    expected_greedy_primary_s,
    expected_oracle_primary_s,
)
from src.executecosimulation import (  # noqa: E402
    KEEP_ALIVE,
    QUEUE_LENGTH,
    execute_simulation,
    extract_task_metrics,
    load_simulation_inputs,
    rtt_from_stats,
)

SIM_INPUT = PROJECT_ROOT / "data/nofs-ids"
TOLERANCE_S = 1.0
OUT_DIR = PROJECT_ROOT / "simulation_data/regime_b_calibration"

# Toy expected last-task RTT (storage_contention.md / cold-start AB audit)
TOY_EXPECTED = {
    "platform_reuse_v1": {"contended": 125.57, "parallel": 31.65, "min_regret_ratio": TOY_MIN_RATIO},
    "node_disk_v2": {"contended": 31.65, "parallel": 31.65, "min_regret_ratio": 0.9},
}


def build_burst_workload(
    n: int,
    *,
    burst_id: str,
    task_type: str = "dnn1",
    timestamps: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Build N-task burst workload. Optional per-event timestamps (default all t=0)."""
    if timestamps is None:
        ts_list = [0.0] * n
    else:
        ts_list = [float(t) for t in timestamps]
        if len(ts_list) != n:
            raise ValueError(
                f"FAIL LOUD: timestamps length {len(ts_list)} != n={n}"
            )
        if any(t < 0.0 for t in ts_list):
            raise ValueError(f"FAIL LOUD: negative timestamps in {ts_list}")
    events = [
        {
            "timestamp": ts_list[i],
            "application": {"name": f"nofs-{task_type}", "dag": {task_type: []}},
            "qos": {"name": "medium", "maxDurationDeviation": 15},
            "node_name": TARGET_CLIENT,
            "burst_id": burst_id,
        }
        for i in range(n)
    ]
    # duration must cover last arrival + headroom for sim end.
    last_ts = max(ts_list) if ts_list else 0.0
    return {
        "rps": max(n, 1),
        "duration": max(1, int(last_ts) + 1),
        "events": events,
    }


def _rpi_node(
    node_name: str,
    peer_names: List[str],
    *,
    n_platforms: int,
    latency: float = 0.001,
) -> Dict[str, Any]:
    if n_platforms < 1:
        raise ValueError(f"n_platforms must be >= 1, got {n_platforms}")
    return {
        "node_name": node_name,
        "type": "rpi",
        "memory": 8,
        "platforms": ["rpiCpu"] * n_platforms,
        "storage": ["flashCard", "someRemote"],
        "network_map": {peer: latency for peer in peer_names},
    }


def build_target_nodes(
    n_servers: int,
    *,
    platforms_scarce: int,
    platforms_other: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[Tuple[str, int], int]]:
    """1 client + N servers; node0 holds N platforms (scarce FilterStore attractor)."""
    if platforms_scarce < n_servers:
        raise ValueError(
            f"scarce node needs ≥{n_servers} platforms for contended N-way, got {platforms_scarce}"
        )
    server_names = [f"node{i}" for i in range(n_servers)]
    all_names = [TARGET_CLIENT, *server_names]
    nodes: List[Dict[str, Any]] = [
        _rpi_node(TARGET_CLIENT, server_names, n_platforms=1),
    ]
    for i, sname in enumerate(server_names):
        peers = [n for n in all_names if n != sname]
        n_plat = platforms_scarce if i == 0 else platforms_other
        nodes.append(_rpi_node(sname, peers, n_platforms=n_plat))

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
        key = (node_name, i)
        if key not in plat_id_by_node_local:
            raise KeyError(
                f"missing platform {key}; scarce node needs ≥{n} platforms "
                f"(have {[k for k in plat_id_by_node_local if k[0] == node_name]})"
            )
        plat_id = plat_id_by_node_local[key]
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


def run_burst_sim(
    sim_inputs: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    forced: Dict[int, Tuple[int, int]],
    det_placements: List[Dict[str, Any]],
    n: int,
    *,
    burst_id: str,
    warmth_physics: str,
    task_type: str = "dnn1",
) -> Dict[str, Any]:
    workload = build_burst_workload(n, burst_id=burst_id, task_type=task_type)
    server_names = [f"node{i}" for i in range(n)]
    infrastructure: Dict[str, Any] = {
        "network": {"bandwidth": 100.0},
        "nodes": nodes,
        "preinitialize_platforms": True,
        "defer_cold_replica_init": True,
        "warmth_physics": warmth_physics,
        "deterministic_replica_placements": {task_type: det_placements},
        "replica_plan": {
            "preinit_clients": [],
            "preinit_servers": server_names,
            "preinit_task_types": [task_type],
            "replicas_config": {task_type: {"per_client": 0, "per_server": 0}},
            "prewarm_config": {},
        },
        "forced_placements": forced,
        "scheduler": {"batch_size": n, "batch_timeout": 0.02},
        "fast_forward_warmup": True,
        "fast_forward_threshold": 1,
    }
    config = {"infrastructure": infrastructure, "workload": workload}

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
    task_rows = attach_burst_ids_from_workload(
        extract_task_metrics(stats),
        workload["events"],
    )
    if len(task_rows) != n:
        raise RuntimeError(
            f"expected {n} task rows for burst {burst_id!r}, got {len(task_rows)} — "
            "SIM_FORCE_FULL_STATS / placement failure"
        )
    return {
        "stats": stats,
        "workload": workload,
        "task_rows": task_rows,
        "total_rtt": rtt_from_stats(stats),
        "regime_b": burst_regime_summary(task_rows),
        "trap": total_rtt_trap_stats(task_rows),
    }


def _import_ab_helpers():
    from scripts_cosim.test_cold_start_queue_last_task_ab import (  # noqa: WPS433
        build_test_nodes,
        contended_placements as ab_contended,
        parallel_placements as ab_parallel,
        load_theory_constants,
    )

    return build_test_nodes, ab_contended, ab_parallel, load_theory_constants


def calibrate_toy(*, warmth_physics: str) -> Dict[str, Any]:
    build_test_nodes, ab_contended, ab_parallel, load_theory_constants = _import_ab_helpers()
    sim_inputs = load_simulation_inputs(SIM_INPUT)
    theory = load_theory_constants(sim_inputs)
    nodes, node_id_by_name, plat_id_by_node_local = build_test_nodes(TOY_N_TASKS)

    fc, dc = ab_contended(TOY_N_TASKS, node_id_by_name, plat_id_by_node_local)
    fp, dp = ab_parallel(TOY_N_TASKS, node_id_by_name, plat_id_by_node_local)

    oracle_run = run_burst_sim(
        sim_inputs,
        nodes,
        fp,
        dp,
        TOY_N_TASKS,
        burst_id="parallel_oracle",
        warmth_physics=warmth_physics,
    )
    oracle_score = float(oracle_run["regime_b"]["regime_b_primary_score_s"])

    contended_run = run_burst_sim(
        sim_inputs,
        nodes,
        fc,
        dc,
        TOY_N_TASKS,
        burst_id="contended_greedy",
        warmth_physics=warmth_physics,
    )
    contended_scored = burst_regime_summary(
        contended_run["task_rows"],
        oracle_rtt=oracle_score,
    )

    expected = TOY_EXPECTED.get(warmth_physics, TOY_EXPECTED["platform_reuse_v1"])
    contended_score = float(contended_scored["regime_b_primary_score_s"])
    regret_ratio = float(contended_scored.get("oracle_regret_ratio", float("nan")))

    checks = {
        "oracle_near_expected": abs(oracle_score - expected["parallel"]) <= TOLERANCE_S,
        "contended_near_expected": abs(contended_score - expected["contended"]) <= TOLERANCE_S,
        "regret_ratio_min": regret_ratio >= expected["min_regret_ratio"],
        "total_rtt_trap_visible": contended_run["trap"]["total_rtt_over_primary_ratio"] >= 3.0,
    }

    return {
        "mode": "toy",
        "warmth_physics": warmth_physics,
        "theory": theory,
        "n_tasks": TOY_N_TASKS,
        "oracle": {
            "burst_id": "parallel_oracle",
            "regime_b_primary_score_s": oracle_score,
            "expected_s": expected["parallel"],
            "total_rtt_s": oracle_run["total_rtt"],
            "burst_summaries": oracle_run["regime_b"]["burst_summaries"],
        },
        "contended": {
            "burst_id": "contended_greedy",
            "regime_b_primary_score_s": contended_score,
            "expected_s": expected["contended"],
            "total_rtt_s": contended_run["total_rtt"],
            "burst_summaries": contended_scored["burst_summaries"],
            "oracle_rtt_s": contended_scored.get("oracle_rtt_s"),
            "oracle_regret_s": contended_scored.get("oracle_regret_s"),
            "oracle_regret_ratio": regret_ratio,
            "total_rtt_trap": contended_run["trap"],
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def calibrate_target() -> Dict[str, Any]:
    """Frozen ≥10× gate on problem-spec cluster+trace (platform_reuse_v1, N=12)."""
    warmth_physics = GATE_WARMTH_PHYSICS
    n = TARGET_N_TASKS
    sim_inputs = load_simulation_inputs(SIM_INPUT)
    nodes, node_id_by_name, plat_id_by_node_local = build_target_nodes(
        TARGET_SERVER_COUNT,
        platforms_scarce=TARGET_PLATFORMS_ON_SCARCE_NODE,
        platforms_other=TARGET_PLATFORMS_ON_OTHER_NODES,
    )
    fc, dc = contended_placements(n, node_id_by_name, plat_id_by_node_local)
    fp, dp = parallel_placements(n, node_id_by_name, plat_id_by_node_local)

    oracle_run = run_burst_sim(
        sim_inputs,
        nodes,
        fp,
        dp,
        n,
        burst_id=f"{TARGET_BURST_ID}_oracle",
        warmth_physics=warmth_physics,
        task_type=TARGET_TASK_TYPE,
    )
    oracle_score = float(oracle_run["regime_b"]["regime_b_primary_score_s"])

    contended_run = run_burst_sim(
        sim_inputs,
        nodes,
        fc,
        dc,
        n,
        burst_id=TARGET_BURST_ID,
        warmth_physics=warmth_physics,
        task_type=TARGET_TASK_TYPE,
    )
    contended_scored = burst_regime_summary(
        contended_run["task_rows"],
        oracle_rtt=oracle_score,
    )
    contended_score = float(contended_scored["regime_b_primary_score_s"])
    regret_ratio = float(contended_scored.get("oracle_regret_ratio", float("nan")))

    exp_oracle = expected_oracle_primary_s()
    exp_greedy = expected_greedy_primary_s(n)

    checks = {
        "oracle_near_theory": abs(oracle_score - exp_oracle) <= TARGET_SCORE_TOLERANCE_S,
        "contended_near_theory": abs(contended_score - exp_greedy) <= TARGET_SCORE_TOLERANCE_S,
        "gate_ratio_ge_10": regret_ratio >= MIN_ORACLE_GREEDY_RATIO,
        "total_rtt_trap_visible": contended_run["trap"]["total_rtt_over_primary_ratio"] >= 5.0,
        "physics_is_gate": warmth_physics == GATE_WARMTH_PHYSICS,
    }

    # Fail loud on gate collapse (even if other checks soft-fail near-theory).
    if checks["gate_ratio_ge_10"]:
        assert_gate_ratio(regret_ratio, context="calibrate_target")
    else:
        try:
            assert_gate_ratio(regret_ratio, context="calibrate_target")
        except AssertionError as exc:
            checks["gate_assertion"] = str(exc)

    return {
        "mode": "target",
        "problem_id": PROBLEM_ID,
        "spec_version": SPEC_VERSION,
        "problem_spec": as_dict(),
        "warmth_physics": warmth_physics,
        "n_tasks": n,
        "expected_ratio_theory": TARGET_EXPECTED_RATIO,
        "oracle": {
            "burst_id": f"{TARGET_BURST_ID}_oracle",
            "regime_b_primary_score_s": oracle_score,
            "expected_s": exp_oracle,
            "total_rtt_s": oracle_run["total_rtt"],
            "burst_summaries": oracle_run["regime_b"]["burst_summaries"],
        },
        "contended": {
            "burst_id": TARGET_BURST_ID,
            "regime_b_primary_score_s": contended_score,
            "expected_s": exp_greedy,
            "total_rtt_s": contended_run["total_rtt"],
            "burst_summaries": contended_scored["burst_summaries"],
            "oracle_rtt_s": contended_scored.get("oracle_rtt_s"),
            "oracle_regret_s": contended_scored.get("oracle_regret_s"),
            "oracle_regret_ratio": regret_ratio,
            "total_rtt_trap": contended_run["trap"],
        },
        "checks": checks,
        "pass": all(
            v for k, v in checks.items() if k != "gate_assertion" and isinstance(v, bool)
        ),
    }


def _print_result(result: Dict[str, Any]) -> None:
    oracle = result["oracle"]
    contended = result["contended"]
    print(f"\nOracle (parallel cold burst)")
    print(
        f"  regime_b_primary_score_s: {oracle['regime_b_primary_score_s']:.2f}s  "
        f"(expected ≈{oracle['expected_s']:.2f}s)"
    )
    print(f"  total_rtt (misleading):   {oracle['total_rtt_s']:.2f}s")

    print(f"\nContended (co-located cold burst — FilterStore trap)")
    print(
        f"  regime_b_primary_score_s: {contended['regime_b_primary_score_s']:.2f}s  "
        f"(expected ≈{contended['expected_s']:.2f}s)"
    )
    print(f"  total_rtt (misleading):   {contended['total_rtt_s']:.2f}s")
    print(f"  total_rtt / primary:      {contended['total_rtt_trap']['total_rtt_over_primary_ratio']:.2f}x")

    if contended.get("oracle_regret_ratio") is not None:
        print(f"\nOracle regret")
        print(f"  oracle_rtt_s:             {contended['oracle_rtt_s']:.2f}s")
        print(f"  oracle_regret_s:          {contended['oracle_regret_s']:.2f}s")
        print(f"  oracle_regret_ratio:      {contended['oracle_regret_ratio']:.2f}x")

    print(f"\nChecks:")
    for name, ok in result["checks"].items():
        if isinstance(ok, bool):
            print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        else:
            print(f"  {name}: {ok}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Regime B metrics / ≥10× gate")
    parser.add_argument(
        "--mode",
        choices=["toy", "target", "both"],
        default="toy",
        help="toy=N4 regression; target=frozen ≥10× gate; both=run sequentially",
    )
    parser.add_argument(
        "--warmth-physics",
        default="platform_reuse_v1",
        choices=["platform_reuse_v1", "node_disk_v2"],
        help="Toy mode only (target always uses gate physics from problem spec)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR,
        help="Directory for calibration JSON output",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    # Always dump frozen spec beside calibration artifacts.
    spec_path = args.output / f"problem_spec_{PROBLEM_ID}.json"
    spec_path.write_text(json.dumps(as_dict(), indent=2) + "\n")

    modes = ["toy", "target"] if args.mode == "both" else [args.mode]
    overall_pass = True

    for mode in modes:
        print("=" * 72)
        if mode == "toy":
            print("Regime B Metrics Calibration — TOY (N=4 regression)")
            print(f"warmth_physics={args.warmth_physics}  N={TOY_N_TASKS}  SIM_FORCE_FULL_STATS=1")
            print("=" * 72)
            result = calibrate_toy(warmth_physics=args.warmth_physics)
            out_path = args.output / f"calibration_{args.warmth_physics}.json"
        else:
            print(f"Regime B Gate Calibration — TARGET ({PROBLEM_ID})")
            print(
                f"warmth_physics={GATE_WARMTH_PHYSICS}  N={TARGET_N_TASKS}  "
                f"min_ratio≥{MIN_ORACLE_GREEDY_RATIO:.0f}×  theory≈{TARGET_EXPECTED_RATIO:.2f}×"
            )
            print("=" * 72)
            result = calibrate_target()
            out_path = args.output / f"calibration_target_{PROBLEM_ID}.json"

        _print_result(result)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nResults written to {out_path}")
        print(f"Problem spec dumped to {spec_path}")

        print("\n" + "=" * 72)
        if result["pass"]:
            print(f"VERDICT: PASS — {mode}")
        else:
            print(f"VERDICT: FAIL — {mode} (see checks above)")
            overall_pass = False
        print("=" * 72)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
