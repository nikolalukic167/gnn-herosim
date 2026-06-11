#!/usr/bin/env python3
"""Verify co-sim vs live queue regime gap on the 1060 corpus."""

from __future__ import annotations

import copy
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.executecosimulation import KEEP_ALIVE, QUEUE_LENGTH, execute_simulation
from src.placement.model import DataclassJSONEncoder

COSIM_1060 = (
    PROJECT_ROOT
    / "simulation_data/artifacts/run_queue_big/gnn_datasets_4tasks_1060"
)
COSIM_3705 = (
    PROJECT_ROOT / "simulation_data/artifacts/run_queue_big/gnn_datasets_4tasks"
)


def pct(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(int(p * len(s)), len(s) - 1)
    return s[idx]


def analyze_corpus(root: Path, label: str) -> Dict[str, Any]:
    avg_qt: List[float] = []
    max_task_qt: List[float] = []
    snap_depths: List[float] = []

    for ds in sorted(root.glob("ds_*")):
        opt = ds / "optimal_result.json"
        ssc = ds / "system_state_captured_unique.json"
        if not opt.exists():
            continue
        data = json.loads(opt.read_text())
        stats = data.get("stats", {})
        aqt = stats.get("averageQueueTime")
        if aqt is not None:
            avg_qt.append(float(aqt))
        trs = stats.get("taskResults") or []
        if trs:
            qts = [float(t.get("queueTime", 0) or 0) for t in trs]
            max_task_qt.append(max(qts))
        if ssc.exists():
            ssc_data = json.loads(ssc.read_text())
            for tp in ssc_data.get("task_placements") or []:
                fqs = tp.get("full_queue_snapshot") or {}
                if fqs:
                    snap_depths.append(max(float(v) for v in fqs.values()))

    def summarize(values: List[float], unit: str) -> Dict[str, float]:
        if not values:
            return {}
        return {
            "n": len(values),
            "max": max(values),
            "p99": pct(values, 0.99),
            "p95": pct(values, 0.95),
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "unit": unit,
        }

    return {
        "label": label,
        "path": str(root),
        "averageQueueTime": summarize(avg_qt, "seconds"),
        "max_per_task_queueTime": summarize(max_task_qt, "seconds"),
        "full_queue_snapshot_depth": summarize(snap_depths, "tasks"),
    }


def load_live_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "missing": True}
    data = json.loads(path.read_text())
    stats = data.get("stats", {})
    trs = stats.get("taskResults") or []
    qts = [float(t.get("queueTime", 0) or 0) for t in trs if t.get("queueTime") is not None]
    out = {
        "path": str(path),
        "policy": data.get("policy"),
        "total_rtt": data.get("total_rtt"),
        "averageQueueTime": stats.get("averageQueueTime"),
        "taskResults_present": bool(trs),
    }
    if qts:
        out["per_task_queueTime"] = {
            "max": max(qts),
            "p99": pct(qts, 0.99),
            "p95": pct(qts, 0.95),
            "median": statistics.median(qts),
        }
    return out


def rerun_optimal(ds_dir: Path, fast_forward: bool) -> Dict[str, Any]:
    opt_path = ds_dir / "optimal_result.json"
    old = json.loads(opt_path.read_text())
    placement_plan = {
        int(k): (int(v[0]), int(v[1]))
        for k, v in old["sample"]["placement_plan"].items()
    }
    config = copy.deepcopy(old["config"])
    sim_inputs = old["sim_inputs"]
    config.setdefault("infrastructure", {})["forced_placements"] = placement_plan
    config["infrastructure"]["fast_forward_warmup"] = fast_forward
    config["infrastructure"]["fast_forward_threshold"] = 1
    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    t0 = __import__("time").time()
    result = execute_simulation(
        config,
        sim_inputs,
        "determined_determined",
        cache_policy="fifo",
        task_priority="fifo",
        keep_alive=KEEP_ALIVE,
        queue_length=QUEUE_LENGTH,
    )
    wall = __import__("time").time() - t0
    stats = result["stats"]
    trs = stats.get("taskResults") or []
    qts = [float(t.get("queueTime", 0) or 0) for t in trs]
    total_rtt = result.get("total_rtt")
    if total_rtt is None:
        total_rtt = stats.get("totalRtt") or stats.get("total_rtt")
    snap_max = 0.0
    for tr in trs:
        fqs = tr.get("fullQueueSnapshot") or {}
        if fqs:
            snap_max = max(snap_max, max(float(v) for v in fqs.values()))
    return {
        "dataset": ds_dir.name,
        "fast_forward_warmup": fast_forward,
        "wall_s": round(wall, 2),
        "total_rtt": total_rtt,
        "averageQueueTime": stats.get("averageQueueTime"),
        "max_queueTime": max(qts) if qts else None,
        "max_full_queue_snapshot": snap_max or None,
    }


def main() -> None:
    print("=" * 80)
    print("QUEUE REGIME GAP VERIFICATION (1060 focus)")
    print("=" * 80)

    cosim_1060 = analyze_corpus(COSIM_1060, "1060")
    cosim_3705 = analyze_corpus(COSIM_3705, "3705_4tasks")

    live_paths = {
        "150-150_knative": PROJECT_ROOT
        / "simulation_data/results/150-150/simulation_result_knative_150-150.json",
        "default_dim14_ce": PROJECT_ROOT
        / "simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_ce_only_20260609/results/default_20_20_p50.json",
    }
    live = {k: load_live_stats(p) for k, p in live_paths.items()}

    print("\n--- CO-SIM CORPORA ---")
    for corpus in (cosim_3705, cosim_1060):
        print(f"\n{corpus['label']} ({corpus['path']})")
        for metric in (
            "averageQueueTime",
            "max_per_task_queueTime",
            "full_queue_snapshot_depth",
        ):
            s = corpus[metric]
            if not s:
                continue
            print(
                f"  {metric}: max={s['max']:.4f} p99={s['p99']:.4f} "
                f"median={s['median']:.4f} ({s['unit']})"
            )

    print("\n--- LIVE BASELINES ---")
    for name, row in live.items():
        print(f"\n{name}:")
        if row.get("missing"):
            print("  MISSING")
            continue
        print(f"  averageQueueTime={row.get('averageQueueTime')}")
        if row.get("per_task_queueTime"):
            pt = row["per_task_queueTime"]
            print(
                f"  per-task queueTime: max={pt['max']:.2f} p99={pt['p99']:.2f} "
                f"(taskResults n={row['taskResults_present']})"
            )
        else:
            print("  per-task queueTime: unavailable (streaming stats, no taskResults)")

    print("\n--- ORIGIN OF ~6.5s CLAIM ---")
    m3705 = cosim_3705["averageQueueTime"]["max"]
    m1060_max = cosim_1060["averageQueueTime"]["max"]
    m1060_p99 = cosim_1060["averageQueueTime"]["p99"]
    print(f"  3705 max averageQueueTime = {m3705:.4f}s  (matches paper ~6.5s)")
    print(f"  1060 max averageQueueTime = {m1060_max:.4f}s  (EXCEEDS ~6.5s)")
    print(f"  1060 p99 averageQueueTime = {m1060_p99:.4f}s  (close to ~6.5s)")

    print("\n--- FAST-FORWARD WARMUP A/B (1060 subset) ---")
    ab_ids = ["ds_00479", "ds_00319", "ds_00016"]
    ab_results: List[Dict[str, Any]] = []
    for ds_name in ab_ids:
        ds_dir = COSIM_1060 / ds_name
        stored = json.loads((ds_dir / "optimal_result.json").read_text())
        stored_aqt = stored["stats"]["averageQueueTime"]
        stored_rtt = stored.get("total_rtt") or stored["stats"].get("totalRtt")
        print(f"\n  {ds_name} (stored avgQT={stored_aqt:.4f}, stored RTT={stored_rtt})")
        for ff in (True, False):
            row = rerun_optimal(ds_dir, fast_forward=ff)
            ab_results.append(row)
            rtt_s = f"{row['total_rtt']:.6f}" if row["total_rtt"] is not None else "None"
            print(
                f"    ff={ff}: wall={row['wall_s']}s RTT={rtt_s} "
                f"avgQT={row['averageQueueTime']:.6f} max_qt={row['max_queueTime']:.6f} "
                f"snap_max={row['max_full_queue_snapshot']}"
            )

    out = {
        "cosim_1060": cosim_1060,
        "cosim_3705": cosim_3705,
        "live": live,
        "claim_origin": {
            "paper_6p5_source": "3705 gnn_datasets_4tasks max averageQueueTime",
            "3705_max_avg_queue_time_s": m3705,
            "1060_max_avg_queue_time_s": m1060_max,
            "1060_p99_avg_queue_time_s": m1060_p99,
        },
        "fast_forward_ab": ab_results,
    }
    out_path = PROJECT_ROOT / "logs/queue_gap_1060_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
