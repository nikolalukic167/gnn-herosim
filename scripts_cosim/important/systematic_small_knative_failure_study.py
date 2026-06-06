#!/usr/bin/env python3
"""
Systematic small-sim study:
- Build heterogeneous small scenarios
- Compute brute-force oracle RTT
- Run knative baseline RTT
- Report where min-queue-like behavior fails (regret vs oracle)
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Must be set before importing simulation modules that read env at import time.
os.environ.setdefault("GNN_CAPTURE_DATASET_STATE", "1")
os.environ.setdefault("SIM_FORCE_FULL_STATS", "1")

from src.executecosimulation import execute_brute_force_optimized
from src.executeknativecosim import run_knative_baseline_for_dataset, setup_logging
from src.generate_infrastructure import generate_deterministic_infrastructure
from src.sample_loader import load_primary_sample_and_mapping


@dataclass
class Scenario:
    name: str
    clients: int
    servers: int
    connection_probability: float
    seed: int
    latency_profile: str  # "sym" | "asym"
    queue_profile: str  # "mid" | "high"
    preinit_profile: str  # "cold" | "warm"


def _queue_params(profile: str) -> Dict[str, Any]:
    if profile == "high":
        return {"type": "normal", "mean": 34, "stddev": 10, "min": 0, "max": 100, "step": 1}
    return {"type": "poisson", "lambda": 18, "min": 0, "max": 56, "step": 1}


def _build_scenarios(size_mode: str) -> List[Scenario]:
    if size_mode == "large":
        sizes: List[Tuple[str, int, int]] = [
            ("balanced", 40, 40),
            ("client_heavy", 50, 35),
            ("server_heavy", 35, 50),
        ]
    else:
        sizes = [
            ("balanced", 8, 8),
            ("client_heavy", 12, 6),
            ("server_heavy", 6, 12),
        ]
    probs = [0.35, 0.6]
    latency_profiles = ["sym", "asym"]

    scenarios: List[Scenario] = []
    idx = 0
    for size_name, clients, servers in sizes:
        for p in probs:
            for lp in latency_profiles:
                queue_profile = "high" if (p < 0.5 or lp == "asym") else "mid"
                preinit_profile = "cold" if lp == "asym" else "warm"
                scenarios.append(
                    Scenario(
                        name=f"{idx:02d}_{size_name}_p{int(p*100)}_{lp}_{queue_profile}_{preinit_profile}",
                        clients=clients,
                        servers=servers,
                        connection_probability=p,
                        seed=101 + idx,
                        latency_profile=lp,
                        queue_profile=queue_profile,
                        preinit_profile=preinit_profile,
                    )
                )
                idx += 1
    return scenarios


def _prepare_config(base_config: Dict[str, Any], s: Scenario, num_tasks: int) -> Dict[str, Any]:
    cfg = deepcopy(base_config)
    cfg["nodes"]["client_nodes"]["count"] = s.clients
    cfg["nodes"]["server_nodes"]["count"] = s.servers
    cfg["network"]["topology"]["connection_probability"] = s.connection_probability
    cfg["network"]["topology"]["seed"] = s.seed

    latency = cfg["network"]["latency"]["device_latencies"]
    for src, dsts in latency.items():
        for dst, rng in dsts.items():
            base_min = float(rng["min"])
            base_max = float(rng["max"])
            if s.latency_profile == "asym":
                factor = 2.0 if src != dst else 1.2
            else:
                factor = 1.0
            rng["min"] = round(base_min * factor, 4)
            rng["max"] = round(base_max * factor, 4)

    if s.preinit_profile == "cold":
        cfg["preinit"] = {"client_percentage": 0.0, "server_percentage": 0.0}
    else:
        cfg["preinit"] = {"client_percentage": 0.4, "server_percentage": 0.6}

    cfg["replicas"] = {
        "dnn1": {"per_client": 1, "per_server": 2},
        "dnn2": {"per_client": 1, "per_server": 2},
    }
    q_params = _queue_params(s.queue_profile)
    cfg["prewarm"] = {
        "dnn1": {
            "distribution": "none",
            "queue_distribution": "statistical",
            "queue_distribution_params": q_params,
        },
        "dnn2": {
            "distribution": "none",
            "queue_distribution": "statistical",
            "queue_distribution_params": q_params,
        },
    }
    cfg.setdefault("scheduler", {})
    cfg["scheduler"]["batch_size"] = num_tasks
    cfg["scheduler"]["batch_timeout"] = 0.1
    return cfg


def _prepare_workload(base_workload: Dict[str, Any], clients: int, num_tasks: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    events = base_workload.get("events", [])
    out_events: List[Dict[str, Any]] = []
    for i in range(num_tasks):
        ev = deepcopy(events[i % len(events)])
        task_type = "dnn1" if i % 2 == 0 else "dnn2"
        ev["application"]["name"] = f"nofs-{task_type}"
        ev["application"]["dag"] = {task_type: []}
        ev["node_name"] = f"client_node{rng.randint(0, max(0, clients - 1))}"
        out_events.append(ev)

    return {"rps": base_workload.get("rps", 1), "duration": 1, "events": out_events}


def _mean_queue_from_knative_capture(captured: Dict[str, Any]) -> float:
    values: List[float] = []
    for tr in captured.get("task_placements", []):
        snap = tr.get("queue_snapshot_at_scheduling") or {}
        if isinstance(snap, dict):
            for v in snap.values():
                if isinstance(v, (int, float)):
                    values.append(float(v))
    if not values:
        return 0.0
    return float(mean(values))


def _offload_rate(captured: Dict[str, Any]) -> float:
    placements = captured.get("task_placements", [])
    if not placements:
        return 0.0
    offloaded = 0
    total = 0
    for tr in placements:
        src = tr.get("source_node")
        dst = tr.get("execution_node")
        if src is None or dst is None:
            continue
        total += 1
        if src != dst:
            offloaded += 1
    if total == 0:
        return 0.0
    return offloaded / total


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Systematic small-sim knative failure study")
    parser.add_argument(
        "--output-subdir",
        default="hetero_small_knative_eval_20260606",
        help="Output directory under simulation_data/",
    )
    parser.add_argument("--max-scenarios", type=int, default=12, help="Max scenarios to execute")
    parser.add_argument("--num-tasks", type=int, default=4, choices=[2, 3, 4, 5], help="Tasks per workload")
    parser.add_argument(
        "--size-mode",
        choices=["small", "large"],
        default="small",
        help="Scenario topology size preset",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--max-combinations-skip",
        type=int,
        default=120000,
        help="Skip scenarios above this placement count",
    )
    args = parser.parse_args()

    os.environ.setdefault("MAX_PLACEMENT_COMBINATIONS_SKIP", str(args.max_combinations_skip))
    os.environ.setdefault("SIM_DEBUG_DETERMINED", "0")

    base_config_path = PROJECT_ROOT / "simulation_data" / "space_with_network.json"
    sample_json_file = PROJECT_ROOT / "simulation_data" / "sample_simple.json"
    samples_file = PROJECT_ROOT / "simulation_data" / "lhs_samples_simple.npy"
    mapping_file = PROJECT_ROOT / "simulation_data" / "lhs_samples_simple_mapping.pkl"
    sim_input_path = PROJECT_ROOT / "data" / "nofs-ids"
    base_workload_path = sim_input_path / "traces" / "workload-10.json"
    output_root = PROJECT_ROOT / "simulation_data" / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    with open(base_config_path, "r") as f:
        base_config = json.load(f)
    with open(base_workload_path, "r") as f:
        base_workload = json.load(f)

    sample, mapping, sample_source = load_primary_sample_and_mapping(
        sample_json_path=sample_json_file,
        samples_npy_path=samples_file,
        mapping_pkl_path=mapping_file,
    )
    apps = list(base_config["wsc"].keys())
    scenarios = _build_scenarios(args.size_mode)[: args.max_scenarios]
    logger = setup_logging(output_root)

    print("=== Systematic Small-Sim Knative Failure Study ===")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Size mode: {args.size_mode}")
    print(f"Workers per oracle run: {args.workers}")
    print(f"MAX_PLACEMENT_COMBINATIONS_SKIP={os.environ['MAX_PLACEMENT_COMBINATIONS_SKIP']}")
    print(f"Sample source: {sample_source}")

    rows: List[Dict[str, Any]] = []
    for i, s in enumerate(scenarios):
        scenario_dir = output_root / f"ds_{i:03d}_{s.name}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        cfg = _prepare_config(base_config, s, args.num_tasks)
        workload = _prepare_workload(base_workload, s.clients, args.num_tasks, s.seed)
        _save_json(scenario_dir / "space_with_network.json", cfg)
        _save_json(scenario_dir / "workload.json", workload)
        _save_json(sim_input_path / "traces" / "workload-10.json", workload)

        infra_file = scenario_dir / "infrastructure.json"
        generate_deterministic_infrastructure(
            str(scenario_dir / "space_with_network.json"),
            sim_input_path,
            str(infra_file),
            s.seed,
        )

        tmp_results = scenario_dir / "_tmp_results"
        tmp_results.mkdir(parents=True, exist_ok=True)

        start = time.time()
        status = "success"
        oracle_rtt = None
        knative_rtt = None
        mean_queue = None
        offload_rate = None
        error_msg = ""

        try:
            result_paths = execute_brute_force_optimized(
                apps=apps,
                config_file=str(scenario_dir / "space_with_network.json"),
                mapping_file=str(mapping_file),
                output_dir=tmp_results,
                sample=sample,
                sim_input_path=sim_input_path,
                workload_base_file=str(sim_input_path / "traces" / "workload-10.json"),
                max_workers=args.workers,
                infrastructure_file=infra_file,
                quiet=True,
                final_dataset_dir=scenario_dir,
                fast_forward_warmup=True,
                fast_forward_threshold=1,
                allow_non_unique_replicas=True,
                mapping_override=mapping,
            )
            best_json = tmp_results / "best.json"
            if not best_json.exists():
                raise RuntimeError("Oracle run finished but best.json is missing (likely skipped/infeasible).")
            with open(best_json, "r") as f:
                best = json.load(f)
            oracle_rtt = float(best["rtt"])
            shutil.copy2(best_json, scenario_dir / "best.json")

            optimal_src = tmp_results / best.get("file", "")
            if optimal_src.exists():
                shutil.copy2(optimal_src, scenario_dir / "optimal_result.json")
            placements_src = tmp_results / "placements.jsonl"
            if placements_src.exists():
                (scenario_dir / "placements").mkdir(exist_ok=True)
                shutil.copy2(placements_src, scenario_dir / "placements" / "placements.jsonl")

            ok = run_knative_baseline_for_dataset(
                dataset_dir=scenario_dir,
                sim_input_path=sim_input_path,
                logger=logger,
                num_tasks=args.num_tasks,
            )
            if not ok:
                raise RuntimeError("Knative baseline run failed.")
            with open(scenario_dir / "system_state_captured_unique.json", "r") as f:
                captured = json.load(f)
            knative_rtt = float(captured.get("total_rtt", 0.0))
            mean_queue = _mean_queue_from_knative_capture(captured)
            offload_rate = _offload_rate(captured)

            _save_json(
                scenario_dir / "scenario_manifest.json",
                {
                    "name": s.name,
                    "clients": s.clients,
                    "servers": s.servers,
                    "connection_probability": s.connection_probability,
                    "seed": s.seed,
                    "latency_profile": s.latency_profile,
                    "queue_profile": s.queue_profile,
                    "preinit_profile": s.preinit_profile,
                },
            )
        except Exception as e:
            status = "failed"
            error_msg = str(e)
        finally:
            if tmp_results.exists():
                shutil.rmtree(tmp_results, ignore_errors=True)

        elapsed = time.time() - start
        regret_abs = None
        regret_pct = None
        if oracle_rtt is not None and knative_rtt is not None and oracle_rtt > 0:
            regret_abs = knative_rtt - oracle_rtt
            regret_pct = (regret_abs / oracle_rtt) * 100.0

        row = {
            "scenario_dir": scenario_dir.name,
            "status": status,
            "error": error_msg,
            "duration_sec": round(elapsed, 2),
            "clients": s.clients,
            "servers": s.servers,
            "connection_probability": s.connection_probability,
            "latency_profile": s.latency_profile,
            "queue_profile": s.queue_profile,
            "preinit_profile": s.preinit_profile,
            "oracle_rtt": oracle_rtt,
            "knative_rtt": knative_rtt,
            "regret_abs": regret_abs,
            "regret_pct": regret_pct,
            "mean_queue_at_sched": mean_queue,
            "offload_rate": offload_rate,
        }
        rows.append(row)
        print(
            f"[{i+1}/{len(scenarios)}] {scenario_dir.name} status={status} "
            f"oracle={oracle_rtt} knative={knative_rtt} regret_pct={regret_pct}"
        )

    _save_json(output_root / "study_results.json", {"rows": rows})

    success_rows = [r for r in rows if r["status"] == "success" and r["regret_pct"] is not None]
    failed_rows = [r for r in rows if r["status"] != "success"]
    failures_over_10 = [r for r in success_rows if r["regret_pct"] > 10.0]
    failures_over_25 = [r for r in success_rows if r["regret_pct"] > 25.0]

    summary = {
        "total_scenarios": len(rows),
        "successful": len(success_rows),
        "failed": len(failed_rows),
        "mean_regret_pct": round(mean([r["regret_pct"] for r in success_rows]), 3) if success_rows else None,
        "median_regret_pct": round(sorted([r["regret_pct"] for r in success_rows])[len(success_rows) // 2], 3)
        if success_rows
        else None,
        "failures_over_10pct": len(failures_over_10),
        "failures_over_25pct": len(failures_over_25),
        "top_failures": sorted(
            [
                {
                    "scenario_dir": r["scenario_dir"],
                    "regret_pct": round(r["regret_pct"], 3),
                    "oracle_rtt": r["oracle_rtt"],
                    "knative_rtt": r["knative_rtt"],
                    "clients": r["clients"],
                    "servers": r["servers"],
                    "connection_probability": r["connection_probability"],
                    "latency_profile": r["latency_profile"],
                    "queue_profile": r["queue_profile"],
                }
                for r in success_rows
            ],
            key=lambda x: x["regret_pct"],
            reverse=True,
        )[:5],
    }
    _save_json(output_root / "study_summary.json", summary)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    if failed_rows:
        print("\nFailed scenarios:")
        for r in failed_rows:
            print(f"- {r['scenario_dir']}: {r['error']}")
    print(f"\nArtifacts: {output_root}")


if __name__ == "__main__":
    main()
