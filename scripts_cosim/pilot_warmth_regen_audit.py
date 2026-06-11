#!/usr/bin/env python3
"""
Pilot warmth regen audit — verify node_disk_v2 label shift before full 1060 regen.

Runs N=4 contended cold scenarios under v1 vs v2 and audits co-sim config defaults.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.test_cold_start_queue_last_task_ab import (  # noqa: E402
    N_DETAIL,
    TOLERANCE_S,
    build_test_nodes,
    contended_placements,
    load_theory_constants,
    run_scenario,
)
from src.executecosimulation import load_simulation_inputs  # noqa: E402
import inspect  # noqa: E402
import src.executecosimulation as executecosim  # noqa: E402


def audit_prepare_simulation_config() -> dict:
    """Verify prepare_simulation_config source sets v2 co-sim defaults."""
    src = inspect.getsource(executecosim.prepare_simulation_config)
    return {
        "warmth_physics_default_node_disk_v2": '"warmth_physics"' in src
        and "node_disk_v2" in src,
        "defer_cold_default_true": "defer_cold_replica_init" in src,
    }


def main() -> int:
    sim_inputs = load_simulation_inputs(PROJECT_ROOT / "data/nofs-ids")
    theory = load_theory_constants(sim_inputs)
    nodes, node_id_by_name, plat_id_by_node_local = build_test_nodes(4)
    fc, dc = contended_placements(N_DETAIL, node_id_by_name, plat_id_by_node_local)

    results = {}
    for physics, expected in (
        ("platform_reuse_v1", 125.57),
        ("node_disk_v2", 31.65),
    ):
        res = run_scenario(
            sim_inputs,
            nodes,
            fc,
            dc,
            N_DETAIL,
            label=f"pilot_{physics}",
            warmth_physics=physics,
        )
        last = res["last_task_rtt"]
        max_q = max(float(t["queueTime"]) for t in res["per_task"])
        results[physics] = {
            "last_task_rtt": last,
            "max_queue_time": max_q,
            "expected_last": expected,
            "pass": abs(last - expected) <= TOLERANCE_S,
        }

    config_audit = audit_prepare_simulation_config()
    out = {
        "theory_t_pull": theory["t_pull"],
        "scenario_results": results,
        "prepare_simulation_config": config_audit,
        "pilot_pass": all(r["pass"] for r in results.values())
        and config_audit.get("warmth_physics_default_node_disk_v2")
        and config_audit.get("defer_cold_default_true"),
    }

    out_dir = PROJECT_ROOT / "simulation_data/pilot_warmth_regen_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0 if out["pilot_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
