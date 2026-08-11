#!/usr/bin/env python3
"""
Materialize Regime B live stub (cluster + burst-tagged workload) from frozen spec.

Does NOT run policies — only writes the env artifacts for later Kn/MLP/GNN baselines.

Usage:
    pipenv run python3 scripts_cosim/build_regime_b_live_stub.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.calibrate_regime_b import (  # noqa: E402
    build_burst_workload,
    build_target_nodes,
    contended_placements,
    parallel_placements,
)
from scripts_cosim.regime_b_problem_spec import (  # noqa: E402
    GATE_WARMTH_PHYSICS,
    LIVE_STUB_DIR,
    MIN_ORACLE_GREEDY_RATIO,
    PROBLEM_ID,
    SPEC_VERSION,
    TARGET_BURST_ID,
    TARGET_N_TASKS,
    TARGET_PLATFORMS_ON_OTHER_NODES,
    TARGET_PLATFORMS_ON_SCARCE_NODE,
    TARGET_SERVER_COUNT,
    TARGET_TASK_TYPE,
    as_dict,
)


def _forced_to_jsonable(
    forced: Dict[int, tuple],
) -> Dict[str, List[int]]:
    return {str(k): [int(v[0]), int(v[1])] for k, v in forced.items()}


def build_stub_payload() -> Dict[str, Any]:
    nodes, node_id_by_name, plat_id_by_node_local = build_target_nodes(
        TARGET_SERVER_COUNT,
        platforms_scarce=TARGET_PLATFORMS_ON_SCARCE_NODE,
        platforms_other=TARGET_PLATFORMS_ON_OTHER_NODES,
    )
    fc, dc = contended_placements(
        TARGET_N_TASKS, node_id_by_name, plat_id_by_node_local
    )
    fp, dp = parallel_placements(
        TARGET_N_TASKS, node_id_by_name, plat_id_by_node_local
    )
    workload = build_burst_workload(
        TARGET_N_TASKS,
        burst_id=TARGET_BURST_ID,
        task_type=TARGET_TASK_TYPE,
    )
    infrastructure: Dict[str, Any] = {
        "network": {"bandwidth": 100.0},
        "nodes": nodes,
        "preinitialize_platforms": True,
        "defer_cold_replica_init": True,
        "warmth_physics": GATE_WARMTH_PHYSICS,
        "replica_plan": {
            "preinit_clients": [],
            "preinit_servers": [f"node{i}" for i in range(TARGET_SERVER_COUNT)],
            "preinit_task_types": [TARGET_TASK_TYPE],
            "replicas_config": {
                TARGET_TASK_TYPE: {"per_client": 0, "per_server": 0}
            },
            "prewarm_config": {},
        },
        "scheduler": {
            "batch_size": TARGET_N_TASKS,
            "batch_timeout": 0.02,
        },
        "fast_forward_warmup": True,
        "fast_forward_threshold": 1,
    }
    return {
        "problem_id": PROBLEM_ID,
        "spec_version": SPEC_VERSION,
        "gate_min_oracle_greedy_ratio": MIN_ORACLE_GREEDY_RATIO,
        "warmth_physics": GATE_WARMTH_PHYSICS,
        "infrastructure": infrastructure,
        "workload": workload,
        "reference_placements": {
            "oracle_parallel": {
                "forced_placements": _forced_to_jsonable(fp),
                "deterministic_replica_placements": {TARGET_TASK_TYPE: dp},
            },
            "greedy_contended": {
                "forced_placements": _forced_to_jsonable(fc),
                "deterministic_replica_placements": {TARGET_TASK_TYPE: dc},
            },
        },
        "problem_spec": as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / LIVE_STUB_DIR,
    )
    args = parser.parse_args()
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    payload = build_stub_payload()
    (out / "meta.json").write_text(json.dumps(payload["problem_spec"], indent=2) + "\n")
    (out / "infrastructure.json").write_text(
        json.dumps(payload["infrastructure"], indent=2) + "\n"
    )
    (out / "workload.json").write_text(json.dumps(payload["workload"], indent=2) + "\n")
    (out / "reference_placements.json").write_text(
        json.dumps(payload["reference_placements"], indent=2) + "\n"
    )
    # Combined space-like config for run_simulation-style drivers.
    space = {
        "problem_id": PROBLEM_ID,
        "infrastructure": payload["infrastructure"],
        "workload": payload["workload"],
        "warmth_physics": GATE_WARMTH_PHYSICS,
    }
    (out / "space_config.json").write_text(json.dumps(space, indent=2) + "\n")

    print(f"Wrote live stub -> {out}")
    print(f"  N={TARGET_N_TASKS}  physics={GATE_WARMTH_PHYSICS}  gate≥{MIN_ORACLE_GREEDY_RATIO:.0f}×")
    print(f"  files: meta.json infrastructure.json workload.json reference_placements.json space_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
