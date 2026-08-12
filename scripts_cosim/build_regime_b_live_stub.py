#!/usr/bin/env python3
"""
Materialize Regime B live stub (cluster + burst-tagged workload) from frozen spec.

Variants:
  scarce_preinit_v2  — GATE: N cold deferred on node0 only (free Kn ≥10×)
  oracle_split_v1    — INTEL: N cold on node0 + 1 cold on every other server
                       (action space open; Kn partial pile; learned can beat Kn)

Does NOT run policies — only writes the env artifacts for later Kn/MLP/GNN baselines.

Usage:
    pipenv run python3 scripts_cosim/build_regime_b_live_stub.py
    pipenv run python3 scripts_cosim/build_regime_b_live_stub.py --variant oracle_split_v1
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
    INTEL_STUB_DIR,
    INTEL_STUB_VARIANT,
    LIVE_STUB_DIR,
    LIVE_STUB_PREINIT_N_COLD,
    LIVE_STUB_SCARCE_NODE,
    LIVE_STUB_VARIANT,
    MIN_FREE_KN_REGRET_RATIO,
    MIN_INTEL_KN_REGRET_RATIO,
    MIN_INTEL_MARGIN_S,
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

KNOWN_VARIANTS = (LIVE_STUB_VARIANT, INTEL_STUB_VARIANT)


def _forced_to_jsonable(
    forced: Dict[int, tuple],
) -> Dict[str, List[int]]:
    return {str(k): [int(v[0]), int(v[1])] for k, v in forced.items()}


def _dedupe_seeds(seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for s in seeds:
        key = (s["node_name"], int(s["platform_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"node_name": s["node_name"], "platform_id": int(s["platform_id"])})
    return out


def _scarce_preinit_seeds(dc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if LIVE_STUB_PREINIT_N_COLD != TARGET_N_TASKS:
        raise RuntimeError(
            f"FAIL LOUD: LIVE_STUB_PREINIT_N_COLD={LIVE_STUB_PREINIT_N_COLD} "
            f"!= TARGET_N_TASKS={TARGET_N_TASKS} — scarce preinit must cover full burst"
        )
    if LIVE_STUB_SCARCE_NODE != "node0":
        raise RuntimeError(
            f"FAIL LOUD: LIVE_STUB_SCARCE_NODE={LIVE_STUB_SCARCE_NODE!r} != 'node0'"
        )
    scarce_preinit = list(dc)
    if len(scarce_preinit) != LIVE_STUB_PREINIT_N_COLD:
        raise RuntimeError(
            f"FAIL LOUD: scarce preinit size {len(scarce_preinit)} "
            f"!= {LIVE_STUB_PREINIT_N_COLD}"
        )
    return scarce_preinit


def _oracle_split_seeds(
    dc: List[Dict[str, Any]], dp: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Union: N cold on node0 + 1 cold on every server (incl. node0 plat0 shared)."""
    seeds = _dedupe_seeds(list(dc) + list(dp))
    expected = TARGET_N_TASKS + (TARGET_SERVER_COUNT - 1)  # 12 + 11 = 23
    if len(seeds) != expected:
        raise RuntimeError(
            f"FAIL LOUD: oracle_split_v1 seeds={len(seeds)} != expected {expected} "
            f"(N on node0 + 1 on each other server)"
        )
    by_node: Dict[str, int] = {}
    for s in seeds:
        by_node[s["node_name"]] = by_node.get(s["node_name"], 0) + 1
    if by_node.get("node0") != TARGET_N_TASKS:
        raise RuntimeError(
            f"FAIL LOUD: oracle_split_v1 node0 seeds={by_node.get('node0')} "
            f"!= {TARGET_N_TASKS}"
        )
    for i in range(1, TARGET_SERVER_COUNT):
        name = f"node{i}"
        if by_node.get(name) != 1:
            raise RuntimeError(
                f"FAIL LOUD: oracle_split_v1 {name} seeds={by_node.get(name)} != 1"
            )
    return seeds


def build_stub_payload(variant: str) -> Dict[str, Any]:
    if variant not in KNOWN_VARIANTS:
        raise ValueError(
            f"unknown stub variant {variant!r}; expected one of {KNOWN_VARIANTS}"
        )
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
    if variant == LIVE_STUB_VARIANT:
        free_seeds = _scarce_preinit_seeds(dc)
    else:
        free_seeds = _oracle_split_seeds(dc, dp)

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
        "stub_variant": variant,
        "deterministic_replica_placements": {TARGET_TASK_TYPE: free_seeds},
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
        "stub_variant": variant,
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
        "--variant",
        choices=list(KNOWN_VARIANTS),
        default=LIVE_STUB_VARIANT,
        help=f"default={LIVE_STUB_VARIANT} (gate); {INTEL_STUB_VARIANT}=intelligence cell",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: gate→LIVE_STUB_DIR, intel→INTEL_STUB_DIR",
    )
    args = parser.parse_args()
    variant: str = args.variant
    out: Path = args.output_dir or (
        PROJECT_ROOT / INTEL_STUB_DIR
        if variant == INTEL_STUB_VARIANT
        else PROJECT_ROOT / LIVE_STUB_DIR
    )
    out.mkdir(parents=True, exist_ok=True)

    payload = build_stub_payload(variant)
    (out / "meta.json").write_text(json.dumps(payload["problem_spec"], indent=2) + "\n")
    (out / "infrastructure.json").write_text(
        json.dumps(payload["infrastructure"], indent=2) + "\n"
    )
    (out / "workload.json").write_text(json.dumps(payload["workload"], indent=2) + "\n")
    (out / "reference_placements.json").write_text(
        json.dumps(payload["reference_placements"], indent=2) + "\n"
    )
    space = {
        "problem_id": PROBLEM_ID,
        "infrastructure": payload["infrastructure"],
        "workload": payload["workload"],
        "warmth_physics": GATE_WARMTH_PHYSICS,
    }
    (out / "space_config.json").write_text(json.dumps(space, indent=2) + "\n")

    n_seeds = len(
        payload["infrastructure"]["deterministic_replica_placements"][TARGET_TASK_TYPE]
    )
    print(f"Wrote live stub -> {out}")
    print(
        f"  N={TARGET_N_TASKS}  physics={GATE_WARMTH_PHYSICS}  "
        f"variant={variant}  seeds={n_seeds}"
    )
    if variant == LIVE_STUB_VARIANT:
        print(
            f"  gate cell: {LIVE_STUB_PREINIT_N_COLD} cold deferred on "
            f"{LIVE_STUB_SCARCE_NODE}; free Kn ≥{MIN_FREE_KN_REGRET_RATIO:.0f}×"
        )
    else:
        print(
            f"  intel cell: union seeds (N on node0 + 1/other); "
            f"Kn ≥{MIN_INTEL_KN_REGRET_RATIO:.0f}×, learned margin ≥{MIN_INTEL_MARGIN_S:.0f}s"
        )
    print(
        "  files: meta.json infrastructure.json workload.json "
        "reference_placements.json space_config.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
