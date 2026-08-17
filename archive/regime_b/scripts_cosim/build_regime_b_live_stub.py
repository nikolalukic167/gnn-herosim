#!/usr/bin/env python3
"""
Materialize Regime B live stub (cluster + burst-tagged workload) from frozen spec.

Variants:
  scarce_preinit_v2  — GATE: N cold deferred on node0 only (free Kn ≥10×)
  oracle_split_v1    — INTEL: N cold on node0 + 1 cold on every other server
                       (action space open; Kn partial pile; learned can beat Kn)

Optional distill randomization (does NOT change the frozen gate stub defaults):
  --arrival-jitter-s   Uniform(0, J) stagger of N arrivals around t=0
  --warm-fraction      Mark that fraction of seed platforms warm (sandbox match)
  --busy-fraction      Seed that fraction with initial_queue=1 (busy FilterStore peers)
  --base-latency-s / --scarce-attract-latency-s  latency diversity overlay
  --seed               RNG seed for jitter + warm/busy sampling

Does NOT run policies — only writes the env artifacts for later Kn/MLP/GNN baselines.

Usage:
    pipenv run python3 scripts_cosim/build_regime_b_live_stub.py
    pipenv run python3 scripts_cosim/build_regime_b_live_stub.py --variant oracle_split_v1
    pipenv run python3 scripts_cosim/build_regime_b_live_stub.py \\
        --variant oracle_split_v1 --arrival-jitter-s 2.0 --warm-fraction 0.25 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
            f"(N on node0 + 1 on every other server)"
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


def _apply_latency_diversity(
    nodes: List[Dict[str, Any]],
    *,
    base_latency_s: float,
    scarce_attract_latency_s: float,
) -> None:
    """Overlay peer latencies; client↔node0 uses scarce_attract (anti-corr lure)."""
    if base_latency_s < 0 or scarce_attract_latency_s < 0:
        raise ValueError(
            f"latencies must be >= 0, got base={base_latency_s} "
            f"scarce={scarce_attract_latency_s}"
        )
    names = [n["node_name"] for n in nodes]
    for node in nodes:
        peers = [p for p in names if p != node["node_name"]]
        nm: Dict[str, float] = {}
        for peer in peers:
            a, b = node["node_name"], peer
            if {a, b} == {"client_node0", "node0"}:
                nm[peer] = float(scarce_attract_latency_s)
            else:
                nm[peer] = float(base_latency_s)
        node["network_map"] = nm
    for node in nodes:
        expected = len(names) - 1
        if len(node["network_map"]) != expected:
            raise RuntimeError(
                f"network_map size {len(node['network_map'])} != {expected} "
                f"for {node['node_name']}"
            )


def sample_arrival_timestamps(
    n: int,
    *,
    jitter_s: float,
    rng: random.Random,
) -> List[float]:
    """Uniform(0, jitter_s) per task, sorted so arrival order is stable."""
    if jitter_s < 0:
        raise ValueError(f"FAIL LOUD: jitter_s={jitter_s} < 0")
    if jitter_s == 0.0:
        return [0.0] * n
    stamps = [float(rng.uniform(0.0, float(jitter_s))) for _ in range(n)]
    stamps.sort()
    return stamps


def apply_init_state_randomization(
    seeds: List[Dict[str, Any]],
    *,
    warm_fraction: float,
    busy_fraction: float,
    busy_queue: int,
    rng: random.Random,
    task_type: str = TARGET_TASK_TYPE,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Mark disjoint warm / busy subsets of seed platforms.

    Returns (seed_list_with_warm_flags, deterministic_queue_distributions).
    warm XOR busy on any single platform — fail loud on overlap.
    """
    if not (0.0 <= warm_fraction <= 1.0):
        raise ValueError(f"FAIL LOUD: warm_fraction={warm_fraction} not in [0,1]")
    if not (0.0 <= busy_fraction <= 1.0):
        raise ValueError(f"FAIL LOUD: busy_fraction={busy_fraction} not in [0,1]")
    if busy_queue < 1 and busy_fraction > 0:
        raise ValueError(
            f"FAIL LOUD: busy_queue={busy_queue} < 1 with busy_fraction={busy_fraction}"
        )
    if warm_fraction + busy_fraction > 1.0 + 1e-9:
        raise ValueError(
            f"FAIL LOUD: warm_fraction+busy_fraction="
            f"{warm_fraction + busy_fraction} > 1 (need disjoint subsets)"
        )

    n = len(seeds)
    n_warm = int(round(warm_fraction * n))
    n_busy = int(round(busy_fraction * n))
    if n_warm + n_busy > n:
        # Rounding edge — shrink busy first.
        n_busy = n - n_warm
    order = list(range(n))
    rng.shuffle(order)
    warm_idx = set(order[:n_warm])
    busy_idx = set(order[n_warm : n_warm + n_busy])
    if warm_idx & busy_idx:
        raise RuntimeError("FAIL LOUD: warm/busy index overlap after sampling")

    out_seeds: List[Dict[str, Any]] = []
    queues: Dict[str, int] = {}
    for i, s in enumerate(seeds):
        row = {"node_name": s["node_name"], "platform_id": int(s["platform_id"])}
        key = f"{row['node_name']}:{row['platform_id']}"
        if i in warm_idx:
            row["warm"] = True
        if i in busy_idx:
            queues[key] = int(busy_queue)
        out_seeds.append(row)

    det_queues: Dict[str, Dict[str, int]] = {task_type: queues} if queues else {}
    return out_seeds, det_queues


def build_stub_payload(
    variant: str,
    *,
    arrival_jitter_s: float = 0.0,
    warm_fraction: float = 0.0,
    busy_fraction: float = 0.0,
    busy_queue: int = 1,
    base_latency_s: float = 0.001,
    scarce_attract_latency_s: float = 0.001,
    seed: Optional[int] = None,
    timestamps: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    if variant not in KNOWN_VARIANTS:
        raise ValueError(
            f"unknown stub variant {variant!r}; expected one of {KNOWN_VARIANTS}"
        )
    rng = random.Random(seed)

    nodes, node_id_by_name, plat_id_by_node_local = build_target_nodes(
        TARGET_SERVER_COUNT,
        platforms_scarce=TARGET_PLATFORMS_ON_SCARCE_NODE,
        platforms_other=TARGET_PLATFORMS_ON_OTHER_NODES,
    )
    _apply_latency_diversity(
        nodes,
        base_latency_s=float(base_latency_s),
        scarce_attract_latency_s=float(scarce_attract_latency_s),
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

    free_seeds, det_queues = apply_init_state_randomization(
        free_seeds,
        warm_fraction=float(warm_fraction),
        busy_fraction=float(busy_fraction),
        busy_queue=int(busy_queue),
        rng=rng,
        task_type=TARGET_TASK_TYPE,
    )

    if timestamps is not None:
        ts = [float(t) for t in timestamps]
    else:
        ts = sample_arrival_timestamps(
            TARGET_N_TASKS, jitter_s=float(arrival_jitter_s), rng=rng
        )
    workload = build_burst_workload(
        TARGET_N_TASKS,
        burst_id=TARGET_BURST_ID,
        task_type=TARGET_TASK_TYPE,
        timestamps=ts,
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
        "distill_randomization": {
            "arrival_jitter_s": float(arrival_jitter_s),
            "warm_fraction": float(warm_fraction),
            "busy_fraction": float(busy_fraction),
            "busy_queue": int(busy_queue),
            "base_latency_s": float(base_latency_s),
            "scarce_attract_latency_s": float(scarce_attract_latency_s),
            "seed": seed,
            "timestamps": ts,
            "n_warm": sum(1 for s in free_seeds if s.get("warm")),
            "n_busy": len(det_queues.get(TARGET_TASK_TYPE, {})),
        },
    }
    if det_queues:
        infrastructure["deterministic_queue_distributions"] = det_queues

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


def default_latency_grid() -> List[Dict[str, float]]:
    """Same cartesian as oracle_split co-sim diversity (12 cells)."""
    bases = (0.0005, 0.001, 0.002, 0.005)
    scarce = (0.0001, 0.0005, 0.001)
    return [
        {"base_latency_s": b, "scarce_attract_latency_s": s}
        for b in bases
        for s in scarce
    ]


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
    parser.add_argument(
        "--arrival-jitter-s",
        type=float,
        default=0.0,
        help="Uniform(0, J) arrival stagger in seconds (0 = all at t=0)",
    )
    parser.add_argument(
        "--warm-fraction",
        type=float,
        default=0.0,
        help="Fraction of seed platforms marked warm (sandbox match, empty queue)",
    )
    parser.add_argument(
        "--busy-fraction",
        type=float,
        default=0.0,
        help="Fraction of seed platforms with initial_queue (busy)",
    )
    parser.add_argument(
        "--busy-queue",
        type=int,
        default=1,
        help="Queue depth for busy platforms (default 1)",
    )
    parser.add_argument("--base-latency-s", type=float, default=0.001)
    parser.add_argument("--scarce-attract-latency-s", type=float, default=0.001)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for jitter + warm/busy sampling",
    )
    args = parser.parse_args()
    variant: str = args.variant
    out: Path = args.output_dir or (
        PROJECT_ROOT / INTEL_STUB_DIR
        if variant == INTEL_STUB_VARIANT
        else PROJECT_ROOT / LIVE_STUB_DIR
    )
    out.mkdir(parents=True, exist_ok=True)

    payload = build_stub_payload(
        variant,
        arrival_jitter_s=float(args.arrival_jitter_s),
        warm_fraction=float(args.warm_fraction),
        busy_fraction=float(args.busy_fraction),
        busy_queue=int(args.busy_queue),
        base_latency_s=float(args.base_latency_s),
        scarce_attract_latency_s=float(args.scarce_attract_latency_s),
        seed=args.seed,
    )
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
    rand = payload["infrastructure"].get("distill_randomization", {})
    print(f"Wrote live stub -> {out}")
    print(
        f"  N={TARGET_N_TASKS}  physics={GATE_WARMTH_PHYSICS}  "
        f"variant={variant}  seeds={n_seeds}"
    )
    print(
        f"  jitter={args.arrival_jitter_s}s  warm={args.warm_fraction}  "
        f"busy={args.busy_fraction}  seed={args.seed}  "
        f"n_warm={rand.get('n_warm')} n_busy={rand.get('n_busy')}"
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
