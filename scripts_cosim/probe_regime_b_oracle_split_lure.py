#!/usr/bin/env python3
"""
Probe anti-corr / oracle-split seed variants on Regime B N=12 cell.

Hypothesis: scarce_preinit_v2 traps ALL free policies on node0 (no intelligence
margin possible). Opening action space (cold seeds on every node) lets Kn
shortest-queue spread → free Kn headroom collapses.

Variants:
  v2_scarce_only   — current: N cold deferred on node0 only
  split_parallel   — 1 cold deferred per server node (oracle action space)
  split_union      — node0 keeps N cold + 1 cold on each other node

Runs: oracle_parallel (forced), free knative_network. Fail loud on errors.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.calibrate_regime_b import (  # noqa: E402
    build_burst_workload,
    build_target_nodes,
    contended_placements,
    parallel_placements,
)
from scripts_cosim.regime_b_metrics import (  # noqa: E402
    attach_burst_ids_from_workload,
    burst_regime_summary,
)
from scripts_cosim.regime_b_problem_spec import (  # noqa: E402
    GATE_WARMTH_PHYSICS,
    PRIMARY_SCORE_KEY,
    PROBLEM_ID,
    TARGET_BURST_ID,
    TARGET_N_TASKS,
    TARGET_PLATFORMS_ON_OTHER_NODES,
    TARGET_PLATFORMS_ON_SCARCE_NODE,
    TARGET_SERVER_COUNT,
    TARGET_TASK_TYPE,
)
from src.executecosimulation import (  # noqa: E402
    KEEP_ALIVE,
    QUEUE_LENGTH,
    execute_simulation,
    extract_task_metrics,
    load_simulation_inputs,
)

SIM_INPUT = PROJECT_ROOT / "data/nofs-ids"
OUT_DIR = PROJECT_ROOT / "simulation_data/regime_b_calibration/oracle_split_lure_probe"
GNN_MODEL = PROJECT_ROOT / "models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt"
MLP_MODEL = (
    PROJECT_ROOT
    / "models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt"
)


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


def _infra(det: List[Dict[str, Any]], nodes: List[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    server_names = [f"node{i}" for i in range(TARGET_SERVER_COUNT)]
    return {
        "network": {"bandwidth": 100.0},
        "nodes": nodes,
        "preinitialize_platforms": True,
        "defer_cold_replica_init": True,
        "warmth_physics": GATE_WARMTH_PHYSICS,
        "stub_variant": variant,
        "deterministic_replica_placements": {TARGET_TASK_TYPE: det},
        "replica_plan": {
            "preinit_clients": [],
            "preinit_servers": server_names,
            "preinit_task_types": [TARGET_TASK_TYPE],
            "replicas_config": {TARGET_TASK_TYPE: {"per_client": 0, "per_server": 0}},
            "prewarm_config": {},
        },
        "scheduler": {"batch_size": TARGET_N_TASKS, "batch_timeout": 0.02},
        "fast_forward_warmup": True,
        "fast_forward_threshold": 1,
    }


def _score(stats: Dict[str, Any], workload: Dict[str, Any]) -> float:
    rows = extract_task_metrics(stats)
    if not rows:
        raise RuntimeError("FAIL LOUD: empty taskResults")
    events = workload.get("events") or []
    tagged = attach_burst_ids_from_workload(rows, events)
    return float(burst_regime_summary(tagged)[PRIMARY_SCORE_KEY])


def _run_determined(
    sim_inputs: Dict[str, Any],
    infra: Dict[str, Any],
    workload: Dict[str, Any],
    forced: Dict[int, Tuple[int, int]],
    det: List[Dict[str, Any]],
) -> float:
    infra = deepcopy(infra)
    # Determined scheduler expects Dict[int, Tuple[int, int]] (not JSON string keys).
    infra["forced_placements"] = {int(k): (int(v[0]), int(v[1])) for k, v in forced.items()}
    infra["deterministic_replica_placements"] = {TARGET_TASK_TYPE: det}
    result = execute_simulation(
        {"infrastructure": infra, "workload": workload},
        sim_inputs,
        scheduling_strategy="determined_determined",
        cache_policy="fifo",
        task_priority="fifo",
        keep_alive=KEEP_ALIVE,
        queue_length=QUEUE_LENGTH,
        models=None,
    )
    return _score(result.get("stats") or {}, workload)


def _run_free(
    sim_inputs: Dict[str, Any],
    infra: Dict[str, Any],
    workload: Dict[str, Any],
    *,
    scheduling_strategy: str,
    models: Any = None,
) -> float:
    infra = deepcopy(infra)
    infra.pop("forced_placements", None)
    result = execute_simulation(
        {"infrastructure": infra, "workload": workload},
        sim_inputs,
        scheduling_strategy=scheduling_strategy,
        cache_policy="fifo",
        task_priority="fifo",
        keep_alive=KEEP_ALIVE,
        queue_length=QUEUE_LENGTH,
        models=models,
    )
    return _score(result.get("stats") or {}, workload)


def _load_mlp_models() -> Dict[str, Any]:
    task_types_data = json.loads((SIM_INPUT / "task-types.json").read_text())
    if not MLP_MODEL.is_file():
        raise FileNotFoundError(f"MLP missing: {MLP_MODEL}")
    return {"mlp_model_path": str(MLP_MODEL), "task_types_data": task_types_data}


def _load_gnn_models() -> Dict[str, Any]:
    from src.executesimulation import load_gnn_model

    task_types_data = json.loads((SIM_INPUT / "task-types.json").read_text())
    if not GNN_MODEL.is_file():
        raise FileNotFoundError(f"GNN missing: {GNN_MODEL}")
    gnn_model, gnn_device = load_gnn_model(GNN_MODEL)
    return {
        "gnn_model": gnn_model,
        "device": gnn_device,
        "task_types_data": task_types_data,
    }


def main() -> int:
    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    os.environ["HEROSIM_WARMTH_PHYSICS"] = GATE_WARMTH_PHYSICS
    os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"

    nodes, node_id_by_name, plat_id_by_node_local = build_target_nodes(
        TARGET_SERVER_COUNT,
        platforms_scarce=TARGET_PLATFORMS_ON_SCARCE_NODE,
        platforms_other=TARGET_PLATFORMS_ON_OTHER_NODES,
    )
    fc, dc = contended_placements(TARGET_N_TASKS, node_id_by_name, plat_id_by_node_local)
    fp, dp = parallel_placements(TARGET_N_TASKS, node_id_by_name, plat_id_by_node_local)
    workload = build_burst_workload(
        TARGET_N_TASKS, burst_id=TARGET_BURST_ID, task_type=TARGET_TASK_TYPE
    )
    sim_inputs = load_simulation_inputs(SIM_INPUT)

    variants: Dict[str, List[Dict[str, Any]]] = {
        "v2_scarce_only": list(dc),
        "split_parallel": list(dp),
        "split_union": _dedupe_seeds(list(dc) + list(dp)),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    print("=" * 72)
    print(f"Regime B oracle-split lure probe — {PROBLEM_ID} N={TARGET_N_TASKS}")
    print("=" * 72)

    # Oracle once on parallel seeds (reference).
    oracle_infra = _infra(dp, nodes, "oracle_ref")
    oracle_s = _run_determined(sim_inputs, oracle_infra, workload, fp, dp)
    print(f"oracle_parallel: {PRIMARY_SCORE_KEY}={oracle_s:.4f}s")

    os.environ.setdefault("GNN_BATCH_SIZE", "4")
    os.environ.setdefault("GNN_BATCH_TIMEOUT", "0.002")
    os.environ.setdefault("INFERENCE_FEATURE_LAYOUT", "dim22")
    os.environ.setdefault("GNN_DECODE_MODE", "argmax")

    free_policies = [
        ("knative", "kn_network_kn_network", None),
        ("mlp", "mlp_batch_mlp_batch", _load_mlp_models),
        ("gnn", "gnn_gnn", _load_gnn_models),
    ]

    for name, seeds in variants.items():
        by_node: Dict[str, int] = {}
        for s in seeds:
            by_node[s["node_name"]] = by_node.get(s["node_name"], 0) + 1
        infra = _infra(seeds, nodes, name)
        print(f"\n--- {name}  seeds={len(seeds)}  by_node={dict(sorted(by_node.items()))} ---")
        policy_scores: Dict[str, float] = {}
        for pol_name, strat, model_fn in free_policies:
            # Full policy suite only on split_union (intelligence candidate).
            if name != "split_union" and pol_name != "knative":
                continue
            models = model_fn() if model_fn else None
            score = _run_free(
                sim_inputs, infra, workload, scheduling_strategy=strat, models=models
            )
            policy_scores[pol_name] = score
            ratio = score / oracle_s if oracle_s > 0 else float("inf")
            print(
                f"  {pol_name:8s} {PRIMARY_SCORE_KEY}={score:.4f}s  ratio={ratio:.3f}×"
            )
        kn_s = policy_scores["knative"]
        kn_ratio = kn_s / oracle_s if oracle_s > 0 else float("inf")
        row = {
            "variant": name,
            "n_seeds": len(seeds),
            "seeds_by_node": by_node,
            "policy_primary_s": policy_scores,
            "knative_primary_s": kn_s,
            "oracle_primary_s": oracle_s,
            "kn_oracle_ratio": kn_ratio,
            "free_kn_headroom_ge_10x": kn_ratio >= 10.0,
        }
        if "mlp" in policy_scores and "gnn" in policy_scores:
            row["mlp_beats_kn"] = policy_scores["mlp"] < kn_s - 1.0
            row["gnn_beats_kn"] = policy_scores["gnn"] < kn_s - 1.0
            row["intelligence_margin_s"] = kn_s - min(
                policy_scores["mlp"], policy_scores["gnn"]
            )
        rows.append(row)

    payload = {
        "problem_id": PROBLEM_ID,
        "warmth_physics": GATE_WARMTH_PHYSICS,
        "n_tasks": TARGET_N_TASKS,
        "oracle_primary_s": oracle_s,
        "variants": rows,
        "decision_notes": {
            "v2_scarce_only": "Gate cell — free Kn ≥10× but action space=node0 only → no intelligence",
            "split_parallel": "Open action space → Kn≡oracle (1.00×) — headroom gone",
            "split_union": "Candidate lure — Kn partial pile; learned can beat if labels/features allow",
        },
    }
    out_path = OUT_DIR / "probe.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
