#!/usr/bin/env python3
"""
Un-single-cell Regime B: same intel physics, more seeds / latency / second burst.

Does NOT replace the frozen oracle_split_v1 31.66s artifact. Adds a small grid so
the case study is not one eval JSON.

Usage:
    pipenv run python3 scripts_cosim/run_regime_b_multicell_eval.py
    pipenv run python3 scripts_cosim/run_regime_b_multicell_eval.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.build_regime_b_live_stub import build_stub_payload  # noqa: E402
from scripts_cosim.calibrate_regime_b import build_burst_workload  # noqa: E402
from scripts_cosim.regime_b_metrics import (  # noqa: E402
    attach_burst_ids_from_workload,
    burst_regime_summary,
    total_rtt_trap_stats,
)
from scripts_cosim.regime_b_problem_spec import (  # noqa: E402
    GATE_WARMTH_PHYSICS,
    INTEL_STUB_VARIANT,
    PRIMARY_SCORE_KEY,
    PROBLEM_ID,
    TARGET_N_TASKS,
    TARGET_TASK_TYPE,
)
from scripts_cosim.run_phase3_ect_pull_distill_eval import _load_gnn  # noqa: E402
from scripts_cosim.run_regime_b_live_stub_baselines import (  # noqa: E402
    _forced_from_json,
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
DEFAULT_GNN = (
    PROJECT_ROOT
    / "models/near-rtt-v2-regime-b-oracle-split-v1-ect-pull-distill-multiseed.pt"
)
DEFAULT_OUT = (
    PROJECT_ROOT
    / "simulation_data/normal_sim_sweeps/regime_b_multicell_20260813"
)
FROZEN_SUMMARY = (
    PROJECT_ROOT
    / "simulation_data/normal_sim_sweeps"
    / "regime_b_phase3_ect_pull_distill_eval_multiseed_cold"
    / "summary.json"
)
SECOND_BURST_T_S = 40.0


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _score(
    stats: Dict[str, Any],
    workload: Dict[str, Any],
    *,
    expected_n: int,
    oracle_rtt: Optional[float] = None,
) -> Dict[str, Any]:
    task_rows = attach_burst_ids_from_workload(
        extract_task_metrics(stats),
        workload.get("events") or [],
    )
    if len(task_rows) != expected_n:
        raise RuntimeError(
            f"FAIL LOUD: expected {expected_n} task rows, got {len(task_rows)} — "
            "set SIM_FORCE_FULL_STATS=1"
        )
    regime = burst_regime_summary(task_rows, oracle_rtt=oracle_rtt)
    trap = total_rtt_trap_stats(task_rows)
    return {
        PRIMARY_SCORE_KEY: regime[PRIMARY_SCORE_KEY],
        "last_task_rtt_s": regime["last_task_rtt_s"],
        "regime_b": regime,
        "total_rtt_trap": trap,
        "total_rtt": rtt_from_stats(stats),
        "num_tasks": len(task_rows),
    }


def _attach_oracle(scored: Dict[str, Any], oracle_score: Optional[float]) -> None:
    if oracle_score is None:
        return
    primary = float(scored[PRIMARY_SCORE_KEY])
    scored["oracle_rtt_s"] = oracle_score
    scored["oracle_regret_s"] = primary - oracle_score
    scored["oracle_regret_ratio"] = (
        primary / oracle_score if oracle_score > 0 else float("inf")
    )


def _dual_burst_payload(seed: int) -> Dict[str, Any]:
    """Two N=12 cold bursts: t=0 and t=40s. Same union seeds / physics."""
    payload = build_stub_payload(
        INTEL_STUB_VARIANT,
        arrival_jitter_s=0.0,
        warm_fraction=0.0,
        busy_fraction=0.0,
        seed=seed,
    )
    first = build_burst_workload(
        TARGET_N_TASKS,
        burst_id="cold_burst_n12_a",
        task_type=TARGET_TASK_TYPE,
        timestamps=[0.0] * TARGET_N_TASKS,
    )
    second = build_burst_workload(
        TARGET_N_TASKS,
        burst_id="cold_burst_n12_b",
        task_type=TARGET_TASK_TYPE,
        timestamps=[SECOND_BURST_T_S] * TARGET_N_TASKS,
    )
    events = list(first["events"]) + list(second["events"])
    payload["workload"] = {
        "rps": TARGET_N_TASKS,
        "duration": int(SECOND_BURST_T_S) + 2,
        "events": events,
    }
    # Oracle: one cold pull / server for each burst (task ids 0..11 and 12..23).
    refs = payload["reference_placements"]["oracle_parallel"]
    forced_a = _forced_from_json(refs["forced_placements"])
    forced = dict(forced_a)
    for tid, pair in forced_a.items():
        forced[int(tid) + TARGET_N_TASKS] = pair
    refs["forced_placements"] = {str(k): [int(v[0]), int(v[1])] for k, v in forced.items()}
    payload["n_tasks"] = 2 * TARGET_N_TASKS
    payload["cell_note"] = (
        f"second burst at t={SECOND_BURST_T_S}s; first-burst platforms may be warm"
    )
    return payload


def _cell_specs(*, smoke: bool) -> List[Dict[str, Any]]:
    cells = [
        {
            "name": "seed7_j0",
            "kind": "single",
            "kwargs": {
                "arrival_jitter_s": 0.0,
                "base_latency_s": 0.001,
                "scarce_attract_latency_s": 0.001,
                "seed": 7,
            },
        },
        {
            "name": "seed11_j0.5",
            "kind": "single",
            "kwargs": {
                "arrival_jitter_s": 0.5,
                "base_latency_s": 0.001,
                "scarce_attract_latency_s": 0.001,
                "seed": 11,
            },
        },
        {
            "name": "seed42_j2.0",
            "kind": "single",
            "kwargs": {
                "arrival_jitter_s": 2.0,
                "base_latency_s": 0.001,
                "scarce_attract_latency_s": 0.001,
                "seed": 42,
            },
        },
        {
            "name": "seed7_lat_hi",
            "kind": "single",
            "kwargs": {
                "arrival_jitter_s": 0.0,
                "base_latency_s": 0.005,
                "scarce_attract_latency_s": 0.0001,
                "seed": 7,
            },
        },
        {
            "name": "dual_burst_t40_seed7",
            "kind": "dual",
            "kwargs": {"seed": 7},
        },
    ]
    if smoke:
        return cells[:1]
    return cells


def _run_determined_raw(
    sim_inputs: Dict[str, Any],
    infrastructure: Dict[str, Any],
    workload: Dict[str, Any],
    *,
    forced: Dict[int, Tuple[int, int]],
    det_placements: List[Dict[str, Any]],
    expected_n: int,
) -> Dict[str, Any]:
    infra = dict(infrastructure)
    infra["forced_placements"] = forced
    infra["deterministic_replica_placements"] = {TARGET_TASK_TYPE: det_placements}
    infra["warmth_physics"] = GATE_WARMTH_PHYSICS
    config = {"infrastructure": infra, "workload": workload}
    result = execute_simulation(
        config,
        sim_inputs,
        scheduling_strategy="determined_determined",
        cache_policy="fifo",
        task_priority="fifo",
        keep_alive=KEEP_ALIVE,
        queue_length=QUEUE_LENGTH,
    )
    return _score(result.get("stats") or {}, workload, expected_n=expected_n)


def _run_policy_raw(
    sim_inputs: Dict[str, Any],
    infrastructure: Dict[str, Any],
    workload: Dict[str, Any],
    *,
    scheduling_strategy: str,
    expected_n: int,
    models: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    infra = dict(infrastructure)
    infra.pop("forced_placements", None)
    infra["warmth_physics"] = GATE_WARMTH_PHYSICS
    config = {"infrastructure": infra, "workload": workload}
    result = execute_simulation(
        config,
        sim_inputs,
        scheduling_strategy=scheduling_strategy,
        cache_policy="fifo",
        task_priority="fifo",
        keep_alive=KEEP_ALIVE,
        queue_length=QUEUE_LENGTH,
        models=models,
    )
    return _score(result.get("stats") or {}, workload, expected_n=expected_n)


def _run_cell_flex(
    *,
    spec: Dict[str, Any],
    sim_inputs: Dict[str, Any],
    distill_models: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    if spec["kind"] == "dual":
        payload = _dual_burst_payload(int(spec["kwargs"]["seed"]))
        expected_n = 2 * TARGET_N_TASKS
    else:
        payload = build_stub_payload(
            INTEL_STUB_VARIANT,
            warm_fraction=0.0,
            busy_fraction=0.0,
            **spec["kwargs"],
        )
        expected_n = TARGET_N_TASKS

    infra = payload["infrastructure"]
    workload = payload["workload"]
    refs = payload["reference_placements"]
    cell_dir = out_dir / "cells" / spec["name"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "infrastructure.json").write_text(json.dumps(infra, indent=2) + "\n")
    (cell_dir / "workload.json").write_text(json.dumps(workload, indent=2) + "\n")

    print(f"\n=== cell {spec['name']} n={expected_n} ===")
    oref = refs["oracle_parallel"]
    oracle = _run_determined_raw(
        sim_inputs,
        infra,
        workload,
        forced=_forced_from_json(oref["forced_placements"]),
        det_placements=oref["deterministic_replica_placements"][TARGET_TASK_TYPE],
        expected_n=expected_n,
    )
    oracle_score = float(oracle[PRIMARY_SCORE_KEY])
    print(f"  oracle={oracle_score:.2f}s")

    ect = _run_policy_raw(
        sim_inputs,
        infra,
        workload,
        scheduling_strategy="kn_network_ect_pull_kn_network_ect_pull",
        expected_n=expected_n,
    )
    _attach_oracle(ect, oracle_score)
    print(f"  ect_pull={ect[PRIMARY_SCORE_KEY]:.2f}s")

    os.environ["GNN_DECODE_MODE"] = "seq_reforward_pull"
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim24"
    os.environ["HEROSIM_WARMTH_PHYSICS"] = GATE_WARMTH_PHYSICS
    os.environ.setdefault("GNN_BATCH_SIZE", "4")
    os.environ.setdefault("GNN_BATCH_TIMEOUT", "0.002")
    distill = _run_policy_raw(
        sim_inputs,
        infra,
        workload,
        scheduling_strategy="gnn_gnn",
        expected_n=expected_n,
        models=distill_models,
    )
    _attach_oracle(distill, oracle_score)
    print(
        f"  distill={distill[PRIMARY_SCORE_KEY]:.2f}s  "
        f"regret={distill.get('oracle_regret_s', float('nan')):.2f}s"
    )

    row: Dict[str, Any] = {
        "name": spec["name"],
        "kind": spec["kind"],
        "n_tasks": expected_n,
        "kwargs": spec["kwargs"],
        "note": payload.get("cell_note"),
        "policies": {
            "oracle_parallel": {PRIMARY_SCORE_KEY: oracle_score, "num_tasks": expected_n},
            "ect_pull": {
                PRIMARY_SCORE_KEY: float(ect[PRIMARY_SCORE_KEY]),
                "oracle_regret_s": ect.get("oracle_regret_s"),
            },
            "distill_seq_reforward_pull": {
                PRIMARY_SCORE_KEY: float(distill[PRIMARY_SCORE_KEY]),
                "oracle_regret_s": distill.get("oracle_regret_s"),
                "decode_mode": "seq_reforward_pull",
            },
        },
    }
    (cell_dir / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-model", type=Path, default=DEFAULT_GNN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
    os.environ["HEROSIM_WARMTH_PHYSICS"] = GATE_WARMTH_PHYSICS

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_inputs = load_simulation_inputs(SIM_INPUT)
    distill_models = _load_gnn(args.gnn_model.resolve(), label="ect_pull_distill")

    frozen = None
    if FROZEN_SUMMARY.is_file():
        frozen = json.loads(FROZEN_SUMMARY.read_text())
        print(f"Loaded frozen cell from {FROZEN_SUMMARY}")
    else:
        print(f"WARN: frozen summary missing at {FROZEN_SUMMARY}")

    cells = []
    for spec in _cell_specs(smoke=bool(args.smoke)):
        cells.append(
            _run_cell_flex(
                spec=spec,
                sim_inputs=sim_inputs,
                distill_models=distill_models,
                out_dir=out_dir,
            )
        )

    summary: Dict[str, Any] = {
        "phase": "regime_b_multicell",
        "problem_id": PROBLEM_ID,
        "warmth_physics": GATE_WARMTH_PHYSICS,
        "gnn_model": distill_models["model_path"],
        "gnn_md5": distill_models["model_md5"],
        "frozen_cell": None,
        "cells": cells,
    }
    if frozen is not None:
        pols = frozen.get("policies") or {}
        summary["frozen_cell"] = {
            "source": str(FROZEN_SUMMARY),
            "oracle_s": (pols.get("oracle_parallel") or {}).get(PRIMARY_SCORE_KEY),
            "ect_pull_s": (pols.get("ect_pull") or {}).get(PRIMARY_SCORE_KEY),
            "distill_s": (pols.get("distill_seq_reforward_pull") or {}).get(
                PRIMARY_SCORE_KEY
            ),
        }

    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {path}")
    print("cell                         n   oracle   ect_pull  distill")
    if summary["frozen_cell"]:
        fc = summary["frozen_cell"]
        print(
            f"{'frozen_oracle_split_v1':<28} {12:3d} "
            f"{fc['oracle_s']:8.2f} {fc['ect_pull_s']:8.2f} {fc['distill_s']:8.2f}"
        )
    for c in cells:
        p = c["policies"]
        print(
            f"{c['name']:<28} {c['n_tasks']:3d} "
            f"{p['oracle_parallel'][PRIMARY_SCORE_KEY]:8.2f} "
            f"{p['ect_pull'][PRIMARY_SCORE_KEY]:8.2f} "
            f"{p['distill_seq_reforward_pull'][PRIMARY_SCORE_KEY]:8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
