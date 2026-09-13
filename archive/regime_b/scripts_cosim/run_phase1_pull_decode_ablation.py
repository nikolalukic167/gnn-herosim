#!/usr/bin/env python3
"""
Phase 1: Decode-state ablation on Regime B oracle_split_v1.

Compares CE GNN under:
  - GNN_DECODE_MODE=argmax              (static features; ~125s pile baseline)
  - GNN_DECODE_MODE=seq_reforward_pull  (pulls_committed ledger + dim24 re-forward)

Primary metric = regime_b_primary_score_s (max burst elapsed). Never total_rtt.

Usage:
    pipenv run python3 scripts_cosim/run_phase1_pull_decode_ablation.py
    pipenv run python3 scripts_cosim/run_phase1_pull_decode_ablation.py \\
        --gnn-model models/near-rtt-v2-regime-b-oracle-split-cosim-dim16-ce-only.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.regime_b_problem_spec import (  # noqa: E402
    GATE_WARMTH_PHYSICS,
    INTEL_STUB_DIR,
    INTEL_STUB_VARIANT,
    PRIMARY_SCORE_KEY,
    PROBLEM_ID,
    TARGET_N_TASKS,
    TARGET_TASK_TYPE,
)
from scripts_cosim.run_regime_b_live_stub_baselines import (  # noqa: E402
    _load_stub,
    run_determined,
    run_policy,
    _forced_from_json,
)
from src.executecosimulation import load_simulation_inputs  # noqa: E402
from src.executesimulation import load_gnn_model  # noqa: E402


SIM_INPUT = PROJECT_ROOT / "data/nofs-ids"
DEFAULT_STUB = PROJECT_ROOT / INTEL_STUB_DIR
DEFAULT_GNN = (
    PROJECT_ROOT
    / "models/near-rtt-v2-regime-b-oracle-split-cosim-dim16-ce-only.pt"
)
DEFAULT_OUT = (
    PROJECT_ROOT
    / "simulation_data/normal_sim_sweeps/regime_b_phase1_pull_decode_ablation"
)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_gnn(model_path: Path) -> Dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"GNN model missing: {model_path}")
    task_types_data = json.loads((SIM_INPUT / "task-types.json").read_text())
    gnn_model, gnn_device = load_gnn_model(model_path)
    gnn_model.eval()
    # Fail-loud: Phase 1 pull ledger needs dim16 platform encoder (dim24 layout).
    plat_in = int(gnn_model.platform_encoder.net[0].in_features)
    if plat_in < 16:
        raise RuntimeError(
            f"FAIL LOUD: Phase 1 requires dim16 platform encoder (got in_features={plat_in}). "
            f"Use a Regime-B pull-obs ckpt, not dim14 CE. path={model_path}"
        )
    return {
        "gnn_model": gnn_model,
        "device": gnn_device,
        "task_types_data": task_types_data,
        "model_path": str(model_path),
        "model_md5": _md5(model_path),
        "platform_in_features": plat_in,
    }


def _run_gnn_decode(
    sim_inputs: Dict[str, Any],
    infra: Dict[str, Any],
    workload: Dict[str, Any],
    *,
    models: Dict[str, Any],
    decode_mode: str,
) -> Dict[str, Any]:
    os.environ["GNN_DECODE_MODE"] = decode_mode
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim24"
    os.environ["HEROSIM_WARMTH_PHYSICS"] = GATE_WARMTH_PHYSICS
    os.environ.setdefault("GNN_BATCH_SIZE", "4")
    os.environ.setdefault("GNN_BATCH_TIMEOUT", "0.002")
    scored = run_policy(
        sim_inputs,
        infra,
        workload,
        policy="gnn",
        models=models,
        scheduling_strategy="gnn_gnn",
    )
    scored["decode_mode"] = decode_mode
    scored["inference_feature_layout"] = "dim24"
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub-dir", type=Path, default=DEFAULT_STUB)
    parser.add_argument("--gnn-model", type=Path, default=DEFAULT_GNN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--skip-refs",
        action="store_true",
        help="Skip oracle/greedy references (use only GNN A/B)",
    )
    parser.add_argument(
        "--modes",
        default="argmax,seq_reforward_pull",
        help="Comma-separated decode modes to compare",
    )
    args = parser.parse_args()

    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
    os.environ["HEROSIM_WARMTH_PHYSICS"] = GATE_WARMTH_PHYSICS

    stub_dir = args.stub_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    infra, workload, refs = _load_stub(stub_dir, expected_variant=INTEL_STUB_VARIANT)
    sim_inputs = load_simulation_inputs(SIM_INPUT)
    models = _load_gnn(args.gnn_model.resolve())

    print("=" * 72)
    print(f"Phase 1 pull-decode ablation — {PROBLEM_ID} / {INTEL_STUB_VARIANT}")
    print(f"stub={stub_dir}")
    print(
        f"gnn={models['model_path']}  md5={models['model_md5'][:12]}…  "
        f"plat_in={models['platform_in_features']}"
    )
    print(f"layout=dim24  N={TARGET_N_TASKS}")
    print("=" * 72)

    summary: Dict[str, Any] = {
        "phase": "phase1_pull_decode_ablation",
        "problem_id": PROBLEM_ID,
        "stub_variant": INTEL_STUB_VARIANT,
        "stub_dir": str(stub_dir),
        "warmth_physics": GATE_WARMTH_PHYSICS,
        "n_tasks": TARGET_N_TASKS,
        "gnn_model": models["model_path"],
        "gnn_md5": models["model_md5"],
        "platform_in_features": models["platform_in_features"],
        "inference_feature_layout": "dim24",
        "policies": {},
    }

    oracle_score: Optional[float] = None
    if not args.skip_refs:
        print("\n--- reference: oracle_parallel ---")
        oref = refs["oracle_parallel"]
        oracle = run_determined(
            sim_inputs,
            infra,
            workload,
            forced=_forced_from_json(oref["forced_placements"]),
            det_placements=oref["deterministic_replica_placements"][TARGET_TASK_TYPE],
            label="oracle_parallel",
        )
        oracle_score = float(oracle[PRIMARY_SCORE_KEY])
        summary["policies"]["oracle_parallel"] = oracle
        (results_dir / "oracle_parallel.json").write_text(json.dumps(oracle, indent=2) + "\n")
        print(f"  {PRIMARY_SCORE_KEY}={oracle_score:.2f}s")

        print("\n--- reference: ect_pull (physics ceiling) ---")
        ect = run_policy(
            sim_inputs,
            infra,
            workload,
            policy="knative_network_ect_pull",
            models=None,
            scheduling_strategy="kn_network_ect_pull_kn_network_ect_pull",
        )
        if oracle_score is not None:
            ect["oracle_rtt_s"] = oracle_score
            ect["oracle_regret_s"] = float(ect[PRIMARY_SCORE_KEY]) - oracle_score
        summary["policies"]["ect_pull"] = ect
        (results_dir / "ect_pull.json").write_text(json.dumps(ect, indent=2) + "\n")
        print(f"  {PRIMARY_SCORE_KEY}={ect[PRIMARY_SCORE_KEY]:.2f}s")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise SystemExit("FAIL LOUD: --modes is empty")

    for mode in modes:
        print(f"\n--- gnn decode={mode} ---")
        scored = _run_gnn_decode(
            sim_inputs, infra, workload, models=models, decode_mode=mode
        )
        if oracle_score is not None:
            primary = float(scored[PRIMARY_SCORE_KEY])
            scored["oracle_rtt_s"] = oracle_score
            scored["oracle_regret_s"] = primary - oracle_score
            scored["oracle_regret_ratio"] = (
                primary / oracle_score if oracle_score > 0 else float("inf")
            )
        key = f"gnn_{mode}"
        summary["policies"][key] = scored
        (results_dir / f"{key}.json").write_text(json.dumps(scored, indent=2) + "\n")
        print(
            f"  {PRIMARY_SCORE_KEY}={scored[PRIMARY_SCORE_KEY]:.2f}s  "
            f"total_rtt_trap={scored['total_rtt']:.2f}s"
            + (
                f"  delta_oracle={scored.get('oracle_regret_s', float('nan')):.2f}s"
                if oracle_score is not None
                else ""
            )
        )

    # Falsification summary
    if "argmax" in modes and "seq_reforward_pull" in modes:
        base = float(summary["policies"]["gnn_argmax"][PRIMARY_SCORE_KEY])
        pull = float(summary["policies"]["gnn_seq_reforward_pull"][PRIMARY_SCORE_KEY])
        delta = base - pull
        summary["falsification"] = {
            "argmax_primary_s": base,
            "seq_reforward_pull_primary_s": pull,
            "improvement_s": delta,
            "broke_125s_pile": pull < 100.0,
            "near_physics_ceiling": pull < 50.0,
            "interpretation": (
                "decision-time pull ledger recovers ect_pull-like intelligence"
                if pull < 50.0
                else (
                    "partial improvement — residual remains"
                    if delta > 5.0
                    else (
                        "pull ledger HURT vs argmax — joint re-forward is hostile "
                        "or CE head ignores updated pull dims"
                        if delta < -5.0
                        else "static features + sequential decode cannot recover ect_pull "
                        "(primary stayed near argmax)"
                    )
                )
            ),
        }
        print("\n=== Phase 1 falsification ===")
        print(f"  argmax:              {base:.2f}s")
        print(f"  seq_reforward_pull:  {pull:.2f}s")
        print(f"  improvement:         {delta:.2f}s")
        print(f"  verdict:             {summary['falsification']['interpretation']}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
