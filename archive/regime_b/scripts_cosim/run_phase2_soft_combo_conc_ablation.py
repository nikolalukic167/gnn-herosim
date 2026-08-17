#!/usr/bin/env python3
"""
Phase 2: soft_combo_conc capacity-matched control on Regime B oracle_split_v1.

Trains joint Boltzmann over b≤4 placements.jsonl combos (+ concentration).
Live decode must match the joint objective: frozen_topk (additive logit sums).
Also reports argmax as a control and CE dim16 argmax as the Phase-1 baseline.

Primary metric = regime_b_primary_score_s (max burst elapsed). Never total_rtt.

Usage:
    pipenv run python3 scripts_cosim/run_phase2_soft_combo_conc_ablation.py
    pipenv run python3 scripts_cosim/run_phase2_soft_combo_conc_ablation.py \\
        --gnn-model models/near-rtt-v2-regime-b-oracle-split-cosim-dim16-soft-combo-conc.pt \\
        --ce-baseline models/near-rtt-v2-regime-b-oracle-split-cosim-dim16-ce-only.pt
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
    _forced_from_json,
    _load_stub,
    run_determined,
    run_policy,
)
from src.executecosimulation import load_simulation_inputs  # noqa: E402
from src.executesimulation import load_gnn_model  # noqa: E402


SIM_INPUT = PROJECT_ROOT / "data/nofs-ids"
DEFAULT_STUB = PROJECT_ROOT / INTEL_STUB_DIR
DEFAULT_GNN = (
    PROJECT_ROOT
    / "models/near-rtt-v2-regime-b-oracle-split-cosim-dim16-soft-combo-conc.pt"
)
DEFAULT_CE = (
    PROJECT_ROOT
    / "models/near-rtt-v2-regime-b-oracle-split-cosim-dim16-ce-only.pt"
)
DEFAULT_OUT = (
    PROJECT_ROOT
    / "simulation_data/normal_sim_sweeps/regime_b_phase2_soft_combo_conc_ablation"
)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_gnn(model_path: Path, *, label: str) -> Dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"FAIL LOUD: {label} GNN model missing: {model_path}")
    task_types_data = json.loads((SIM_INPUT / "task-types.json").read_text())
    gnn_model, gnn_device = load_gnn_model(model_path)
    gnn_model.eval()
    plat_in = int(gnn_model.platform_encoder.net[0].in_features)
    if plat_in < 16:
        raise RuntimeError(
            f"FAIL LOUD: Phase 2 requires dim16 platform encoder (got in_features={plat_in}). "
            f"label={label} path={model_path}"
        )
    return {
        "gnn_model": gnn_model,
        "device": gnn_device,
        "task_types_data": task_types_data,
        "model_path": str(model_path),
        "model_md5": _md5(model_path),
        "platform_in_features": plat_in,
        "label": label,
    }


def _run_gnn_decode(
    sim_inputs: Dict[str, Any],
    infra: Dict[str, Any],
    workload: Dict[str, Any],
    *,
    models: Dict[str, Any],
    decode_mode: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    os.environ["GNN_DECODE_MODE"] = decode_mode
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim24"
    os.environ["HEROSIM_WARMTH_PHYSICS"] = GATE_WARMTH_PHYSICS
    os.environ.setdefault("GNN_BATCH_SIZE", "4")
    os.environ.setdefault("GNN_BATCH_TIMEOUT", "0.002")
    if top_k is not None:
        os.environ["GNN_DECODE_TOP_K"] = str(int(top_k))
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
    scored["model_label"] = models.get("label")
    scored["gnn_model"] = models["model_path"]
    scored["gnn_md5"] = models["model_md5"]
    if top_k is not None:
        scored["decode_top_k"] = int(top_k)
    return scored


def _attach_oracle(scored: Dict[str, Any], oracle_score: Optional[float]) -> None:
    if oracle_score is None:
        return
    primary = float(scored[PRIMARY_SCORE_KEY])
    scored["oracle_rtt_s"] = oracle_score
    scored["oracle_regret_s"] = primary - oracle_score
    scored["oracle_regret_ratio"] = (
        primary / oracle_score if oracle_score > 0 else float("inf")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub-dir", type=Path, default=DEFAULT_STUB)
    parser.add_argument("--gnn-model", type=Path, default=DEFAULT_GNN)
    parser.add_argument("--ce-baseline", type=Path, default=DEFAULT_CE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-refs", action="store_true")
    parser.add_argument("--skip-ce-baseline", action="store_true")
    parser.add_argument(
        "--modes",
        default="frozen_topk,argmax",
        help="Comma-separated soft_combo_conc decode modes (primary=frozen_topk)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.environ.get("GNN_DECODE_TOP_K", "10")),
        help="GNN_DECODE_TOP_K for frozen_topk joint search",
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
    soft_models = _load_gnn(args.gnn_model.resolve(), label="soft_combo_conc")

    print("=" * 72)
    print(f"Phase 2 soft_combo_conc ablation — {PROBLEM_ID} / {INTEL_STUB_VARIANT}")
    print(f"stub={stub_dir}")
    print(
        f"soft_combo={soft_models['model_path']}  md5={soft_models['model_md5'][:12]}…  "
        f"plat_in={soft_models['platform_in_features']}"
    )
    print(f"layout=dim24  N={TARGET_N_TASKS}  top_k={args.top_k}")
    print("=" * 72)

    summary: Dict[str, Any] = {
        "phase": "phase2_soft_combo_conc_ablation",
        "problem_id": PROBLEM_ID,
        "stub_variant": INTEL_STUB_VARIANT,
        "stub_dir": str(stub_dir),
        "warmth_physics": GATE_WARMTH_PHYSICS,
        "n_tasks": TARGET_N_TASKS,
        "gnn_model": soft_models["model_path"],
        "gnn_md5": soft_models["model_md5"],
        "platform_in_features": soft_models["platform_in_features"],
        "inference_feature_layout": "dim24",
        "train_objective": "soft_combo_conc",
        "decode_top_k": int(args.top_k),
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
        _attach_oracle(ect, oracle_score)
        summary["policies"]["ect_pull"] = ect
        (results_dir / "ect_pull.json").write_text(json.dumps(ect, indent=2) + "\n")
        print(f"  {PRIMARY_SCORE_KEY}={ect[PRIMARY_SCORE_KEY]:.2f}s")

    if not args.skip_ce_baseline:
        print("\n--- baseline: CE dim16 argmax (Phase 1 floor) ---")
        ce_models = _load_gnn(args.ce_baseline.resolve(), label="ce_only")
        ce_scored = _run_gnn_decode(
            sim_inputs, infra, workload, models=ce_models, decode_mode="argmax"
        )
        _attach_oracle(ce_scored, oracle_score)
        summary["policies"]["ce_argmax"] = ce_scored
        summary["ce_baseline_model"] = ce_models["model_path"]
        summary["ce_baseline_md5"] = ce_models["model_md5"]
        (results_dir / "ce_argmax.json").write_text(json.dumps(ce_scored, indent=2) + "\n")
        print(
            f"  {PRIMARY_SCORE_KEY}={ce_scored[PRIMARY_SCORE_KEY]:.2f}s  "
            f"md5={ce_models['model_md5'][:12]}…"
        )

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise SystemExit("FAIL LOUD: --modes is empty")
    if "frozen_topk" not in modes:
        print(
            "WARN: primary decode frozen_topk not in --modes; "
            "Phase 2 falsification expects joint combo decode"
        )

    for mode in modes:
        print(f"\n--- soft_combo_conc decode={mode} ---")
        top_k = args.top_k if mode in ("frozen_topk", "topk", "topk_joint") else None
        scored = _run_gnn_decode(
            sim_inputs,
            infra,
            workload,
            models=soft_models,
            decode_mode=mode,
            top_k=top_k,
        )
        _attach_oracle(scored, oracle_score)
        key = f"soft_combo_conc_{mode}"
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

    # Falsification vs CE argmax floor + ect_pull ceiling
    primary_key = "soft_combo_conc_frozen_topk"
    if primary_key in summary["policies"]:
        joint = float(summary["policies"][primary_key][PRIMARY_SCORE_KEY])
        soft_argmax = None
        if "soft_combo_conc_argmax" in summary["policies"]:
            soft_argmax = float(summary["policies"]["soft_combo_conc_argmax"][PRIMARY_SCORE_KEY])
        ce_floor = None
        if "ce_argmax" in summary["policies"]:
            ce_floor = float(summary["policies"]["ce_argmax"][PRIMARY_SCORE_KEY])
        ect = None
        if "ect_pull" in summary["policies"]:
            ect = float(summary["policies"]["ect_pull"][PRIMARY_SCORE_KEY])

        vs_ce = (ce_floor - joint) if ce_floor is not None else None
        near_ceiling = joint < 50.0
        still_pile = joint >= 90.0
        if near_ceiling:
            interpretation = (
                "joint soft_combo_conc + frozen_topk recovers ect_pull-like intelligence"
            )
        elif still_pile:
            interpretation = (
                "hostile — b≤4 co-sim combos + joint score insufficient for N=12 "
                "continuous stream (stayed in 90–125s pile band)"
            )
        elif vs_ce is not None and vs_ce > 5.0:
            interpretation = (
                "partial — joint training beats CE argmax but residual to ect_pull remains"
            )
        else:
            interpretation = (
                "null / hostile vs CE — joint objective did not close Regime B residual"
            )

        summary["falsification"] = {
            "soft_combo_conc_frozen_topk_primary_s": joint,
            "soft_combo_conc_argmax_primary_s": soft_argmax,
            "ce_argmax_primary_s": ce_floor,
            "ect_pull_primary_s": ect,
            "improvement_vs_ce_s": vs_ce,
            "near_physics_ceiling": near_ceiling,
            "still_in_pile_band": still_pile,
            "interpretation": interpretation,
        }
        print("\n=== Phase 2 falsification ===")
        print(f"  soft_combo frozen_topk: {joint:.2f}s")
        if soft_argmax is not None:
            print(f"  soft_combo argmax:      {soft_argmax:.2f}s")
        if ce_floor is not None:
            print(f"  CE argmax floor:        {ce_floor:.2f}s")
        if ect is not None:
            print(f"  ect_pull ceiling:       {ect:.2f}s")
        if vs_ce is not None:
            print(f"  improvement vs CE:      {vs_ce:.2f}s")
        print(f"  verdict:                {interpretation}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
