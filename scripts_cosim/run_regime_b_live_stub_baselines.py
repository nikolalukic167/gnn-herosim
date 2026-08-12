#!/usr/bin/env python3
"""
Regime B live-stub zero-shot baselines: Kn / MLP / GNN on frozen N=12 cell.

Variants:
  scarce_preinit_v2 — GATE: free Kn ≥10× required; intel margin not expected
  oracle_split_v1   — INTEL: Kn ≥3× + learned beats Kn by ≥30s

Deploy ckpts: contention_v2 873/v5.5 (same as sealed holdout).
Primary score = regime_b_primary_score_s — never total_rtt.

Usage:
    pipenv run python3 scripts_cosim/run_regime_b_live_stub_baselines.py
    pipenv run python3 scripts_cosim/run_regime_b_live_stub_baselines.py \\
        --variant oracle_split_v1
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

from scripts_cosim.regime_b_metrics import (  # noqa: E402
    attach_burst_ids_from_workload,
    burst_regime_summary,
    total_rtt_trap_stats,
)
from scripts_cosim.regime_b_problem_spec import (  # noqa: E402
    GATE_WARMTH_PHYSICS,
    INTEL_STUB_DIR,
    INTEL_STUB_VARIANT,
    LIVE_STUB_DIR,
    LIVE_STUB_VARIANT,
    MIN_FREE_KN_REGRET_RATIO,
    MIN_INTEL_KN_REGRET_RATIO,
    MIN_INTEL_MARGIN_S,
    MIN_ORACLE_GREEDY_RATIO,
    PRIMARY_SCORE_KEY,
    PROBLEM_ID,
    TARGET_N_TASKS,
    TARGET_SCORE_TOLERANCE_S,
    TARGET_SERVER_COUNT,
    TARGET_TASK_TYPE,
    assert_gate_ratio,
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
DEFAULT_STUB_BY_VARIANT = {
    LIVE_STUB_VARIANT: PROJECT_ROOT / LIVE_STUB_DIR,
    INTEL_STUB_VARIANT: PROJECT_ROOT / INTEL_STUB_DIR,
}
DEFAULT_OUT_BY_VARIANT = {
    LIVE_STUB_VARIANT: PROJECT_ROOT
    / "simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_scarce_preinit_v2_zeroshot",
    INTEL_STUB_VARIANT: PROJECT_ROOT
    / "simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_oracle_split_v1_zeroshot",
}
GNN_MODEL = Path(
    os.environ.get(
        "GNN_MODEL",
        str(PROJECT_ROOT / "models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt"),
    )
)
MLP_MODEL = Path(
    os.environ.get(
        "MLP_MODEL",
        str(
            PROJECT_ROOT
            / "models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt"
        ),
    )
)
# Sealed-holdout / memory.md md5 prefixes for fail-loud identity check.
# Override with EXPECTED_GNN_MD5_PREFIX / EXPECTED_MLP_MD5_PREFIX for Regime B ckpts.
# Set to "" or "any" to skip prefix check (still requires file exists).
EXPECTED_GNN_MD5_PREFIX = os.environ.get("EXPECTED_GNN_MD5_PREFIX", "3efed472")
EXPECTED_MLP_MD5_PREFIX = os.environ.get("EXPECTED_MLP_MD5_PREFIX", "aa40dc51")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_model_identity(path: Path, expected_prefix: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    digest = _md5(path)
    skip = (not expected_prefix) or expected_prefix.lower() in {"any", "*"}
    if not skip and not digest.startswith(expected_prefix):
        raise RuntimeError(
            f"FAIL LOUD: {label} md5={digest} does not start with {expected_prefix} "
            f"(expected deploy ckpt). path={path}"
        )
    return digest


def _load_stub(
    stub_dir: Path, *, expected_variant: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    infra_path = stub_dir / "infrastructure.json"
    workload_path = stub_dir / "workload.json"
    ref_path = stub_dir / "reference_placements.json"
    for p in (infra_path, workload_path, ref_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"live stub incomplete: missing {p} — run build_regime_b_live_stub.py "
                f"--variant {expected_variant}"
            )
    infra = json.loads(infra_path.read_text())
    workload = json.loads(workload_path.read_text())
    refs = json.loads(ref_path.read_text())
    if infra.get("warmth_physics") != GATE_WARMTH_PHYSICS:
        raise RuntimeError(
            f"FAIL LOUD: stub warmth_physics={infra.get('warmth_physics')!r} "
            f"!= {GATE_WARMTH_PHYSICS}"
        )
    stub_variant = infra.get("stub_variant")
    if stub_variant != expected_variant:
        raise RuntimeError(
            f"FAIL LOUD: stub_variant={stub_variant!r} != {expected_variant!r} — "
            f"rebuild with build_regime_b_live_stub.py --variant {expected_variant}"
        )
    det = infra.get("deterministic_replica_placements") or {}
    seeded = det.get(TARGET_TASK_TYPE) or []
    by_node: Dict[str, int] = {}
    for p in seeded:
        by_node[str(p.get("node_name"))] = by_node.get(str(p.get("node_name")), 0) + 1

    if expected_variant == LIVE_STUB_VARIANT:
        if len(seeded) != TARGET_N_TASKS:
            raise RuntimeError(
                f"FAIL LOUD: scarce preinit has {len(seeded)} {TARGET_TASK_TYPE} replicas, "
                f"expected N={TARGET_N_TASKS} on node0"
            )
        if any(p.get("node_name") != "node0" for p in seeded):
            raise RuntimeError(
                f"FAIL LOUD: scarce preinit must be entirely on node0; got {seeded}"
            )
    elif expected_variant == INTEL_STUB_VARIANT:
        expected_n = TARGET_N_TASKS + (TARGET_SERVER_COUNT - 1)
        if len(seeded) != expected_n:
            raise RuntimeError(
                f"FAIL LOUD: oracle_split_v1 has {len(seeded)} seeds, expected {expected_n}"
            )
        if by_node.get("node0") != TARGET_N_TASKS:
            raise RuntimeError(
                f"FAIL LOUD: oracle_split_v1 node0={by_node.get('node0')} != {TARGET_N_TASKS}"
            )
        for i in range(1, TARGET_SERVER_COUNT):
            name = f"node{i}"
            if by_node.get(name) != 1:
                raise RuntimeError(
                    f"FAIL LOUD: oracle_split_v1 {name}={by_node.get(name)} != 1"
                )
    else:
        raise RuntimeError(f"FAIL LOUD: unknown expected_variant={expected_variant!r}")

    n_events = len(workload.get("events") or [])
    if n_events != TARGET_N_TASKS:
        raise RuntimeError(
            f"FAIL LOUD: stub workload has {n_events} events, expected N={TARGET_N_TASKS}"
        )
    return infra, workload, refs


def _score_run(
    stats: Dict[str, Any],
    workload: Dict[str, Any],
    *,
    oracle_rtt: Optional[float] = None,
) -> Dict[str, Any]:
    task_rows = attach_burst_ids_from_workload(
        extract_task_metrics(stats),
        workload.get("events") or [],
    )
    if len(task_rows) != TARGET_N_TASKS:
        raise RuntimeError(
            f"expected {TARGET_N_TASKS} task rows, got {len(task_rows)} — "
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


def _forced_from_json(raw: Dict[str, List[int]]) -> Dict[int, Tuple[int, int]]:
    return {int(k): (int(v[0]), int(v[1])) for k, v in raw.items()}


def run_determined(
    sim_inputs: Dict[str, Any],
    infrastructure: Dict[str, Any],
    workload: Dict[str, Any],
    *,
    forced: Dict[int, Tuple[int, int]],
    det_placements: List[Dict[str, Any]],
    label: str,
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
    scored = _score_run(result.get("stats") or {}, workload)
    scored["policy"] = label
    scored["scheduling_strategy"] = "determined_determined"
    return scored


def run_policy(
    sim_inputs: Dict[str, Any],
    infrastructure: Dict[str, Any],
    workload: Dict[str, Any],
    *,
    policy: str,
    models: Optional[Dict[str, Any]],
    scheduling_strategy: str,
) -> Dict[str, Any]:
    infra = dict(infrastructure)
    # Free placement — no forced map; keep replica_plan / warmth from stub.
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
    scored = _score_run(result.get("stats") or {}, workload)
    scored["policy"] = policy
    scored["scheduling_strategy"] = scheduling_strategy
    return scored


def _load_models_for(policy: str) -> Optional[Dict[str, Any]]:
    if policy in ("knative", "ect", "ect_pull"):
        return None
    task_types_data = json.loads((SIM_INPUT / "task-types.json").read_text())
    if policy == "mlp":
        _assert_model_identity(MLP_MODEL, EXPECTED_MLP_MD5_PREFIX, "MLP")
        return {"mlp_model_path": str(MLP_MODEL), "task_types_data": task_types_data}
    if policy == "gnn":
        from src.executesimulation import load_gnn_model

        digest = _assert_model_identity(GNN_MODEL, EXPECTED_GNN_MD5_PREFIX, "GNN")
        gnn_model, gnn_device = load_gnn_model(GNN_MODEL)
        print(f"  GNN md5={digest} device={gnn_device}")
        return {
            "gnn_model": gnn_model,
            "device": gnn_device,
            "task_types_data": task_types_data,
        }
    raise ValueError(f"unknown policy {policy!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=[LIVE_STUB_VARIANT, INTEL_STUB_VARIANT],
        default=LIVE_STUB_VARIANT,
    )
    parser.add_argument("--stub-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--policies",
        default=None,
        help=(
            "Comma-separated: knative,ect,ect_pull,mlp,gnn "
            "(default: knative,mlp,gnn on gate; knative,ect_pull,mlp,gnn on intel)"
        ),
    )
    parser.add_argument("--skip-refs", action="store_true")
    args = parser.parse_args()

    variant: str = args.variant
    stub_dir: Path = args.stub_dir or DEFAULT_STUB_BY_VARIANT[variant]
    output_dir: Path = args.output_dir or DEFAULT_OUT_BY_VARIANT[variant]
    if args.policies is None:
        # Intel cell always includes physics ceiling ect_pull; gate stays Kn/MLP/GNN.
        default_policies = (
            "knative,ect_pull,mlp,gnn"
            if variant == INTEL_STUB_VARIANT
            else "knative,mlp,gnn"
        )
    else:
        default_policies = args.policies
    requested = [p.strip() for p in default_policies.split(",") if p.strip()]

    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
    os.environ["HEROSIM_WARMTH_PHYSICS"] = GATE_WARMTH_PHYSICS
    os.environ.setdefault("GNN_BATCH_SIZE", "4")
    os.environ.setdefault("GNN_BATCH_TIMEOUT", "0.002")
    os.environ.setdefault("INFERENCE_FEATURE_LAYOUT", "dim22")
    os.environ.setdefault("GNN_DECODE_MODE", "argmax")

    infra, workload, refs = _load_stub(stub_dir, expected_variant=variant)
    sim_inputs = load_simulation_inputs(SIM_INPUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Regime B live-stub zero-shot — {PROBLEM_ID}  variant={variant}")
    print(f"stub={stub_dir}  physics={GATE_WARMTH_PHYSICS}  N={TARGET_N_TASKS}")
    print("=" * 72)

    summary: Dict[str, Any] = {
        "problem_id": PROBLEM_ID,
        "stub_dir": str(stub_dir),
        "stub_variant": variant,
        "warmth_physics": GATE_WARMTH_PHYSICS,
        "n_tasks": TARGET_N_TASKS,
        "models": {
            "gnn": str(GNN_MODEL),
            "mlp": str(MLP_MODEL),
            "gnn_md5": _assert_model_identity(GNN_MODEL, EXPECTED_GNN_MD5_PREFIX, "GNN")
            if "gnn" in requested
            else None,
            "mlp_md5": _assert_model_identity(MLP_MODEL, EXPECTED_MLP_MD5_PREFIX, "MLP")
            if "mlp" in requested
            else None,
        },
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

        print("\n--- reference: greedy_contended ---")
        cref = refs["greedy_contended"]
        contended = run_determined(
            sim_inputs,
            infra,
            workload,
            forced=_forced_from_json(cref["forced_placements"]),
            det_placements=cref["deterministic_replica_placements"][TARGET_TASK_TYPE],
            label="greedy_contended",
        )
        c_primary = float(contended[PRIMARY_SCORE_KEY])
        contended["oracle_rtt_s"] = oracle_score
        contended["oracle_regret_s"] = c_primary - float(oracle_score)
        contended["oracle_regret_ratio"] = c_primary / float(oracle_score)
        assert_gate_ratio(contended["oracle_regret_ratio"], context="live_stub_refs")
        summary["policies"]["greedy_contended"] = contended
        (results_dir / "greedy_contended.json").write_text(
            json.dumps(contended, indent=2) + "\n"
        )
        print(
            f"  {PRIMARY_SCORE_KEY}={c_primary:.2f}s  "
            f"regret_ratio={contended['oracle_regret_ratio']:.2f}x"
        )

    policy_map = {
        "knative": ("knative_network", "kn_network_kn_network"),
        "ect": ("knative_network_ect", "kn_network_ect_kn_network_ect"),
        "ect_pull": (
            "knative_network_ect_pull",
            "kn_network_ect_pull_kn_network_ect_pull",
        ),
        "mlp": ("mlp_batch", "mlp_batch_mlp_batch"),
        "gnn": ("gnn", "gnn_gnn"),
    }
    for key in requested:
        if key not in policy_map:
            raise SystemExit(
                f"unknown policy {key!r}; expected knative,ect,ect_pull,mlp,gnn"
            )
        policy_name, strategy = policy_map[key]
        print(f"\n--- policy: {key} ({strategy}) ---")
        models = _load_models_for(key)
        scored = run_policy(
            sim_inputs,
            infra,
            workload,
            policy=policy_name,
            models=models,
            scheduling_strategy=strategy,
        )
        if oracle_score is not None:
            primary = float(scored[PRIMARY_SCORE_KEY])
            scored["oracle_rtt_s"] = oracle_score
            scored["oracle_regret_s"] = primary - oracle_score
            scored["oracle_regret_ratio"] = (
                primary / oracle_score if oracle_score > 0 else float("inf")
            )
        summary["policies"][key] = scored
        out_path = results_dir / f"{key}.json"
        out_path.write_text(json.dumps(scored, indent=2) + "\n")
        print(
            f"  {PRIMARY_SCORE_KEY}={scored[PRIMARY_SCORE_KEY]:.2f}s  "
            f"total_rtt_trap={scored['total_rtt']:.2f}s"
            + (
                f"  vs_oracle={scored.get('oracle_regret_ratio', float('nan')):.2f}x"
                if oracle_score is not None
                else ""
            )
        )

    ranked = sorted(
        (
            (name, float(p[PRIMARY_SCORE_KEY]))
            for name, p in summary["policies"].items()
            if name in requested
        ),
        key=lambda x: x[1],
    )
    summary["ranking_primary_asc"] = [{"policy": n, PRIMARY_SCORE_KEY: s} for n, s in ranked]
    if ranked:
        summary["winner"] = ranked[0][0]

    # Physics ceiling (ect_pull) — not part of intelligence margin; fail loud if it
    # regresses away from oracle on the intel cell.
    if (
        variant == INTEL_STUB_VARIANT
        and "ect_pull" in summary["policies"]
        and oracle_score is not None
    ):
        ect_s = float(summary["policies"]["ect_pull"][PRIMARY_SCORE_KEY])
        ect_gap = ect_s - float(oracle_score)
        summary["physics_ceiling_policy"] = "ect_pull"
        summary["physics_ceiling_s"] = ect_s
        summary["physics_ceiling_gap_s"] = ect_gap
        if ect_gap > TARGET_SCORE_TOLERANCE_S:
            raise RuntimeError(
                f"FAIL LOUD: ect_pull physics ceiling={ect_s:.2f}s "
                f"exceeds oracle={oracle_score:.2f}s by {ect_gap:.2f}s "
                f"(tol={TARGET_SCORE_TOLERANCE_S:.1f}s). Marginal FilterStore cost broken."
            )
        summary["physics_ceiling_pass"] = True

    kn = summary["policies"].get("knative")
    if kn is not None and oracle_score is not None:
        kn_ratio = float(kn.get("oracle_regret_ratio") or 0.0)
        summary["free_kn_regret_ratio"] = kn_ratio
        if variant == LIVE_STUB_VARIANT:
            if kn_ratio < MIN_FREE_KN_REGRET_RATIO:
                raise RuntimeError(
                    f"FAIL LOUD: free Kn oracle_regret_ratio={kn_ratio:.3f}x "
                    f"< required {MIN_FREE_KN_REGRET_RATIO:.1f}x on {LIVE_STUB_VARIANT}. "
                    "Scarce preinit did not collapse Kn into FilterStore."
                )
            summary["free_kn_headroom_pass"] = True
        elif variant == INTEL_STUB_VARIANT:
            if kn_ratio < MIN_INTEL_KN_REGRET_RATIO:
                raise RuntimeError(
                    f"FAIL LOUD: intel-cell Kn ratio={kn_ratio:.3f}x "
                    f"< {MIN_INTEL_KN_REGRET_RATIO:.1f}x — lure collapsed to near-oracle."
                )
            summary["intel_kn_headroom_pass"] = True
            learned = [
                (n, float(summary["policies"][n][PRIMARY_SCORE_KEY]))
                for n in ("mlp", "gnn")
                if n in summary["policies"]
            ]
            if learned:
                best_name, best_s = min(learned, key=lambda x: x[1])
                kn_s = float(kn[PRIMARY_SCORE_KEY])
                margin = kn_s - best_s
                summary["intelligence_margin_s"] = margin
                summary["intelligence_best_policy"] = best_name
                if margin < MIN_INTEL_MARGIN_S:
                    raise RuntimeError(
                        f"FAIL LOUD: intelligence margin={margin:.2f}s "
                        f"({best_name} {best_s:.2f}s vs Kn {kn_s:.2f}s) "
                        f"< required {MIN_INTEL_MARGIN_S:.1f}s on {INTEL_STUB_VARIANT}. "
                        "Consider Regime B retrain."
                    )
                summary["intelligence_margin_pass"] = True

    summary_path = output_dir / "compare.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 72)
    print("Ranking (lower regime_b_primary_score_s wins):")
    for n, s in ranked:
        print(f"  {n:12s}  {s:.2f}s")
    if ranked:
        print(f"WINNER: {ranked[0][0]}")
    if summary.get("physics_ceiling_pass"):
        print(
            f"Physics ceiling ect_pull={summary['physics_ceiling_s']:.2f}s "
            f"(gap={summary['physics_ceiling_gap_s']:.2f}s) PASS"
        )
    if kn is not None and oracle_score is not None:
        print(f"Free Kn vs oracle: {kn['oracle_regret_ratio']:.2f}x")
        if variant == LIVE_STUB_VARIANT:
            print(f"  gate ≥{MIN_FREE_KN_REGRET_RATIO:.0f}× PASS")
        elif summary.get("intelligence_margin_pass"):
            print(
                f"  intel margin={summary['intelligence_margin_s']:.1f}s "
                f"({summary['intelligence_best_policy']} vs Kn) "
                f"≥{MIN_INTEL_MARGIN_S:.0f}s PASS"
            )
    print(f"Wrote {summary_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
