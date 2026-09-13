#!/usr/bin/env python3
"""
Regime B oracle_split co-sim — all-cold union seeds with FilterStore-scale labels.

Live intel cell is N=12. Co-sim stays at COSIM_NUM_TASKS=4 BF for placement space,
but MUST use the same lever: scarce attractor + all-cold union seeds under
platform_reuse_v1 (not scarce-warm ER). Fail loud if best RTT is warm-path.

Usage:
    pipenv run python3 scripts_cosim/generate_regime_b_oracle_split_cosim.py --n-datasets 3
    pipenv run python3 scripts_cosim/generate_regime_b_oracle_split_cosim.py --smoke
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    COSIM_NUM_TASKS,
    COSIM_OUTPUT_SUBDIR,
    GATE_WARMTH_PHYSICS,
    PRIMARY_SCORE_KEY,
    PROBLEM_ID,
    T_BASELINE_S,
    T_PULL_S,
    TARGET_SCORE_TOLERANCE_S,
    TARGET_TASK_TYPE,
    TOY_MIN_RATIO,
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
DEFAULT_OUT = (
    PROJECT_ROOT / "simulation_data" / f"{COSIM_OUTPUT_SUBDIR}_oracle_split_cosim"
)


def _dedupe_seeds(seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for s in seeds:
        key = (s["node_name"], int(s["platform_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _oracle_split_seeds_n(
    n: int,
    dc: List[Dict[str, Any]],
    dp: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    seeds = _dedupe_seeds(list(dc) + list(dp))
    expected = n + (n - 1)  # N on node0 + 1 on each other server
    if len(seeds) != expected:
        raise RuntimeError(
            f"FAIL LOUD: oracle_split seeds={len(seeds)} != expected {expected} "
            f"(N={n} on node0 + 1/other)"
        )
    by_node: Dict[str, int] = {}
    for s in seeds:
        by_node[s["node_name"]] = by_node.get(s["node_name"], 0) + 1
    if by_node.get("node0") != n:
        raise RuntimeError(
            f"FAIL LOUD: node0 seeds={by_node.get('node0')} != N={n}"
        )
    for i in range(1, n):
        name = f"node{i}"
        if by_node.get(name) != 1:
            raise RuntimeError(
                f"FAIL LOUD: {name} seeds={by_node.get(name)} != 1"
            )
    return seeds


def _build_dataset_infra(
    n: int,
    *,
    nodes: List[Dict[str, Any]],
    free_seeds: List[Dict[str, Any]],
    node_id_by_name: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "network": {"bandwidth": 100.0},
        "nodes": nodes,
        "preinitialize_platforms": True,
        "defer_cold_replica_init": True,
        "warmth_physics": GATE_WARMTH_PHYSICS,
        "stub_variant": "oracle_split_cosim",
        "deterministic_replica_placements": {TARGET_TASK_TYPE: free_seeds},
        "replica_plan": {
            "preinit_clients": [],
            "preinit_servers": [f"node{i}" for i in range(n)],
            "preinit_task_types": [TARGET_TASK_TYPE],
            "replicas_config": {TARGET_TASK_TYPE: {"per_client": 0, "per_server": 0}},
            "prewarm_config": {},
        },
        "scheduler": {"batch_size": n, "batch_timeout": 0.02},
        "fast_forward_warmup": True,
        "fast_forward_threshold": 1,
        "_node_id_by_name": node_id_by_name,
    }


def _seed_to_forced(
    seeds: List[Dict[str, Any]],
    node_id_by_name: Dict[str, int],
    plan: Tuple[int, ...],
) -> Dict[int, Tuple[int, int]]:
    forced: Dict[int, Tuple[int, int]] = {}
    for task_id, seed_idx in enumerate(plan):
        s = seeds[seed_idx]
        node_id = int(node_id_by_name[s["node_name"]])
        plat_id = int(s["platform_id"])
        forced[task_id] = (node_id, plat_id)
    return forced


def _run_forced(
    sim_inputs: Dict[str, Any],
    infrastructure: Dict[str, Any],
    workload: Dict[str, Any],
    forced: Dict[int, Tuple[int, int]],
    *,
    quiet: bool,
) -> Dict[str, Any]:
    infra = dict(infrastructure)
    infra.pop("_node_id_by_name", None)
    infra["forced_placements"] = forced
    config = {"infrastructure": infra, "workload": workload}
    prev_capture = os.environ.get("GNN_CAPTURE_DATASET_STATE")
    os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    os.environ["HEROSIM_WARMTH_PHYSICS"] = GATE_WARMTH_PHYSICS
    try:
        ctx = redirect_stdout(StringIO()), redirect_stderr(StringIO())
        if quiet:
            with ctx[0], ctx[1]:
                result = execute_simulation(
                    config,
                    sim_inputs,
                    scheduling_strategy="determined_determined",
                    cache_policy="fifo",
                    task_priority="fifo",
                    keep_alive=KEEP_ALIVE,
                    queue_length=QUEUE_LENGTH,
                )
        else:
            result = execute_simulation(
                config,
                sim_inputs,
                scheduling_strategy="determined_determined",
                cache_policy="fifo",
                task_priority="fifo",
                keep_alive=KEEP_ALIVE,
                queue_length=QUEUE_LENGTH,
            )
    finally:
        if prev_capture is None:
            os.environ.pop("GNN_CAPTURE_DATASET_STATE", None)
        else:
            os.environ["GNN_CAPTURE_DATASET_STATE"] = prev_capture
    stats = result.get("stats") or {}
    task_rows = attach_burst_ids_from_workload(
        extract_task_metrics(stats), workload["events"]
    )
    regime = burst_regime_summary(task_rows)
    primary = float(regime[PRIMARY_SCORE_KEY])
    return {
        "stats": stats,
        "task_rows": task_rows,
        "regime_b": regime,
        "primary_s": primary,
        "total_rtt": rtt_from_stats(stats),
        "result": result,
    }


def _assert_filterstore_scale(best_primary: float, contended_primary: float) -> None:
    if abs(best_primary - T_BASELINE_S) > TARGET_SCORE_TOLERANCE_S + 1.0:
        raise RuntimeError(
            f"FAIL LOUD: best primary={best_primary:.2f}s not FilterStore oracle "
            f"(~{T_BASELINE_S:.2f}s). Labels still warm-path?"
        )
    ratio = contended_primary / best_primary if best_primary > 0 else 0.0
    if ratio < TOY_MIN_RATIO:
        raise RuntimeError(
            f"FAIL LOUD: contended/oracle={ratio:.2f}x < {TOY_MIN_RATIO:.1f}x "
            f"(contended={contended_primary:.2f}s best={best_primary:.2f}s)"
        )


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
    by_name = {n["node_name"]: n for n in nodes}
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
    # Fail loud if maps incomplete.
    for node in nodes:
        expected = len(names) - 1
        if len(node["network_map"]) != expected:
            raise RuntimeError(
                f"network_map size {len(node['network_map'])} != {expected} "
                f"for {node['node_name']}"
            )


def _default_diversity_grid() -> List[Dict[str, float]]:
    """Cartesian latency diversity for oracle_split co-sim (48 cells with ×4 seeds)."""
    bases = (0.0005, 0.001, 0.002, 0.005)
    scarce = (0.0001, 0.0005, 0.001)
    return [
        {"base_latency_s": b, "scarce_attract_latency_s": s}
        for b in bases
        for s in scarce
    ]


def generate_one_dataset(
    ds_dir: Path,
    *,
    n: int,
    quiet: bool,
    max_combos: Optional[int],
    base_latency_s: float = 0.001,
    scarce_attract_latency_s: float = 0.001,
    diversity_seed: int = 0,
    resume: bool = False,
) -> Dict[str, Any]:
    jsonl_ok = (ds_dir / "placements" / "placements.jsonl").is_file() and (
        (ds_dir / "placements" / "placements.jsonl").stat().st_size > 0
    )
    best_ok = (ds_dir / "best.json").is_file()
    if resume and jsonl_ok and best_ok:
        best = json.loads((ds_dir / "best.json").read_text())
        meta = {
            "dataset": ds_dir.name,
            "best_s": float(best.get("regime_b_primary_score_s") or best.get("rtt")),
            "skipped": True,
            "base_latency_s": base_latency_s,
            "scarce_attract_latency_s": scarce_attract_latency_s,
            "diversity_seed": diversity_seed,
        }
        print(f"  {ds_dir.name}: SKIP resume best={meta['best_s']:.2f}s")
        return meta

    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    ds_dir.mkdir(parents=True, exist_ok=True)

    nodes, node_id_by_name, plat_id_by_node_local = build_target_nodes(
        n, platforms_scarce=n, platforms_other=1
    )
    # Tiny deterministic jitter from diversity_seed (±10% on base, not on scarce).
    jitter = 1.0 + 0.02 * ((diversity_seed % 5) - 2)
    base_j = float(base_latency_s) * jitter
    _apply_latency_diversity(
        nodes,
        base_latency_s=base_j,
        scarce_attract_latency_s=float(scarce_attract_latency_s),
    )
    fc, dc = contended_placements(n, node_id_by_name, plat_id_by_node_local)
    fp, dp = parallel_placements(n, node_id_by_name, plat_id_by_node_local)
    free_seeds = _oracle_split_seeds_n(n, dc, dp)

    infrastructure = _build_dataset_infra(
        n,
        nodes=nodes,
        free_seeds=free_seeds,
        node_id_by_name=node_id_by_name,
    )

    workload = build_burst_workload(
        n, burst_id=f"cold_burst_n{n}", task_type=TARGET_TASK_TYPE
    )
    sim_inputs = load_simulation_inputs(SIM_INPUT)

    oracle = _run_forced(sim_inputs, infrastructure, workload, fp, quiet=quiet)
    contended = _run_forced(sim_inputs, infrastructure, workload, fc, quiet=quiet)

    n_seeds = len(free_seeds)
    all_plans = list(itertools.product(range(n_seeds), repeat=n))
    if max_combos is not None and len(all_plans) > max_combos:
        keep: List[Tuple[int, ...]] = []
        seed_key = {
            (s["node_name"], int(s["platform_id"])): i for i, s in enumerate(free_seeds)
        }
        for forced in (fp, fc):
            try:
                plan = tuple(
                    seed_key[
                        (
                            next(
                                nm
                                for nm, nid in node_id_by_name.items()
                                if nid == forced[t][0]
                            ),
                            forced[t][1],
                        )
                    ]
                    for t in range(n)
                )
                keep.append(plan)
            except (KeyError, StopIteration):
                pass
        ranked = sorted(
            all_plans,
            key=lambda p: (
                len({free_seeds[i]["node_name"] for i in p}),
                len(set(p)),
            ),
            reverse=True,
        )
        for p in ranked:
            if p not in keep:
                keep.append(p)
            if len(keep) >= max_combos:
                break
        all_plans = keep

    placements_dir = ds_dir / "placements"
    placements_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = placements_dir / "placements.jsonl"

    best_primary = float("inf")
    best_plan: Optional[Tuple[int, ...]] = None
    best_payload: Optional[Dict[str, Any]] = None
    best_forced: Optional[Dict[int, Tuple[int, int]]] = None
    rows_written = 0
    t0 = time.time()

    with jsonl_path.open("w") as jf:
        for plan in all_plans:
            forced = _seed_to_forced(free_seeds, node_id_by_name, plan)
            scored = _run_forced(
                sim_inputs, infrastructure, workload, forced, quiet=True
            )
            primary = scored["primary_s"]
            placement_plan = {
                str(tid): [int(nid), int(pid)] for tid, (nid, pid) in forced.items()
            }
            row = {
                "placement_plan": placement_plan,
                "rtt": primary,
                "total_rtt": scored["total_rtt"],
                "regime_b_primary_score_s": primary,
            }
            jf.write(json.dumps(row) + "\n")
            rows_written += 1
            if primary < best_primary:
                best_primary = primary
                best_plan = plan
                best_payload = scored
                best_forced = forced

    if best_payload is None or best_plan is None or best_forced is None:
        raise RuntimeError("FAIL LOUD: no placements evaluated")

    _assert_filterstore_scale(best_primary, float(contended["primary_s"]))

    infra_out = dict(infrastructure)
    infra_out.pop("_node_id_by_name", None)
    (ds_dir / "infrastructure.json").write_text(
        json.dumps(infra_out, indent=2) + "\n"
    )
    (ds_dir / "workload.json").write_text(json.dumps(workload, indent=2) + "\n")
    (ds_dir / "best.json").write_text(
        json.dumps(
            {
                "file": "optimal_result.json",
                "rtt": best_primary,
                "regime_b_primary_score_s": best_primary,
                "plan_seed_indices": list(best_plan),
            },
            indent=2,
        )
        + "\n"
    )
    opt_result = dict(best_payload["result"])
    opt_result["sample"] = {
        "placement_plan": {
            str(tid): [int(nid), int(pid)] for tid, (nid, pid) in best_forced.items()
        }
    }
    (ds_dir / "optimal_result.json").write_text(json.dumps(opt_result, indent=2) + "\n")
    # SSC for CACHE 5.6 pull-obs / shared_fate
    from src.executecosimulation import build_system_state_captured
    from src.placement.model import DataclassJSONEncoder

    ssc = build_system_state_captured(opt_result.get("stats") or {})
    (ds_dir / "system_state_captured_unique.json").write_text(
        json.dumps(ssc, indent=2, cls=DataclassJSONEncoder) + "\n"
    )
    (ds_dir / "placement_metadata.json").write_text(
        json.dumps(
            {
                "num_placements": rows_written,
                "completed": rows_written,
                "n_tasks": n,
                "n_seeds": n_seeds,
                "oracle_parallel_s": oracle["primary_s"],
                "greedy_contended_s": contended["primary_s"],
                "best_s": best_primary,
                "oracle_regret_ratio": contended["primary_s"] / best_primary,
                "warmth_physics": GATE_WARMTH_PHYSICS,
                "problem_id": PROBLEM_ID,
                "label_regime": "oracle_split_all_cold_union",
                "base_latency_s": base_latency_s,
                "scarce_attract_latency_s": scarce_attract_latency_s,
                "base_latency_jittered_s": base_j,
                "diversity_seed": diversity_seed,
            },
            indent=2,
        )
        + "\n"
    )
    elapsed = time.time() - t0
    meta = {
        "dataset": ds_dir.name,
        "best_s": best_primary,
        "oracle_s": oracle["primary_s"],
        "contended_s": contended["primary_s"],
        "n_placements": rows_written,
        "elapsed_s": elapsed,
        "t_pull_s": T_PULL_S,
        "base_latency_s": base_latency_s,
        "scarce_attract_latency_s": scarce_attract_latency_s,
        "diversity_seed": diversity_seed,
    }
    print(
        f"  {ds_dir.name}: best={best_primary:.2f}s  "
        f"oracle={oracle['primary_s']:.2f}s  contended={contended['primary_s']:.2f}s  "
        f"placements={rows_written}  lat=({base_latency_s},{scarce_attract_latency_s})  "
        f"({elapsed:.1f}s)"
    )
    return meta


def _generate_one_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return generate_one_dataset(
        Path(job["ds_dir"]),
        n=int(job["n"]),
        quiet=bool(job["quiet"]),
        max_combos=job.get("max_combos"),
        base_latency_s=float(job["base_latency_s"]),
        scarce_attract_latency_s=float(job["scarce_attract_latency_s"]),
        diversity_seed=int(job["diversity_seed"]),
        resume=bool(job.get("resume", False)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-datasets", type=int, default=None,
                        help="Override diversity grid size (default: 48 = 12 lat × 4 seeds)")
    parser.add_argument("--n-tasks", type=int, default=COSIM_NUM_TASKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scale", action="store_true",
                        help="Full diversity grid (48 ds) for training corpus")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true", default=True)
    args = parser.parse_args()

    n = int(args.n_tasks)
    if n < 2:
        raise SystemExit("--n-tasks must be >= 2")

    grid = _default_diversity_grid()
    seeds = (0, 1, 2, 3)
    cells: List[Dict[str, Any]] = []
    for g in grid:
        for seed in seeds:
            cells.append({**g, "diversity_seed": seed})

    if args.smoke:
        cells = cells[:1]
        max_combos = 64 if args.max_combos is None else args.max_combos
    elif args.scale:
        max_combos = args.max_combos
    elif args.n_datasets is not None:
        cells = cells[: int(args.n_datasets)]
        max_combos = args.max_combos
    else:
        cells = cells[:3]
        max_combos = args.max_combos

    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, cell in enumerate(cells):
        jobs.append(
            {
                "ds_dir": str(out / f"ds_{i:05d}"),
                "n": n,
                "quiet": bool(args.quiet),
                "max_combos": max_combos,
                "base_latency_s": cell["base_latency_s"],
                "scarce_attract_latency_s": cell["scarce_attract_latency_s"],
                "diversity_seed": cell["diversity_seed"],
                "resume": bool(args.resume),
            }
        )

    print("=" * 72)
    print(f"Regime B oracle_split co-sim  N={n}  physics={GATE_WARMTH_PHYSICS}")
    print(f"out={out}  n_datasets={len(jobs)}  max_combos={max_combos}  workers={args.workers}")
    print("=" * 72)

    summaries: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for job in jobs:
            print(f"\n--- {Path(job['ds_dir']).name} ---")
            summaries.append(_generate_one_job(job))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_generate_one_job, job): job for job in jobs}
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    summaries.append(fut.result())
                except Exception as exc:
                    raise RuntimeError(
                        f"FAIL LOUD: dataset {job['ds_dir']} failed: {exc}"
                    ) from exc

    summaries.sort(key=lambda m: m["dataset"])
    summary_path = out / "corpus_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "problem_id": PROBLEM_ID,
                "warmth_physics": GATE_WARMTH_PHYSICS,
                "n_tasks": n,
                "n_datasets": len(summaries),
                "datasets": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    bests = [float(s["best_s"]) for s in summaries if "best_s" in s]
    if bests:
        print(
            f"\nCorpus: n={len(bests)} best_s in "
            f"[{min(bests):.2f}, {max(bests):.2f}]"
        )
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
