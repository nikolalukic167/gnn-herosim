#!/usr/bin/env python3
"""
Phase 3: multi-seed ect_pull harvest → unified dim24 distill corpus.

Loops infrastructure latency × RNG seeds × jitter/init variants, running live
DES under knative_network_ect_pull with ECT_PULL_DISTILL_DIR set. Appends frames
into one harvest directory (no wipe unless --fresh).

Target: 5,000–10,000 frames (default --target-frames 6000).

Usage:
    pipenv run python3 scripts_cosim/harvest_ect_pull_distill_corpus.py
    pipenv run python3 scripts_cosim/harvest_ect_pull_distill_corpus.py \\
        --target-frames 6000 --fresh
    pipenv run python3 scripts_cosim/harvest_ect_pull_distill_corpus.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.build_regime_b_live_stub import (  # noqa: E402
    build_stub_payload,
    default_latency_grid,
)
from scripts_cosim.regime_b_problem_spec import (  # noqa: E402
    GATE_WARMTH_PHYSICS,
    INTEL_STUB_VARIANT,
    PRIMARY_SCORE_KEY,
    PROBLEM_ID,
    TARGET_N_TASKS,
)
from scripts_cosim.run_regime_b_live_stub_baselines import (  # noqa: E402
    run_policy,
)
from src.executecosimulation import load_simulation_inputs  # noqa: E402
from src.policy.knative_network_ect_pull.distill_log import (  # noqa: E402
    next_frame_index,
    reset_frame_counter,
)

SIM_INPUT = PROJECT_ROOT / "data/nofs-ids"
DEFAULT_OUT = (
    PROJECT_ROOT
    / "simulation_data/graphs_cache_regime_b_ect_pull_distill_multiseed"
)
DEFAULT_HARVEST = (
    PROJECT_ROOT
    / "simulation_data/regime_b_ect_pull_distill_harvest_multiseed"
)

# Init-state presets (warm_frac, busy_frac). Disjoint; sum ≤ 1.
# NOTE: under platform_reuse_v1, busy queues set previous_task → warm sandbox →
# zero pull cost. That poisons FilterStore ordinal imitation (375s collapse).
# Default harvest stays cold-only; warm/busy available via --init-presets for ablations.
DEFAULT_INIT_PRESETS: List[Tuple[float, float]] = [
    (0.0, 0.0),
]
DEFAULT_JITTER_S = (0.0, 2.0)
# Teacher must not collapse into contended pile (~125s). Busy/jitter can lift
# primary above the cold oracle (~31s); 100s is still well below pile.
DEFAULT_MAX_TEACHER_PRIMARY_S = 100.0


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _existing_frame_count(frames_dir: Path) -> int:
    if not frames_dir.is_dir():
        return 0
    return len(list(frames_dir.glob("frame_*.pt")))


def _max_frame_idx(frames_dir: Path) -> int:
    """Highest existing frame_XXXXXX index, or -1 if none."""
    if not frames_dir.is_dir():
        return -1
    best = -1
    for path in frames_dir.glob("frame_*.pt"):
        try:
            idx = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"FAIL LOUD: bad frame name {path.name}") from exc
        if idx > best:
            best = idx
    return best


def _build_cache(frames_dir: Path, cache_dir: Path, *, run_meta: Dict[str, Any]) -> int:
    frame_paths = sorted(frames_dir.glob("frame_*.pt"))
    if not frame_paths:
        raise RuntimeError(f"FAIL LOUD: no frames in {frames_dir}")

    graphs = []
    dataset_ids: List[str] = []
    optimal_rtt: Dict[str, float] = {}
    for i, path in enumerate(frame_paths):
        g = torch.load(path, map_location="cpu", weights_only=False)
        if int(g.platform_features.size(-1)) < 16:
            raise RuntimeError(
                f"FAIL LOUD: frame {path.name} plat_dim={int(g.platform_features.size(-1))} < 16"
            )
        if not hasattr(g, "y") or g.y is None:
            raise RuntimeError(f"FAIL LOUD: frame {path.name} missing y")
        if not hasattr(g, "teacher_soft") or g.teacher_soft is None:
            raise RuntimeError(f"FAIL LOUD: frame {path.name} missing teacher_soft")
        ds_id = str(getattr(g, "dataset_id", f"ect_pull_distill/frame_{i:06d}"))
        g.dataset_id = ds_id
        graphs.append(g)
        dataset_ids.append(ds_id)
        optimal_rtt[ds_id] = float(getattr(g, "env_now", 0.0))

    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "graphs.pkl").open("wb") as fh:
        pickle.dump(graphs, fh)
    with (cache_dir / "dataset_ids.pkl").open("wb") as fh:
        pickle.dump(dataset_ids, fh)
    with (cache_dir / "optimal_rtt.pkl").open("wb") as fh:
        pickle.dump(optimal_rtt, fh)

    meta = {
        "cache_version": 5.6,
        "feature_layout": "dim24",
        "platform_feature_dim": 16,
        "task_feature_dim": 3,
        "n_graphs": len(graphs),
        "corpus": "ect_pull_distill_multiseed",
        "stub_variant": INTEL_STUB_VARIANT,
        "problem_id": PROBLEM_ID,
        "run": run_meta,
    }
    (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return len(graphs)


def _iter_rollout_configs(
    *,
    n_rollouts: int,
    latency_grid: List[Dict[str, float]],
    jitter_values: List[float],
    init_presets: List[Tuple[float, float]],
    base_seed: int,
) -> List[Dict[str, Any]]:
    """Deterministic round-robin over latency × jitter × init, seeded per cell."""
    configs: List[Dict[str, Any]] = []
    combo_id = 0
    while len(configs) < n_rollouts:
        for lat in latency_grid:
            for jitter in jitter_values:
                for warm_f, busy_f in init_presets:
                    if len(configs) >= n_rollouts:
                        break
                    seed = int(base_seed) + combo_id
                    configs.append(
                        {
                            "rollout_idx": len(configs),
                            "combo_id": combo_id,
                            "seed": seed,
                            "arrival_jitter_s": float(jitter),
                            "warm_fraction": float(warm_f),
                            "busy_fraction": float(busy_f),
                            "base_latency_s": float(lat["base_latency_s"]),
                            "scarce_attract_latency_s": float(
                                lat["scarce_attract_latency_s"]
                            ),
                        }
                    )
                    combo_id += 1
                if len(configs) >= n_rollouts:
                    break
            if len(configs) >= n_rollouts:
                break
    return configs


def _harvest_one(
    sim_inputs: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    harvest_dir: Path,
    tau: float,
    max_teacher_primary_s: float,
    busy_queue: int,
) -> Dict[str, Any]:
    run_id = (
        f"ms{cfg['rollout_idx']:04d}_s{cfg['seed']}"
        f"_j{cfg['arrival_jitter_s']:.1f}"
        f"_w{cfg['warm_fraction']:.2f}_b{cfg['busy_fraction']:.2f}"
        f"_lat{cfg['base_latency_s']}_{cfg['scarce_attract_latency_s']}"
    )
    os.environ["ECT_PULL_DISTILL_DIR"] = str(harvest_dir)
    os.environ["ECT_PULL_DISTILL_TAU"] = str(float(tau))
    os.environ["ECT_PULL_DISTILL_RUN_ID"] = run_id

    frames_before = next_frame_index()
    payload = build_stub_payload(
        INTEL_STUB_VARIANT,
        arrival_jitter_s=float(cfg["arrival_jitter_s"]),
        warm_fraction=float(cfg["warm_fraction"]),
        busy_fraction=float(cfg["busy_fraction"]),
        busy_queue=int(busy_queue),
        base_latency_s=float(cfg["base_latency_s"]),
        scarce_attract_latency_s=float(cfg["scarce_attract_latency_s"]),
        seed=int(cfg["seed"]),
    )
    infra = payload["infrastructure"]
    workload = payload["workload"]

    scored = run_policy(
        sim_inputs,
        infra,
        workload,
        policy="knative_network_ect_pull",
        models=None,
        scheduling_strategy="kn_network_ect_pull_kn_network_ect_pull",
    )
    primary = float(scored[PRIMARY_SCORE_KEY])
    if primary > float(max_teacher_primary_s):
        raise RuntimeError(
            f"FAIL LOUD: ect_pull teacher primary={primary:.2f}s > "
            f"{max_teacher_primary_s}s on run_id={run_id} — refusing broken trajectory"
        )

    frames_after = next_frame_index()
    n_new = frames_after - frames_before
    if n_new != TARGET_N_TASKS:
        raise RuntimeError(
            f"FAIL LOUD: run_id={run_id} harvested {n_new} frames "
            f"!= TARGET_N_TASKS={TARGET_N_TASKS} "
            f"(before={frames_before} after={frames_after})"
        )

    row = {
        "run_id": run_id,
        "primary_score_s": primary,
        "n_frames": n_new,
        "frames_before": frames_before,
        "frames_after": frames_after,
        **cfg,
        "distill_randomization": infra.get("distill_randomization"),
    }
    with (harvest_dir / "rollouts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harvest-dir", type=Path, default=DEFAULT_HARVEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument(
        "--target-frames",
        type=int,
        default=6000,
        help="Stop once ≥ this many frames (default 6000 ∈ 5k–10k)",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=None,
        help="Override target with exact rollout count (each yields N=12 frames)",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--busy-queue", type=int, default=1)
    parser.add_argument(
        "--max-teacher-primary-s",
        type=float,
        default=DEFAULT_MAX_TEACHER_PRIMARY_S,
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Wipe harvest-dir before starting (default: append)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 rollouts only (sanity)",
    )
    parser.add_argument(
        "--jitter-values",
        default="0,2.0",
        help="Comma-separated arrival jitter seconds",
    )
    parser.add_argument(
        "--init-presets",
        default="0:0",
        help="Comma-separated warm:busy fractions (default cold-only 0:0). "
        "Warm/busy under platform_reuse_v1 skip pulls — use only for ablations.",
    )
    args = parser.parse_args()

    if args.target_frames < TARGET_N_TASKS and args.n_rollouts is None and not args.smoke:
        raise ValueError(
            f"FAIL LOUD: --target-frames={args.target_frames} < N={TARGET_N_TASKS}"
        )

    jitter_values = [float(x.strip()) for x in str(args.jitter_values).split(",") if x.strip()]
    if not jitter_values:
        raise ValueError("FAIL LOUD: empty --jitter-values")

    init_presets: List[Tuple[float, float]] = []
    for part in str(args.init_presets).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"FAIL LOUD: init preset {part!r} must be warm:busy (e.g. 0.25:0)"
            )
        w_s, b_s = part.split(":", 1)
        init_presets.append((float(w_s), float(b_s)))
    if not init_presets:
        raise ValueError("FAIL LOUD: empty --init-presets")

    harvest_dir = args.harvest_dir.resolve()
    frames_dir = harvest_dir / "frames"
    if args.fresh and harvest_dir.exists():
        shutil.rmtree(harvest_dir)
    harvest_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing = _existing_frame_count(frames_dir)
    start_idx = _max_frame_idx(frames_dir) + 1
    reset_frame_counter(start_idx)

    if args.smoke:
        n_rollouts = 2
        target_frames = existing + 2 * TARGET_N_TASKS
    elif args.n_rollouts is not None:
        n_rollouts = int(args.n_rollouts)
        target_frames = existing + n_rollouts * TARGET_N_TASKS
    else:
        need = max(0, int(args.target_frames) - existing)
        n_rollouts = (need + TARGET_N_TASKS - 1) // TARGET_N_TASKS
        target_frames = int(args.target_frames)

    latency_grid = default_latency_grid()
    configs = _iter_rollout_configs(
        n_rollouts=n_rollouts,
        latency_grid=latency_grid,
        jitter_values=jitter_values,
        init_presets=init_presets,
        base_seed=int(args.base_seed),
    )

    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
    os.environ["HEROSIM_WARMTH_PHYSICS"] = GATE_WARMTH_PHYSICS
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim24"

    sim_inputs = load_simulation_inputs(SIM_INPUT)

    print("=" * 72)
    print(f"Phase 3 ect_pull MULTI-SEED harvest — {PROBLEM_ID} / {INTEL_STUB_VARIANT}")
    print(f"harvest={harvest_dir}")
    print(
        f"existing_frames={existing}  start_idx={start_idx}  "
        f"n_rollouts={n_rollouts}  target_frames≥{target_frames}"
    )
    print(
        f"latency_cells={len(latency_grid)}  jitter={jitter_values}  "
        f"init_presets={init_presets}  tau={args.tau}"
    )
    print("=" * 72)

    t0 = time.time()
    rollout_rows: List[Dict[str, Any]] = []
    for cfg in configs:
        if _existing_frame_count(frames_dir) >= target_frames and not args.smoke:
            break
        print(
            f"[rollout {cfg['rollout_idx']+1}/{n_rollouts}] "
            f"seed={cfg['seed']} jitter={cfg['arrival_jitter_s']} "
            f"warm={cfg['warm_fraction']} busy={cfg['busy_fraction']} "
            f"lat=({cfg['base_latency_s']},{cfg['scarce_attract_latency_s']})"
        )
        row = _harvest_one(
            sim_inputs,
            cfg,
            harvest_dir=harvest_dir,
            tau=float(args.tau),
            max_teacher_primary_s=float(args.max_teacher_primary_s),
            busy_queue=int(args.busy_queue),
        )
        rollout_rows.append(row)
        print(
            f"  → primary={row['primary_score_s']:.2f}s  "
            f"frames={row['n_frames']}  total_now={next_frame_index()}"
        )

    n_frames = _build_cache(
        frames_dir,
        args.cache_dir.resolve(),
        run_meta={
            "mode": "multiseed",
            "n_rollouts_completed": len(rollout_rows),
            "tau": float(args.tau),
            "base_seed": int(args.base_seed),
            "target_frames": target_frames,
            "fresh": bool(args.fresh),
            "smoke": bool(args.smoke),
            "wall_s": time.time() - t0,
            "teacher_policy": "knative_network_ect_pull",
            "max_teacher_primary_s": float(args.max_teacher_primary_s),
            "primaries": [r["primary_score_s"] for r in rollout_rows],
        },
    )
    if n_frames < target_frames and not args.smoke:
        # Allow exact multiple: if target not multiple of 12, we may overshoot;
        # under-shoot is fail-loud.
        if n_frames < int(args.target_frames):
            raise RuntimeError(
                f"FAIL LOUD: only {n_frames} frames < target {args.target_frames}"
            )

    primaries = [r["primary_score_s"] for r in rollout_rows]
    summary = {
        "phase": "phase3_ect_pull_harvest_multiseed",
        "problem_id": PROBLEM_ID,
        "stub_variant": INTEL_STUB_VARIANT,
        "n_frames": n_frames,
        "n_rollouts": len(rollout_rows),
        "existing_frames_at_start": existing,
        "tau": float(args.tau),
        "harvest_dir": str(harvest_dir),
        "cache_dir": str(args.cache_dir.resolve()),
        "graphs_md5": _md5_file(args.cache_dir.resolve() / "graphs.pkl"),
        "primary_min_s": min(primaries) if primaries else None,
        "primary_max_s": max(primaries) if primaries else None,
        "primary_mean_s": (sum(primaries) / len(primaries)) if primaries else None,
        "wall_s": time.time() - t0,
    }
    (harvest_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.cache_dir.resolve() / "harvest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Wrote {n_frames} frames → {args.cache_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
