#!/usr/bin/env python3
"""
Phase 3: harvest ect_pull trajectories on oracle_split_v1 → graphs.pkl distill corpus.

Runs live DES under knative_network_ect_pull with ECT_PULL_DISTILL_DIR set.
Each decision dumps a dim24 PyG frame (pull-ledger injected) + soft ECT targets.

Usage:
    pipenv run python3 scripts_cosim/harvest_ect_pull_distill_corpus.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.regime_b_problem_spec import (  # noqa: E402
    GATE_WARMTH_PHYSICS,
    INTEL_STUB_DIR,
    INTEL_STUB_VARIANT,
    PRIMARY_SCORE_KEY,
    PROBLEM_ID,
    TARGET_N_TASKS,
)
from scripts_cosim.run_regime_b_live_stub_baselines import (  # noqa: E402
    _load_stub,
    run_policy,
)
from src.executecosimulation import load_simulation_inputs  # noqa: E402


SIM_INPUT = PROJECT_ROOT / "data/nofs-ids"
DEFAULT_STUB = PROJECT_ROOT / INTEL_STUB_DIR
DEFAULT_OUT = (
    PROJECT_ROOT
    / "simulation_data/graphs_cache_regime_b_ect_pull_distill_oracle_split_v1"
)
DEFAULT_HARVEST = (
    PROJECT_ROOT
    / "simulation_data/regime_b_ect_pull_distill_harvest_oracle_split_v1"
)


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        # Stub optimal_rtt so stock tooling that probes the cache doesn't crash.
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
        "corpus": "ect_pull_distill",
        "stub_variant": INTEL_STUB_VARIANT,
        "problem_id": PROBLEM_ID,
        "run": run_meta,
    }
    (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return len(graphs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub-dir", type=Path, default=DEFAULT_STUB)
    parser.add_argument("--harvest-dir", type=Path, default=DEFAULT_HARVEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--run-id", default="oracle_split_v1")
    args = parser.parse_args()

    harvest_dir = args.harvest_dir.resolve()
    if harvest_dir.exists():
        shutil.rmtree(harvest_dir)
    harvest_dir.mkdir(parents=True, exist_ok=True)
    (harvest_dir / "frames").mkdir(parents=True, exist_ok=True)

    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
    os.environ["HEROSIM_WARMTH_PHYSICS"] = GATE_WARMTH_PHYSICS
    os.environ["INFERENCE_FEATURE_LAYOUT"] = "dim24"
    os.environ["ECT_PULL_DISTILL_DIR"] = str(harvest_dir)
    os.environ["ECT_PULL_DISTILL_TAU"] = str(float(args.tau))
    os.environ["ECT_PULL_DISTILL_RUN_ID"] = str(args.run_id)

    stub_dir = args.stub_dir.resolve()
    infra, workload, _refs = _load_stub(stub_dir, expected_variant=INTEL_STUB_VARIANT)
    sim_inputs = load_simulation_inputs(SIM_INPUT)

    print("=" * 72)
    print(f"Phase 3 ect_pull harvest — {PROBLEM_ID} / {INTEL_STUB_VARIANT}")
    print(f"stub={stub_dir}")
    print(f"harvest={harvest_dir}")
    print(f"tau={args.tau}  N={TARGET_N_TASKS}  layout=dim24")
    print("=" * 72)

    scored = run_policy(
        sim_inputs,
        infra,
        workload,
        policy="knative_network_ect_pull",
        models=None,
        scheduling_strategy="kn_network_ect_pull_kn_network_ect_pull",
    )
    primary = float(scored[PRIMARY_SCORE_KEY])
    print(f"ect_pull {PRIMARY_SCORE_KEY}={primary:.2f}s")
    if primary > 50.0:
        raise RuntimeError(
            f"FAIL LOUD: ect_pull teacher primary={primary:.2f}s > 50s — "
            "refusing to harvest a broken teacher trajectory"
        )

    n_frames = _build_cache(
        harvest_dir / "frames",
        args.cache_dir.resolve(),
        run_meta={
            "stub_dir": str(stub_dir),
            "primary_score_s": primary,
            "tau": float(args.tau),
            "run_id": str(args.run_id),
            "teacher_policy": "knative_network_ect_pull",
        },
    )
    if n_frames != TARGET_N_TASKS:
        raise RuntimeError(
            f"FAIL LOUD: harvested {n_frames} frames != TARGET_N_TASKS={TARGET_N_TASKS}"
        )

    summary = {
        "phase": "phase3_ect_pull_harvest",
        "problem_id": PROBLEM_ID,
        "stub_variant": INTEL_STUB_VARIANT,
        "n_frames": n_frames,
        "primary_score_s": primary,
        "tau": float(args.tau),
        "harvest_dir": str(harvest_dir),
        "cache_dir": str(args.cache_dir.resolve()),
        "graphs_md5": _md5_file(args.cache_dir.resolve() / "graphs.pkl"),
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
