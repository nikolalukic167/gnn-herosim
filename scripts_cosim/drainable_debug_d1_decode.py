#!/usr/bin/env python3
"""drainable_debug_v1 D1, step 4: decode both checkpoints on every captured state.

Emits `{dataset_id: {arm: {task_id: [node_id, platform_id]}}}` — the input
`drainable_debug_one_step_read.py --decoded` expects, so the checkpoint plans are scored
against the same enumerated sweep, by the same code, as the reactive plan and the optimum.

The decoder is the SERVING one the T1b arms were gated with (`EVAL_DECODE_REPLICA_REUSE=1`,
`EVAL_DECODE_RELAX=1`), not the evaluator's route_b default — the same correction
`peer_affinity_warm_w0_read.read_cold_checkpoints` carries, and for the same reason: the
route_b decoder has no plan at all on a 10-task peer group over ~3 candidates per type and
reports every dataset infeasible.

`load_arm` verifies the checkpoint's sidecar against `GNN_DISABLE_MESSAGE_PASSING`, so the
flag follows the arm being loaded. Serving the `gnn` weights with message passing off is a
train/serve mismatch, not an ablation, and it is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Platform feature column holding normalized queue depth. Named in
# `src/placement/queue_features.py:3` -- "Platform queue depth reaches the models twice: as a
# normalized depth (platform dim 7) and as a usage ratio against target concurrency (dim 13)."
# Only dim 7 is scaled: dim 13 is log1p-compressed on a different divisor, so scaling both
# would confound "sensitivity to depth" with "sensitivity to the compression".
QUEUE_DEPTH_DIM = 7
EXPECTED_PLATFORM_DIMS = 14


def scale_queue_depth(graphs: List[Any], scale: float) -> int:
    """Multiply platform dim 7 by `scale`, in place. Returns how many graphs were touched.

    `scale == 1.0` must be a no-op: offline_live_transfer_v1's R2 relies on a scaled decode
    being comparable to the unscaled one, so an accidental rewrite at scale 1 would make the
    surrogate measure the rewrite instead of the sensitivity.
    """
    if scale == 1.0:
        return 0
    if scale <= 0.0:
        raise SystemExit(f"FAIL LOUD: --queue-scale must be > 0, got {scale}")
    touched = 0
    for graph in graphs:
        feats = getattr(graph, "platform_features", None)
        if feats is None:
            raise SystemExit("FAIL LOUD: a graph has no platform_features to scale")
        if feats.shape[1] != EXPECTED_PLATFORM_DIMS:
            raise SystemExit(
                f"FAIL LOUD: platform_features has {feats.shape[1]} columns, expected "
                f"{EXPECTED_PLATFORM_DIMS}; dim {QUEUE_DEPTH_DIM} may not be the queue column"
            )
        feats[:, QUEUE_DEPTH_DIM] = feats[:, QUEUE_DEPTH_DIM] * scale
        touched += 1
    return touched


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--tag", default="peer-affinity-v1-t1b")
    ap.add_argument("--lr", default="lr2e3")
    ap.add_argument("--arms", nargs="+", default=["gnn", "mpoff"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha-key", default="2.5")
    ap.add_argument("--task-types", default="data/nofs-ids/task-types.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--queue-scale", type=float, default=1.0,
                    help="multiply platform dim 7 (normalized queue depth) by this before "
                         "decoding. 1.0 (default) is a strict no-op. Used by "
                         "offline_live_transfer_v1 R2 to measure how much a checkpoint's plan "
                         "moves when queue depth is pushed toward the live range the corpus "
                         "never contains.")
    args = ap.parse_args(argv)

    os.environ.setdefault("EVAL_DECODE_REPLICA_REUSE", "1")
    os.environ.setdefault("EVAL_DECODE_RELAX", "1")

    from scripts_cosim.eval_route_b_stage2_arm import evaluate_dataset, load_arm

    with open(args.cache_dir / "graphs.pkl", "rb") as fh:
        graphs = pickle.load(fh)
    with open(args.cache_dir / "dataset_ids.pkl", "rb") as fh:
        dataset_ids = pickle.load(fh)
    print(f"[d1-decode] {len(graphs)} graphs in {args.cache_dir}", flush=True)
    touched = scale_queue_depth(graphs, args.queue_scale)
    if touched:
        print(f"[d1-decode] queue depth (platform dim {QUEUE_DEPTH_DIM}) scaled by "
              f"{args.queue_scale} on {touched} graphs", flush=True)

    task_types_db = json.loads(Path(args.task_types).read_text())
    sim_root = REPO_ROOT / "simulation_data"

    out: Dict[str, Dict[str, Dict[int, List[int]]]] = {}
    stats: Dict[str, Dict[str, int]] = {}
    for arm_name in args.arms:
        os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1" if arm_name == "mpoff" else "0"
        ck = args.models_dir / f"{args.tag}-{arm_name}-{args.lr}-seed{args.seed}.pt"
        if not ck.is_file():
            raise SystemExit(f"FAIL LOUD: missing checkpoint {ck}")
        arm = load_arm(ck)
        n_ok = n_infeasible = 0
        for graph, did in zip(graphs, dataset_ids):
            res = evaluate_dataset(graph, did, arm=arm, alpha_key=args.alpha_key,
                                   simulation_data_root=sim_root, task_types_db=task_types_db)
            combo = res.get("decoded_combo")
            if res.get("infeasible") or combo is None:
                n_infeasible += 1
                continue
            # `decoded_combo` is in task-id order 0..n-1, the same ids the sweep rows use.
            out.setdefault(did, {})[arm_name] = {
                str(t): [int(p[0]), int(p[1])] for t, p in enumerate(combo)
            }
            n_ok += 1
        stats[arm_name] = {"decoded": n_ok, "infeasible": n_infeasible}
        print(f"[d1-decode] {arm_name} seed {args.seed}: {n_ok} decoded, {n_infeasible} infeasible",
              flush=True)
        if n_ok == 0:
            raise SystemExit(
                f"FAIL LOUD: {ck} decoded no feasible plan on any of {len(graphs)} datasets"
            )

    # Key by the cache's FULL dataset id, `<corpus dir name>/ds_XXXXX`. The bare name is not
    # unique: each source corpus numbers its datasets from zero, so keying on it silently
    # collapsed 151 datasets into 84 and handed one source's plan to another source's sweep
    # (caught 2026-09-14 by the read's own not-in-the-sweep guard, which is what it is for).
    if len(out) != len(graphs):
        raise SystemExit(
            f"FAIL LOUD: {len(out)} keyed datasets from {len(graphs)} graphs -- dataset ids "
            "are colliding and plans would be scored against the wrong sweep"
        )
    args.out.write_text(json.dumps({"_stats": stats, **out}, indent=1))
    print(f"[wrote] {args.out} ({len(out)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
