"""Write a split_artifact_v1 whose test/val sets are FIXED BLOCKS of a corpus, by seed.

`make_split_artifact.py` draws a random parent split of one cache. A learning curve
needs the opposite: one held-out set that every rung shares while the train set grows.
route_b fit-ceiling Phase 2 (docs/lineages/route_b_v1.md, 2026-09-06) generates the
held-out block into its own dataset dir (`gnn_datasets_dag4_route_b_pilot_v1_x_holdout`,
seeds 5001-5021) and assigns test/val by the dataset's generation seed; every other
parent in the rung's cache is train. The artifact is validated with the same loader the
trainers use, so coverage/overlap failures surface here, not in a SLURM task.

Usage:
    PIPENV_IGNORE_VIRTUALENVS=1 PYTHONPATH=. pipenv run python3 \
        scripts_cosim/make_split_artifact_by_block.py \
        --cache-dir simulation_data/graphs_cache_route_b_fit_p2_r1 \
        --holdout-dir gnn_datasets_dag4_route_b_pilot_v1_x_holdout \
        --test-seeds 5001-5017 --val-seeds 5018-5021 \
        --output experiments/route_b_fit_p2_split_r1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "src" / "notebooks"
for p in (str(REPO_ROOT), str(NOTEBOOKS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from non_unique_lib.training_contract import (  # noqa: E402
    SPLIT_ARTIFACT_SCHEMA,
    SPLIT_NAMES,
    assert_split_artifact_covers,
    canonical_parent_id,
    load_split_artifact,
)


def parse_seed_range(text: str) -> Set[int]:
    lo, hi = (int(x) for x in text.split("-"))
    if hi < lo:
        raise SystemExit(f"bad seed range {text!r}")
    return set(range(lo, hi + 1))


def dataset_seed(simulation_data: Path, parent_id: str) -> int:
    infra = simulation_data / parent_id / "infrastructure.json"
    with infra.open() as f:
        meta = json.load(f).get("metadata", {})
    seed = meta.get("seed")
    if seed is None:
        raise SystemExit(f"{infra}: no metadata.seed")
    return int(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--holdout-dir", required=True,
                    help="dataset dir NAME (under simulation_data/) holding test+val parents")
    ap.add_argument("--test-seeds", required=True, help="inclusive range, e.g. 5001-5017")
    ap.add_argument("--val-seeds", required=True, help="inclusive range, e.g. 5018-5021")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--simulation-data", type=Path, default=REPO_ROOT / "simulation_data")
    args = ap.parse_args()

    if args.output.exists():
        raise SystemExit(
            f"Refusing to overwrite {args.output} — a split artifact is frozen once runs "
            "depend on it. Delete it explicitly if regeneration is intended."
        )
    test_seeds, val_seeds = parse_seed_range(args.test_seeds), parse_seed_range(args.val_seeds)
    if test_seeds & val_seeds:
        raise SystemExit("test and val seed ranges overlap")

    with (args.cache_dir / "dataset_ids.pkl").open("rb") as f:
        dataset_ids = pickle.load(f)
    parents: List[str] = []
    seen: Set[str] = set()
    for dsid in dataset_ids:
        p = canonical_parent_id(dsid)
        if p not in seen:
            seen.add(p)
            parents.append(p)

    buckets: Dict[str, List[str]] = {name: [] for name in SPLIT_NAMES}
    holdout_prefix = args.holdout_dir.rstrip("/") + "/"
    for p in parents:
        if not p.startswith(holdout_prefix):
            buckets["train"].append(p)
            continue
        seed = dataset_seed(args.simulation_data, p)
        if seed in test_seeds:
            buckets["test"].append(p)
        elif seed in val_seeds:
            buckets["val"].append(p)
        else:
            raise SystemExit(
                f"{p}: seed {seed} is in the hold-out dir but in neither --test-seeds nor "
                "--val-seeds — the hold-out block must be fully assigned"
            )
    for name in SPLIT_NAMES:
        if not buckets[name]:
            raise SystemExit(f"split {name!r} is empty")

    payload = {
        "schema": SPLIT_ARTIFACT_SCHEMA,
        "cache_dir": str(args.cache_dir),
        "n_parents": len(parents),
        "assignment": "by_block",
        "holdout_dir": args.holdout_dir,
        "test_seeds": args.test_seeds,
        "val_seeds": args.val_seeds,
        "train": sorted(buckets["train"]),
        "val": sorted(buckets["val"]),
        "test": sorted(buckets["test"]),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    sha256 = hashlib.sha256(raw).hexdigest()

    reloaded, reloaded_sha = load_split_artifact(args.output)
    if reloaded_sha != sha256 or reloaded != payload:
        raise SystemExit(f"Round-trip mismatch on {args.output}")
    assert_split_artifact_covers(reloaded, dataset_ids, artifact_path=str(args.output))

    sizes = " / ".join(f"{name}={len(payload[name])}" for name in SPLIT_NAMES)
    print(f"[split-artifact] wrote {args.output}")
    print(f"[split-artifact] parents: {payload['n_parents']} ({sizes})")
    print(f"[split-artifact] sha256: {sha256}")


if __name__ == "__main__":
    main()
