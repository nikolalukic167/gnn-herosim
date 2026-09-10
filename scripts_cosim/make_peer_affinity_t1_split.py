#!/usr/bin/env python3
"""peer_affinity_v1 T1: the shared split artifact with R2 as the held-out block.

Registration (docs/lineages/peer_affinity_v1.md, "Training registration T1"): test = every
dataset of the R2 corpus (seeds 7018-7034; no model has seen them), train/val = the
training corpus split ~80/20 by a fixed seed. Written once, committed, and pointed at by
every arm and seed (GNN: NEAR_RTT_SPLIT_ARTIFACT; MLP: --split-artifact).

Usage:
    PIPENV_IGNORE_VIRTUALENVS=1 PYTHONPATH=. pipenv run python3 \
        scripts_cosim/make_peer_affinity_t1_split.py \
        --cache-dir simulation_data/graphs_cache_peer_affinity_v1_t1 \
        --heldout-corpus gnn_datasets_peer_affinity_v1_c3_x200_r2 \
        --output experiments/peer_affinity_v1_t1_split.json
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "src" / "notebooks"
for p in (str(REPO_ROOT), str(NOTEBOOKS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from non_unique_lib.training_contract import load_split_artifact, write_split_artifact  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--heldout-corpus", required=True, help="corpus dir name whose datasets form the test block")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output} — a split artifact is frozen once runs depend on it.")
    ids = pickle.load(open(args.cache_dir / "dataset_ids.pkl", "rb"))
    ids = sorted({str(i) for i in ids})
    test = [i for i in ids if i.startswith(args.heldout_corpus + "/")]
    rest = [i for i in ids if i not in set(test)]
    if not test or not rest:
        raise SystemExit(f"held-out {len(test)} / training {len(rest)} — both must be non-empty")
    rng = random.Random(args.random_state)
    shuffled = rest[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(args.val_fraction * len(rest))))
    val = sorted(shuffled[:n_val]); train = sorted(shuffled[n_val:])
    write_split_artifact(args.output, {"train": train, "val": val, "test": sorted(test)})
    payload, sha = load_split_artifact(args.output)
    print(f"[split] wrote {args.output}: train={len(payload['train'])} val={len(payload['val'])} test={len(payload['test'])} sha256={sha}")


if __name__ == "__main__":
    main()
