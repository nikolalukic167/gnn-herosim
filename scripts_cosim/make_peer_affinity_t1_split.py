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

from non_unique_lib.training_contract import (  # noqa: E402
    SPLIT_ARTIFACT_SCHEMA,
    canonical_parent_id,
    load_split_artifact,
)
import hashlib
import json


def _cell_seed(ds_dir: Path) -> int:
    meta = json.loads((ds_dir / "infrastructure.json").read_text()).get("metadata") or {}
    cell = (meta.get("warm_snapshot") or {}).get("cell_seed")
    if cell is None:
        raise SystemExit(f"{ds_dir}: no metadata.warm_snapshot.cell_seed -- cannot group by cell")
    return int(cell)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--heldout-corpus", required=True, help="corpus dir name whose datasets form the test block")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--val-group-by", choices=("dataset", "cell"), default="dataset",
        help="cell (backlog_corpus_v1): val holds out WHOLE capture cells (warm_snapshot.cell_seed), so "
             "no val group has a same-run neighbour in train; cells are taken in a seeded order until "
             "val reaches --val-fraction of the training datasets")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output} — a split artifact is frozen once runs depend on it.")
    raw_ids = pickle.load(open(args.cache_dir / "dataset_ids.pkl", "rb"))
    meta = json.loads((args.cache_dir / "metadata.json").read_text())
    parent_ids = meta.get("parent_dataset_ids")
    if parent_ids is not None and len(parent_ids) != len(raw_ids):
        raise SystemExit("metadata parent_dataset_ids and dataset_ids.pkl disagree in length")
    ids = sorted({canonical_parent_id(parent_ids[i] if parent_ids is not None else d) for i, d in enumerate(raw_ids)})
    test = [i for i in ids if i.startswith(args.heldout_corpus + "/")]
    rest = [i for i in ids if i not in set(test)]
    if not test or not rest:
        raise SystemExit(f"held-out {len(test)} / training {len(rest)} — both must be non-empty")
    rng = random.Random(args.random_state)
    val_cells = None
    if args.val_group_by == "cell":
        cell_of = {i: _cell_seed(REPO_ROOT / "simulation_data" / i) for i in rest}
        cells = sorted(set(cell_of.values()))
        rng.shuffle(cells)
        target = args.val_fraction * len(rest)
        val_cells, n = [], 0
        for c in cells:
            if n >= target:
                break
            val_cells.append(c)
            n += sum(1 for v in cell_of.values() if v == c)
        val = sorted(i for i in rest if cell_of[i] in set(val_cells))
        train = sorted(i for i in rest if cell_of[i] not in set(val_cells))
        if not train:
            raise SystemExit("cell-grouped split left no training datasets")
    else:
        shuffled = rest[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(args.val_fraction * len(rest))))
        val = sorted(shuffled[:n_val]); train = sorted(shuffled[n_val:])
    payload = {
        "schema": SPLIT_ARTIFACT_SCHEMA,
        "cache_dir": str(args.cache_dir),
        "n_parents": len(ids),
        "random_state": int(args.random_state),
        "heldout_corpus": args.heldout_corpus,
        "val_fraction": float(args.val_fraction),
        **({"val_group_by": "cell", "val_cells": sorted(val_cells)} if val_cells is not None else {}),
        "train": train,
        "val": val,
        "test": sorted(test),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    payload, sha = load_split_artifact(args.output)
    print(f"[split] wrote {args.output}: train={len(payload['train'])} val={len(payload['val'])} test={len(payload['test'])} sha256={sha}")


if __name__ == "__main__":
    main()
