#!/usr/bin/env python3
"""r1_attribution_v1: the train / val / test split artifact, by TOPOLOGY.

The corpus build fixes train and held-out topologies (corpus_prod/split.json, from the node's 19 test topologies and the feasibility list).
This freezes the artifact the trainer reads (split_artifact_v1, parents = cache dataset ids):

  test  = every dataset of a held-out topology (corpus_prod/split.json "heldout"; they are the 19 test topologies and never trained on)
  val   = every dataset of VAL_FRACTION of the TRAIN topologies, drawn with a fixed seed from train only
  train = the remaining train topologies

so validation is topology-held-out as well and no held-out topology can leak into either. A dataset's topology is read from its
generation_provenance.json (`--source-tag wf1p_cc40s<topology>_<rung>_<window>`).

`--dry-run-test K` is for a dry run before the held-out build exists: K train topologies are carved into `test` so the artifact loads
(the loader refuses an empty split). The artifact then carries "dry_run": true and must never be used for a gate.

  make_r1a_split.py --cache-dir <v5 cache> --corpus-split corpus_prod/split.json --out experiments/r1_attribution_v1_split.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "notebooks"))

from non_unique_lib.training_contract import SPLIT_ARTIFACT_SCHEMA, canonical_parent_id  # noqa: E402



def topology_of(dataset_dir: Path) -> str:
    prov = json.loads((dataset_dir / "generation_provenance.json").read_text())
    argv = prov.get("argv") or []
    for i, a in enumerate(argv):
        if a == "--source-tag" and i + 1 < len(argv):
            m = re.search(r"cc40s(\d+)", argv[i + 1])
            if m:
                return m.group(1)
    raise RuntimeError(f"{dataset_dir}: no --source-tag with a cc40s<topology> in generation_provenance.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--corpus-split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run-test", type=int, default=0, help="carve this many TRAIN topologies into test when no held-out dataset exists")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"FAIL LOUD: {a.out} exists; a split artifact is frozen once runs depend on it")
    corpus = json.loads(a.corpus_split.read_bytes())
    train_topos, heldout_topos = [str(t) for t in corpus["train"]], [str(t) for t in corpus["heldout"]]
    if set(train_topos) & set(heldout_topos):
        raise SystemExit("FAIL LOUD: corpus split lists a topology as both train and heldout")
    meta = json.loads((a.cache_dir / "metadata.json").read_text())
    ids = pickle.load(open(a.cache_dir / "dataset_ids.pkl", "rb"))
    parents_meta = meta.get("parent_dataset_ids")
    by_topo: Dict[str, List[str]] = {}
    dirs = {Path(d).name: Path(d) for d in meta["base_dirs"]}
    seen = set()
    for k, dsid in enumerate(ids):
        parent = canonical_parent_id(parents_meta[k] if parents_meta else dsid)
        if parent in seen:
            continue
        seen.add(parent)
        corpus_name, ds = parent.split("/", 1)
        topo = topology_of(dirs[corpus_name] / ds)
        by_topo.setdefault(topo, []).append(parent)
    unknown = sorted(set(by_topo) - set(train_topos) - set(heldout_topos))
    if unknown:
        raise SystemExit(f"FAIL LOUD: datasets of topologies {unknown} are in neither list of {a.corpus_split}")
    wrong = sorted(t for t, ps in by_topo.items() if t in heldout_topos and not all("heldout" in p for p in ps)) + \
            sorted(t for t, ps in by_topo.items() if t in train_topos and any("heldout" in p for p in ps))
    if wrong:
        raise SystemExit(f"FAIL LOUD: topologies {wrong} sit in the wrong corpus directory for corpus split {a.corpus_split}")
    present_train = sorted(t for t in by_topo if t in train_topos)
    present_held = sorted(t for t in by_topo if t in heldout_topos)
    rng = random.Random(a.seed)
    pool = list(present_train)
    rng.shuffle(pool)
    test_topos = list(present_held)
    dry = False
    if not test_topos:
        if a.dry_run_test < 1:
            raise SystemExit("FAIL LOUD: no held-out dataset in the cache; pass --dry-run-test K for a dry run")
        dry = True
        test_topos = sorted(pool[:a.dry_run_test])
        pool = pool[a.dry_run_test:]
    n_val = max(1, round(a.val_fraction * len(pool)))
    val_topos = sorted(pool[:n_val])
    train_set = sorted(pool[n_val:])
    if not train_set:
        raise SystemExit("FAIL LOUD: no train topology left after the validation draw")
    pick = lambda topos: sorted(p for t in topos for p in by_topo[t])
    payload = {
        "schema": SPLIT_ARTIFACT_SCHEMA, "cache_dir": str(a.cache_dir), "n_parents": len(seen), "random_state": a.seed,
        "split_by": "topology", "dry_run": dry,
        "corpus_split": {"path": str(a.corpus_split), "sha256": hashlib.sha256(a.corpus_split.read_bytes()).hexdigest()},
        "topologies": {"train": train_set, "val": val_topos, "test": test_topos},
        "train": pick(train_set), "val": pick(val_topos), "test": pick(test_topos),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(raw)
    print(f"[split] {a.out} sha256 {hashlib.sha256(raw).hexdigest()} dry_run={dry}: "
          f"train {len(payload['train'])} datasets / {len(train_set)} topologies, val {len(payload['val'])} / {len(val_topos)}, "
          f"test {len(payload['test'])} / {len(test_topos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
