"""Mint the split artifact for a cache that EXTENDS the T1b corpus.

    python3 scripts_cosim/peer_only_v1_split.py --cache DIR --base experiments/peer_affinity_v1_t1b_split.json \
        --out experiments/peer_only_v1_1670_split.json

Test and val are the base artifact's verbatim; train is the base's train plus every parent
of the cache not in the base. Selection and held-out are therefore identical between the
base corpus's arms and the extended corpus's arms (peer_only_v1 B1 pairs like with like).
Fails loud if any base parent is absent from the cache, and verifies coverage exactly the
way the trainer will (assert_split_artifact_covers).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Sequence

from src.notebooks.non_unique_lib.training_contract import (
    assert_split_artifact_covers, canonical_parent_id, load_split_artifact,
)


def mint(cache: Path, base: Path, out: Path) -> dict:
    ids = [str(i) for i in pickle.load(open(cache / "dataset_ids.pkl", "rb"))]
    present = {canonical_parent_id(i) for i in ids}
    b, _ = load_split_artifact(base)
    old = set(b["train"]) | set(b["val"]) | set(b["test"])
    missing = sorted(old - present)
    if missing:
        raise SystemExit(f"FAIL LOUD: {len(missing)} base parents absent from {cache}, e.g. {missing[:3]}")
    new = sorted(present - old)
    payload = {
        "schema": b["schema"], "cache_dir": str(cache), "heldout_corpus": b.get("heldout_corpus"),
        "n_parents": len(present), "random_state": b.get("random_state", 42),
        "derived_from": str(base),
        "derivation": (f"test and val are {base.name}'s verbatim ({len(b['test'])} + {len(b['val'])}); train = its "
                       f"{len(b['train'])} + the {len(new)} parents of this cache not in it"),
        "train": sorted(set(b["train"]) | set(new)), "val": sorted(b["val"]), "test": sorted(b["test"]),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    p, sha = load_split_artifact(out)
    assert_split_artifact_covers(p, ids, artifact_path=str(out))
    return {"train": len(p["train"]), "val": len(p["val"]), "test": len(p["test"]), "n": p["n_parents"],
            "sha256": sha, "new": len(new)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=Path("experiments/peer_affinity_v1_t1b_split.json"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    r = mint(a.cache, a.base, a.out)
    print(f"[split] {a.out}: train={r['train']} val={r['val']} test={r['test']} n={r['n']} new={r['new']} sha={r['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
