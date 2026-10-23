#!/usr/bin/env python3
"""live_link_features_v1 corpus plumbing: the recaptured, relabelled union corpus -> a combined view and a by-TOPOLOGY split (the filed Arm 1 split's unit).

  v7_corpus.py view  --corpus CORPUS --arms A [A2 ...] --out VIEW
      VIEW/gnn_datasets_wf1_<A> -> CORPUS/<A> for every arm name (the base-dir name the cache and the split ids use); the corpus is read, never written.
  v7_corpus.py split --manifests GLOB [GLOB ...] --out SPLIT.json [--val-topos 4] [--test-topos 1] [--seed 42] [--accel-split experiments/accel_replica_v1_split.json]
      One split_artifact_v1 over every BUILT dataset of the manifests (rows with status "built": dataset, state_id "<arm>:<topology>:...", topology). The unit is the topology, as the
      filed uncut_arm1_split.json: val and test topologies are drawn from the topology list by the same nested sklearn train_test_split the trainers' split artifact uses
      (random_state = --seed), every other topology is train; no topology is in two parts. `test` is a placeholder (NEAR_RTT_SKIP_FINAL_TEST=1), as in the filed split.
      Fails loud on a dataset in two splits, on a manifest row without a topology, on an id that is an accel_replica_v1 test name, and on an empty part.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]


def dataset_rows(manifests: Sequence[str]) -> List[Dict[str, Any]]:
    """Built datasets of the manifests: {id: "gnn_datasets_wf1_<arm>/<dataset>", topology}. A dataset appears once."""
    files: List[str] = []
    for pat in manifests:
        files += sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"FAIL LOUD: no manifest matches {list(manifests)}")
    rows: Dict[str, Dict[str, Any]] = {}
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "built":
                continue
            arm = str(r["state_id"]).split(":")[0]
            if r.get("topology") is None:
                raise SystemExit(f"FAIL LOUD: {f}: built dataset {r.get('dataset')} has no topology")
            did = f"gnn_datasets_wf1_{arm}/{r['dataset']}"
            if did in rows:
                raise SystemExit(f"FAIL LOUD: dataset {did} is built twice in the manifests")
            rows[did] = {"id": did, "topology": str(r["topology"])}
    if not rows:
        raise SystemExit("FAIL LOUD: the manifests hold no built dataset")
    return [rows[k] for k in sorted(rows)]


def topology_split(rows: Sequence[Dict[str, Any]], n_val: int, n_test: int, seed: int, accel_test: Sequence[str] = ()) -> Dict[str, Any]:
    from sklearn.model_selection import train_test_split

    topos = sorted({r["topology"] for r in rows})
    if n_val < 1 or n_test < 1 or len(topos) <= n_val + n_test:
        raise SystemExit(f"FAIL LOUD: {len(topos)} topologies cannot give {n_val} val + {n_test} test and a train part")
    train_t, hold_t = train_test_split(topos, test_size=n_val + n_test, random_state=seed)
    val_t, test_t = train_test_split(sorted(hold_t), test_size=n_test, random_state=seed)
    part = {t: "train" for t in train_t}
    part.update({t: "val" for t in val_t})
    part.update({t: "test" for t in test_t})
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for r in rows:
        out[part[r["topology"]]].append(r["id"])
    for k, v in out.items():
        if not v:
            raise SystemExit(f"FAIL LOUD: the {k} part is empty")
    ids = [i for v in out.values() for i in v]
    if len(set(ids)) != len(ids):
        raise SystemExit("FAIL LOUD: a dataset id is in two parts")
    hit = set(ids) & set(accel_test)
    if hit:
        raise SystemExit(f"FAIL LOUD: {len(hit)} dataset ids are accel_replica_v1 test names, e.g. {sorted(hit)[:3]}")
    return {"schema": "split_artifact_v1", "n_parents": len(ids), "split_by": "topology", "random_state": seed, "test_placeholder": True, "heldout_as_val": True,
            "train": sorted(out["train"]), "val": sorted(out["val"]), "test": sorted(out["test"]),
            "topologies": {"train": sorted(train_t), "val": sorted(val_t), "test": sorted(test_t)}, "counts": {k: len(v) for k, v in out.items()}}


def write_split(payload: Dict[str, Any], out: Path) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def make_view(corpus: Path, arms: Sequence[str], out: Path) -> List[str]:
    if out.exists():
        raise SystemExit(f"FAIL LOUD: {out} exists")
    out.mkdir(parents=True)
    names = []
    for a in arms:
        src = corpus / a
        if not src.is_dir():
            raise SystemExit(f"FAIL LOUD: {src} is not a directory")
        os.symlink(os.path.realpath(src), out / f"gnn_datasets_wf1_{a}")
        names.append(f"gnn_datasets_wf1_{a}")
    return names


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("view")
    v.add_argument("--corpus", type=Path, required=True)
    v.add_argument("--arms", nargs="+", required=True)
    v.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("split")
    s.add_argument("--manifests", nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--val-topos", type=int, default=4)
    s.add_argument("--test-topos", type=int, default=1)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--accel-split", type=Path, default=REPO / "experiments" / "accel_replica_v1_split.json")
    a = ap.parse_args(argv)
    if a.cmd == "view":
        print(json.dumps({"view": str(a.out), "base_dirs": make_view(a.corpus, a.arms, a.out)}))
        return 0
    accel = json.load(open(a.accel_split))["test"]
    payload = topology_split(dataset_rows(a.manifests), a.val_topos, a.test_topos, a.seed, accel)
    sha = write_split(payload, a.out)
    print(json.dumps({"split": str(a.out), "sha256": sha, "n_parents": payload["n_parents"], "counts": payload["counts"], "topologies": payload["topologies"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
