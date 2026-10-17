"""Relabel sidecar vs the Arm 1 cache: for every dataset, the sidecar's label plan has one [node, platform] per task and each one is a candidate of that task in the
cached graph (the trainer's NEAR_RTT_LABEL_OVERRIDE_JSON loader fails loud otherwise, train_near_rtt.py:1744). Also: the sidecar and the cache cover the same ids.

  uncut_relabel_cache_check.py <cache_dir> <sidecar.json>      exit 1 on any failure; prints counts per split
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List


def check_entry(g, entry: Dict[str, Any]) -> List[str]:
    plan = entry["plans"][0]
    n = len(g.task_logit_to_placement)
    if len(plan) != n:
        return [f"plan has {len(plan)} tasks, graph {n}"]
    bad = []
    for t, (node, plat) in enumerate(plan):
        cands = {tuple(int(v) for v in c) for c in g.task_logit_to_placement[t]}
        if (int(node), int(plat)) not in cands:
            bad.append(f"task {t}: ({node}, {plat}) is not a candidate")
    return bad


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cache, side = Path(argv[0]), json.load(open(argv[1]))
    side.pop("_meta", None)
    graphs = pickle.load(open(cache / "graphs.pkl", "rb"))
    ids = [str(x) for x in pickle.load(open(cache / "dataset_ids.pkl", "rb"))]
    if set(ids) != set(side):
        print(f"FAIL LOUD: cache has {len(ids)} ids, sidecar {len(side)}; only in cache {sorted(set(ids) - set(side))[:3]}, only in sidecar {sorted(set(side) - set(ids))[:3]}")
        return 1
    failed, per_split = [], {}
    for gid, g in zip(ids, graphs):
        e = side[gid]
        bad = check_entry(g, e)
        c = per_split.setdefault(e["split"], {"n": 0, "failed": 0})
        c["n"] += 1
        if bad:
            c["failed"] += 1
            failed.append((gid, bad))
    print(f"graphs checked: {len(ids)}; failed: {len(failed)}; per split: {per_split}")
    for gid, bad in failed[:10]:
        print(" ", gid, bad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
