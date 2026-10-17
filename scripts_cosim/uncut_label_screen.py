#!/usr/bin/env python3
"""Offline screen of simulator-checked labels against the current formula labels (docs/lineages/uncut_groups_s0_v1.md, Record 2026-10-17
"Simulator-label screen: REGISTERED"). No training. MYOPIC-REPLAY, INTERIM-DESCRIPTIVE.

  uncut_label_screen.py draw --split experiments/uncut_arm1_split.json --corpus /share/nikola.lukic/uncut_arm1/corpus --out DRAW.json

`draw` reads only dataset headers (n_tasks, plan_space_size, label_source, state_id); it reads no replayed RTT and no label outcome.
It takes 125 states per group-size bucket (<=10, 11-50, 51-100, 101-256 tasks) from the TRAIN key of the Arm 1 split, seed 20261017, or all of a bucket
that has fewer, and fails loud if any drawn state sits on a gate topology (19537-19548), a TE3 id (19554-19589) or an accel_replica_v1 test id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

SEED = 20261017
PER_BUCKET = 125
BUCKETS = (("le10", 1, 10), ("11_50", 11, 50), ("51_100", 51, 100), ("101_256", 101, 256))
FORBIDDEN_TOPOLOGIES = set(range(19537, 19549)) | set(range(19554, 19590))
SMALL_PLAN_SPACE = 600


def bucket_of(n: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    raise ValueError(f"FAIL LOUD: group of {n} tasks is outside the registered buckets 1-256")


def topology_of(state_id: str) -> int:
    m = re.search(r"cc40s(\d+)__", state_id)
    if not m:
        raise ValueError(f"FAIL LOUD: cannot read a topology from state_id {state_id!r}")
    return int(m.group(1))


def draw(frame: Dict[str, Dict[str, Any]], accel_test_ids: Sequence[str] = (), seed: int = SEED, per_bucket: int = PER_BUCKET) -> Dict[str, Any]:
    """frame: dataset id -> {"n_tasks", "state_id", "plan_space_size", "label_source"} for the TRAIN key only. Deterministic in (frame, seed)."""
    rng = random.Random(seed)
    by_bucket: Dict[str, List[str]] = {name: [] for name, _, _ in BUCKETS}
    for gid in sorted(frame):
        by_bucket[bucket_of(int(frame[gid]["n_tasks"]))].append(gid)
    picked: List[str] = []
    counts: Dict[str, Dict[str, int]] = {}
    for name, _, _ in BUCKETS:
        pool = by_bucket[name]
        take = sorted(rng.sample(pool, per_bucket)) if len(pool) > per_bucket else list(pool)
        counts[name] = {"frame": len(pool), "drawn": len(take)}
        picked += take
    bad = [g for g in picked if topology_of(frame[g]["state_id"]) in FORBIDDEN_TOPOLOGIES]
    if bad:
        raise SystemExit(f"FAIL LOUD: drawn states on forbidden topologies (gate 19537-19548 or TE3 19554-19589): {bad[:5]}")
    clash = sorted(set(picked) & set(accel_test_ids))
    if clash:
        raise SystemExit(f"FAIL LOUD: drawn states that are accel_replica_v1 test ids: {clash[:5]}")
    states = [{"gid": g, "bucket": bucket_of(int(frame[g]["n_tasks"])), "n_tasks": int(frame[g]["n_tasks"]), "topology": topology_of(frame[g]["state_id"]),
               "plan_space_size": frame[g]["plan_space_size"], "label_source": frame[g]["label_source"], "state_id": frame[g]["state_id"],
               "small": frame[g]["plan_space_size"] is not None and int(frame[g]["plan_space_size"]) <= SMALL_PLAN_SPACE} for g in picked]
    sha = hashlib.sha256("\n".join(picked).encode()).hexdigest()
    return {"seed": seed, "per_bucket": per_bucket, "counts": counts, "n": len(states), "state_list_sha256": sha, "states": states}


def load_frame(split_path: str, corpus: str) -> Dict[str, Dict[str, Any]]:
    train = json.load(open(split_path))["train"]
    frame = {}
    for gid in train:
        kind, ds = gid.split("/")
        sub = kind.replace("gnn_datasets_wf1_", "")
        h = json.loads(open(f"{corpus}/{sub}/{ds}/placements/placements.jsonl").readline())
        if not h.get("header"):
            raise SystemExit(f"FAIL LOUD: {gid}: first line of placements.jsonl is not the header")
        frame[gid] = {k: h.get(k) for k in ("n_tasks", "plan_space_size", "label_source", "state_id")}
    return frame


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--split", required=True)
    d.add_argument("--corpus", required=True)
    d.add_argument("--accel-split", default=str(Path(__file__).resolve().parents[1] / "experiments" / "accel_replica_v1_split.json"))
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    accel = json.load(open(a.accel_split))["test"]
    res = draw(load_frame(a.split, a.corpus), accel_test_ids=accel)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("seed", "n", "counts", "state_list_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
