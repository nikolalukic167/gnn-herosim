#!/usr/bin/env python3
"""cd_gap_v1 D2 -- in-batch platform stacking in LIVE runs (docs/lineages/cd_gap_v1.md).

A peer group arrives as one burst and is decided as one batch; its members share a `scheduledTime`.
Per batch, over pairs of members placed on the SAME node: how many share the same platform (one FIFO
queue, served one after the other) rather than sibling platforms of that node (served in parallel,
still co-located, zero extra exchange). Also the queue time of members that were 2nd+ on their
platform within the batch vs the rest.

  cd_gap_v1_stacking.py RAW.json [RAW.json ...]
"""
from __future__ import annotations

import collections
import json
import sys
from statistics import mean, median


def analyse(path: str) -> dict:
    st = json.load(open(path))["stats"]
    tasks = st["taskResults"]
    if not tasks:
        raise SystemExit(f"FAIL LOUD: {path} has no taskResults")
    by_batch = collections.defaultdict(list)
    for t in tasks:
        by_batch[round(float(t["scheduledTime"]), 9)].append(t)
    same_node = same_plat = 0
    sizes, max_on = [], []
    q_first, q_stacked = [], []
    for members in by_batch.values():
        sizes.append(len(members))
        per_node = collections.Counter(m["executionNode"] for m in members)
        per_plat = collections.defaultdict(list)
        for m in members:
            per_plat[(m["executionNode"], m["executionPlatform"])].append(m)
        same_node += sum(c * (c - 1) // 2 for c in per_node.values())
        same_plat += sum(len(v) * (len(v) - 1) // 2 for v in per_plat.values())
        max_on.append(max(len(v) for v in per_plat.values()))
        for v in per_plat.values():
            v = sorted(v, key=lambda m: float(m["startedTime"]))
            q_first.append(float(v[0]["queueTime"]))
            q_stacked.extend(float(m["queueTime"]) for m in v[1:])
    return {
        "file": path.rsplit("/", 1)[-1],
        "batches": len(by_batch), "median_batch_size": median(sizes),
        "same_node_pairs_per_batch": same_node / len(by_batch),
        "same_platform_share_of_same_node_pairs": same_plat / same_node if same_node else 0.0,
        "mean_max_on_one_platform": mean(max_on),
        "queue_first_on_platform": mean(q_first) if q_first else 0.0,
        "queue_stacked_behind_batchmate": mean(q_stacked) if q_stacked else 0.0,
        "stacked_task_share": len(q_stacked) / len(tasks),
        "elapsed": float(st["averageElapsedTime"]), "queue": float(st["averageQueueTime"]),
    }


def main(argv) -> int:
    for p in argv:
        r = analyse(p)
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
