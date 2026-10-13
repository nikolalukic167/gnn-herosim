#!/usr/bin/env python3
"""uncut_groups_s0_v1 K1(b): draw the sealed sample of 200 batch states before any k = 1 hand cell runs
(docs/lineages/uncut_groups_s0_v1.md, kill rules).

A state is (topology, window, rung, trigger task id): the first-dispatched member of one peer group of more than 10 tasks.
The snapshot hook captures that group's batch in every hand cell of the topology x window x rung (R1A_SNAPSHOT_PLAN), so
the deciding arm, picked after the cells run, has the same states. Per window x rung: N states, split over the size buckets
11-50 / 51-100 / 101+ in proportion to the window's task-weighted shares (largest remainder), each a seeded draw without replacement of a
(group, topology) pair in its bucket. Workloads are the gate's own (identical across topologies within a window and rung).

  uncut_groups_k1_draw.py --inputs <gate>/inputs_group --topos 19461 ... 19472 --seed 20261013 --out-states S.json --out-plan P.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys

BUCKETS = ((11, 50, "11_50"), (51, 100, "51_100"), (101, None, "gt100"))
WINDOWS = ("g0", "g1", "g2", "g3")
RUNGS = ("moderate", "heavy")


def bucket_of(size: int):
    return next((tag for lo, hi, tag in BUCKETS if size >= lo and (hi is None or size <= hi)), None)


def groups_of(wl: dict):
    """(trigger, size) per peer group: trigger = the group's first event index, which is its first task id."""
    first, size = {}, collections.Counter()
    for i, e in enumerate(wl["events"]):
        g = e["peer_group"]
        size[g] += 1
        first.setdefault(g, i)
    return [(first[g], size[g]) for g in sorted(first, key=lambda g: first[g])]


def allocate(n: int, weights: dict) -> dict:
    tot = sum(weights.values())
    raw = {k: n * w / tot for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: n - sum(out.values())]:
        out[k] += 1
    return out


def draw(workloads: dict, topos: list, seed: int, per_cell: int = 25):
    """workloads[(window, rung)] -> workload dict. Returns (states, plan)."""
    rng = random.Random(seed)
    states, plan = [], collections.defaultdict(list)
    for w in WINDOWS:
        for r in RUNGS:
            gs = [(t, s, bucket_of(s)) for t, s in groups_of(workloads[(w, r)]) if s > 10]
            weights = collections.Counter()
            for _t, s, b in gs:
                weights[b] += s
            alloc = allocate(per_cell, weights)
            for b in sorted(alloc):
                # a state is (group, topology): a big group may recur on another topology, never twice on one
                pool = [(t, s, topo) for t, s, bb in gs if bb == b for topo in topos]
                if alloc[b] > len(pool):
                    raise SystemExit(f"FAIL LOUD: {w} {r} bucket {b}: {alloc[b]} states wanted from {len(pool)} (group, topology) pairs")
                for t, s, topo in rng.sample(pool, alloc[b]):
                    sid = f"{r}_{w}_{len([x for x in states if x['rung'] == r and x['window'] == w]):02d}"
                    states.append({"state_id": sid, "topology": topo, "window": w, "rung": r, "trigger_task_id": t, "group_size": s, "bucket": b})
                    plan[f"{topo}__{w}{r}"].append(t)
    return states, {k: sorted(v) for k, v in sorted(plan.items())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--topos", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--per-cell", type=int, default=25)
    ap.add_argument("--out-states", required=True)
    ap.add_argument("--out-plan", required=True)
    a = ap.parse_args(argv)
    for p in (a.out_states, a.out_plan):
        if os.path.exists(p):
            raise SystemExit(f"FAIL LOUD: {p} exists; the sealed draw is made once")
    wls = {(w, r): json.load(open(os.path.join(a.inputs, f"wf1_{r}", "wl", f"grounded_{w}_n50000.json"))) for w in WINDOWS for r in RUNGS}
    states, plan = draw(wls, sorted(a.topos), a.seed, a.per_cell)
    for p, obj in ((a.out_states, states), (a.out_plan, plan)):
        with open(p, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
        print(p, hashlib.sha256(open(p, "rb").read()).hexdigest())
    print(json.dumps({"states": len(states), "by_bucket": collections.Counter(s["bucket"] for s in states)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
