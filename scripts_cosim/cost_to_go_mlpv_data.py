#!/usr/bin/env python3
"""cost_to_go_v1 rung (b): the MLP V's data. One record per state: per plan, the post-plan per-replica feature matrix, S, Q5.

  cost_to_go_mlpv_data.py build OUT.jsonl --features FEATS.jsonl --tops TOPS.jsonl [--h 5] [--lookback 60]

Per replica of the post-plan state (S7's post_commit_features rows, in-flight from the snapshot ghosts as cost_to_go_handv_inflight
computes it), in FEATURES order:
  own     this plan's committed service on the replica's platform (s)
  drain   the platform's queue drain (s)
  inflt   in-flight remaining from the ghosts, b1 = sum over the platform's tasks (s)  -- zeroed by the no-in-flight variant
  cold    the type's cold start on this platform (s), the quantity S charges
  idle    seconds since the replica was last used
  used    1 if the plan places a task on it
  scalein 1 if unused and idle + H reaches the KPA stable window
  need    pre-t0 arrival rate of its type x H / replicas of the type (60 s lookback; no event after t0 is read)
  type    one-hot over TYPES
Seconds go through log1p. S per plan follows the hand V's convention: the policy plan's S is the argmin-S of the tops row, slot sk's S
is top_s[k]. A state is kept only if every plan has features, S and Q.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_to_go_handv import type_rates  # noqa: E402
from cost_to_go_handv_inflight import add_inflight  # noqa: E402

TYPES = ("dnn1", "dnn2", "rf", "cnn")
FEATURES = ("own", "drain", "inflt", "cold", "idle", "used", "scalein", "need") + tuple(f"type_{t}" for t in TYPES)
LOG = {"own", "drain", "inflt", "cold", "idle", "need"}


def replica_matrix(f, rates, H):
    n_of = {t: len(ks) for t, ks in f["type_platforms"].items()}
    out = []
    for r in sorted(f["replicas"], key=lambda r: (r["type"], r["key"])):
        k, t = r["key"], r["type"]
        if t not in TYPES:
            raise ValueError(f"FAIL LOUD: unknown task type {t!r}")
        fly = float(f["inflight_b1"][k])
        own = float(f["load_after"][k]) - float(f["drain"][k]) - (float(f.get("inflight", {}).get(k, 0.0)) if f.get("pg_inflight") else 0.0)
        idle = max(0.0, float(r["idle_s"]))
        v = {"own": max(0.0, own), "drain": float(f["drain"][k]), "inflt": fly, "cold": float(r["cold_s"]), "idle": idle,
             "used": float(bool(r["used"])),
             "scalein": float((not r["used"]) and idle + H >= float(f["scale_in_after_s"])),
             "need": rates.get(t, 0.0) * H / max(1, n_of.get(t, 1))}
        for tt in TYPES:
            v[f"type_{tt}"] = float(t == tt)
        out.append([math.log1p(v[x]) if x in LOG else v[x] for x in FEATURES])
    return out


def build(a):
    tops = {}
    for l in open(a.tops):
        r = json.loads(l)
        if "error" not in r and r.get("argmin_s"):
            tops[r["ds"]] = r["argmin_s"]
    by = defaultdict(dict)
    for l in open(a.features):
        f = json.loads(l)
        by[f["ds"]][f["slot"]] = f
    ghost_cache, wl_cache = {}, {}
    kept = skipped = 0
    with open(a.out, "w") as fh:
        for ds, slots in sorted(by.items()):
            if ds not in tops or "policy" not in slots:
                skipped += 1
                continue
            fid = json.load(open(f"{ds}/infrastructure.json"))["live_snapshot_seed"]["fidelity_replay"]["snapshot"]["fidelity"]
            ghosts, queued = fid.get("ghosts") or [], [int(x["gid"]) for x in fid.get("queued") or []]
            pol = slots["policy"]
            if pol["workload"] not in wl_cache:
                wl_cache.clear()
                wl_cache[pol["workload"]] = json.load(open(pol["workload"]))["events"]
            rates = type_rates(wl_cache[pol["workload"]], float(pol["t0"]), a.lookback)
            top = tops[ds]
            plans = []
            n_s = 0
            for slot in sorted(slots, key=lambda s: (s != "policy", s)):
                if slot != "policy":
                    # the 5-plan set of the S0 reader (_plan_set): the policy plan and the first 4 S-ranked plans that differ from it
                    if slots[slot]["plan"] == pol["plan"] or n_s >= 4:
                        continue
                    n_s += 1
                f = add_inflight(slots[slot], ghosts, queued)
                if slot == "policy":
                    s = float(top["s"])
                else:
                    k = int(slot[1:])
                    if k >= len(top["top_s"]):
                        continue
                    s = float(top["top_s"][k]["s"])
                plans.append({"slot": slot, "s": s, "q": float(f["q"]), "x": replica_matrix(f, rates, a.h)})
            if len(plans) < 2:
                skipped += 1
                continue
            argv = json.load(open(os.path.join(ds, "generation_provenance.json")))["argv"]
            cell = os.path.basename(argv[argv.index("--snapshots") + 1]).replace("shard_", "").replace(".jsonl", "")
            fh.write(json.dumps({"ds": ds, "cell": cell, "topo": int(cell.split("_")[0][4:]), "plans": plans}) + "\n")
            kept += 1
    print(f"{kept} states written to {a.out}; skipped {skipped} (no tops row, no policy features, or < 2 plans)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out")
    b.add_argument("--features", required=True)
    b.add_argument("--tops", required=True)
    b.add_argument("--h", type=float, default=5.0)
    b.add_argument("--lookback", type=float, default=60.0)
    a = ap.parse_args()
    build(a)


if __name__ == "__main__":
    main()
