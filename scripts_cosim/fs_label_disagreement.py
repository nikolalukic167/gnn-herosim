#!/usr/bin/env python3
"""Fair-share label-disagreement test (direction only, nothing is trained on it). If the Arm 1 training labels had been made under fair-share exchange physics, how often would the best plan change?

  sample : a stratified, seeded sample of Arm 1 states -> a split file for `uncut_label_screen.py relabel` and the sample list. The seed and the allocation are fixed here, before any result.
  read   : the fair-share-off relabel (must reproduce the filed label) against the fair-share-on relabel (HEROSIM_EXCHANGE_FAIRSHARE=1, HEROSIM_EXCHANGE_TCP=1, --record-pair)."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20261022
PER_BUCKET = 125
BUCKETS = ("le10", "11_50", "51_100", "101_256")


def cmd_sample(a):
    labels = json.load(open(a.labels))
    base = json.load(open(a.base_split))
    by = defaultdict(list)
    for gid, e in labels.items():
        if gid != "_meta" and e["split"] == "train":
            by[e["bucket"]].append(gid)
    rng = random.Random(SEED)
    pick = []
    for b in BUCKETS:
        pool = sorted(by[b])
        pick += sorted(rng.sample(pool, min(PER_BUCKET, len(pool))))
    out = dict(base)
    out.update(train=pick, val=[], test=[], n_parents=len(pick), fs_label_test=True)
    Path(a.out_split).write_text(json.dumps(out))
    lst = {"seed": SEED, "per_bucket": PER_BUCKET, "pool": {b: len(by[b]) for b in BUCKETS}, "n": len(pick), "gids": pick,
           "sha256": hashlib.sha256("\n".join(pick).encode()).hexdigest()}
    Path(a.out_list).write_text(json.dumps(lst, indent=1))
    print(json.dumps({k: v for k, v in lst.items() if k != "gids"}))


def _rows(files):
    out = {}
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                out[r["gid"]] = r
    return out


def cmd_read(a):
    filed = json.load(open(a.labels))
    off, on = _rows(a.off), _rows(a.on)
    sample = json.load(open(a.sample_list))["gids"]
    if set(off) != set(sample) or set(on) != set(sample):
        raise SystemExit(f"FAIL LOUD: off has {len(off)}, on has {len(on)} rows, the sample {len(sample)}")
    same_off = [g for g in sample if off[g]["plans"] == filed[g]["plans"] and abs(off[g]["cd_rtt"] - filed[g]["cd_rtt"]) <= 1e-9 * abs(filed[g]["cd_rtt"])]
    res = {"n": len(sample), "off_reproduces_filed": len(same_off), "off_not_reproduced": [g for g in sample if g not in set(same_off)][:10]}
    per = defaultdict(lambda: {"n": 0, "changed": 0, "gap": [], "task_changed": 0, "task_total": 0, "task_changed_in_changed": 0, "task_total_in_changed": 0, "inflation": []})
    for g in sample:
        e = on[g]
        for key in ("ALL", e["bucket"], "exhaustive (change rate exact)" if e["method"] == "exhaustive" else "sampled candidates (change rate is a lower bound)"):
            p = per[key]
            filed_plan, new_plan = filed[g]["plans"][0], e["plans"][0]
            moved = sum(1 for x, y in zip(filed_plan, new_plan) if x[0] != y[0])
            ch = bool(e["differs"])
            p["n"] += 1
            p["changed"] += ch
            p["task_changed"] += moved
            p["task_total"] += len(filed_plan)
            p["inflation"].append(e["rtt_current"] / e["stored_label_rtt"] - 1.0)
            if ch:
                p["gap"].append(1.0 - e["rtt_sim"] / e["rtt_current"])
                p["task_changed_in_changed"] += moved
                p["task_total_in_changed"] += len(filed_plan)
    def med(x):
        x = sorted(x)
        return x[len(x) // 2] if x else None
    res["by_bucket"] = {k: {"n": p["n"], "changed": p["changed"], "share_changed": p["changed"] / p["n"],
                            "gap_under_fairshare_median": med(p["gap"]), "gap_mean": (sum(p["gap"]) / len(p["gap"])) if p["gap"] else None,
                            "tasks_node_changed_share_all": p["task_changed"] / p["task_total"],
                            "tasks_node_changed_share_in_changed_states": (p["task_changed_in_changed"] / p["task_total_in_changed"]) if p["task_total_in_changed"] else None,
                            "filed_label_rtt_inflation_under_fairshare_median": med(p["inflation"])} for k, p in per.items()}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--labels", required=True)
    s.add_argument("--base-split", required=True)
    s.add_argument("--out-split", required=True)
    s.add_argument("--out-list", required=True)
    r = sub.add_parser("read")
    r.add_argument("--labels", required=True)
    r.add_argument("--sample-list", required=True)
    r.add_argument("--off", nargs="+", required=True)
    r.add_argument("--on", nargs="+", required=True)
    r.add_argument("--out", default="")
    a = ap.parse_args()
    {"sample": cmd_sample, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    main()
