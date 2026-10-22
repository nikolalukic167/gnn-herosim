#!/usr/bin/env python3
"""Assemble the fair-share label override for the Arm 1 corpus: the filed sim_labels_arm1.json with the TRAIN entries replaced by the fair-share relabel (same keys per entry, same `_meta` plus a fair-share note);
val and test entries are copied as filed. Also: the identity of the 500-state test's fair-share rows against the full run, and the change statistics.

  fs_label_full.py build --labels sim_labels_arm1.json --rows fs_*_of_20.jsonl --test-rows fs_*_of_4.jsonl --out label_override_fs.json --stats stats.json"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

DROP = ("gid", "wall_s", "stored_label_rtt")


def _rows(files):
    out = {}
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r["gid"] in out:
                    raise SystemExit(f"FAIL LOUD: {r['gid']} appears twice")
                out[r["gid"]] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--test-rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()
    filed = json.load(open(a.labels))
    rows = _rows(a.rows)
    train = sorted(g for g, e in filed.items() if g != "_meta" and e["split"] == "train")
    if set(rows) != set(train):
        raise SystemExit(f"FAIL LOUD: the relabel covers {len(rows)} states, the filed train entries are {len(train)}; missing {sorted(set(train) - set(rows))[:5]}, extra {sorted(set(rows) - set(train))[:5]}")
    out = {"_meta": {**filed["_meta"], "fair_share": "train entries relabelled under HEROSIM_EXCHANGE_FAIRSHARE=1 (serial partners) + HEROSIM_EXCHANGE_TCP=1, same candidate set and method; val/test entries as filed",
                     "filed_labels_sha256": hashlib.sha256(open(a.labels, "rb").read()).hexdigest()}}
    changed, big = defaultdict(lambda: [0, 0, 0]), 0
    for g in sorted(k for k in filed if k != "_meta"):
        old = filed[g]
        if old["split"] != "train":
            out[g] = old
            continue
        r = rows[g]
        new = {k: v for k, v in r.items() if k not in DROP}
        if set(new) != set(old):
            raise SystemExit(f"FAIL LOUD: {g}: keys {sorted(set(new) ^ set(old))} differ from the filed entry's")
        if new["split"] != old["split"] or new["bucket"] != old["bucket"] or new["n_tasks"] != old["n_tasks"] or new["current_plan"] != old["plans"][0]:
            raise SystemExit(f"FAIL LOUD: {g}: the relabelled row is not the filed state (split/bucket/n_tasks/current plan)")
        out[g] = new
        c = changed[old["bucket"]]
        c[0] += 1
        c[1] += int(new["differs"])
        c[2] += int(new["differs"] and (1.0 - new["rtt_sim"] / new["rtt_current"]) >= 0.05)
    tot = [sum(v[i] for v in changed.values()) for i in range(3)]
    stats = {"states_relabelled": tot[0], "changed": tot[1], "share_changed": tot[1] / tot[0], "gap_ge_5pct": tot[2], "share_gap_ge_5pct": tot[2] / tot[0],
             "by_bucket": {b: {"n": v[0], "share_changed": v[1] / v[0], "share_gap_ge_5pct": v[2] / v[0]} for b, v in sorted(changed.items())}}
    test = _rows(a.test_rows)
    ident = [g for g in test if g in rows and {k: v for k, v in test[g].items() if k != "wall_s"} == {k: v for k, v in rows[g].items() if k != "wall_s"}]
    stats["test500_identical_to_full_run"] = f"{len(ident)} of {len(test)}"
    stats["test500_not_identical"] = [g for g in test if g not in set(ident)][:10]
    raw = json.dumps(out, indent=0, sort_keys=False) + "\n"
    Path(a.out).write_text(raw)
    stats["override_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    stats["override_md5"] = hashlib.md5(raw.encode()).hexdigest()
    Path(a.stats).write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
