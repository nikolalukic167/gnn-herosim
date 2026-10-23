#!/usr/bin/env python3
"""Merge the shards of the union corpus' stored-flow relabel into the label-override JSON and report how the labels moved (live_link_features_v1).

  llf_label_merge.py --split SPLIT --corpus CORPUS --states STATES.json --in relabel_*_of_N.jsonl --out label_override.json --stats stats.json

Same entry shape as `uncut_label_screen.py merge` (trainer-read keys `plans`, `cd_rtt`), without the two test-only keys `stored_label_rtt` and `stored_flows`
(the pair is kept in the stats; the switch is named once in `_meta`). The stats compare, per size bucket: the final label with the first-pass label (`differs`), and
the first-pass and the final label with the plan the bar SERVED at that state (`chosen.jsonl`, compared on the tasks the live decision placed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import uncut_label_screen as L  # noqa: E402

DROP = ("gid", "wall_s", "stored_label_rtt", "stored_flows")


def served_map(snapfile: str, cache: Dict[str, Dict[int, Any]]) -> Dict[int, Any]:
    if snapfile not in cache:
        cache[snapfile] = {int(r["trigger_task_id"]): r for r in (json.loads(l) for l in open(snapfile + ".chosen.jsonl") if l.strip())}
    return cache[snapfile]


def differs_from_served(plan: List[List[int]], trace_ids: List[int], rec: Dict[str, Any]) -> bool:
    live = {int(t): [int(n), int(p)] for t, n, p in rec["chosen"]}
    return any(plan[i] != live[int(g)] for i, g in enumerate(trace_ids) if int(g) in live)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", required=True)
    a = ap.parse_args(argv)
    entries, pair, flagged = {}, {}, set()
    for f in a.inp:
        for line in open(f):
            e = json.loads(line)
            if e["gid"] in entries:
                raise SystemExit(f"FAIL LOUD: {e['gid']} appears in two shards")
            flagged.add(bool(e.get("stored_flows")))
            pair[e["gid"]] = e.get("stored_label_rtt")
            entries[e["gid"]] = {k: v for k, v in e.items() if k not in DROP}
    expect = {s["gid"] for s in L.all_states(a.split, a.corpus)}
    if set(entries) != expect:
        raise SystemExit(f"FAIL LOUD: {len(entries)} of {len(expect)} states; missing {sorted(expect - set(entries))[:5]}, extra {sorted(set(entries) - expect)[:5]}")
    if flagged != {True}:
        raise SystemExit("FAIL LOUD: not every shard row was relabelled with --stored-flows")
    body = {"_meta": {"version": L.RELABEL_VERSION, "stored_flows": True, "method": "exhaustive replay for plan spaces <= 600; else the 8 lowest-S plans plus the current label, replayed under option D with the "
                      "stored flows of the snapshot; the fastest replayed plan is the label; the current label's replayed RTT is kept in the stats (it was stored under uncontended physics)", "n": len(entries)},
            **dict(sorted(entries.items()))}
    raw = json.dumps(body, indent=0, sort_keys=False) + "\n"
    Path(a.out).write_text(raw)
    states = {s["key"]: s for s in json.load(open(a.states))["states"]}
    cache: Dict[str, Dict[int, Any]] = {}
    agg: Dict[str, Dict[str, Any]] = {}
    ratio = []
    for gid, e in entries.items():
        ds = L._dataset_dir(a.corpus, gid)
        head, _rows = L._stored_rows(ds)
        st = states[head["state_id"]]
        rec = served_map(st["snapfile"], cache).get(int(st["trigger"]))
        trace = [int(x) for x in head["trace_task_ids"]]
        b = agg.setdefault(e["bucket"], {"n": 0, "differs_from_first_pass": 0, "served_checked": 0, "first_pass_differs_from_served": 0, "final_differs_from_served": 0})
        b["n"] += 1
        b["differs_from_first_pass"] += int(e["differs"])
        if rec is not None and rec["chosen"]:
            b["served_checked"] += 1
            b["first_pass_differs_from_served"] += int(differs_from_served(e["current_plan"], trace, rec))
            b["final_differs_from_served"] += int(differs_from_served(e["plans"][0], trace, rec))
        if pair[gid] is not None and pair[gid] > 0:
            ratio.append(float(e["rtt_current"]) / float(pair[gid]))
    tot = {k: sum(b[k] for b in agg.values()) for k in ("n", "differs_from_first_pass", "served_checked", "first_pass_differs_from_served", "final_differs_from_served")}
    stats = {"n": len(entries), "sha256": hashlib.sha256(raw.encode()).hexdigest(), "md5": hashlib.md5(raw.encode()).hexdigest(), "by_bucket": agg, "total": tot,
             "share": {k: tot[k] / tot["served_checked"] if k.endswith("served") else tot[k] / tot["n"] for k in ("differs_from_first_pass", "first_pass_differs_from_served", "final_differs_from_served")}}
    if ratio:
        ratio.sort()
        stats["current_label_rtt_under_flows_over_stored"] = {"median": ratio[len(ratio) // 2], "p10": ratio[len(ratio) // 10], "p90": ratio[9 * len(ratio) // 10]}
    Path(a.stats).write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps({"n": stats["n"], "sha256": stats["sha256"], "share": stats["share"]}))


if __name__ == "__main__":
    main()
