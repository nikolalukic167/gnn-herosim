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

DROP = ("gid", "wall_s", "stored_label_rtt", "stored_flows", "winner_pool", "sfs_wall_s", "sfs_new_plans", "hand_replan_differs")


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
    ap.add_argument("--carry-first-pass", nargs="*", default=[], help="datasets whose relabel did not finish: their FIRST-PASS label is carried, flagged relabel_unfinished (named in the stats)")
    a = ap.parse_args(argv)
    entries, pair, flagged, pools, sfs_wall, hand_diff = {}, {}, set(), {}, {}, {}
    raw_rows: Dict[str, Dict[str, Any]] = {}
    for f in a.inp:
        for line in open(f):
            if not line.strip():
                continue
            e = json.loads(line)
            if e["gid"] in entries:
                # a state may be in two files when a task was re-split while the first task was still running; the rows are deterministic, so they must be equal but for the wall times
                same = {k: v for k, v in e.items() if k not in ("wall_s", "sfs_wall_s", "replay_setup_s")} == {k: v for k, v in raw_rows[e["gid"]].items() if k not in ("wall_s", "sfs_wall_s", "replay_setup_s")}
                if not same:
                    raise SystemExit(f"FAIL LOUD: {e['gid']} appears twice with different rows")
                continue
            raw_rows[e["gid"]] = e
            flagged.add(bool(e.get("stored_flows")))
            pair[e["gid"]] = e.get("stored_label_rtt")
            pools[e["gid"]] = e.get("winner_pool")
            hand_diff[e["gid"]] = e.get("hand_replan_differs")
            sfs_wall[e["gid"]] = e.get("sfs_wall_s")
            entries[e["gid"]] = {k: v for k, v in e.items() if k not in DROP}
    states_all = L.all_states(a.split, a.corpus)
    expect = {s["gid"] for s in states_all}
    carried = sorted(set(a.carry_first_pass or []))
    for gid in carried:
        if gid in entries:
            raise SystemExit(f"FAIL LOUD: {gid} has a relabel row and is listed in --carry-first-pass")
        st = next(s for s in states_all if s["gid"] == gid)
        head, rows = L._stored_rows(L._dataset_dir(a.corpus, gid))
        n = int(head["n_tasks"])
        plan, rtt = L._plan_from_row(rows[0], n), float(rows[0]["rtt"])
        entries[gid] = {"plans": [plan], "cd_rtt": rtt, "split": st["split"], "bucket": st["bucket"], "n_tasks": n, "method": "first_pass_carried", "current_label_source": head["label_source"],
                        "differs": False, "gain": 0.0, "rtt_current": rtt, "rtt_sim": rtt, "n_plans_replayed": 0, "current_plan": plan, "best_tag": "label", "relabel_unfinished": True}
        pair[gid], pools[gid], sfs_wall[gid], hand_diff[gid] = None, None, None, None
        flagged.add(True)
    if set(entries) != expect:
        raise SystemExit(f"FAIL LOUD: {len(entries)} of {len(expect)} states; missing {sorted(expect - set(entries))[:5]}, extra {sorted(set(entries) - expect)[:5]}")
    if flagged != {True}:
        raise SystemExit("FAIL LOUD: not every shard row was relabelled with --stored-flows")
    body = {"_meta": {"version": L.RELABEL_VERSION, "stored_flows": True, "method": "exhaustive replay for plan spaces <= 600; else the 8 lowest-S plans plus the current label, replayed under option D with the "
                      "stored flows of the snapshot; the fastest replayed plan is the label; the current label's replayed RTT is kept in the stats (it was stored under uncontended physics)", "n": len(entries), **({"relabel_unfinished": carried} if carried else {})},
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
        if hand_diff[gid] is not None and e["current_label_source"] == "hand":
            b["hand_states"] = b.get("hand_states", 0) + 1
            b["hand_replan_differs"] = b.get("hand_replan_differs", 0) + int(hand_diff[gid])
        if pools[gid] is not None:
            b.setdefault("winner_pool", {})
            b["winner_pool"][pools[gid]] = b["winner_pool"].get(pools[gid], 0) + 1
            b["sfs_wall_s"] = b.get("sfs_wall_s", 0.0) + float(sfs_wall[gid] or 0.0)
        if rec is not None and rec["chosen"]:
            b["served_checked"] += 1
            b["first_pass_differs_from_served"] += int(differs_from_served(e["current_plan"], trace, rec))
            b["final_differs_from_served"] += int(differs_from_served(e["plans"][0], trace, rec))
        if pair[gid] is not None and pair[gid] > 0:
            ratio.append(float(e["rtt_current"]) / float(pair[gid]))
    pool_tot: Dict[str, int] = {}
    for b in agg.values():
        for k, v in (b.get("winner_pool") or {}).items():
            pool_tot[k] = pool_tot.get(k, 0) + v
    tot = {k: sum(b[k] for b in agg.values()) for k in ("n", "differs_from_first_pass", "served_checked", "first_pass_differs_from_served", "final_differs_from_served")}
    hs = sum(b.get("hand_states", 0) for b in agg.values())
    stats = {"n": len(entries), "hand_replan_differs_share": (sum(b.get("hand_replan_differs", 0) for b in agg.values()) / hs) if hs else None, "sha256": hashlib.sha256(raw.encode()).hexdigest(), "md5": hashlib.md5(raw.encode()).hexdigest(), "by_bucket": agg, "total": tot,
             "winner_pool_total": pool_tot, "sfs_wall_s_total": sum(b.get("sfs_wall_s", 0.0) for b in agg.values()), "share": {k: tot[k] / tot["served_checked"] if k.endswith("served") else tot[k] / tot["n"] for k in ("differs_from_first_pass", "first_pass_differs_from_served", "final_differs_from_served")}}
    if ratio:
        ratio.sort()
        stats["current_label_rtt_under_flows_over_stored"] = {"median": ratio[len(ratio) // 2], "p10": ratio[len(ratio) // 10], "p90": ratio[9 * len(ratio) // 10]}
    Path(a.stats).write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps({"n": stats["n"], "sha256": stats["sha256"], "share": stats["share"]}))


if __name__ == "__main__":
    main()
