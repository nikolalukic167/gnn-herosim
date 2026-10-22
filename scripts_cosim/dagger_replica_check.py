#!/usr/bin/env python3
"""Replica-set assertion for the own-state (DAgger) round (SCIENTIST ruling 2026-10-22): for every RETAINED state, and for each task type, the replay's replica set must contain every
replica of the captured served plan. The replay's replica set is what `build_live_snapshot_seed` seeds: the captured `replicas_by_type` plus the batch tasks' candidates, by
(node name, platform id). The served plan is the capture's `.chosen.jsonl` record ([task, node, platform] per placed task; deferred tasks are not in it). A state whose served plan names
a replica the replay lacks is a failure. Any failure exits non-zero and stops labelling.

  dagger_replica_check.py --states STATES.json [--labels LABELS.jsonl ...] [--out REPORT.json]    (labels: only states with a non-dropped label are checked; none given: every state)"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def snapshot_of(snapfile: str, trigger: int) -> Dict[str, Any]:
    """The FIRST capture led by `trigger` (the state, as `dagger_corpus.first_captures` defines it)."""
    for line in open(snapfile):
        if line.strip():
            rec = json.loads(line)
            if int(rec["trigger_task_id"]) == int(trigger):
                return rec
    raise SystemExit(f"FAIL LOUD: trigger {trigger} is not in {snapfile}")


def check_state(snap: Dict[str, Any], chosen: Dict[str, Any]) -> List[str]:
    """Problems of one state (an empty list is a pass). `chosen` is the .chosen.jsonl record of the state's trigger."""
    from src.placement.live_snapshot_seed import build_live_snapshot_seed

    seeded = build_live_snapshot_seed(snap)["replicas_by_type"]                  # type -> [spec dicts]
    replay = {t: {(str(x["node_name"]), int(x["platform_id"])) for x in specs} for t, specs in seeded.items()}
    payload = snap.get("replicas_by_type") or {}
    name_of = {int(e["node_id"]): str(e["node_name"]) for v in payload.values() for e in v if "node_id" in e}
    type_of = {int(t["task_id"]): str(t["task_type"]) for t in snap["tasks"]}
    problems: List[str] = []
    for tid, node, plat in chosen.get("chosen") or []:
        node_name = node if isinstance(node, str) else name_of.get(int(node))
        ty = type_of.get(int(tid))
        if ty is None:
            problems.append(f"task {tid}: not in the capture's tasks")
        elif node_name is None:
            problems.append(f"task {tid} ({ty}): served node {node} is not in the capture's replica payload")
        elif (node_name, int(plat)) not in replay.get(ty, set()):
            problems.append(f"task {tid} ({ty}): served replica ({node_name}, {plat}) is not in the replay's replica set")
    return problems


def main(argv: Sequence[str] = None) -> int:
    from src.placement import live_audit

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True)
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    states = json.load(open(a.states))["states"]
    keep = None
    if a.labels:
        keep = {r["state_id"] for f in a.labels for r in map(json.loads, filter(str.strip, open(f))) if r["source"] != "dropped"}
    cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
    checked, failed = 0, []
    for st in states:
        if keep is not None and st["key"] not in keep:
            continue
        cho = cache.setdefault(st["snapfile"], live_audit.load_chosen(st["snapfile"]))
        rec = cho.get(int(st["trigger"]))
        if rec is None:
            failed.append({"state": st["key"], "problems": ["no .chosen.jsonl record for the trigger"]})
            continue
        probs = check_state(snapshot_of(st["snapfile"], st["trigger"]), rec)
        checked += 1
        if probs:
            failed.append({"state": st["key"], "problems": probs[:5], "n_problems": len(probs)})
    res = {"states_file": a.states, "checked": checked, "failed": len(failed), "failures": failed[:20]}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
