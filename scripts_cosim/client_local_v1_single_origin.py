#!/usr/bin/env python3
"""client_local_v1 part B: give every peer group one origin client.

  client_local_v1_single_origin.py <src wl dir> <dst wl dir>

The grounded mint (grounded_workload_v1_mint.py) gives each task the client of the k-th base event, so 97 % of
multi-task groups span several clients (3.56 distinct clients per 4.09-task group at x2 g0). In the Alibaba trace a
group is one request (one trace id) from one caller. This rewrites only `node_name`: every task of a group takes the
client of the group's earliest task (ties by event order). Timestamps, types, demand, QoS and groups are unchanged;
the workload records what was moved under "client_local_v1_single_origin".
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict


def rewrite(wl: dict) -> dict:
    events = wl["events"]
    first: dict = {}
    for i, e in sorted(enumerate(events), key=lambda ie: (float(ie[1]["timestamp"]), ie[0])):
        first.setdefault(e["peer_group"], e["node_name"])
    clients_before = defaultdict(set)
    moved = 0
    for e in events:
        clients_before[e["peer_group"]].add(e["node_name"])
        origin = first[e["peer_group"]]
        if e["node_name"] != origin:
            e["node_name"] = origin
            moved += 1
    spanning = sum(1 for c in clients_before.values() if len(c) > 1)
    after = defaultdict(set)
    for e in events:
        after[e["peer_group"]].add(e["node_name"])
    if any(len(c) != 1 for c in after.values()):
        raise SystemExit("FAIL LOUD: a group still spans several clients after the rewrite")
    per_client = defaultdict(int)
    for e in events:
        per_client[e["node_name"]] += 1
    wl["client_local_v1_single_origin"] = {
        "rule": "every task of a peer group takes the client of the group's earliest task",
        "n_tasks": len(events), "n_groups": len(first), "tasks_moved": moved,
        "groups_spanning_clients_before": spanning, "clients_used": len(per_client),
        "max_tasks_per_client": max(per_client.values()), "min_tasks_per_client": min(per_client.values()),
    }
    return wl


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1:]
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        if not name.endswith(".json"):
            continue
        wl = rewrite(json.load(open(os.path.join(src, name))))
        tmp = os.path.join(dst, name + ".partial")
        with open(tmp, "w") as fh:
            json.dump(wl, fh)
        os.replace(tmp, os.path.join(dst, name))
        print(f"[single-origin] {name}: {json.dumps(wl['client_local_v1_single_origin'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
