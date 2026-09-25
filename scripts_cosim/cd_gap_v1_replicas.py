#!/usr/bin/env python3
"""cd_gap_v1 D2 follow-up -- HOW does an arm spread a batch's co-located members?

Per live run (raw result with taskResults and scaleEvents):
  * same-node batch pairs split into: same platform / sibling of the same device type / different type;
  * distinct (node, platform) pairs ever used, and per node the number of platforms used;
  * autoscaler activity: scale-up events and their count.

  cd_gap_v1_replicas.py RAW.json [RAW.json ...]
"""
from __future__ import annotations

import collections
import json
import sys


def analyse(path: str) -> dict:
    st = json.load(open(path))["stats"]
    tasks = st["taskResults"]
    ptype = {}
    for n in st["nodeResults"]:
        for p in n["platformResults"]:
            ptype[str(p["platformId"])] = p["platformType"]["shortName"]
    by_batch = collections.defaultdict(list)
    for t in tasks:
        by_batch[round(float(t["scheduledTime"]), 9)].append(t)
    same_plat = sib_same_type = diff_type = 0
    for members in by_batch.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a["executionNode"] != b["executionNode"]:
                    continue
                if a["executionPlatform"] == b["executionPlatform"]:
                    same_plat += 1
                elif ptype[str(a["executionPlatform"])] == ptype[str(b["executionPlatform"])]:
                    sib_same_type += 1
                else:
                    diff_type += 1
    used = collections.defaultdict(set)
    for t in tasks:
        used[t["executionNode"]].add(str(t["executionPlatform"]))
    tot = same_plat + sib_same_type + diff_type
    ups = [e for e in st.get("scaleEvents") or [] if e.get("action") == "up"]
    downs = [e for e in st.get("scaleEvents") or [] if e.get("action") == "down"]
    return {
        "file": path.rsplit("/", 1)[-1],
        "same_node_pairs": tot,
        "share_same_platform": same_plat / tot, "share_sibling_same_type": sib_same_type / tot,
        "share_other_type_same_node": diff_type / tot,
        "distinct_node_platforms_used": sum(len(v) for v in used.values()),
        "platforms_used_per_node": {k: len(v) for k, v in sorted(used.items())},
        "scale_up_events": len(ups), "scale_up_count": sum(int(e.get("count") or 0) for e in ups),
        "scale_down_events": len(downs),
        "elapsed": float(st["averageElapsedTime"]), "queue": float(st["averageQueueTime"]),
    }


def main(argv) -> int:
    for p in argv:
        print(json.dumps(analyse(p)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
