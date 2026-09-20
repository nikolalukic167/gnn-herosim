#!/usr/bin/env python3
"""joint_burst_v2 gate helper: copy a reused v1-screen summary into the v2 result dir, tagged
with the v2 arm name. Called as: joint_burst_v2_reuse.py <src.summary.json> <dst.summary.json> <arm_name>
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    src, dst, arm_name = sys.argv[1:4]
    d = json.load(open(src))
    d["arm"] = arm_name
    d["arm_kind"] = "peer_greedy_network_batch"
    d["reused_from"] = src
    json.dump(d, open(dst, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
