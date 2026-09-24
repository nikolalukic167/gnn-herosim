#!/usr/bin/env python3
"""cd_gap_v1 B -- build the lighter-rate input set for the burst load ladder (docs/lineages/cd_gap_v1.md).

Every event timestamp of each burst window workload is multiplied by --factor (groups stay dispatched
together: members share a timestamp, and a product of equal numbers is equal). Every policy time
constant scales by the same factor (the `drainable_regime_v1` hard-stop rule): the cells'
`scheduler.batch_timeout`. Checkpoints and the jb2 split are linked, not copied. The output keeps the
source file names so `fresh_topo_burst_v1_gate.py` runs unchanged against --out as its --inputs.

  cd_gap_v1_build_b.py --src <fresh inputs> --out <dir> --factor 2 --topologies 9119 9414 ...
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import sys


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=float, required=True)
    ap.add_argument("--topologies", type=int, nargs="+", required=True)
    a = ap.parse_args()
    if a.factor <= 0:
        raise SystemExit("FAIL LOUD: --factor must be > 0")
    for sub in ("cfg", "wl"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    for name in ("models", "joint_burst_v2_split.json"):
        dst = os.path.join(a.out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(os.path.join(a.src, name)), dst)
    for path in sorted(glob.glob(os.path.join(a.src, "wl", "*.json"))):
        wl = json.load(open(path))
        before = [float(e["timestamp"]) for e in wl["events"]]
        for e in wl["events"]:
            e["timestamp"] = float(e["timestamp"]) * a.factor
        wl["duration"] = float(wl["duration"]) * a.factor
        wl["cd_gap_v1_rate_scale"] = {"factor": a.factor, "source": os.path.basename(path), "source_sha256": _sha(path),
                                      "span_before_s": max(before) - min(before),
                                      "span_after_s": (max(before) - min(before)) * a.factor}
        groups = {}
        for e, t0 in zip(wl["events"], before):
            groups.setdefault(e.get("peer_group"), set()).add(e["timestamp"])
        spread = max(len(v) for v in groups.values())
        out = os.path.join(a.out, "wl", os.path.basename(path))
        json.dump(wl, open(out + ".partial", "w"))
        os.replace(out + ".partial", out)
        print(f"wl {os.path.basename(path)}: {len(before)} events, span {max(before)-min(before):.0f} -> "
              f"{(max(before)-min(before))*a.factor:.0f} s, max distinct timestamps per group {spread}")
    for t in a.topologies:
        src = os.path.join(a.src, "cfg", f"cc40s{t}.json")
        cfg = json.load(open(src))
        new = copy.deepcopy(cfg)
        old_to = float(new["scheduler"]["batch_timeout"])
        new["scheduler"]["batch_timeout"] = old_to * a.factor
        new.setdefault("cd_gap_v1_rate_scale", {})
        new["cd_gap_v1_rate_scale"] = {"factor": a.factor, "source_sha256": _sha(src),
                                       "batch_timeout_before": old_to, "batch_timeout_after": old_to * a.factor}
        json.dump(new, open(os.path.join(a.out, "cfg", f"cc40s{t}.json"), "w"), indent=1)
    print(f"cfg: {len(a.topologies)} cells, batch_timeout x{a.factor}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
