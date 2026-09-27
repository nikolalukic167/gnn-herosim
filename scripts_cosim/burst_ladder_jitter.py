#!/usr/bin/env python3
"""burst_ladder_v1 Amendment 1 -- a perturbed draw of a built ladder rung (docs/lineages/burst_ladder_v1.md).

Every peer group's shared timestamp moves LATER by u * min(cap, gap_frac * gap to the next group), u ~ U(0, 1)
from Random(seed). Groups stay dispatched together (one timestamp per group), the event order is unchanged
(a shift is below the gap to the next group, so the peer_exchange table's event indices stay valid), and the
same draw applies to every arm. The rung's cfg (with its rate scale) is copied and its split/models links are
kept, so fresh_topo_burst_v1_gate.py runs unchanged against --out.

  burst_ladder_jitter.py --src <inputs>/ladder/x15 --out <inputs>/ladder/x15d1 --seed 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from typing import List

CAP_S = 0.5
GAP_FRAC = 0.25


def jitter_timestamps(ts: List[float], groups: List[object], seed: int, cap: float = CAP_S,
                      gap_frac: float = GAP_FRAC) -> List[float]:
    """Per-group forward shift; returns the new per-event timestamps."""
    order: List[object] = []
    t_of = {}
    for t, g in zip(ts, groups):
        if g not in t_of:
            order.append(g)
            t_of[g] = t
        elif t_of[g] != t:
            raise ValueError(f"group {g!r} has more than one timestamp")
    for a, b in zip(order, order[1:]):
        if t_of[b] < t_of[a]:
            raise ValueError("groups are not in timestamp order")
    rng = random.Random(seed)
    shift = {}
    for i, g in enumerate(order):
        gap = (t_of[order[i + 1]] - t_of[g]) if i + 1 < len(order) else cap / gap_frac
        shift[g] = rng.random() * min(cap, gap_frac * gap)
    return [t + shift[g] for t, g in zip(ts, groups)]


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
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.src, ".built")):
        raise SystemExit(f"FAIL LOUD: {a.src} is not a built ladder rung")
    os.makedirs(os.path.join(a.out, "wl"), exist_ok=True)
    shutil.copytree(os.path.join(a.src, "cfg"), os.path.join(a.out, "cfg"), dirs_exist_ok=True)
    for name in ("models", "joint_burst_v2_split.json", "backlog_corpus_v1_split.json"):
        dst = os.path.join(a.out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(a.src, name)), dst)
    for fn in sorted(os.listdir(os.path.join(a.src, "wl"))):
        path = os.path.join(a.src, "wl", fn)
        wl = json.load(open(path))
        ev = wl["events"]
        ts = [float(e["timestamp"]) for e in ev]
        new = jitter_timestamps(ts, [e.get("peer_group") for e in ev], a.seed)
        if any(y < x for x, y in zip(new, new[1:])):
            raise SystemExit("FAIL LOUD: jitter reordered events")
        for e, t in zip(ev, new):
            e["timestamp"] = t
        shifts = [n - o for n, o in zip(new, ts)]
        wl["burst_ladder_jitter"] = {"seed": a.seed, "cap_s": CAP_S, "gap_frac": GAP_FRAC, "source_sha256": _sha(path),
                                     "mean_shift_s": sum(shifts) / len(shifts), "max_shift_s": max(shifts)}
        out = os.path.join(a.out, "wl", fn)
        json.dump(wl, open(out + ".partial", "w"))
        os.replace(out + ".partial", out)
        print(f"{fn}: seed {a.seed}, {len(ev)} events, mean shift {sum(shifts)/len(shifts):.3f} s, "
              f"max {max(shifts):.3f} s")
    open(os.path.join(a.out, ".built"), "w").close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
