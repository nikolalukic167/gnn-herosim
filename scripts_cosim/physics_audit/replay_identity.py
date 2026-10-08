#!/usr/bin/env python3
"""physics_audit_v1 -- do two result JSONs agree on everything the simulation decides?

Wall-clock fields (decision and scheduling time measured with a host timer, see I10) differ between any two
runs of the same seed and are masked; everything else must be equal. With --strict nothing is masked.

  replay_identity.py A.json B.json [--strict] [--ignore-code] [--show 20]

--ignore-code also masks run_provenance.code (git describe, dirty flag, diff hash): use it only to compare a run
of edited code against a run of the code before the edit, where that block must differ.

Exit 0 when equal, 1 when not (the differing paths are printed).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List

# host-timer fields: measured outside simulated time, never decided by the simulation
WALL_CLOCK_KEYS = frozenset({
    "gnn_decision_time", "total_inference_time", "averageGNNDecisionTime", "total_rtt_plus_inference",
    "schedulingTime", "wall_seconds", "wall_clock_s", "average_inference_time_s",
    "hypothetical_total_with_inference_s", "inference_fraction_of_combined", "total_inference_time_s",
})
# lists serialised from Python sets of objects (SystemState.result): their order follows object hashes, which
# differ between processes; compared as multisets
SET_LIST_PARENTS = frozenset({"available_resources", "replicas"})


def diff_paths(a: Any, b: Any, path: str, masked: frozenset, out: List[str], limit: int,
               strict_order: bool = False) -> None:
    if len(out) >= limit:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in masked:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'B' if k not in a else 'A'}")
                continue
            diff_paths(a[k], b[k], f"{path}/{k}", masked, out, limit, strict_order)
    elif isinstance(a, list) and isinstance(b, list):
        if not strict_order and path.rsplit("/", 2)[-2:-1] and path.rsplit("/", 2)[-2] in SET_LIST_PARENTS:
            a, b = sorted(a, key=json.dumps), sorted(b, key=json.dumps)
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{path}[{i}]", masked, out, limit, strict_order)
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and a != a and b != b):
        out.append(f"{path}: {a!r} != {b!r}")


def compare(a_path: str, b_path: str, strict: bool = False, limit: int = 50,
            ignore_code: bool = False) -> List[str]:
    with open(a_path) as f:
        a = json.load(f)
    with open(b_path) as f:
        b = json.load(f)
    if ignore_code:
        for d in (a, b):
            (d.get("run_provenance") or {}).pop("code", None)
    out: List[str] = []
    diff_paths(a, b, "", frozenset() if strict else WALL_CLOCK_KEYS, out, limit, strict)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--ignore-code", action="store_true")
    ap.add_argument("--show", type=int, default=20)
    args = ap.parse_args()
    diffs = compare(args.a, args.b, args.strict, args.show, args.ignore_code)
    if not diffs:
        print("IDENTICAL" + ("" if args.strict else " (wall-clock fields masked)"))
        return 0
    print(f"DIFFERENT ({len(diffs)}{'+' if len(diffs) >= args.show else ''} paths)")
    for d in diffs:
        print("  " + d)
    return 1


if __name__ == "__main__":
    sys.exit(main())
