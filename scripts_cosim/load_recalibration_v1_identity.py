#!/usr/bin/env python3
"""load_recalibration_v1: instrumentation identity check. Compares the gate summaries of the same cells run from two
worktrees (before / after the placement_wait and arrival_end fields): every key but the new instrumentation fields and the run's own
bookkeeping must be equal, nested values included.

  load_recalibration_v1_identity.py <dir before> <dir after>        exit 1 on any difference
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NEW = {"placement_wait", "arrival_end", "backlog_profile", "lock_wait", "effective_queue_share", "backlog_profile_v2"}  # fields a later instrumentation adds
# total_rtt_plus_inference adds the measured inference wall time (not simulated; total_rtt is compared)
BOOKKEEPING = {"wallclock_s", "code", "total_rtt_plus_inference"}


def compare(before: Path, after: Path) -> list:
    names = sorted(p.name for p in before.glob("*.summary.json"))
    problems = []
    if names != sorted(p.name for p in after.glob("*.summary.json")) or not names:
        return [f"cell sets differ or empty: {names} vs {sorted(p.name for p in after.glob('*.summary.json'))}"]
    for n in names:
        a, b = json.loads((before / n).read_text()), json.loads((after / n).read_text())
        if not (set(b) - set(a)) or not (set(b) - set(a)) <= NEW:
            problems.append(f"{n}: new keys {sorted(set(b) - set(a))}, expected a non-empty subset of {sorted(NEW)}")
        problems += [f"{n}: {k} differs" for k in sorted(set(a) - BOOKKEEPING) if a[k] != b.get(k)]
    print(f"{len(names)} cells compared; {len(problems)} differences")
    return problems


if __name__ == "__main__":
    probs = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    for p in probs:
        print(p)
    raise SystemExit(1 if probs else 0)
