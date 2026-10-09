#!/usr/bin/env python3
"""r1_attribution_v1 gate: two runs of the same cells from two commits must agree on every simulated field.

  r1a_gate_identity.py <dir before> <dir after>     compares the summaries both dirs hold (after may hold more); exit 1 on any difference
Skipped: the run's own bookkeeping (wall clock, commit, the measured inference time).
"""
import json
import sys
from pathlib import Path

SKIP = {"wallclock_s", "code", "total_rtt_plus_inference", "env"}
# env keys that differ by construction: the checkpoint directory of the run, and provenance keys recorded since the earlier commit
ENV_OK = {"GNN_MODEL_PATH", "GNN_CD_REFINE", "HEROSIM_CD_RANDOM_SEED"}


def compare(before: Path, after: Path) -> list:
    names = sorted(p.name for p in before.glob("*.summary.json") if (after / p.name).exists())
    problems = [] if names else ["no common cells"]
    for n in names:
        a, b = json.loads((before / n).read_text()), json.loads((after / n).read_text())
        problems += [f"{n}: {k}" for k in sorted(set(a) | set(b)) if k not in SKIP and a.get(k) != b.get(k)]
        problems += [f"{n}: env {k}" for k in sorted(set(a["env"]) | set(b["env"])) if k not in ENV_OK and a["env"].get(k) != b["env"].get(k)]
    print(f"{len(names)} cells compared; {len(problems)} differences")
    return problems


if __name__ == "__main__":
    probs = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    print("\n".join(probs))
    raise SystemExit(1 if probs else 0)
