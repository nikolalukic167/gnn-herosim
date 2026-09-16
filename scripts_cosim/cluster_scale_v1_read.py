"""cluster_scale_v1 -- the registered read. Every bar below is a module constant.

Registered in docs/lineages/cluster_scale_v1.md before any arm ran. S0.a, S0.c and S0.d were
read by other tools at registration time (the cell minter carries S0_CORPUS_CANDIDATE_MAX);
this module owns S0.b, the primary, and restates the others so the bars have one home.

S0.b asks whether peer-group assembly cost falls when arrivals speed up. Amendment 1 (2026-09-16)
decoupled it from the capacity rungs: collection happens strictly before any decode and never
consults the model (src/policy/gnn/scheduler.py:443), so it is measured at 6 servers by varying
only the workload. The latency columns of those runs are meaningless -- the rungs are overloaded
by disclosure -- and this module refuses to report them.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

# --- registered bars -------------------------------------------------------------------
S0_COLLECTION_MAX_S = 2.0        # S0.b primary: median collection at the fastest readable rung
S0_RHO_BAND = (0.33, 3.0)        # S0.a: reactive mean queue vs the baseline rung's
S0_MAX_WALLCLOCK_MIN = 45        # S0.c: median wall-clock per arm at the fastest readable rung
S0_CORPUS_CANDIDATE_MAX = 5.0    # S0.d: max candidates per task in the arms' training corpus
S0_MIN_CELLS = 3                 # a rung read on fewer cells than this is not read at all
S0_RECONSTRUCTION_TOL = 0.01     # the decomposition must reconstruct averageWaitTime

# The rung ladder, fastest last. Arrival rates are the workloads' own, measured at registration.
S0B_RUNGS: Sequence[tuple] = (
    ("f4000", 0.460),
    ("f1000", 1.842),
    ("f300", 6.139),
)

VERDICT_SCALES = "ASSEMBLY-IS-ARRIVAL-BOUND"
VERDICT_DOES_NOT = "ASSEMBLY-DOES-NOT-SCALE"
VERDICT_UNREADABLE = "RUNG-UNREADABLE"


def median(xs: Sequence[float]) -> Optional[float]:
    v = sorted(float(x) for x in xs)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def load_arms(summary_dir: str) -> List[dict]:
    """Every S0.b summary in a directory, keyed by the rung tag its arm name carries.

    Job 769390 wrote all three rungs to one path because the arm name had no rung tag; the
    arm name is the key here precisely so that cannot recur silently.
    """
    arms = []
    for path in sorted(glob.glob(os.path.join(summary_dir, "*.summary.json"))):
        doc = json.load(open(path))
        arm = doc.get("arm") or ""
        parts = arm.split("__")
        if len(parts) != 3:
            raise ValueError(
                f"FAIL LOUD: {path} carries arm {arm!r}, which has no rung tag. A summary "
                f"without its rung cannot be assigned to one -- see job 769390.")
        doc["_rung"] = parts[2]
        doc["_seed"] = parts[0]
        arms.append(doc)
    return arms


def rung_summary(arms: Sequence[dict], rung: str) -> dict:
    """The per-rung statistic S0.b reads: median collection over that rung's cells."""
    rows = [a for a in arms if a.get("_rung") == rung]
    out = {"rung": rung, "n_cells": len(rows), "cells": sorted(a["_seed"] for a in rows)}
    if not rows:
        return {**out, "readable": False, "reason": "no cells"}

    for a in rows:
        r = a.get("residence_summary")
        if not r:
            raise ValueError(f"FAIL LOUD: {a.get('arm')} has no residence_summary")
        err = r.get("reconstruction_error")
        if err is None or err > S0_RECONSTRUCTION_TOL:
            raise ValueError(
                f"FAIL LOUD: {a.get('arm')} decomposition does not reconcile ({err})")

    out["collection_s"] = median([a["residence_summary"]["collection_s"] for a in rows])
    out["head_of_line_s"] = median([a["residence_summary"]["head_of_line_s"] for a in rows])
    out["placement_s"] = median([a["residence_summary"]["placement_s"] for a in rows])
    out["wait_s"] = median([a["residence_summary"]["average_wait_time_s"] for a in rows])
    out["mean_batch_size"] = median([a["residence_summary"]["mean_batch_size"] for a in rows])
    out["n_batches"] = median([a["residence_summary"]["n_batches"] for a in rows])
    out["wallclock_min"] = median([a["wallclock_s"] / 60.0 for a in rows])
    out["readable"] = len(rows) >= S0_MIN_CELLS
    if not out["readable"]:
        out["reason"] = f"{len(rows)} cells < S0_MIN_CELLS={S0_MIN_CELLS}"
    return out


def read_s0b(summary_dir: str, rungs: Sequence[tuple] = S0B_RUNGS) -> dict:
    """S0.b: monotone decline in median collection, reaching S0_COLLECTION_MAX_S at the fastest."""
    arms = load_arms(summary_dir)
    per_rung = [rung_summary(arms, tag) for tag, _ in rungs]
    for row, (_, rate) in zip(per_rung, rungs):
        row["arrivals_per_s"] = rate

    readable = [r for r in per_rung if r.get("readable")]
    result = {"rungs": per_rung, "n_readable": len(readable),
              "collection_max_s": S0_COLLECTION_MAX_S}

    if len(readable) < 2:
        return {**result, "verdict": VERDICT_UNREADABLE,
                "reason": f"{len(readable)} readable rungs, need >= 2 to see a trend"}

    cols = [r["collection_s"] for r in readable]
    monotone = all(b < a for a, b in zip(cols, cols[1:]))
    fastest = readable[-1]
    meets_bar = fastest["collection_s"] <= S0_COLLECTION_MAX_S

    result.update(
        monotone=monotone,
        fastest_rung=fastest["rung"],
        fastest_collection_s=fastest["collection_s"],
        meets_bar=meets_bar,
        baseline_collection_s=readable[0]["collection_s"],
        fold_reduction=(readable[0]["collection_s"] / fastest["collection_s"]
                        if fastest["collection_s"] else None),
        # S0.c is reported beside the primary; it gates S1, never S0.
        wallclock_ok=fastest["wallclock_min"] <= S0_MAX_WALLCLOCK_MIN,
        verdict=VERDICT_SCALES if (monotone and meets_bar) else VERDICT_DOES_NOT,
    )
    return result


def format_s0b(res: dict) -> str:
    lines = ["S0.b -- does peer-group assembly cost fall when arrivals speed up?",
             f"  bar: median collection <= {S0_COLLECTION_MAX_S} s at the fastest readable "
             f"rung, falling monotonically",
             "",
             f"  {'rung':>6} {'arr/s':>7} {'cells':>5} {'collect':>9} {'HOL':>8} "
             f"{'place':>7} {'batch':>7} {'wall/min':>9}"]
    for r in res["rungs"]:
        if r.get("collection_s") is None:
            lines.append(f"  {r['rung']:>6} {r['arrivals_per_s']:>7.3f} {r['n_cells']:>5} "
                         f"{'--':>9}  ({r.get('reason','')})")
            continue
        flag = "" if r.get("readable") else f"  UNREADABLE ({r.get('reason')})"
        lines.append(
            f"  {r['rung']:>6} {r['arrivals_per_s']:>7.3f} {r['n_cells']:>5} "
            f"{r['collection_s']:>9.3f} {r['head_of_line_s']:>8.3f} {r['placement_s']:>7.3f} "
            f"{r['mean_batch_size']:>7.2f} {r['wallclock_min']:>9.1f}{flag}")
    lines.append("")
    if res["verdict"] == VERDICT_UNREADABLE:
        lines.append(f"  VERDICT: {res['verdict']} -- {res.get('reason')}")
        return "\n".join(lines)
    lines += [f"  monotone decline: {res['monotone']}",
              f"  fastest rung {res['fastest_rung']}: {res['fastest_collection_s']:.3f} s "
              f"vs bar {S0_COLLECTION_MAX_S} -> {'MEETS' if res['meets_bar'] else 'MISSES'}",
              f"  reduction from {res['baseline_collection_s']:.3f} s: "
              f"{res['fold_reduction']:.2f}x",
              f"  S0.c wall-clock <= {S0_MAX_WALLCLOCK_MIN} min: {res['wallclock_ok']}",
              "",
              f"  VERDICT: {res['verdict']}"]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "simulation_data/cluster_scale_v1/s0b"
    print(format_s0b(read_s0b(d)))
