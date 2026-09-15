#!/usr/bin/env python3
"""offline_live_transfer_v1 -- R3: is a bad seed born bad, or does it become bad?

Registered in docs/lineages/offline_live_transfer_v1.md on 2026-09-15. Bars are the module
constants below and were committed before the captures were read.

R1 measured that the offline score cannot rank checkpoints. R4 measured that it CAN rank
epochs within a run. Neither says where the within-arm live spread comes from -- and that
spread is the largest effect in this program: two checkpoints of the same recipe, same
corpus, same split, same lr, differing only in training seed, serve the identical trace at
43.99 s and 74.76 s.

R3 asks WHEN the gap appears. Live latency at this cell is queue time (R0: Spearman 1.0000
in all three families), so the question is asked of queue time, decile by decile of the
trace's own arrival order:

  PRESENT-FROM-THE-START  the decile-1 gap is >= R3_START_MIN_FRAC of the full-trace gap.
                          The worse seed is already worse on its first decisions, before it
                          can have shaped the state it faces -- the live regime is outside
                          what it was trained on from the beginning.

  COMPOUNDS               the decile-1 gap is <= R3_COMPOUND_MAX_FRAC of the full-trace gap.
                          The seeds start level and diverge -- the model builds its own bad
                          states, which is what drainable_debug_v1's D1/D2 point at (near
                          perfect one-step plans; 77-90 % of excess queue is cross-batch).

  MIXED                   in between; reported with the decile curve and nothing more.

R3 explains. It cannot make R1 or R2 fire, and it does not close the lineage.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- BARS
R3_DECILES = 10
R3_START_MIN_FRAC = 0.50        # decile-1 share of the gap => PRESENT-FROM-THE-START
R3_COMPOUND_MAX_FRAC = 0.20     # decile-1 share of the gap => COMPOUNDS
R3_MIN_TASKS_PER_DECILE = 500   # below this a decile is not read
# The gap must be worth decomposing at all. Two seeds that serve the trace within this many
# seconds of mean queue time are not a spread, and the decile shares would be noise ratios.
R3_MIN_FULL_GAP_S = 1.0

ORDER_KEY = "dispatchedTime"    # when the task entered the system: the trace's own order
QUEUE_KEY = "queueTime"
ELAPSED_KEY = "elapsedTime"


class DecileReadError(RuntimeError):
    """Fail loud: a silent hole here turns a mechanism claim into an artifact."""


def _num(rec: Dict[str, Any], key: str) -> float:
    """0.0 is a real value; a missing key is not. Do not conflate them."""
    val = rec.get(key)
    if val is None:
        raise DecileReadError(f"record {rec.get('taskId')!r} has no {key}")
    return float(val)


def load_arm(path: Path) -> Dict[int, Tuple[float, float, float]]:
    """taskId -> (order, queue, elapsed). Only these survive; the raw file is ~780 MB."""
    doc = json.loads(path.read_text())
    stats = doc.get("stats") or doc
    recs = stats.get("taskResults")
    if not recs:
        raise DecileReadError(
            f"{path} carries no taskResults. Re-run with KEEP_RAW=1 / SIM_FORCE_FULL_STATS=1 "
            "-- the streaming stats path writes an empty list above 10,000 events and this "
            "read would report zeros for everything."
        )
    out: Dict[int, Tuple[float, float, float]] = {}
    for rec in recs:
        tid = rec.get("taskId")
        if tid is None:
            raise DecileReadError(f"{path}: a task record has no taskId")
        out[int(tid)] = (_num(rec, ORDER_KEY), _num(rec, QUEUE_KEY), _num(rec, ELAPSED_KEY))
    return out


def decile_bounds(orders: Sequence[float], n: int = R3_DECILES) -> List[float]:
    """Equal-count boundaries over the trace's arrival order.

    Boundaries come from ONE arm and are applied to both: the two arms serve the same trace,
    so a task belongs to the same decile in each, and per-arm boundaries would silently
    compare different task sets.
    """
    xs = sorted(orders)
    if len(xs) < n:
        raise DecileReadError(f"{len(xs)} tasks cannot be split into {n} deciles")
    return [xs[min(len(xs) - 1, (i + 1) * len(xs) // n)] for i in range(n - 1)]


def decile_of(order: float, bounds: Sequence[float]) -> int:
    for i, b in enumerate(bounds):
        if order < b:
            return i
    return len(bounds)


def decile_queue_summary(recs: List[Dict[str, Any]], n: int = R3_DECILES) -> Dict[str, Any]:
    """Per-decile mean queue time, computed from a run's OWN task records.

    serving_stability_v1 needs this for 99 arms. A retained raw result is ~780 MB, so keeping
    one per arm is not a storage plan -- and a read that can only afford two captures is how
    this module's own parent ended at n = 2. Called from a gate's summary step, it turns the
    records into ten numbers before they are discarded.

    Fails loud on an empty record list rather than returning zeros: the streaming stats path
    writes no taskResults above 10,000 events unless SIM_FORCE_FULL_STATS=1, and silent zeros
    here would read as a perfectly flat, perfectly stable arm.
    """
    if not recs:
        raise DecileReadError(
            "no task records to summarise -- run with SIM_FORCE_FULL_STATS=1, or this arm "
            "would be recorded as having a flat queue for the whole trace"
        )
    orders = [_num(r, ORDER_KEY) for r in recs]
    bounds = decile_bounds(orders, n)
    buckets: List[List[float]] = [[] for _ in range(n)]
    elapsed: List[List[float]] = [[] for _ in range(n)]
    for rec in recs:
        d = decile_of(_num(rec, ORDER_KEY), bounds)
        buckets[d].append(_num(rec, QUEUE_KEY))
        elapsed[d].append(_num(rec, ELAPSED_KEY))
    rows = [{"decile": i + 1, "n": len(b),
             "mean_queue_s": st.fmean(b) if b else None,
             "mean_elapsed_s": st.fmean(e) if e else None}
            for i, (b, e) in enumerate(zip(buckets, elapsed))]
    depths = [r["mean_queue_s"] for r in rows if r["mean_queue_s"] is not None]
    if not depths:
        raise DecileReadError("every decile is empty after bucketing")
    lo = min(depths)
    return {
        "deciles": rows,
        "n_tasks": len(recs),
        "mean_queue_s": st.fmean(_num(r, QUEUE_KEY) for r in recs),
        "mean_elapsed_s": st.fmean(_num(r, ELAPSED_KEY) for r in recs),
        # S2's statistic. A zero minimum would make the ratio infinite, so it is reported as
        # None and the read decides what to do -- not silently clamped here.
        "queue_max_over_min": (max(depths) / lo) if lo > 0 else None,
    }


def read(
    best: Dict[int, Tuple[float, float, float]],
    worst: Dict[int, Tuple[float, float, float]],
) -> Dict[str, Any]:
    shared = sorted(set(best) & set(worst))
    if not shared:
        raise DecileReadError("the two captures share no taskId")
    if len(shared) != len(best) or len(shared) != len(worst):
        raise DecileReadError(
            f"captures cover different tasks ({len(best)} vs {len(worst)}, {len(shared)} "
            "shared) -- the same trace must produce the same task set"
        )

    bounds = decile_bounds([best[t][0] for t in shared])
    per_decile: List[Dict[str, Any]] = []
    for d in range(R3_DECILES):
        ids = [t for t in shared if decile_of(best[t][0], bounds) == d]
        if len(ids) < R3_MIN_TASKS_PER_DECILE:
            raise DecileReadError(
                f"decile {d + 1} holds {len(ids)} tasks, below {R3_MIN_TASKS_PER_DECILE}"
            )
        b = st.fmean(best[t][1] for t in ids)
        w = st.fmean(worst[t][1] for t in ids)
        per_decile.append({"decile": d + 1, "n": len(ids), "best_queue_s": b,
                           "worst_queue_s": w, "gap_s": w - b})

    full_best = st.fmean(best[t][1] for t in shared)
    full_worst = st.fmean(worst[t][1] for t in shared)
    full_gap = full_worst - full_best
    if full_gap < R3_MIN_FULL_GAP_S:
        return {
            "verdict": "VOID-NO-GAP",
            "why": (f"full-trace queue gap {full_gap:.4f} s is below "
                    f"{R3_MIN_FULL_GAP_S} s; there is no spread to decompose"),
            "full": {"best_queue_s": full_best, "worst_queue_s": full_worst,
                     "gap_s": full_gap},
            "per_decile": per_decile,
        }

    # The bar is "what SHARE of the gap lives in decile 1", so the normaliser is the total
    # gap mass across deciles, not the per-task mean. Dividing by the mean makes the shares
    # sum to R3_DECILES instead of 1, so a gap entirely inside decile 1 reads 10.0 against a
    # 0.50 bar and every verdict becomes PRESENT-FROM-THE-START. Caught by the tests before
    # this read ever touched a capture.
    gap_mass = sum(row["gap_s"] for row in per_decile)
    if gap_mass <= 0:
        return {
            "verdict": "VOID-NO-GAP",
            "why": f"total decile gap mass {gap_mass:.4f} s is not positive",
            "full": {"best_queue_s": full_best, "worst_queue_s": full_worst,
                     "gap_s": full_gap},
            "per_decile": per_decile,
        }
    for row in per_decile:
        row["share"] = row["gap_s"] / gap_mass
    first_frac = per_decile[0]["share"]
    # DESCRIPTIVE, not a bar. The COMPOUNDS bar (decile-1 share <= 0.20) is cleared by a gap
    # that GROWS from zero and also by one that is simply UNIFORM -- a flat gap puts 1/10 of
    # the mass in decile 1, below the bar. Those are different mechanisms, so the shape is
    # reported next to the verdict and a COMPOUNDS reading is only a closed-loop claim when
    # the curve actually rises.
    shares = [row["share"] for row in per_decile]
    second_half = sum(shares[R3_DECILES // 2:])
    if first_frac >= R3_START_MIN_FRAC:
        verdict = "PRESENT-FROM-THE-START"
    elif first_frac <= R3_COMPOUND_MAX_FRAC:
        verdict = "COMPOUNDS"
    else:
        verdict = "MIXED"
    return {
        "lineage": "offline_live_transfer_v1",
        "read": "R3",
        "bars": {"deciles": R3_DECILES, "start_min_frac": R3_START_MIN_FRAC,
                 "compound_max_frac": R3_COMPOUND_MAX_FRAC,
                 "min_tasks_per_decile": R3_MIN_TASKS_PER_DECILE,
                 "min_full_gap_s": R3_MIN_FULL_GAP_S},
        "n_tasks": len(shared),
        "full": {"best_queue_s": full_best, "worst_queue_s": full_worst, "gap_s": full_gap,
                 "best_elapsed_s": st.fmean(best[t][2] for t in shared),
                 "worst_elapsed_s": st.fmean(worst[t][2] for t in shared)},
        "per_decile": per_decile,
        "gap_mass_s": gap_mass,
        "second_half_share": second_half,
        "shape": ("RISING" if second_half > 0.60 else
                  "FLAT" if 0.40 <= second_half <= 0.60 else "FALLING"),
        "first_decile_frac": first_frac,
        "last_decile_frac": per_decile[-1]["share"],
        "verdict": verdict,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--best", type=Path, required=True)
    ap.add_argument("--worst", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    print(f"[r3] loading {args.best}", flush=True)
    best = load_arm(args.best)
    print(f"[r3] loading {args.worst}", flush=True)
    worst = load_arm(args.worst)
    result = read(best, worst)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\n[R3] {result.get('n_tasks', '?')} tasks, {R3_DECILES} deciles by {ORDER_KEY}")
    print(f"     {'decile':>7} {'n':>6} {'best queue':>12} {'worst queue':>12} {'gap':>10}"
          f" {'share':>8}")
    for row in result["per_decile"]:
        share = row.get("share")
        share_s = f"{share:>8.3f}" if share is not None else f"{'-':>8}"
        print(f"     {row['decile']:>7} {row['n']:>6} {row['best_queue_s']:>12.3f}"
              f" {row['worst_queue_s']:>12.3f} {row['gap_s']:>10.3f} {share_s}")
    f = result["full"]
    print(f"     {'FULL':>7} {result.get('n_tasks', 0):>6} {f['best_queue_s']:>12.3f}"
          f" {f['worst_queue_s']:>12.3f} {f['gap_s']:>10.3f}")
    if "first_decile_frac" in result:
        print(f"[R3] decile-1 share {result['first_decile_frac']:.3f} "
              f"(>= {R3_START_MIN_FRAC} => born bad, <= {R3_COMPOUND_MAX_FRAC} => compounds); "
              f"decile-10 share {result['last_decile_frac']:.3f}")
    if "shape" in result:
        print(f"[R3] shape {result['shape']} (second half carries "
              f"{result['second_half_share']:.3f} of the gap) -- DESCRIPTIVE, not a bar: a "
              f"COMPOUNDS verdict is a closed-loop claim only when the curve RISES")
    print(f"[R3] {result['verdict']}")
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
