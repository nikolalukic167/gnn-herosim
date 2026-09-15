#!/usr/bin/env python3
"""queue_range_v1 -- Q0..Q4.

Registered in docs/lineages/queue_range_v1.md on 2026-09-15. Every bar below is a module
constant and was committed before the arms it reads had been submitted.

The claim: the learned arms lose their early advantage over reactive Knative because the
queue column they rank candidates with leaves the contract it was trained under, and leaves
it as the trace gets busier. On the arms' own corpus the divisor is 1.0 in all 516 datasets,
so dim7 IS the raw queue depth, candidate p50 12 and max 42. Live there are two ways out of
that contract and they are opposite -- OUT-OF-RANGE (divisor pinned at 1.0 by idle platforms,
depth unbounded) and COMPRESSED (divisor inflated, real differences squeezed to nothing).
Q0 measures which, and can refute the mechanism outright.

Q0 blocking, and can kill the lineage -- the column must actually be out of contract, and
     out of contract MORE late than early.
Q1 blocking -- the knob must have reached the feature builder. A knob that did nothing reads
     exactly like one that did not help (drainable_regime_v1's registered general rule).
Q2 primary -- paired Wilcoxon, intervention vs control, Holm over the REGISTERED family size.
Q3 primary -- does the best intervention beat reactive. Registered expectation: NEGATIVE on
     2 of 3 cells; the early margin is worth ~4 s against deficits of 28.9 / 1.4 / 41.9 s.
Q4 diagnostic -- does the early advantage extend past decile 2. Control value: 2.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.placement.queue_features import CORPUS_DIM7_CANDIDATE_MAX  # noqa: E402

# --------------------------------------------------------------------------- BARS
Q0_OUT_OF_RANGE_FRAC = 0.20
Q0_BLIND_FRAC = 0.20
Q0_EARLY_LATE_RATIO = 3.0
Q0_MIN_CELLS = 2

Q1_MIN_PINNED_FRAC = 0.99
Q1_MAX_OUT_OF_RANGE_FRAC = 0.0

Q2_ALPHA = 0.05
Q2_HOLM_N = 6                 # 2 interventions x 3 cells. Registered; never shrunk.
Q2_MIN_SEEDS = 12
Q2_MIN_CELLS = 2

Q3_MIN_CELLS = 2
Q3_REGISTERED_EXPECTATION = "NEGATIVE on >= 2 of 3 cells"

Q4_MIN_DECILES = 4
Q4_MIN_CELLS = 2
Q4_CONTROL_DECILES = 2        # measured by serving_stability_v1 S1, not assumed

EARLY_DECILES = (1, 2)
LATE_DECILES = (9, 10)

CELLS = ("cell_s7901_f4000_pg16", "cell_s9001_f4000_pg16", "cell_s9002_f4000_pg16")
POLICIES = ("gnn", "mpoff")
CONTROL_ARM = "plain"
INTERVENTIONS = ("pinned", "inrange")

# gnn seed 3's checkpoint deterministically livelocks the simulator; declared, not discovered.
KNOWN_UNSERVABLE: Tuple[Tuple[str, int], ...] = (("gnn", 3),)
# The jam_probe scoping read used these on this cell. A bar cannot be applied to the data
# that suggested it.
BURNED: Dict[str, Tuple[Tuple[str, int], ...]] = {
    "cell_s7901_f4000_pg16": (("gnn", 8), ("gnn", 14)),
}


class QueueRangeReadError(RuntimeError):
    """Fail loud: a quietly dropped arm is a quietly different experiment."""


# --------------------------------------------------------------- per-batch summarisation
def queue_range_decile_summary(
    records: Sequence[Sequence[float]], bounds: Sequence[float]
) -> Dict[str, Any]:
    """Bucket the scheduler's per-batch queue-column records onto the task deciles.

    `records` rows are [sim_time, divisor, raw_spread, dim7_spread, blind, dim7_max] and
    `bounds` are `decile_queue_summary`'s own edges, so a batch and the tasks it placed land
    in the same decile. Called from the gate's summary step: 288 arms cannot each keep a
    ~780 MB raw result, and the trace itself is ~8,000 rows per arm.
    """
    if not records:
        raise QueueRangeReadError(
            "no per-batch queue-column records -- the scheduler served no graph through the "
            "feature builder, so this arm cannot be read as being in or out of contract"
        )
    n = len(bounds) + 1
    buckets: List[List[Sequence[float]]] = [[] for _ in range(n)]
    for row in records:
        t = float(row[0])
        idx = n - 1
        for i, b in enumerate(bounds):
            if t < b:
                idx = i
                break
        buckets[idx].append(row)
    rows = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            rows.append({"decile": i + 1, "n": 0})
            continue
        rows.append({
            "decile": i + 1,
            "n": len(bucket),
            "median_divisor": st.median(float(r[1]) for r in bucket),
            "pinned_frac": sum(1 for r in bucket if float(r[1]) == 1.0) / len(bucket),
            "blind_frac": sum(1 for r in bucket if float(r[4])) / len(bucket),
            "out_of_range_frac": sum(
                1 for r in bucket if float(r[5]) > CORPUS_DIM7_CANDIDATE_MAX
            ) / len(bucket),
            "median_dim7_max": st.median(float(r[5]) for r in bucket),
            "median_dim7_spread": st.median(float(r[3]) for r in bucket),
        })
    return {
        "deciles": rows,
        "n_batches": len(records),
        "pinned_frac": sum(1 for r in records if float(r[1]) == 1.0) / len(records),
        "blind_frac": sum(1 for r in records if float(r[4])) / len(records),
        "out_of_range_frac": sum(
            1 for r in records if float(r[5]) > CORPUS_DIM7_CANDIDATE_MAX
        ) / len(records),
        "max_dim7": max(float(r[5]) for r in records),
        "max_divisor": max(float(r[1]) for r in records),
    }


# --------------------------------------------------------------------------- statistics
def _signed_rank(values: Sequence[float]) -> Tuple[float, int]:
    """Wilcoxon W+ over the non-zero differences, with average ranks for ties."""
    nz = [v for v in values if v != 0.0]
    if not nz:
        return 0.0, 0
    order = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    ranks = [0.0] * len(nz)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return sum(r for r, v in zip(ranks, nz) if v > 0), len(nz)


def wilcoxon_p(values: Sequence[float]) -> Optional[float]:
    """Two-sided p for the signed-rank statistic, normal approximation with tie correction.

    PAIRED, deliberately: the arms share seeds and the same trace, and an unpaired test on
    seed-paired arms is the error `docs/gates/gate-tools.md` records from serving_stability_v1
    S3-d, where it moved a verdict. Pinned against scipy in the test file.
    """
    w_plus, n = _signed_rank(values)
    if n < 6:
        return None
    mean = n * (n + 1) / 4.0
    nz = [abs(v) for v in values if v != 0.0]
    counts: Dict[float, int] = {}
    for v in nz:
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(c ** 3 - c for c in counts.values())
    var = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var <= 0:
        return None
    z = (w_plus - mean) / math.sqrt(var)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def holm(pvalues: Sequence[Optional[float]], family_n: int, alpha: float) -> List[bool]:
    """Holm-Bonferroni over the REGISTERED family size, not the number computed."""
    if family_n < len([p for p in pvalues if p is not None]):
        raise QueueRangeReadError(
            f"family size {family_n} is smaller than the {len(pvalues)} tests computed -- "
            "the registered family is never shrunk to fit"
        )
    indexed = sorted(
        (i for i, p in enumerate(pvalues) if p is not None),
        key=lambda i: pvalues[i],                    # type: ignore[index]
    )
    out = [False] * len(pvalues)
    for rank, i in enumerate(indexed):
        if pvalues[i] < alpha / (family_n - rank):   # type: ignore[operator]
            out[i] = True
        else:
            break
    return out


# --------------------------------------------------------------------------- loading
def seeds_for(arms: Dict[str, Dict[str, Any]], cell: str, arm: str, policy: str) -> List[int]:
    burned = set(BURNED.get(cell, ()))
    keep: List[int] = []
    for seed in range(1, 17):
        if (policy, seed) in burned or (policy, seed) in KNOWN_UNSERVABLE:
            continue
        if f"{cell}__{arm}_{policy}_s{seed}" in arms:
            keep.append(seed)
    return keep


def load(results: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(results.glob("*.summary.json")):
        doc = json.loads(path.read_text())
        name = doc.get("arm")
        if not name:
            raise QueueRangeReadError(f"{path}: no arm name")
        out[name] = doc
    if not out:
        raise QueueRangeReadError(f"{results}: no summaries")
    return out


def load_reactive(results: Path, cell: str) -> Dict[str, Any]:
    """Reactive comes from serving_stability_v1's own results dir; it is not re-run."""
    path = results / f"{cell}__reactive.summary.json"
    if not path.exists():
        raise QueueRangeReadError(f"{path} missing -- Q3 has nothing to compare against")
    return json.loads(path.read_text())


def _elapsed(doc: Dict[str, Any]) -> float:
    v = doc.get("averageElapsedTime")
    if v is None:
        raise QueueRangeReadError(f"{doc.get('arm')}: no averageElapsedTime")
    return float(v)


def _decile_means(doc: Dict[str, Any]) -> List[Optional[float]]:
    rows = (doc.get("decile_summary") or {}).get("deciles") or []
    if len(rows) != 10:
        raise QueueRangeReadError(f"{doc.get('arm')}: {len(rows)} deciles, need 10")
    return [r.get("mean_queue_s") for r in rows]


def _qr(doc: Dict[str, Any]) -> Dict[str, Any]:
    qr = doc.get("queue_range_summary")
    if not qr:
        raise QueueRangeReadError(
            f"{doc.get('arm')}: no queue_range_summary -- this arm was produced by a tree "
            "without the mechanism instrument and cannot be read against Q0/Q1"
        )
    return qr


def _decile_frac(qr: Dict[str, Any], key: str, deciles: Iterable[int]) -> Optional[float]:
    """Batch-weighted share over the named deciles; None when they are all empty."""
    num = den = 0.0
    for row in qr["deciles"]:
        if row["decile"] in tuple(deciles) and row["n"]:
            num += row[key] * row["n"]
            den += row["n"]
    return (num / den) if den else None


# --------------------------------------------------------------------------- the bars
def read_q0(arms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_cell: Dict[str, Any] = {}
    firing = 0
    for cell in CELLS:
        seeds = seeds_for(arms, cell, CONTROL_ARM, "gnn")
        rows = []
        for seed in seeds:
            qr = _qr(arms[f"{cell}__{CONTROL_ARM}_gnn_s{seed}"])
            row = {"seed": seed}
            for key, label in (("out_of_range_frac", "oor"), ("blind_frac", "blind")):
                row[f"early_{label}"] = _decile_frac(qr, key, EARLY_DECILES)
                row[f"late_{label}"] = _decile_frac(qr, key, LATE_DECILES)
            row["max_dim7"] = qr["max_dim7"]
            row["max_divisor"] = qr["max_divisor"]
            rows.append(row)
        if len(rows) < Q2_MIN_SEEDS:
            per_cell[cell] = {"verdict": "VOID-TOO-FEW-SEEDS", "n": len(rows)}
            continue

        def med(key: str) -> Optional[float]:
            vals = [r[key] for r in rows if r[key] is not None]
            return st.median(vals) if vals else None

        summary: Dict[str, Any] = {"n": len(rows)}
        fires = False
        for label, bar in (("oor", Q0_OUT_OF_RANGE_FRAC), ("blind", Q0_BLIND_FRAC)):
            early, late = med(f"early_{label}"), med(f"late_{label}")
            ok = (late is not None and late > bar
                  and (early is None or early == 0.0
                       or late >= Q0_EARLY_LATE_RATIO * early))
            summary[label] = {"early": early, "late": late, "bar": bar, "fires": bool(ok)}
            fires = fires or bool(ok)
        summary["max_dim7"] = med("max_dim7")
        summary["max_divisor"] = med("max_divisor")
        summary["fires"] = fires
        per_cell[cell] = summary
        firing += int(fires)
    return {
        "per_cell": per_cell, "cells_firing": firing,
        "fires": firing >= Q0_MIN_CELLS,
        "verdict": ("COLUMN-LEAVES-CONTRACT" if firing >= Q0_MIN_CELLS
                    else "COLUMN-IN-CONTRACT"),
        "bars": {"out_of_range_frac": Q0_OUT_OF_RANGE_FRAC, "blind_frac": Q0_BLIND_FRAC,
                 "early_late_ratio": Q0_EARLY_LATE_RATIO, "min_cells": Q0_MIN_CELLS,
                 "corpus_dim7_max": CORPUS_DIM7_CANDIDATE_MAX},
    }


def read_q1(arms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_arm: Dict[str, Any] = {}
    all_ok = True
    for cell in CELLS:
        for policy in POLICIES:
            for arm in (CONTROL_ARM,) + INTERVENTIONS:
                seeds = seeds_for(arms, cell, arm, policy)
                if not seeds:
                    continue
                qrs = [_qr(arms[f"{cell}__{arm}_{policy}_s{s}"]) for s in seeds]
                pinned = st.median(q["pinned_frac"] for q in qrs)
                oor = st.median(q["out_of_range_frac"] for q in qrs)
                batches = min(q["n_batches"] for q in qrs)
                if arm == CONTROL_ARM:
                    ok = batches > 0
                elif arm == "pinned":
                    ok = batches > 0 and pinned >= Q1_MIN_PINNED_FRAC
                else:
                    ok = (batches > 0 and pinned >= Q1_MIN_PINNED_FRAC
                          and oor <= Q1_MAX_OUT_OF_RANGE_FRAC)
                per_arm[f"{cell}|{policy}|{arm}"] = {
                    "n": len(seeds), "min_batches": batches,
                    "median_pinned_frac": pinned, "median_out_of_range_frac": oor,
                    "ok": bool(ok),
                }
                all_ok = all_ok and bool(ok)
    return {"per_arm": per_arm, "all_ok": all_ok,
            "verdict": "KNOB-BOUND" if all_ok else "KNOB-NOT-BOUND",
            "bars": {"min_pinned_frac": Q1_MIN_PINNED_FRAC,
                     "max_out_of_range_frac": Q1_MAX_OUT_OF_RANGE_FRAC}}


def read_q2(arms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    tests: List[Tuple[str, str, str, Optional[float], Dict[str, Any]]] = []
    for policy in POLICIES:
        pvals: List[Optional[float]] = []
        meta: List[Tuple[str, str, Dict[str, Any]]] = []
        for intervention in INTERVENTIONS:
            for cell in CELLS:
                a = seeds_for(arms, cell, CONTROL_ARM, policy)
                b = seeds_for(arms, cell, intervention, policy)
                shared = sorted(set(a) & set(b))
                if len(shared) < Q2_MIN_SEEDS:
                    meta.append((intervention, cell,
                                 {"verdict": "VOID-TOO-FEW-SEEDS", "n": len(shared)}))
                    pvals.append(None)
                    continue
                deltas = [
                    _elapsed(arms[f"{cell}__{CONTROL_ARM}_{policy}_s{s}"])
                    - _elapsed(arms[f"{cell}__{intervention}_{policy}_s{s}"])
                    for s in shared
                ]
                p = wilcoxon_p(deltas)
                meta.append((intervention, cell, {
                    "n": len(shared), "median_gain_s": st.median(deltas),
                    "seeds_better": sum(1 for d in deltas if d > 0), "p": p,
                    "control_median_s": st.median(
                        _elapsed(arms[f"{cell}__{CONTROL_ARM}_{policy}_s{s}"]) for s in shared),
                    "arm_median_s": st.median(
                        _elapsed(arms[f"{cell}__{intervention}_{policy}_s{s}"]) for s in shared),
                }))
                pvals.append(p)
        flags = holm(pvals, Q2_HOLM_N, Q2_ALPHA)
        for (intervention, cell, row), flag in zip(meta, flags):
            row["holm_significant"] = bool(flag)
            row["fires"] = bool(flag and row.get("median_gain_s", 0.0) > 0)
            tests.append((policy, intervention, cell, row.get("p"), row))

    per_policy: Dict[str, Any] = {}
    for policy in POLICIES:
        cells_by_arm = {
            i: sum(1 for pol, iv, _c, _p, r in tests
                   if pol == policy and iv == i and r.get("fires"))
            for i in INTERVENTIONS
        }
        per_policy[policy] = {"cells_firing": cells_by_arm,
                              "fires": any(v >= Q2_MIN_CELLS for v in cells_by_arm.values())}
    fires = per_policy["gnn"]["fires"]
    return {
        "tests": [{"policy": p, "intervention": i, "cell": c, **r}
                  for p, i, c, _pv, r in tests],
        "per_policy": per_policy, "fires": fires,
        "verdict": "RANGE-FIX-HELPS" if fires else "RANGE-FIX-DOES-NOT-HELP",
        "bars": {"alpha": Q2_ALPHA, "holm_n": Q2_HOLM_N, "min_seeds": Q2_MIN_SEEDS,
                 "min_cells": Q2_MIN_CELLS, "test": "wilcoxon signed-rank (PAIRED)"},
    }


def read_q3(arms: Dict[str, Dict[str, Any]], reactive: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_cell: Dict[str, Any] = {}
    best_cells = {i: 0 for i in INTERVENTIONS}
    for cell in CELLS:
        ref = _elapsed(reactive[cell])
        row: Dict[str, Any] = {"reactive_s": ref}
        for intervention in INTERVENTIONS:
            seeds = seeds_for(arms, cell, intervention, "gnn")
            if len(seeds) < Q2_MIN_SEEDS:
                row[intervention] = {"verdict": "VOID-TOO-FEW-SEEDS", "n": len(seeds)}
                continue
            vals = [_elapsed(arms[f"{cell}__{intervention}_gnn_s{s}"]) for s in seeds]
            med = st.median(vals)
            below = sum(1 for v in vals if v < ref)
            row[intervention] = {"n": len(seeds), "median_s": med,
                                 "pct_vs_reactive": 100.0 * (ref - med) / ref,
                                 "seeds_below": below, "beats": med < ref}
            best_cells[intervention] += int(med < ref)
        per_cell[cell] = row
    fires = any(v >= Q3_MIN_CELLS for v in best_cells.values())
    return {"per_cell": per_cell, "cells_beating": best_cells, "fires": fires,
            "verdict": "BEATS-REACTIVE" if fires else "REACTIVE-STILL-WINS",
            "registered_expectation": Q3_REGISTERED_EXPECTATION,
            "bars": {"min_cells": Q3_MIN_CELLS}}


def read_q4(arms: Dict[str, Dict[str, Any]], reactive: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_cell: Dict[str, Any] = {}
    cells = {i: 0 for i in (CONTROL_ARM,) + INTERVENTIONS}
    for cell in CELLS:
        ref = _decile_means(reactive[cell])
        row: Dict[str, Any] = {}
        for arm in (CONTROL_ARM,) + INTERVENTIONS:
            seeds = seeds_for(arms, cell, arm, "gnn")
            if len(seeds) < Q2_MIN_SEEDS:
                row[arm] = {"verdict": "VOID-TOO-FEW-SEEDS", "n": len(seeds)}
                continue
            counts = []
            for seed in seeds:
                mine = _decile_means(arms[f"{cell}__{arm}_gnn_s{seed}"])
                counts.append(sum(
                    1 for m, r in zip(mine, ref)
                    if m is not None and r is not None and m < r
                ))
            med = st.median(counts)
            row[arm] = {"n": len(seeds), "median_deciles_ahead": med,
                        "min": min(counts), "max": max(counts),
                        "fires": med >= Q4_MIN_DECILES}
            cells[arm] += int(med >= Q4_MIN_DECILES)
        per_cell[cell] = row
    fires = any(cells[i] >= Q4_MIN_CELLS for i in INTERVENTIONS)
    return {"per_cell": per_cell, "cells_firing": cells, "fires": fires,
            "verdict": "ADVANTAGE-EXTENDS" if fires else "ADVANTAGE-DOES-NOT-EXTEND",
            "bars": {"min_deciles": Q4_MIN_DECILES, "min_cells": Q4_MIN_CELLS,
                     "control_value": Q4_CONTROL_DECILES}}


def read(arms: Dict[str, Dict[str, Any]],
         reactive: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    q0 = read_q0(arms)
    q1 = read_q1(arms)
    q2 = read_q2(arms)
    q3 = read_q3(arms, reactive)
    q4 = read_q4(arms, reactive)
    if not q1["all_ok"]:
        outcome = "VOID-KNOB-NOT-BOUND"
    elif q3["fires"]:
        outcome = "RANGE-FIX-BEATS-REACTIVE"
    elif q2["fires"] and q0["fires"]:
        outcome = "RANGE-IS-A-LEVER-NOT-ENOUGH"
    elif q2["fires"]:
        outcome = "FIX-HELPS-MECHANISM-UNCONFIRMED"
    elif q0["fires"]:
        outcome = "COLUMN-OUT-OF-CONTRACT-BUT-NOT-THE-LEVER"
    else:
        outcome = "QUEUE-RANGE-NOT-THE-LEVER"
    return {"lineage": "queue_range_v1", "Q0": q0, "Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4,
            "outcome": outcome,
            "excluded": {"burned": {c: [list(x) for x in v] for c, v in BURNED.items()},
                         "known_unservable": [list(x) for x in KNOWN_UNSERVABLE]}}


# --------------------------------------------------------------------------- printing
def _fmt(v: Optional[float], nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--reactive-results", type=Path, required=True,
                    help="serving_stability_v1's results dir, for the reactive arms")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    arms = load(args.results)
    reactive = {c: load_reactive(args.reactive_results, c) for c in CELLS}
    result = read(arms, reactive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"[inputs] {len(arms)} intervention summaries, {len(reactive)} reactive")
    print(f"\n[Q0] is the served queue column out of its trained contract, and more so late?"
          f"  (corpus dim7 max {CORPUS_DIM7_CANDIDATE_MAX})")
    for cell, row in result["Q0"]["per_cell"].items():
        if "verdict" in row:
            print(f"     {cell}  {row['verdict']} (n={row['n']})")
            continue
        print(f"     {cell}  n={row['n']}  median max dim7 {_fmt(row['max_dim7'],1)}  "
              f"median max divisor {_fmt(row['max_divisor'],1)}")
        for label, name in (("oor", "out-of-range"), ("blind", "blind")):
            a = row[label]
            print(f"       {name:12s} early {_fmt(a['early'])}  late {_fmt(a['late'])}"
                  f"  bar >{a['bar']} and >={Q0_EARLY_LATE_RATIO}x early"
                  f"  -> {'FIRES' if a['fires'] else 'no'}")
    print(f"     {result['Q0']['verdict']}  ({result['Q0']['cells_firing']}/{len(CELLS)} cells)")

    print(f"\n[Q1] did the knob reach the feature builder (blocking)")
    for name, row in result["Q1"]["per_arm"].items():
        print(f"     {name:48s} n={row['n']:2d} batches>={row['min_batches']:5d}"
              f" pinned {row['median_pinned_frac']:.3f} oor {row['median_out_of_range_frac']:.3f}"
              f" -> {'ok' if row['ok'] else 'FAIL'}")
    print(f"     {result['Q1']['verdict']}")

    print(f"\n[Q2] paired Wilcoxon, control - intervention, Holm over n={Q2_HOLM_N}, "
          f"alpha={Q2_ALPHA}")
    for row in result["Q2"]["tests"]:
        if "verdict" in row:
            print(f"     {row['policy']:6s} {row['intervention']:8s} {row['cell']}  "
                  f"{row['verdict']} (n={row['n']})")
            continue
        print(f"     {row['policy']:6s} {row['intervention']:8s} {row['cell']}  n={row['n']:2d}"
              f"  {row['control_median_s']:8.3f} -> {row['arm_median_s']:8.3f}"
              f"  gain {row['median_gain_s']:+7.3f} s  {row['seeds_better']}/{row['n']}"
              f"  p={_fmt(row['p'],4)}  -> {'FIRES' if row['fires'] else 'no'}")
    print(f"     {result['Q2']['verdict']}")

    print(f"\n[Q3] vs reactive knative_network  (registered expectation: "
          f"{Q3_REGISTERED_EXPECTATION})")
    for cell, row in result["Q3"]["per_cell"].items():
        print(f"     {cell}  reactive {row['reactive_s']:.2f} s")
        for intervention in INTERVENTIONS:
            a = row[intervention]
            if "verdict" in a:
                print(f"       {intervention:8s} {a['verdict']}")
                continue
            print(f"       {intervention:8s} n={a['n']:2d} median {a['median_s']:8.2f} s"
                  f"  {a['pct_vs_reactive']:+7.2f}%  {a['seeds_below']}/{a['n']} below"
                  f"  -> {'BEATS' if a['beats'] else 'no'}")
    print(f"     {result['Q3']['verdict']}")

    print(f"\n[Q4] deciles ahead of reactive (control measured {Q4_CONTROL_DECILES}, "
          f"bar >= {Q4_MIN_DECILES} on >= {Q4_MIN_CELLS} cells)")
    for cell, row in result["Q4"]["per_cell"].items():
        print(f"     {cell}")
        for arm in (CONTROL_ARM,) + INTERVENTIONS:
            a = row[arm]
            if "verdict" in a:
                print(f"       {arm:8s} {a['verdict']}")
                continue
            print(f"       {arm:8s} n={a['n']:2d} median {a['median_deciles_ahead']:.1f}"
                  f"  range {a['min']}..{a['max']}"
                  f"  -> {'FIRES' if a['fires'] else 'no'}")
    print(f"     {result['Q4']['verdict']}")

    print(f"\n[OUTCOME] {result['outcome']}")
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
