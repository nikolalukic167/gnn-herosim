#!/usr/bin/env python3
"""serving_stability_v1 -- S3, the live gate that closes the lineage (rule 6).

Registered in docs/lineages/serving_stability_v1.md on 2026-09-15. Every bar below is a
module constant and was committed before the guarded arms had finished.

S1 measured that the learned arms beat reactive Knative while queues are shallow -- 3/3
cells, every seed. S2 did NOT establish that they fail to stabilise (1/3 cells), and an
exploratory within-cell read found stability does not predict latency across seeds
(pooled-z rho = +0.088, p = 0.40, with two cells significant in OPPOSITE directions). So the
mechanism story is unsupported by correlation, and S3 is the only causal read: force
stability and see what happens.

  S3-a deadlock control (BLOCKING)  0 hangs per cell. GNN_PREFIX_PLATFORM_CAP deadlocks 3/16
      seeds in this regime and this guardrail is one design decision away from it. A hang
      means it is not a serving default, reported with its count, and S3-c is CONFOUNDED.

  S3-b bind control (BLOCKING)  the arm's own counters must show the mask active on
      >= S3B_MIN_BIND_PCT of decisions. A knob that did nothing reads exactly like one that
      did not help -- drainable_regime_v1 lost three reads that way.

  S3-c primary  guarded gnn vs knative_network: median latency below AND >= S3_MIN_SEEDS
      seeds below, on >= S3_MIN_CELLS of 3 cells => LEARNED-BEATS-REACTIVE, which no
      measurement in this program has produced at a drainable load.

  S3-d paired  guarded vs unguarded, same seeds and cells => GUARDRAIL-HELPS.

No GNN-vs-pointwise claim may be founded here: a queue guardrail is the kind of per-platform
count-and-depth function the count theorem covers, and mpoff is predicted to gain as much.
mpoff is read identically and reported.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.offline_live_transfer_v1_pairing_read import spearman  # noqa: E402,F401
from scripts_cosim.serving_stability_v1_s1_s2_read import (  # noqa: E402
    ARMS, BURNED, CELLS, KNOWN_UNSERVABLE, StabilityReadError,
)

# --------------------------------------------------------------------------- BARS
S3_MIN_SEEDS = 12
S3_MIN_CELLS = 2
S3_ALPHA = 0.05
S3B_MIN_BIND_PCT = 5.0
S3A_MAX_HANGS = 0
S3_K = 3.0
REGISTERED_C5_PREDICTION = "TIE: mpoff gains as much (count theorem)"


def mann_whitney_u_p(xs: List[float], ys: List[float]) -> float:
    """Two-sided Mann-Whitney U with a normal approximation and tie correction."""
    n1, n2 = len(xs), len(ys)
    if n1 < 3 or n2 < 3:
        raise StabilityReadError(f"need >= 3 per group, got {n1} and {n2}")
    pooled = sorted((v, 0) for v in xs)
    pooled += [(v, 1) for v in ys]
    pooled.sort()
    ranks: List[float] = [0.0] * len(pooled)
    i = 0
    ties = 0.0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        t = j - i + 1
        ties += t ** 3 - t
        for k in range(i, j + 1):
            ranks[k] = shared
        i = j + 1
    r1 = sum(r for r, (_v, g) in zip(ranks, pooled) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sd2 = (n1 * n2 / 12.0) * ((n + 1) - ties / (n * (n - 1)))
    if sd2 <= 0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / (sd2 ** 0.5)
    # two-sided normal tail
    return max(0.0, min(1.0, 2.0 * 0.5 * (1.0 - _erf(z / (2 ** 0.5)))))


def _erf(x: float) -> float:
    import math
    return math.erf(x)


def bind_pct(summary: Dict[str, Any]) -> float:
    c = summary.get("schedulerCounters") or {}
    for key in ("queue_guard_decisions", "queue_guard_steps_active"):
        if key not in c:
            raise StabilityReadError(
                f"{summary.get('arm')}: no {key} -- the guardrail's counters did not survive "
                "the orchestrator whitelist, so S3-b cannot be read and an unread S3-b is how "
                "three drainable_regime_v1 reads were confounded"
            )
    decisions = float(c["queue_guard_decisions"])
    if decisions <= 0:
        return 0.0
    return 100.0 * float(c["queue_guard_steps_active"]) / decisions


def seeds_for(guarded: Dict[str, Dict[str, Any]], cell: str, arm: str) -> List[int]:
    burned = set(BURNED.get(cell, ()))
    return [s for s in range(1, 17)
            if (arm, s) not in burned and (arm, s) not in KNOWN_UNSERVABLE
            and f"{cell}__guarded_{arm}_s{s}" in guarded]


def read(
    guarded: Dict[str, Dict[str, Any]],
    unguarded: Dict[str, Dict[str, Any]],
    expected_per_cell: int = 16,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "lineage": "serving_stability_v1", "read": "S3",
        "bars": {"k": S3_K, "min_seeds": S3_MIN_SEEDS, "min_cells": S3_MIN_CELLS,
                 "alpha": S3_ALPHA, "bind_min_pct": S3B_MIN_BIND_PCT,
                 "max_hangs": S3A_MAX_HANGS},
        "S3a": {"per_cell": {}}, "S3b": {"per_cell": {}},
        "S3c": {"per_cell": {}}, "S3d": {"per_cell": {}},
    }
    hangs_total = 0
    bind_ok = True
    for cell in CELLS:
        # --- S3-a: an arm that is absent hung (the job caps at 30 min on purpose) ---
        missing = {arm: [s for s in range(1, 17)
                         if (arm, s) not in set(BURNED.get(cell, ()))
                         and (arm, s) not in KNOWN_UNSERVABLE
                         and f"{cell}__guarded_{arm}_s{s}" not in guarded]
                   for arm in ARMS}
        n_missing = sum(len(v) for v in missing.values())
        hangs_total += n_missing
        out["S3a"]["per_cell"][cell] = {"missing": {a: v for a, v in missing.items() if v},
                                        "hangs": n_missing}

        # --- S3-b: did the mask actually bind? ---
        binds: Dict[str, Any] = {}
        for arm in ARMS:
            pcts = [bind_pct(guarded[f"{cell}__guarded_{arm}_s{s}"])
                    for s in seeds_for(guarded, cell, arm)]
            if not pcts:
                binds[arm] = {"verdict": "VOID-NO-ARMS"}
                bind_ok = False
                continue
            med = st.median(pcts)
            ok = med >= S3B_MIN_BIND_PCT
            binds[arm] = {"n": len(pcts), "median_bind_pct": med,
                          "min_bind_pct": min(pcts), "passes": ok}
            bind_ok = bind_ok and ok
        out["S3b"]["per_cell"][cell] = binds

        ref = unguarded.get(f"{cell}__reactive")
        if ref is None:
            raise StabilityReadError(f"{cell}: no reactive arm to compare against")
        r = float(ref["averageElapsedTime"])

        c_row: Dict[str, Any] = {"reactive_s": r}
        d_row: Dict[str, Any] = {}
        for arm in ARMS:
            seeds = seeds_for(guarded, cell, arm)
            g = [float(guarded[f"{cell}__guarded_{arm}_s{s}"]["averageElapsedTime"])
                 for s in seeds]
            if len(g) < S3_MIN_SEEDS:
                c_row[arm] = {"verdict": "VOID-TOO-FEW-SEEDS", "n": len(g)}
                d_row[arm] = {"verdict": "VOID-TOO-FEW-SEEDS", "n": len(g)}
                continue
            med = st.median(g)
            below = sum(1 for v in g if v < r)
            c_row[arm] = {"n": len(g), "median_s": med, "seeds_below_reactive": below,
                          "pct_vs_reactive": 100.0 * (r - med) / r,
                          "fires": med < r and below >= S3_MIN_SEEDS}

            paired = [(guarded[f"{cell}__guarded_{arm}_s{s}"]["averageElapsedTime"],
                       unguarded[f"{cell}__{arm}_s{s}"]["averageElapsedTime"])
                      for s in seeds if f"{cell}__{arm}_s{s}" in unguarded]
            if len(paired) < S3_MIN_SEEDS:
                d_row[arm] = {"verdict": "VOID-NO-PAIRS", "n": len(paired)}
                continue
            gs = [p[0] for p in paired]
            us = [p[1] for p in paired]
            better = sum(1 for a, b in paired if a < b)
            p = mann_whitney_u_p(gs, us)
            d_row[arm] = {"n": len(paired), "guarded_median_s": st.median(gs),
                          "unguarded_median_s": st.median(us),
                          "pct_vs_unguarded": 100.0 * (st.median(us) - st.median(gs))
                          / st.median(us),
                          "seeds_better": better, "p": p,
                          "fires": st.median(gs) < st.median(us)
                          and better >= S3_MIN_SEEDS and p < S3_ALPHA}
        out["S3c"]["per_cell"][cell] = c_row
        out["S3d"]["per_cell"][cell] = d_row

    out["S3a"]["hangs"] = hangs_total
    out["S3a"]["passes"] = hangs_total <= S3A_MAX_HANGS
    out["S3a"]["verdict"] = ("NO-DEADLOCK" if out["S3a"]["passes"]
                             else "DEADLOCKS-NOT-A-SERVING-DEFAULT")
    out["S3b"]["passes"] = bind_ok
    out["S3b"]["verdict"] = "GUARDRAIL-BOUND" if bind_ok else "GUARDRAIL-DID-NOTHING-VOID"

    for key, label_yes, label_no in (
            ("S3c", "LEARNED-BEATS-REACTIVE", "REACTIVE-STILL-WINS"),
            ("S3d", "GUARDRAIL-HELPS", "GUARDRAIL-DOES-NOT-HELP")):
        cells_firing = sum(1 for c in CELLS
                           if (out[key]["per_cell"][c].get("gnn") or {}).get("fires"))
        out[key]["gnn_cells_firing"] = cells_firing
        out[key]["mpoff_cells_firing"] = sum(
            1 for c in CELLS if (out[key]["per_cell"][c].get("mpoff") or {}).get("fires"))
        fires = cells_firing >= S3_MIN_CELLS
        if not out["S3b"]["passes"]:
            out[key]["verdict"] = "VOID-S3B"
        elif not out["S3a"]["passes"]:
            out[key]["verdict"] = f"CONFOUNDED-S3A ({label_yes if fires else label_no})"
        else:
            out[key]["verdict"] = label_yes if fires else label_no
        out[key]["fires"] = fires

    out["registered_prediction"] = REGISTERED_C5_PREDICTION
    if out["S3c"]["fires"] and out["S3d"]["fires"]:
        out["outcome"] = "STABILITY-WAS-THE-LEVER"
    elif out["S3d"]["fires"]:
        out["outcome"] = "HELPS-NOT-ENOUGH"
    else:
        out["outcome"] = "STABILITY-NOT-THE-LEVER"
    return out


def load(path: Path, prefix_filter: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f in sorted(path.glob("*.summary.json")):
        doc = json.loads(f.read_text())
        arm = doc.get("arm")
        if not arm:
            raise StabilityReadError(f"{f}: no arm name")
        out[arm] = doc
    if not out:
        raise StabilityReadError(f"{path}: no summaries")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--guarded", type=Path, required=True)
    ap.add_argument("--unguarded", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    result = read(load(args.guarded), load(args.unguarded))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\n[S3-a] deadlock control (blocking): {result['S3a']['hangs']} hang(s), bar "
          f"<= {S3A_MAX_HANGS} -> {result['S3a']['verdict']}")
    for cell, row in result["S3a"]["per_cell"].items():
        if row["hangs"]:
            print(f"        {cell}: {row['missing']}")
    print(f"\n[S3-b] bind control (blocking): mask active on >= {S3B_MIN_BIND_PCT} % of decisions")
    for cell, row in result["S3b"]["per_cell"].items():
        for arm in ARMS:
            a = row.get(arm) or {}
            if "verdict" in a:
                print(f"        {cell} {arm:6s} {a['verdict']}")
            else:
                print(f"        {cell} {arm:6s} n={a['n']:2d} median bind "
                      f"{a['median_bind_pct']:6.2f} %  min {a['min_bind_pct']:6.2f} %"
                      f" -> {'pass' if a['passes'] else 'FAIL'}")
    print(f"       {result['S3b']['verdict']}")

    for key, title in (("S3c", "vs reactive knative_network"),
                       ("S3d", "guarded vs unguarded, paired")):
        print(f"\n[{key}] {title}")
        for cell, row in result[key]["per_cell"].items():
            head = (f"reactive {row['reactive_s']:.2f} s" if key == "S3c" else "")
            print(f"        {cell}  {head}")
            for arm in ARMS:
                a = row.get(arm) or {}
                if "verdict" in a:
                    print(f"          {arm:6s} {a['verdict']}")
                elif key == "S3c":
                    print(f"          {arm:6s} n={a['n']:2d} median {a['median_s']:7.2f} s"
                          f"  {a['pct_vs_reactive']:+7.2f}%  {a['seeds_below_reactive']}"
                          f"/{a['n']} below -> {'FIRES' if a['fires'] else 'no'}")
                else:
                    print(f"          {arm:6s} n={a['n']:2d} guarded {a['guarded_median_s']:7.2f}"
                          f" vs unguarded {a['unguarded_median_s']:7.2f}"
                          f"  {a['pct_vs_unguarded']:+7.2f}%  {a['seeds_better']}/{a['n']}"
                          f"  p={a['p']:.4f} -> {'FIRES' if a['fires'] else 'no'}")
        print(f"       {result[key]['verdict']}  (gnn cells {result[key]['gnn_cells_firing']}"
              f"/{len(CELLS)}, mpoff {result[key]['mpoff_cells_firing']}/{len(CELLS)})")

    print(f"\n[OUTCOME] {result['outcome']}")
    print(f"[note] registered prediction for gnn vs mpoff: {REGISTERED_C5_PREDICTION}")
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
