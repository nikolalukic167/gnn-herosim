#!/usr/bin/env python3
"""scheduler_residence_v1 R3 -- does topology lopsidedness predict the queue blow-up?

Registered in docs/lineages/scheduler_residence_v1.md, Amendment 1, on 2026-09-15. Every bar
below is a module constant and was committed before any R3 cell had been served a task.

R1 found the one cell of three WITHOUT a queue blow-up separated from the other two with no
overlap: min reachable servers per client 2 against 1 and 1, clients-per-server imbalance 3
against 8 and 7. Three cells is three cells -- the shape that killed serving_stability_v1's
mechanism claim. R3 is the sample size, and it is the lineage's live gate (rule 6).

The unit of replication is the CELL. The dependent variable is each cell's median excess queue
over ITS OWN reactive arm, which removes whatever the topology costs everybody.

R3_PRIMARY   spearman(min_reachable_servers, excess_queue). Negative = fewer reachable
             servers, more excess.
R3_CONTROL   the same correlation against REACTIVE'S OWN queue. If lopsidedness predicts
             reactive just as strongly it is a fact about the environment, not about the
             learned arm, and the primary reads CONFOUNDED-ENVIRONMENT whatever its rho.
             drainable_objective_v1's C0 and peer_affinity_v1's H5 both died of a control that
             moved with the treatment; this one is registered in advance.
R3_SECOND    spearman(hosting_node_spread, excess_queue), R1's other separator. Holm over 2.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- BARS
R3_MIN_ABS_RHO = 0.60
R3_ALPHA = 0.05
R3_HOLM_N = 2                    # primary + second. Registered; never shrunk.
R3_MIN_CELLS_READ = 10
R3_N_CELLS = 12
# Amendment 2 (2026-09-16): 4 of the 12 batch-1 cells hang on ALL NINE arms including
# reactive -- the documented starved-client spin, a property of the topology draw and not of
# any policy. 8 more cells were selected from the SAME manifest by the SAME deterministic
# spanning rule. The read pools both batches; R3_MIN_CELLS_READ is unchanged.
R3_N_CELLS_B2 = 8
R3_N_CELLS_TOTAL = R3_N_CELLS + R3_N_CELLS_B2
R3_SEEDS_PER_ARM = 4
R3_MIN_SEEDS_PER_CELL = 3        # below this a (cell, arm) is dropped and the read says so
R3_PRIMARY_KEY = "min_reachable_servers"
R3_SECOND_KEY = "hosting_node_spread"
R3_EXPECTED_SIGN = -1            # fewer reachable servers => MORE excess queue
R3_REGISTERED_EXPECTATION = "UNCERTAIN, leaning positive"
# The control fires CONFOUNDED-ENVIRONMENT when reactive's own queue tracks the structure at
# least this fraction as strongly as the excess does.
R3_CONTROL_RATIO = 0.75

ARMS = ("gnn", "mpoff")


class R3ReadError(RuntimeError):
    """Fail loud: a quietly dropped cell is a quietly different experiment."""


# --------------------------------------------------------------------------- statistics
def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """rho and a two-sided p via the t approximation. Ties get average ranks."""
    if len(xs) != len(ys):
        raise R3ReadError(f"spearman got {len(xs)} vs {len(ys)} values")
    n = len(xs)
    if n < 4:
        return None, None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return None, None
    rho = num / den
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    # Student-t survival via the incomplete beta, two-sided.
    df = n - 2
    x = df / (df + t * t)
    p = _betainc(df / 2.0, 0.5, x)
    return rho, max(0.0, min(1.0, p))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b), continued fraction. Pinned against scipy."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-16) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def holm(pvalues: Sequence[Optional[float]], family_n: int, alpha: float) -> List[bool]:
    if family_n < len([p for p in pvalues if p is not None]):
        raise R3ReadError("the registered family is never shrunk to fit")
    idx = sorted((i for i, p in enumerate(pvalues) if p is not None),
                 key=lambda i: pvalues[i])          # type: ignore[index]
    out = [False] * len(pvalues)
    for rank, i in enumerate(idx):
        if pvalues[i] < alpha / (family_n - rank):  # type: ignore[operator]
            out[i] = True
        else:
            break
    return out


# --------------------------------------------------------------------------- loading
def load(results: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(results.glob("*.summary.json")):
        doc = json.loads(path.read_text())
        name = doc.get("arm")
        if not name:
            raise R3ReadError(f"{path}: no arm name")
        out[name] = doc
    if not out:
        raise R3ReadError(f"{results}: no summaries")
    return out


def _queue(doc: Dict[str, Any]) -> float:
    v = doc.get("averageQueueTime")
    if v is None:
        raise R3ReadError(f"{doc.get('arm')}: no averageQueueTime")
    return float(v)


def read(manifests: Sequence[Dict[str, Any]],
         arms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Pool every registered batch. The expected total is a constant, never inferred."""
    selected: List[Dict[str, Any]] = []
    for m in manifests:
        selected.extend(m.get("selected") or [])
    expected = R3_N_CELLS if len(manifests) == 1 else R3_N_CELLS_TOTAL
    if len(selected) != expected:
        raise R3ReadError(
            f"manifests carry {len(selected)} selected cells, registered {expected}")
    if len({e["cell"] for e in selected}) != len(selected):
        raise R3ReadError("a cell appears in more than one batch -- it would be counted twice")

    rows: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for entry in selected:
        cell = entry["cell"]
        ref = arms.get(f"{cell}__reactive")
        if ref is None:
            dropped.append({"cell": cell, "why": "no reactive arm"})
            continue
        row: Dict[str, Any] = {
            "cell": cell, "seed": entry["seed"],
            "structure": entry["structure"], "reactive_queue_s": _queue(ref),
        }
        ok = True
        for arm in ARMS:
            vals = [_queue(arms[k]) for k in
                    (f"{cell}__{arm}_s{s}" for s in (1, 2, 4, 5)) if k in arms]
            if len(vals) < R3_MIN_SEEDS_PER_CELL:
                ok = False
                row[arm] = {"verdict": "DROPPED-TOO-FEW-SEEDS", "n": len(vals)}
                continue
            row[arm] = {"n": len(vals), "median_queue_s": st.median(vals),
                        "excess_queue_s": st.median(vals) - row["reactive_queue_s"]}
        if not ok:
            dropped.append({"cell": cell, "why": "an arm had too few seeds"})
            continue
        rows.append(row)

    out: Dict[str, Any] = {
        "lineage": "scheduler_residence_v1", "stage": "R3",
        "bars": {"min_abs_rho": R3_MIN_ABS_RHO, "alpha": R3_ALPHA, "holm_n": R3_HOLM_N,
                 "min_cells_read": R3_MIN_CELLS_READ, "n_cells": len(selected),
                 "expected_sign": R3_EXPECTED_SIGN, "control_ratio": R3_CONTROL_RATIO},
        "registered_expectation": R3_REGISTERED_EXPECTATION,
        "cells": rows, "dropped": dropped, "n_read": len(rows),
    }
    if len(rows) < R3_MIN_CELLS_READ:
        out["verdict"] = "VOID-TOO-FEW-CELLS"
        return out

    tests: Dict[str, Any] = {}
    pvals: List[Optional[float]] = []
    keys = (R3_PRIMARY_KEY, R3_SECOND_KEY)
    for key in keys:
        xs = [float(r["structure"][key]) for r in rows]
        ys = [float(r["gnn"]["excess_queue_s"]) for r in rows]
        rho, p = spearman(xs, ys)
        # The control: does the same structure predict REACTIVE's own queue?
        crho, cp = spearman(xs, [float(r["reactive_queue_s"]) for r in rows])
        mrho, mp = spearman(xs, [float(r["mpoff"]["excess_queue_s"]) for r in rows])
        tests[key] = {"rho": rho, "p": p, "control_rho": crho, "control_p": cp,
                      "mpoff_rho": mrho, "mpoff_p": mp}
        pvals.append(p)
    for key, flag in zip(keys, holm(pvals, R3_HOLM_N, R3_ALPHA)):
        t = tests[key]
        t["holm_significant"] = bool(flag)
        t["fires"] = bool(
            flag and t["rho"] is not None
            and abs(t["rho"]) >= R3_MIN_ABS_RHO
            and (t["rho"] < 0) == (R3_EXPECTED_SIGN < 0)
        )
        # Registered in advance: a control that moves with the treatment voids the reading.
        t["confounded"] = bool(
            t["rho"] is not None and t["control_rho"] is not None and t["rho"] != 0
            and abs(t["control_rho"]) >= R3_CONTROL_RATIO * abs(t["rho"])
            and (t["control_rho"] < 0) == (t["rho"] < 0)
        )
    out["tests"] = tests
    primary = tests[R3_PRIMARY_KEY]
    if primary["confounded"]:
        out["verdict"] = "CONFOUNDED-ENVIRONMENT"
    elif primary["fires"]:
        out["verdict"] = "LOPSIDEDNESS-PREDICTS"
    else:
        out["verdict"] = "LOPSIDEDNESS-DOES-NOT-PREDICT"
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, action="append", required=True,
                    help="repeat for each registered batch; the read pools them")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    result = read([json.loads(m.read_text()) for m in args.manifest],
                  load(args.results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\n[R3] {result['n_read']} of {result['bars']['n_cells']} cells read"
          + (f"; dropped {len(result['dropped'])}" if result["dropped"] else ""))
    for d in result["dropped"]:
        print(f"     DROPPED {d['cell']}: {d['why']}")
    print(f"  {'cell':30s} {'min_reach':>9s} {'imbal':>6s} {'reactive':>9s} "
          f"{'gnn':>9s} {'excess':>9s} {'mpoff ex':>9s}")
    for r in result["cells"]:
        print(f"  {r['cell']:30s} {r['structure'][R3_PRIMARY_KEY]:9.1f} "
              f"{r['structure'][R3_SECOND_KEY]:6.1f} {r['reactive_queue_s']:9.3f} "
              f"{r['gnn']['median_queue_s']:9.3f} {r['gnn']['excess_queue_s']:+9.3f} "
              f"{r['mpoff']['excess_queue_s']:+9.3f}")

    if "tests" in result:
        print(f"\n[R3] Spearman vs each cell's excess queue over its OWN reactive arm  "
              f"(bar |rho| >= {R3_MIN_ABS_RHO}, sign {R3_EXPECTED_SIGN:+d}, "
              f"Holm over n={R3_HOLM_N}, alpha={R3_ALPHA})")
        for key, t in result["tests"].items():
            def f(v, nd=3):
                return "  n/a" if v is None else f"{v:+.{nd}f}"
            print(f"     {key:24s} rho {f(t['rho'])}  p {f(t['p'], 4)}"
                  f"  -> {'FIRES' if t['fires'] else 'no'}"
                  + ("  [CONFOUNDED]" if t["confounded"] else ""))
            print(f"       {'control: reactive queue':24s} rho {f(t['control_rho'])}"
                  f"  p {f(t['control_p'], 4)}")
            print(f"       {'pointwise control mpoff':24s} rho {f(t['mpoff_rho'])}"
                  f"  p {f(t['mpoff_p'], 4)}")
    print(f"\n[VERDICT] {result['verdict']}   (registered expectation: "
          f"{R3_REGISTERED_EXPECTATION})")
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
