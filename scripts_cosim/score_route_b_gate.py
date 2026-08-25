#!/usr/bin/env python3
"""route_b_v1 — the registered gate, evaluated exactly as written in LINEAGES.md.

Takes the frozen per-dataset reports for Arm S and Arm B0 (written by
score_route_b_contention.py --include-per-dataset) and prints the registered
PASS / FAIL / VOID row for the primary cell: Arm S, tight alpha = 2.0, objective rtt.
Thresholds are constants here on purpose — this script takes no threshold arguments,
the same discipline as score_p5b_collapse_pairs.py.

Registered conditions (pre-registration + the disclosed amendments A1-A3):
  PASS iff ALL of
    (1) frac(R_exact > 5%) >= 0.10 with the 95% Wilson CI excluding 0.10 from below
    (2) median repair_fraction < 0.5 for BOTH count repairs over firing datasets
        (saturated repairs excluded and counted)
    (3') unconstrained rung fires R_exact > 5% on < 2% of datasets AND each binding
        rung fires above the unconstrained rung
    (4) spread-view firing fraction nonzero at the tight rung
  FAIL iff CI excludes 0.10 from above, OR condition (2) fails
  VOID iff Arm B0 fires R_exact > 5% on > 2% of datasets, OR the CI straddles 0.10
       (escalate n 200 -> 400 -> 800; ladder exhausted -> VOID-UNDERPOWERED)
"""

from __future__ import annotations

import argparse
import json
import math
import sys

TIGHT, LOOSE = "2.0", "3.0"
MATERIAL_PCT = 5.0
PASS_FRAC = 0.10
B0_VOID_FRAC = 0.02
UNCONSTRAINED_MAX_FRAC = 0.02
REPAIR_MAX = 0.5
LADDER = (200, 400, 800)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def scored_rows(report: dict, alpha_key: str):
    rows = report["per_dataset"][alpha_key]
    usable = [r for r in rows
              if not r.get("no_feasible_rows") and "r_exact_pct" in r]
    dropped = len(rows) - len(usable)
    return usable, dropped


def firing_frac(rows):
    k = sum(1 for r in rows if r["r_exact_pct"] > MATERIAL_PCT)
    return k, len(rows), (k / len(rows) if rows else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-s", required=True, help="Arm S report JSON (rtt)")
    ap.add_argument("--arm-b0", required=True, help="Arm B0 report JSON (rtt)")
    args = ap.parse_args()

    rep_s = json.load(open(args.arm_s))[0]
    rep_b0 = json.load(open(args.arm_b0))[0]

    verdict = None
    reasons = []

    # --- primary statistic, Arm S tight -----------------------------------
    rows_tight, dropped_tight = scored_rows(rep_s, TIGHT)
    k, n, frac = firing_frac(rows_tight)
    lo, hi = wilson_ci(k, n)
    print(f"Arm S tight (alpha={TIGHT}): {k}/{n} datasets with R_exact > "
          f"{MATERIAL_PCT}%  frac={frac:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
          + (f"  ({dropped_tight} datasets without feasible rows)" if dropped_tight
             else ""))

    cond1 = frac >= PASS_FRAC and lo > PASS_FRAC
    fail_frac = hi < PASS_FRAC
    straddle = not cond1 and not fail_frac

    # --- condition 2: repairs ---------------------------------------------
    firing = [r for r in rows_tight if r["r_exact_pct"] > MATERIAL_PCT]
    cond2, repair_stats = True, {}
    for name in ("1int", "kint"):
        fractions, saturated = [], 0
        for r in firing:
            repaired = r.get(f"r_exact_repaired_{name}_pct")
            if r.get(f"repair_{name}_saturated") or repaired is None:
                saturated += 1
                continue
            fractions.append(1.0 - repaired / r["r_exact_pct"])
        med = sorted(fractions)[len(fractions) // 2] if fractions else 0.0
        repair_stats[name] = (med, len(fractions), saturated)
        print(f"repair {name}: median repair_fraction={med:.3f} over "
              f"{len(fractions)} firing datasets ({saturated} saturated, excluded)")
        if fractions and med >= REPAIR_MAX:
            cond2 = False
            reasons.append(f"repair {name} closes {med:.0%} >= 50% — count-shaped")

    # --- condition 3': free-choice attribution ----------------------------
    rows_unc, _ = scored_rows(rep_s, "None")
    ku, nu, frac_u = firing_frac(rows_unc)
    rows_loose, _ = scored_rows(rep_s, LOOSE)
    kl, nl, frac_l = firing_frac(rows_loose)
    cond3 = frac_u < UNCONSTRAINED_MAX_FRAC and frac > frac_u and frac_l > frac_u
    print(f"attribution: unconstrained {ku}/{nu} ({frac_u:.3f}), "
          f"loose {kl}/{nl} ({frac_l:.3f}), tight {frac:.3f}  -> "
          f"{'ok' if cond3 else 'FAILS'}")

    # --- condition 4: spread view -----------------------------------------
    spread_firing = sum(1 for r in rows_tight
                        if r.get("r_exact_spread_pct", 0.0) > MATERIAL_PCT)
    spread_n = sum(1 for r in rows_tight if "r_exact_spread_pct" in r)
    cond4 = spread_firing > 0
    print(f"spread view (tight): {spread_firing}/{spread_n} firing -> "
          f"{'ok' if cond4 else 'FAILS'}")

    # --- Arm B0 validity ---------------------------------------------------
    rows_b0, _ = scored_rows(rep_b0, TIGHT)
    kb, nb, frac_b = firing_frac(rows_b0)
    b0_void = frac_b > B0_VOID_FRAC
    print(f"Arm B0 tight: {kb}/{nb} ({frac_b:.3f}) with R_exact > {MATERIAL_PCT}% -> "
          f"{'VOID (instrumentation suspect)' if b0_void else 'ok'}")

    # --- verdict -----------------------------------------------------------
    if b0_void:
        verdict = "VOID"
        reasons.append(f"Arm B0 fires {frac_b:.1%} > {B0_VOID_FRAC:.0%}")
    elif straddle:
        next_rung = next((r for r in LADDER if r > n), None)
        if next_rung:
            verdict = "VOID"
            reasons.append(f"CI straddles {PASS_FRAC}: escalate to n={next_rung}")
        else:
            verdict = "VOID-UNDERPOWERED"
            reasons.append("power ladder exhausted")
    elif fail_frac:
        verdict = "FAIL"
        reasons.append(f"CI [{lo:.3f}, {hi:.3f}] excludes {PASS_FRAC} from above")
    elif not cond2:
        verdict = "FAIL"
    elif cond1 and cond3 and cond4:
        verdict = "PASS"
    else:
        # cond1 held with CI clear, but 3'/4 failed — the amended registration maps
        # this to FAIL (the effect exists but is not attributable as registered).
        verdict = "FAIL"
        if not cond3:
            reasons.append("free-choice attribution (3') failed")
        if not cond4:
            reasons.append("spread view (4) empty — collision-channel only")

    print(f"\nVERDICT: {verdict}" + (f"  [{'; '.join(reasons)}]" if reasons else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
