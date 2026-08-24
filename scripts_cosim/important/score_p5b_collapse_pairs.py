#!/usr/bin/env python3
"""Score the P5b candidate-relative control against its pre-registered decision rule.

P5b (program_verdict_v1) asks whether the MLP's collapse is architectural or just a
missing feature. Two paired arms, each differing from its baseline in the feature set
alone:

    mlpcandrel   vs mlp          (graphs_cache_full_corpus_siv1_dim14)
    mlpcandreltf vs mlptempfix   (graphs_cache_full_corpus_siv1_dim14_tempfix)

Everything registered below was fixed in LINEAGES.md BEFORE the gate array was submitted.
This script only applies it — it takes no thresholds from the command line and has no
options that could change the verdict, because the whole point of the exercise is that
the 30/30 record it is testing was scored under a rule written after the numbers were in.

Reads the summary written by extract_gate_stats_summary.py (never the 80MB results).

Usage:
  score_p5b_collapse_pairs.py --summary simulation_data/gate_stats_summary.json
"""

import argparse
import json
import sys
from math import comb
from pathlib import Path

# ---- Pre-registered constants. Do not tune. ---------------------------------------
#
# Detector: chosen_queue_vs_min p95 from the .decode_stats.json sidecar. Measured on all
# 120 runs of the exploratory campaign: collapse 13,485-23,866, healthy 449-1,387 -- a
# 9.7x gap with no overlap. The median is normal in both, which is the direct evidence
# that collapse is a minority-of-decisions tail that compounds.
DETECTOR = ("chosen_queue_vs_min", "p95")
PRIMARY_THRESHOLD = 5_000.0
SENSITIVITY_THRESHOLDS = (2_000.0, 5_000.0, 10_000.0)
# No run has ever landed between these. A cell that does is reported, because the
# detector's separation is an empirical fact about 120 runs, not a law.
UNOBSERVED_BAND = (1_387.0, 13_485.0)

ALPHA = 0.05
HARDEN_MIN_RETAINED = 5   # of the baseline's collapses still collapsing
HARDEN_MIN_TOTAL = 5      # total collapses out of 30

PAIRS = [("mlpcandrel", "mlp"), ("mlpcandreltf", "mlptempfix")]


def binom_sf(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binom(n, 0.5) — exact one-sided McNemar."""
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)


def detector_value(entry: dict, where: str) -> float:
    ds = entry.get("decode_stats")
    if not ds:
        raise SystemExit(
            f"FAIL LOUD: {where} has no decode_stats sidecar — the pre-registered detector "
            f"is unmeasurable for this run and it cannot be silently dropped from the pairing."
        )
    block = ds.get(DETECTOR[0])
    if not block or DETECTOR[1] not in block:
        raise SystemExit(f"FAIL LOUD: {where} decode_stats has no {'.'.join(DETECTOR)}")
    return float(block[DETECTOR[1]])


def collect_pairs(summary: dict, arm: str, baseline: str):
    """[(block, cell, arm_p95, baseline_p95)] over every cell present in BOTH arms."""
    rows, missing = [], []
    for block in sorted(summary):
        for cell in sorted(summary[block]):
            arms = summary[block][cell]
            if arm not in arms and baseline not in arms:
                continue
            if arm not in arms or baseline not in arms:
                missing.append(f"{block}/{cell}: has {sorted(set(arms) & {arm, baseline})}")
                continue
            rows.append((
                block, cell,
                detector_value(arms[arm], f"{block}/{cell}/{arm}"),
                detector_value(arms[baseline], f"{block}/{cell}/{baseline}"),
            ))
    return rows, missing


def evaluate(rows, threshold: float):
    b = c = both = neither = 0
    for _blk, _cell, arm_v, base_v in rows:
        arm_bad, base_bad = arm_v >= threshold, base_v >= threshold
        if base_bad and not arm_bad:
            b += 1
        elif arm_bad and not base_bad:
            c += 1
        elif arm_bad and base_bad:
            both += 1
        else:
            neither += 1
    p = binom_sf(b, b + c)
    base_total, arm_total = b + both, c + both
    if p <= ALPHA and b > c:
        verdict = "REFUTE"
    elif both >= HARDEN_MIN_RETAINED and arm_total >= HARDEN_MIN_TOTAL and p > ALPHA:
        verdict = "HARDEN"
    else:
        verdict = "INDETERMINATE"
    return {
        "threshold": threshold, "b_fixed": b, "c_broken": c, "both_collapsed": both,
        "neither": neither, "baseline_collapses": base_total, "arm_collapses": arm_total,
        "n_pairs": len(rows), "p_one_sided": p, "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path,
                    default=Path("simulation_data/gate_stats_summary.json"))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())
    report = {
        "detector": ".".join(DETECTOR),
        "primary_threshold": PRIMARY_THRESHOLD,
        "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
        "alpha": ALPHA,
        "pairs": {},
    }
    verdicts = []

    for arm, baseline in PAIRS:
        rows, missing = collect_pairs(summary, arm, baseline)
        print(f"\n=== {arm} vs {baseline} ===")
        if missing:
            for m in missing:
                print(f"  FAIL LOUD (unpaired): {m}")
            raise SystemExit(
                f"FAIL LOUD: {len(missing)} cell(s) present in one arm only. A paired test "
                f"cannot drop half a pair — rerun the missing tasks."
            )
        if not rows:
            raise SystemExit(f"FAIL LOUD: no cells found for {arm} / {baseline}")

        print(f"{'block/cell':46s} {baseline:>12s} {arm:>12s}   pair")
        for blk, cell, arm_v, base_v in rows:
            arm_bad = arm_v >= PRIMARY_THRESHOLD
            base_bad = base_v >= PRIMARY_THRESHOLD
            tag = {(True, False): "FIXED", (False, True): "BROKEN",
                   (True, True): "both-collapse", (False, False): "both-healthy"}[
                (base_bad, arm_bad)]
            band = ""
            for label, v in ((baseline, base_v), (arm, arm_v)):
                if UNOBSERVED_BAND[0] < v < UNOBSERVED_BAND[1]:
                    band += f"  !! {label} p95={v:,.0f} in the unobserved band"
            print(f"{blk + '/' + cell:46s} {base_v:12,.0f} {arm_v:12,.0f}   {tag}{band}")

        primary = evaluate(rows, PRIMARY_THRESHOLD)
        sens = [evaluate(rows, t) for t in SENSITIVITY_THRESHOLDS]
        stable = len({s["verdict"] for s in sens}) == 1
        pair_verdict = primary["verdict"] if stable else "INDETERMINATE"

        print(
            f"\n  baseline collapses {primary['baseline_collapses']}/{primary['n_pairs']}, "
            f"{arm} {primary['arm_collapses']}/{primary['n_pairs']}  "
            f"(fixed={primary['b_fixed']} broken={primary['c_broken']} "
            f"both={primary['both_collapsed']})"
        )
        print(f"  exact one-sided McNemar p = {primary['p_one_sided']:.4f} @ threshold "
              f"{PRIMARY_THRESHOLD:,.0f} -> {primary['verdict']}")
        print("  sensitivity: " + "  ".join(
            f"{s['threshold']:,.0f}->{s['verdict']}(p={s['p_one_sided']:.3f})" for s in sens))
        if not stable:
            print("  !! verdict NOT stable across thresholds -> INDETERMINATE")
        print(f"  PAIR VERDICT: {pair_verdict}")

        report["pairs"][arm] = {
            "baseline": baseline, "primary": primary, "sensitivity": sens,
            "threshold_stable": stable, "verdict": pair_verdict,
            "cells": [
                {"block": b, "cell": c, "arm_p95": a, "baseline_p95": v}
                for b, c, a, v in rows
            ],
        }
        verdicts.append(pair_verdict)

    if all(v == "REFUTE" for v in verdicts):
        overall = "REFUTE"
        note = ("A candidate-relative queue feature stops the collapse. The reliability "
                "separation is FEATURE ENGINEERING, not architecture — the paper claim "
                "must shrink accordingly.")
    elif all(v == "HARDEN" for v in verdicts):
        overall = "HARDEN"
        note = ("The collapse survives being handed the set-relative view directly. The "
                "architectural reading of the GNN's reliability advantage stands.")
    else:
        overall = "INDETERMINATE"
        note = ("The two paired arms do not agree, or a verdict was threshold-sensitive. "
                "Report as indeterminate — do not read it as either result.")

    report["overall_verdict"] = overall
    report["note"] = note
    print(f"\n=== P5b VERDICT: {overall} ===\n{note}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
