#!/usr/bin/env python3
"""Score the P5b draw study against its pre-registered rule (LINEAGES p5b_draw_study).

The question P5b could not answer: when the candidate-relative feature moved collapse
7/30 -> 2/30 on one cache and 7/30 -> 17/30 on the other, was that the cache, or the
training draw? Both arms passed --random-state 42 -- and nothing seeded torch, so their
weight inits were drawn from OS entropy anyway.

This scores a 4x4 grid ({dim14, tempfix} x {dim22, dim25cr} x seeds 1-4, 30 cells each)
under the criterion registered before it ran. Like score_p5b_collapse_pairs.py it takes no
threshold arguments: the thresholds ARE the pre-registration.

Usage:
  score_p5b_draw_study.py --summary simulation_data/draw_study_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

# ---- Pre-registered constants. Do not tune. ---------------------------------------
#
# Collapse is scored on total_rtt, NOT on chosen_queue_vs_min p95: that detector is
# measured-invalid for candidate-relative arms (it fires on 5/30 healthy mlpcandrel cells,
# one of them a -2.5% win, because the CR columns make non-min-queue choice deliberate).
PRIMARY_THRESHOLD_PCT = 50.0
SENSITIVITY_PCT = (30.0, 50.0, 100.0)

LOTTERY_RANGE = 5   # a within-condition range this large => draw-dominated
STABLE_RANGE = 2    # every condition at or below this => stable

CACHES = ("dim14", "tempfix")
LAYOUTS = ("dim22", "dim25cr")
SEEDS = (1, 2, 3, 4)


def arm_name(cache: str, layout: str, seed: int) -> str:
    return f"ds{cache}{layout}s{seed}"


def collapse_counts(summary: dict, threshold: float):
    """{(cache, layout, seed): (n_collapsed, n_cells)} plus the per-cell margins."""
    out, margins = {}, {}
    for cache in CACHES:
        for layout in LAYOUTS:
            for seed in SEEDS:
                arm = arm_name(cache, layout, seed)
                n = bad = 0
                for blk in sorted(summary):
                    for cell in sorted(summary[blk]):
                        e = summary[blk][cell]
                        if arm not in e:
                            continue
                        if "knative" not in e:
                            raise SystemExit(f"FAIL LOUD: {blk}/{cell} has {arm} but no knative")
                        pct = 100.0 * (e[arm]["total_rtt"] - e["knative"]["total_rtt"]) \
                            / e["knative"]["total_rtt"]
                        margins.setdefault((cache, layout, seed), []).append((blk, cell, pct))
                        n += 1
                        bad += pct >= threshold
                if n == 0:
                    raise SystemExit(f"FAIL LOUD: no results for arm {arm}")
                if n != 30:
                    raise SystemExit(f"FAIL LOUD: arm {arm} has {n} cells, expected 30")
                out[(cache, layout, seed)] = (bad, n)
    return out, margins


def evaluate(counts, threshold):
    # Q1 -- within-condition spread across seeds.
    ranges = {}
    for cache in CACHES:
        for layout in LAYOUTS:
            vals = [counts[(cache, layout, s)][0] for s in SEEDS]
            ranges[(cache, layout)] = (min(vals), max(vals), max(vals) - min(vals), vals)
    worst = max(r[2] for r in ranges.values())
    if worst >= LOTTERY_RANGE:
        q1 = "LOTTERY"
    elif all(r[2] <= STABLE_RANGE for r in ranges.values()):
        q1 = "STABLE"
    else:
        q1 = "PARTIAL"

    # Q2 -- sign of the candrel effect, paired within (cache, seed).
    deltas = {
        (cache, seed): counts[(cache, "dim25cr", seed)][0] - counts[(cache, "dim22", seed)][0]
        for cache in CACHES for seed in SEEDS
    }
    pos = {c: all(deltas[(c, s)] > 0 for s in SEEDS) for c in CACHES}
    neg = {c: all(deltas[(c, s)] < 0 for s in SEEDS) for c in CACHES}
    separated = (pos[CACHES[0]] and neg[CACHES[1]]) or (neg[CACHES[0]] and pos[CACHES[1]])
    q2 = "CACHE-DETERMINED" if separated else "DRAW-DOMINATED"

    # Q3 -- descriptive only.
    pooled = {
        layout: sorted(counts[(c, layout, s)][0] for c in CACHES for s in SEEDS)
        for layout in LAYOUTS
    }
    return {"threshold": threshold, "ranges": ranges, "q1": q1, "worst_range": worst,
            "deltas": deltas, "q2": q2, "pooled": pooled}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path,
                    default=Path("simulation_data/draw_study_summary.json"))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())
    primary_counts, margins = collapse_counts(summary, PRIMARY_THRESHOLD_PCT)
    primary = evaluate(primary_counts, PRIMARY_THRESHOLD_PCT)

    print(f"=== collapse counts (total_rtt >= +{PRIMARY_THRESHOLD_PCT:.0f}% vs same-cell knative) ===")
    print(f"{'condition':22s} " + " ".join(f"{'s' + str(s):>6s}" for s in SEEDS)
          + f" {'range':>7s}")
    for cache in CACHES:
        for layout in LAYOUTS:
            lo, hi, rng, vals = primary["ranges"][(cache, layout)]
            flag = "  <-- >= LOTTERY_RANGE" if rng >= LOTTERY_RANGE else ""
            print(f"{cache + '/' + layout:22s} "
                  + " ".join(f"{v:>4d}/30" for v in vals) + f" {rng:>7d}{flag}")

    print(f"\nQ1 within-condition spread: worst range = {primary['worst_range']}/30 "
          f"(LOTTERY >= {LOTTERY_RANGE}, STABLE <= {STABLE_RANGE}) -> {primary['q1']}")

    print("\nQ2 candrel effect, paired within (cache, seed): "
          "delta = collapses(dim25cr) - collapses(dim22)")
    for cache in CACHES:
        ds = [primary["deltas"][(cache, s)] for s in SEEDS]
        signs = "".join("+" if d > 0 else ("-" if d < 0 else "0") for d in ds)
        print(f"  {cache:10s} " + " ".join(f"{d:+4d}" for d in ds) + f"   signs={signs}")
    print(f"  -> {primary['q2']}")

    print("\nQ3 (descriptive) pooled collapse counts over caches and seeds:")
    for layout in LAYOUTS:
        v = primary["pooled"][layout]
        print(f"  {layout:8s} n=8  min={v[0]:2d}  median={(v[3] + v[4]) / 2:4.1f}  "
              f"max={v[-1]:2d}   {v}")

    # Sensitivity: the registered rule requires the verdicts to hold at all three.
    print("\n=== sensitivity (verdict must hold at all three thresholds) ===")
    sens = []
    for t in SENSITIVITY_PCT:
        c, _ = collapse_counts(summary, t)
        e = evaluate(c, t)
        sens.append(e)
        print(f"  +{t:>5.0f}%  worst_range={e['worst_range']:2d} -> Q1={e['q1']:8s}  Q2={e['q2']}")
    q1_stable = len({e["q1"] for e in sens}) == 1
    q2_stable = len({e["q2"] for e in sens}) == 1

    q1 = primary["q1"] if q1_stable else "INDETERMINATE"
    q2 = primary["q2"] if q2_stable else "INDETERMINATE"
    if not q1_stable:
        print("  !! Q1 not stable across thresholds -> INDETERMINATE")
    if not q2_stable:
        print("  !! Q2 not stable across thresholds -> INDETERMINATE")

    print(f"\n=== DRAW STUDY VERDICT ===\nQ1 (is reliability a draw lottery?): {q1}")
    print(f"Q2 (is the candrel sign cache-determined?): {q2}")
    if q1 == "LOTTERY":
        print("\nPointwise reliability on this benchmark is a property of the training draw.\n"
              "The P5b 7->2 / 7->17 split is not attributable to the feature or the cache,\n"
              "and neither is the 7/30-vs-7/30 result the architectural reading rested on.\n"
              "The GNN's 0/120 record survives as the only reliability claim -- and now needs\n"
              "its own multi-seed check before it can be written either.")
    elif q1 == "STABLE" and q2 == "CACHE-DETERMINED":
        print("\nThe feature effect is real and attributable to the corpus. P5b's split was a\n"
              "genuine cache interaction, not noise.")
    else:
        print("\nMixed. Report the table; do not summarise it as either result.")

    if args.json_out:
        rep = {
            "criterion": f"total_rtt >= +{PRIMARY_THRESHOLD_PCT}% vs same-cell knative",
            "q1": q1, "q2": q2,
            "q1_threshold_stable": q1_stable, "q2_threshold_stable": q2_stable,
            "counts": {f"{c}/{l}/s{s}": primary_counts[(c, l, s)][0]
                       for c in CACHES for l in LAYOUTS for s in SEEDS},
            "ranges": {f"{c}/{l}": primary["ranges"][(c, l)][:3] for c in CACHES for l in LAYOUTS},
            "deltas": {f"{c}/s{s}": primary["deltas"][(c, s)] for c in CACHES for s in SEEDS},
            "sensitivity": [{"threshold": e["threshold"], "q1": e["q1"], "q2": e["q2"],
                             "worst_range": e["worst_range"]} for e in sens],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
