#!/usr/bin/env python3
"""Score gnn_draw_study_v1 against its pre-registered rule (LINEAGES gnn_draw_study_v1).

The question. p5b_draw_study established that the MLP's live reliability is a property of
the training draw: collapse counts swing 0 -> 26 of 30 cells on the seed alone. Against
that measured distribution, both GNN arms scored 0/30 — but that is 2-3 draws against a
coin that lands badly about half the time, so p ~ 0.125. This scores 8 seeded GNN draws of
the deployed config over the same 30 cells, so the GNN's record can be stated as evidence
or retired the same way the MLP's was.

A sibling of score_p5b_draw_study.py rather than an edit to it: that script's grid and its
Q2 (the candidate-relative sign test) are MLP-specific. The collapse rule, its thresholds
and the LOTTERY/STABLE constants are shared verbatim and are the pre-registration — like
that script, this one takes no threshold arguments.

The MLP comparison distribution is READ FROM THE SAME SUMMARY, not hardcoded. Both arms are
then scored by one function at one threshold on one set of cells; a number typed in from a
previous session's write-up could silently drift from the data it claims to describe.

Usage:
  score_gnn_draw_study.py --summary simulation_data/gate_stats_summary.json
"""

import argparse
import json
import sys
from math import comb
from pathlib import Path

# ---- Pre-registered constants. Do not tune. ---------------------------------------
#
# Identical to score_p5b_draw_study.py, deliberately: the GNN draws are being compared
# against MLP draws scored by this exact rule, and a different threshold would compare two
# different questions.
PRIMARY_THRESHOLD_PCT = 50.0
SENSITIVITY_PCT = (30.0, 50.0, 100.0)

LOTTERY_RANGE = 5   # a range this large across seeds => draw-dominated
STABLE_RANGE = 2    # at or below this => stable

# Q2 significance. One-sided Fisher exact on clean-draw counts.
ALPHA = 0.05

# Power, computed against the FROZEN p5b comparison group (7/16 clean at +50%) BEFORE this
# study ran, and registered here because it decides what the 8-draw design can conclude:
#
#     GNN clean 8/8 -> p = 0.0087   significant
#     GNN clean 7/8 -> p = 0.0507   MISSES, by 0.0007
#     GNN clean 6/8 -> p = 0.1557
#
# So at n=8 a single unclean draw takes the study from decisive to not-established. That is
# not a reason to read 7/8 as a positive after the fact; it is a reason to have decided in
# advance what to do about it. Registered escalation, in the same shape as
# gate_statistics.py's tier ladder and for the same reason ("a run below the power table is
# VOID, not FAIL"):
#
#     ESCALATE_CLEAN_MIN..n-1 clean  -> ESCALATE to N_ESCALATED draws, verdict is VOID
#     below that                     -> NOT-ESTABLISHED, and it is a real negative
#
# At n=12 the rule tolerates 2 unclean draws (10/12 -> p = 0.0398), which is why that is the
# escalation tier rather than a larger one.
N_ESCALATED = 12
ESCALATE_CLEAN_MIN = 7

GNN_SEEDS = tuple(range(1, 9))
GNN_ARMS = {seed: f"gnndraws{seed}" for seed in GNN_SEEDS}

# The MLP draw distribution p5b_draw_study measured, by arm name, on these same cells.
MLP_ARMS = [
    f"ds{cache}{layout}s{seed}"
    for cache in ("dim14", "tempfix")
    for layout in ("dim22", "dim25cr")
    for seed in (1, 2, 3, 4)
]


def collapse_counts(summary: dict, arms, threshold: float, *, expect_cells: int = 30):
    """{arm: n_collapsed} over `expect_cells` cells, plus the per-cell margins."""
    counts, margins = {}, {}
    for arm in arms:
        n = bad = 0
        for blk in sorted(summary):
            for cell in sorted(summary[blk]):
                entry = summary[blk][cell]
                if arm not in entry:
                    continue
                if "knative" not in entry:
                    raise SystemExit(f"FAIL LOUD: {blk}/{cell} has {arm} but no knative")
                pct = 100.0 * (entry[arm]["total_rtt"] - entry["knative"]["total_rtt"]) \
                    / entry["knative"]["total_rtt"]
                margins.setdefault(arm, []).append((blk, cell, pct))
                n += 1
                bad += pct >= threshold
        if n == 0:
            raise SystemExit(f"FAIL LOUD: no results for arm {arm}")
        if n != expect_cells:
            raise SystemExit(f"FAIL LOUD: arm {arm} has {n} cells, expected {expect_cells}")
        counts[arm] = bad
    return counts, margins


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided p for the 2x2 [[a,b],[c,d]], testing that row 1 has MORE column-1 outcomes.

    Written out rather than imported: scipy in the cluster `gnn` env has been found with
    two disagreeing installs of itself (datalab-pitfalls #10), and a verdict script is the
    last place that should depend on which one loads. The arithmetic here is exact integer
    hypergeometric summation.
    """
    row1, row2 = a + b, c + d
    col1 = a + c
    total = row1 + row2
    denom = comb(total, col1)
    return sum(
        comb(row1, k) * comb(row2, col1 - k)
        for k in range(a, min(row1, col1) + 1)
    ) / denom


def evaluate(gnn_counts, mlp_counts, threshold):
    vals = [gnn_counts[GNN_ARMS[s]] for s in GNN_SEEDS]
    rng = max(vals) - min(vals)
    if rng >= LOTTERY_RANGE:
        q1 = "LOTTERY"
    elif rng <= STABLE_RANGE:
        q1 = "STABLE"
    else:
        q1 = "PARTIAL"

    # A draw is "clean" when it collapses on no cell at all. This is the property the
    # reliability claim is about — not the mean collapse count, which would let one very
    # bad draw and seven perfect ones average into a middling number.
    gnn_clean = sum(1 for v in vals if v == 0)
    mlp_vals = [mlp_counts[a] for a in MLP_ARMS]
    mlp_clean = sum(1 for v in mlp_vals if v == 0)

    p = fisher_exact_greater(
        gnn_clean, len(vals) - gnn_clean, mlp_clean, len(mlp_vals) - mlp_clean
    )
    if p < ALPHA:
        q2 = "GNN-MORE-RELIABLE"
    elif ESCALATE_CLEAN_MIN <= gnn_clean < len(vals):
        # Under-powered, not negative — the registered ladder, decided before the run.
        q2 = "ESCALATE"
    else:
        q2 = "NOT-ESTABLISHED"

    return {
        "threshold": threshold, "q1": q1, "range": rng, "gnn_counts": vals,
        "gnn_clean": gnn_clean, "n_gnn": len(vals),
        "mlp_counts": mlp_vals, "mlp_clean": mlp_clean, "n_mlp": len(mlp_vals),
        "p_value": p, "q2": q2,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    # The p5b scorer's default points at a file that does not exist; this one names the
    # file the extractor actually writes.
    ap.add_argument("--summary", type=Path,
                    default=Path("simulation_data/gate_stats_summary.json"))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())
    gnn_counts, gnn_margins = collapse_counts(summary, GNN_ARMS.values(), PRIMARY_THRESHOLD_PCT)
    mlp_counts, _ = collapse_counts(summary, MLP_ARMS, PRIMARY_THRESHOLD_PCT)
    primary = evaluate(gnn_counts, mlp_counts, PRIMARY_THRESHOLD_PCT)

    print(f"=== collapse counts (total_rtt >= +{PRIMARY_THRESHOLD_PCT:.0f}% vs same-cell knative) ===")
    print("GNN draws (deployed config, seeds 1-8):")
    print("  " + " ".join(f"s{s}={gnn_counts[GNN_ARMS[s]]:>2d}/30" for s in GNN_SEEDS))
    print(f"  range = {primary['range']}/30  (LOTTERY >= {LOTTERY_RANGE}, STABLE <= {STABLE_RANGE})"
          f"  -> Q1 = {primary['q1']}")

    print("\nMLP draws measured by p5b_draw_study, same cells, same rule:")
    print("  " + " ".join(f"{v:>2d}/30" for v in primary["mlp_counts"]))

    print(f"\nQ2 clean draws (zero collapsed cells):")
    print(f"  GNN {primary['gnn_clean']}/{primary['n_gnn']}   "
          f"MLP {primary['mlp_clean']}/{primary['n_mlp']}   "
          f"one-sided Fisher p = {primary['p_value']:.4g}  (alpha {ALPHA})")
    print(f"  -> {primary['q2']}")

    # Worst per-cell margins are what a reader will want if a draw is not clean.
    print("\nworst cell per GNN draw:")
    for s in GNN_SEEDS:
        arm = GNN_ARMS[s]
        blk, cell, pct = max(gnn_margins[arm], key=lambda r: r[2])
        print(f"  s{s}: {pct:+8.2f}%  {blk}/{cell}")

    print("\n=== sensitivity (verdict must hold at all three thresholds) ===")
    sens = []
    for t in SENSITIVITY_PCT:
        gc, _ = collapse_counts(summary, GNN_ARMS.values(), t)
        mc, _ = collapse_counts(summary, MLP_ARMS, t)
        e = evaluate(gc, mc, t)
        sens.append(e)
        print(f"  +{t:>5.0f}%  range={e['range']:2d} -> Q1={e['q1']:8s}  "
              f"clean {e['gnn_clean']}/{e['n_gnn']} vs {e['mlp_clean']}/{e['n_mlp']}  "
              f"p={e['p_value']:.4g} -> Q2={e['q2']}")
    q1_stable = len({e["q1"] for e in sens}) == 1
    q2_stable = len({e["q2"] for e in sens}) == 1

    q1 = primary["q1"] if q1_stable else "INDETERMINATE"
    q2 = primary["q2"] if q2_stable else "INDETERMINATE"
    if not q1_stable:
        print("  !! Q1 not stable across thresholds -> INDETERMINATE")
    if not q2_stable:
        print("  !! Q2 not stable across thresholds -> INDETERMINATE")

    print(f"\n=== GNN DRAW STUDY VERDICT ===")
    print(f"Q1 (is GNN reliability itself a draw lottery?): {q1}")
    print(f"Q2 (are GNN draws more reliable than MLP draws?): {q2}")
    if q1 == "STABLE" and q2 == "GNN-MORE-RELIABLE":
        print("\nThe GNN's reliability record is a property of the architecture, not of the\n"
              "draws. It can be written as a claim.")
    elif q2 == "ESCALATE":
        print(f"\nUNDER-POWERED, not negative: {primary['gnn_clean']}/{primary['n_gnn']} clean draws "
              f"gives p={primary['p_value']:.4f}.\n"
              f"The registered ladder escalates to {N_ESCALATED} draws (train seeds "
              f"{primary['n_gnn'] + 1}..{N_ESCALATED} and re-gate;\n"
              f"the existing draws stay valid and are not re-run). Verdict is VOID until then —\n"
              f"do NOT report this as either result.")
    elif q1 == "LOTTERY":
        print("\nGNN reliability is ALSO a draw lottery. The 0/30 arms were lucky draws, and\n"
              "the claim retires the same way the MLP's did. This closes the last open\n"
              "empirical question in favour of the program's terminal negative.")
    else:
        print("\nMixed. Report the table; do not summarise it as either result.")

    if args.json_out:
        rep = {
            "criterion": f"total_rtt >= +{PRIMARY_THRESHOLD_PCT}% vs same-cell knative",
            "clean_draw_definition": "a draw with zero collapsed cells out of 30",
            "q1": q1, "q2": q2,
            "q1_threshold_stable": q1_stable, "q2_threshold_stable": q2_stable,
            "alpha": ALPHA,
            "gnn_counts": {f"s{s}": gnn_counts[GNN_ARMS[s]] for s in GNN_SEEDS},
            "gnn_range": primary["range"],
            "gnn_clean": primary["gnn_clean"], "n_gnn": primary["n_gnn"],
            "mlp_counts": dict(zip(MLP_ARMS, primary["mlp_counts"])),
            "mlp_clean": primary["mlp_clean"], "n_mlp": primary["n_mlp"],
            "p_value": primary["p_value"],
            "sensitivity": [
                {"threshold": e["threshold"], "q1": e["q1"], "q2": e["q2"],
                 "range": e["range"], "gnn_clean": e["gnn_clean"],
                 "mlp_clean": e["mlp_clean"], "p_value": e["p_value"]}
                for e in sens
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
