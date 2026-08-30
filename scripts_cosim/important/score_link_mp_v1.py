#!/usr/bin/env python3
"""Score link_mp_v1 against its registration.

Registration: docs/lineages/link_mp_v1.md, written BEFORE any arm was trained. As with
score_mp_ablation.py, the constants here ARE the registration: this script takes no tuning
arguments and refuses to compute a verdict until the code/venue identity of every arm is
asserted from the summary itself.

The question, in brief: mp_ablation_v1 could not distinguish "message passing is unhelpful"
from "message passing runs over the WRONG graph" (bipartite + same-node edges cannot
express shared-link contention). This lineage builds the link-aware graph (network-entity
contract core_v1: physical nodes + core links + route edges, src/placement/network_graph.py)
and asks the paired question again, on a corpus whose backbone bandwidth BINDS.

Design:
  - Three arm families, 16 seeds each, all trained on the SAME binding-backbone corpus at
    the pinned commit, differing in exactly one factor each:
      lgon     core_v1 network entities in the message-passing graph (MP ON)
      lgctrl   MP ON over the old graph (no network entities)      -> attribution control
      lgmpoff  MP OFF (GNN_DISABLE_MESSAGE_PASSING=1 train+serve)  -> the pointwise bar
  - Gated on the 20 BACKBONE cells only. A core_v1 model fails loud on a fabric-less
    graph by design, and the latency effect this lineage chases lives on backbone cells
    (objective_pivot_v1: -25.1% backbone vs +2.5% flat).
  - PRIMARY (directional, registered one-sided): exact Wilcoxon signed-rank over the 16
    per-seed differences in mean margin vs same-cell Knative, H1 = lgon margin is LOWER
    (better) than lgmpoff margin. One-sided because the hypothesis is directional and was
    fixed before any data existed; the opposite tail is still reported and has its own
    registered verdict (OPPOSITE_DIRECTION), so a harm cannot hide behind the sidedness.
  - SECONDARY S1 (attribution): same statistic, lgon vs lgctrl. Only a PASS here licenses
    crediting the LINK ENTITIES specifically rather than "message passing on this corpus".
  - SECONDARY S2 (reliability): paired sign test on severe-collapse counts (>= +50%),
    lgon vs lgmpoff, non-tied pairs only; ties expected and reported.
  - VOID: any arm whose recorded provenance mismatches its family's pin row below, or
    that is missing any of its 20 cells. A VOID is not a FAIL; fix and re-run the arm.
"""

import argparse
import json
import sys
from itertools import product
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_gnn_draw_study import collapse_counts  # noqa: E402

# ---- Registered constants. Do not tune. -------------------------------------------
SEEDS = tuple(range(1, 17))
LGON = {s: f"lgon{s}" for s in SEEDS}
LGCTRL = {s: f"lgctrl{s}" for s in SEEDS}
LGMPOFF = {s: f"lgmpoff{s}" for s in SEEDS}

PRIMARY_ALPHA = 0.05
SEVERE_PCT = 50.0
DESCRIPTIVE_PCT = (30.0, 100.0)
N_CELLS = 20

BACKBONE_BLOCKS = ("drawgate/backbone", "promo175/backbone",
                   "bbrob/bb_core8_bw1p5", "bbrob/bb_core4_bw0p5")

# Set by the registration amendment that pins the arms' commit; None refuses to score.
PIN_COMMIT = "8aef27a98fd636000008468d75a52d645f999969"

_COMMON_PROVENANCE = {
    "code_dirty": False,
    "INFERENCE_FEATURE_LAYOUT": "dim22",
    "QUEUE_FEATURE_CONTRACT": "scale_invariant_v1",
    "warmth_physics": "node_disk_v2",
    "GNN_DECODE_MODE": "argmax",
    "GNN_BATCH_SIZE": "4",
    "GNN_BATCH_TIMEOUT": "0.002",
    "HEROSIM_GNN_DEVICE": "cpu",
    "GNN_MP_NODE_EDGES": None,
}
# Per-family levers. Serving a graph the checkpoint was not trained on is the 2026-08-16
# 12.4x failure; each family's lever is asserted, never assumed, and each family must
# ALSO assert the other families' levers are absent.
FAMILY_PROVENANCE = {
    "lgon": {**_COMMON_PROVENANCE,
             "NETWORK_GRAPH_CONTRACT": "core_v1",
             "GNN_DISABLE_MESSAGE_PASSING": None},
    "lgctrl": {**_COMMON_PROVENANCE,
               "NETWORK_GRAPH_CONTRACT": None,
               "GNN_DISABLE_MESSAGE_PASSING": None},
    "lgmpoff": {**_COMMON_PROVENANCE,
                "NETWORK_GRAPH_CONTRACT": None,
                "GNN_DISABLE_MESSAGE_PASSING": "1"},
}


def assert_provenance(summary: dict) -> list:
    """[(arm, reasons)] for arms failing the pin; empty means all 48 arms are clean."""
    if PIN_COMMIT is None:
        return [("<registration>", {"pin": ["PIN_COMMIT is None: the registration has "
                                            "not been amended with the arms' commit"]})]
    void = []
    for family, arms in (("lgon", LGON), ("lgctrl", LGCTRL), ("lgmpoff", LGMPOFF)):
        required = dict(FAMILY_PROVENANCE[family])
        required["code_commit"] = PIN_COMMIT
        for seed in SEEDS:
            arm = arms[seed]
            mismatches, n_cells = {}, 0
            for blk in sorted(summary):
                for cell in sorted(summary[blk]):
                    entry = summary[blk][cell]
                    if arm not in entry:
                        continue
                    n_cells += 1
                    prov = entry[arm].get("provenance") or {}
                    for key, want in required.items():
                        got = prov.get(key)
                        if got != want:
                            mismatches.setdefault(key, set()).add(f"{blk}/{cell}: {got!r}")
            if n_cells == 0:
                void.append((arm, {"missing": ["no cells in summary"]}))
            elif n_cells != N_CELLS:
                void.append((arm, {"cells": [f"{n_cells} cells, expected {N_CELLS}"]}))
            elif mismatches:
                void.append((arm, {k: sorted(v)[:3] for k, v in mismatches.items()}))
    return void


def mean_margin(summary: dict, arm: str) -> float:
    """Mean per-cell % margin vs same-cell Knative over every cell the arm has."""
    pcts = []
    for blk in sorted(summary):
        for cell in sorted(summary[blk]):
            entry = summary[blk][cell]
            if arm not in entry:
                continue
            if "knative" not in entry:
                raise SystemExit(f"FAIL LOUD: {blk}/{cell} has {arm} but no knative")
            kn = entry["knative"]["total_rtt"]
            pcts.append(100.0 * (entry[arm]["total_rtt"] - kn) / kn)
    if not pcts:
        raise SystemExit(f"FAIL LOUD: arm {arm} has no cells")
    return sum(pcts) / len(pcts)


def _signed_ranks(diffs):
    """Wilcoxon W+ : sum of ranks of |d| over positive d, midranks for tied |d|."""
    nz = [d for d in diffs if d != 0.0]
    mags = sorted(range(len(nz)), key=lambda i: abs(nz[i]))
    ranks = [0.0] * len(nz)
    i = 0
    while i < len(mags):
        j = i
        while j + 1 < len(mags) and abs(nz[mags[j + 1]]) == abs(nz[mags[i]]):
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[mags[k]] = r
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    return w_plus, ranks


def wilcoxon_exact_one_sided_greater(diffs):
    """Exact one-sided P(W+ >= obs) by full 2^n enumeration (n <= 16).

    `diffs` must be oriented so that the REGISTERED hypothesis predicts positive values;
    the returned p is small exactly when the data land on the hypothesized side.
    """
    w_obs, ranks = _signed_ranks(diffs)
    n = len(ranks)
    if n == 0:
        return 1.0, w_obs, 0
    hits = 0
    for signs in product((0, 1), repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s)
        if w >= w_obs - 1e-12:
            hits += 1
    return hits / (2 ** n), w_obs, n


def sign_test_exact(n_pos: int, n_eff: int) -> float:
    """One-sided P(X >= n_pos) for X ~ Binomial(n_eff, 1/2)."""
    if n_eff == 0:
        return 1.0
    return sum(comb(n_eff, i) for i in range(n_pos, n_eff + 1)) / 2 ** n_eff


def paired_wilcoxon(summary, better_arms, worse_arms, label):
    """One-sided Wilcoxon: H1 = `better_arms` has LOWER margin than `worse_arms`."""
    mb = {s: mean_margin(summary, better_arms[s]) for s in SEEDS}
    mw = {s: mean_margin(summary, worse_arms[s]) for s in SEEDS}
    diffs = [mw[s] - mb[s] for s in SEEDS]  # >0 supports H1 (the "worse" arm is worse)
    p_h1, w_obs, n_nz = wilcoxon_exact_one_sided_greater(diffs)
    p_opp, _, _ = wilcoxon_exact_one_sided_greater([-d for d in diffs])
    print(f"\n{label}")
    print("  seed:        " + " ".join(f"s{s}" for s in SEEDS))
    print("  hypothesized-better: " + " ".join(f"{mb[s]:+.1f}" for s in SEEDS))
    print("  hypothesized-worse:  " + " ".join(f"{mw[s]:+.1f}" for s in SEEDS))
    print("  diff (worse-better): " + " ".join(f"{d:+.1f}" for d in diffs))
    print(f"  mean diff = {sum(diffs)/len(diffs):+.2f} pp (positive supports H1)   "
          f"W+ = {w_obs:.1f}, n_nonzero = {n_nz}")
    print(f"  one-sided exact p(H1) = {p_h1:.5f}   opposite tail p = {p_opp:.5f}")
    return {"better_margin": mb, "worse_margin": mw, "diffs": diffs,
            "mean_diff_pp": sum(diffs) / len(diffs), "w_plus": w_obs,
            "n_nonzero": n_nz, "p_one_sided_h1": p_h1, "p_one_sided_opposite": p_opp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path,
                    default=Path("simulation_data/gate_stats_summary.json"))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    summary = json.loads(args.summary.read_text())

    void = assert_provenance(summary)
    if void:
        print("=== VOID: arms failed the pin (registration §Design/VOID) ===")
        for arm, why in void:
            print(f"  {arm}:")
            for key, ex in sorted(why.items()):
                print(f"    {key}: " + "; ".join(ex))
        print("\nFix and re-run the named arms. NO verdict computed; a VOID is not a FAIL.")
        return 2
    print(f"[PIN] all 48 arms verified: {PIN_COMMIT[:12]}, clean, per-family levers "
          f"asserted — proceeding")

    report = {"pin_commit": PIN_COMMIT, "alpha": PRIMARY_ALPHA, "n_pairs": len(SEEDS),
              "n_cells": N_CELLS}

    # ---- PRIMARY: lgon vs lgmpoff, one-sided (registered direction). --------------
    report["primary"] = paired_wilcoxon(
        summary, LGON, LGMPOFF,
        "PRIMARY — link-graph MP-ON vs MP-OFF, per-seed mean margin vs Knative "
        "(20 backbone cells)")
    # ---- SECONDARY S1: lgon vs lgctrl (attribution). ------------------------------
    report["s1_attribution"] = paired_wilcoxon(
        summary, LGON, LGCTRL,
        "S1 (attribution) — link-graph MP-ON vs old-graph MP-ON")
    # Context only, not part of any verdict: does plain MP still lose to MP-OFF here?
    report["context_ctrl_vs_mpoff"] = paired_wilcoxon(
        summary, LGMPOFF, LGCTRL,
        "CONTEXT [not a verdict input] — MP-OFF vs old-graph MP-ON (mp_ablation_v1's "
        "question, on this corpus)")

    # ---- SECONDARY S2: reliability, lgon vs lgmpoff. ------------------------------
    on_arms = [LGON[s] for s in SEEDS]
    off_arms = [LGMPOFF[s] for s in SEEDS]
    c_on, _ = collapse_counts(summary, on_arms, SEVERE_PCT, expect_cells=N_CELLS)
    c_off, _ = collapse_counts(summary, off_arms, SEVERE_PCT, expect_cells=N_CELLS)
    von = [c_on[LGON[s]] for s in SEEDS]
    voff = [c_off[LGMPOFF[s]] for s in SEEDS]
    # Direction, named explicitly (the mp_ablation scorer inverted this once): H1 is that
    # lgon collapses LESS, i.e. a supporting pair has lgon count < lgmpoff count.
    support = sum(1 for a, b in zip(von, voff) if a < b)
    against = sum(1 for a, b in zip(von, voff) if a > b)
    ties = sum(1 for a, b in zip(von, voff) if a == b)
    n_eff = support + against
    p_sign = sign_test_exact(support, n_eff)
    print(f"\nS2 (reliability) — paired sign test on severe collapse (>= +{SEVERE_PCT:.0f}%)")
    print(f"  lgon counts:    {von}")
    print(f"  lgmpoff counts: {voff}")
    print(f"  lgon better in {support}, worse in {against}, tied in {ties} of 16 pairs")
    print(f"  non-tied n = {n_eff}, one-sided exact p = {p_sign:.4g}"
          + ("   (ties expected by construction — see registration)" if ties else ""))
    report["s2_reliability"] = {"lgon_counts": von, "lgmpoff_counts": voff,
                                "support": support, "against": against, "ties": ties,
                                "n_effective": n_eff, "p_one_sided": p_sign}

    report["descriptive"] = {}
    for t in DESCRIPTIVE_PCT:
        a, _ = collapse_counts(summary, on_arms, t, expect_cells=N_CELLS)
        b, _ = collapse_counts(summary, off_arms, t, expect_cells=N_CELLS)
        c, _ = collapse_counts(summary, [LGCTRL[s] for s in SEEDS], t,
                               expect_cells=N_CELLS)
        report["descriptive"][f"+{t:.0f}%"] = {
            "lgon": [a[LGON[s]] for s in SEEDS],
            "lgmpoff": [b[LGMPOFF[s]] for s in SEEDS],
            "lgctrl": [c[LGCTRL[s]] for s in SEEDS],
        }
        print(f"\n+{t:.0f}% [descriptive]  lgon    {report['descriptive'][f'+{t:.0f}%']['lgon']}"
              f"\n                   lgmpoff {report['descriptive'][f'+{t:.0f}%']['lgmpoff']}"
              f"\n                   lgctrl  {report['descriptive'][f'+{t:.0f}%']['lgctrl']}")

    # ---- Reading rule, applied mechanically (fixed before any data existed). ------
    p1 = report["primary"]["p_one_sided_h1"]
    p1_opp = report["primary"]["p_one_sided_opposite"]
    s1 = report["s1_attribution"]["p_one_sided_h1"]
    if p1 <= PRIMARY_ALPHA:
        if s1 <= PRIMARY_ALPHA:
            verdict = "LINK_MP_WINS_ATTRIBUTED"
            msg = ("Message passing over the link-aware graph beats BOTH the no-MP model "
                   "and MP over the\nold graph. The mp_ablation_v1 ambiguity resolves to "
                   "(b): MP was running over the wrong\ngraph. The link entities carry "
                   "the credit.")
        else:
            verdict = "LINK_MP_WINS_UNATTRIBUTED"
            msg = ("lgon beats the no-MP model but is not distinguishable from plain "
                   "MP-ON on this\ncorpus, so the win cannot be attributed to the link "
                   "entities specifically. Claim only\n'MP helps on this corpus'; do not "
                   "claim the link graph resolved mp_ablation_v1.")
    elif p1_opp <= PRIMARY_ALPHA:
        verdict = "OPPOSITE_DIRECTION"
        msg = ("The no-MP model beats link-graph MP at the registered threshold. Even "
               "over a graph\nthat CAN express shared-link contention, and a corpus whose "
               "bandwidth binds, message\npassing hurts. Report as-is; the supervised MP "
               "question is closed on both graphs.")
    else:
        verdict = "NO_DIFFERENCE_DETECTED"
        msg = ("No difference detected at the registered threshold. This is a FAILURE TO "
               "DETECT, not\nequivalence (that would need a registered TOST). The "
               "mp_ablation_v1 ambiguity between\n'MP unhelpful' and 'wrong graph' stays "
               "open on the supervised objective; per the\nregistration, do not re-run "
               "with tweaks — the remaining MP question moves to the\nclosed-loop phase.")
    report["verdict"] = verdict
    print(f"\n=== LINK MP V1 VERDICT: {verdict} ===\n{msg}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
