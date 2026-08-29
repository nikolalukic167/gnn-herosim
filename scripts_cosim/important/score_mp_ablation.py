#!/usr/bin/env python3
"""Score mp_ablation_v1 against its registration.

Registration: docs/lineages/mp_ablation_v1.md, written 2026-08-29 BEFORE any run. As with
score_objective_pivot_phase1.py, the thresholds and statistics here ARE the registration:
this script takes no tuning arguments, and refuses to compute a verdict until the
venue/code identity of every MP-OFF arm is asserted.

The design, in brief:
  - PAIRED. GNN_DISABLE_MESSAGE_PASSING=1 skips the GIN forward call but still constructs
    the module, so at a fixed seed both arms start from bit-identical weights. Seed s of
    `gnnmpoff` pairs with seed s of `gnndraws`.
  - PRIMARY: two-sided EXACT Wilcoxon signed-rank on the 16 per-seed differences in mean
    margin vs same-cell Knative. n=16, so the null is enumerated exactly (2^16 = 65,536
    sign patterns) -- no Monte Carlo, reproducible by construction.
  - CO-PRIMARY: one-sided exact sign test on per-seed severe-collapse counts (>= +50%),
    over NON-TIED pairs only. Ties are expected by construction (MP-ON is mostly zero) and
    the tie count is reported; a null here is a failure to detect, never equivalence.
  - VOID: any MP-OFF arm whose recorded provenance mismatches the pin table, which for this
    screen REQUIRES GNN_DISABLE_MESSAGE_PASSING == "1".

Usage:
  score_mp_ablation.py --summary simulation_data/gate_stats_summary.json \
      [--json-out simulation_data/mp_ablation_verdict.json]
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
MP_ON = {s: f"gnndraws{s}" for s in SEEDS}
MP_OFF = {s: f"gnnmpoff{s}" for s in SEEDS}

PRIMARY_ALPHA = 0.05
SEVERE_PCT = 50.0
DESCRIPTIVE_PCT = (30.0, 100.0)

BACKBONE_BLOCKS = {"drawgate/backbone", "promo175/backbone",
                   "bbrob/bb_core8_bw1p5", "bbrob/bb_core4_bw0p5"}
FLAT_BLOCKS = {"drawgate/nobackbone", "promo175/nobackbone"}

PIN_COMMIT = "c08aa7ee140fd51e3d384f97df3f31b126df96ab"
REQUIRED_PROVENANCE = {
    "code_commit": PIN_COMMIT,
    "code_dirty": False,
    "INFERENCE_FEATURE_LAYOUT": "dim22",
    "QUEUE_FEATURE_CONTRACT": "scale_invariant_v1",
    "warmth_physics": "node_disk_v2",
    "GNN_DECODE_MODE": "argmax",
    "GNN_BATCH_SIZE": "4",
    "GNN_BATCH_TIMEOUT": "0.002",
    "HEROSIM_GNN_DEVICE": "cpu",
    "TOPOLOGY_FEATURE_CONTRACT": "src_index_v0",
    "GNN_MP_NODE_EDGES": None,
    # THE lever. Serving with message passing a checkpoint trained without it is the
    # 2026-08-16 failure that cost 12.4x live RTT, and it would silently invalidate the
    # whole screen -- so it is asserted, not assumed.
    "GNN_DISABLE_MESSAGE_PASSING": "1",
}


def assert_mp_off_provenance(summary: dict) -> list:
    """Return [(arm, reasons)] for MP-OFF arms failing the pin; empty means all clean."""
    void = []
    for seed in SEEDS:
        arm = MP_OFF[seed]
        mismatches, n_cells = {}, 0
        for blk in sorted(summary):
            for cell in sorted(summary[blk]):
                entry = summary[blk][cell]
                if arm not in entry:
                    continue
                n_cells += 1
                prov = entry[arm].get("provenance") or {}
                for key, want in REQUIRED_PROVENANCE.items():
                    got = prov.get(key)
                    if got != want:
                        mismatches.setdefault(key, set()).add(f"{blk}/{cell}: {got!r}")
        if n_cells == 0:
            void.append((arm, {"missing": ["no cells in summary"]}))
        elif n_cells != 30:
            void.append((arm, {"cells": [f"{n_cells} cells, expected 30"]}))
        elif mismatches:
            void.append((arm, {k: sorted(v)[:3] for k, v in mismatches.items()}))
    return void


def mean_margin(summary: dict, arm: str, blocks=None) -> float:
    """Mean per-cell % margin vs same-cell knative, optionally restricted to blocks."""
    pcts = []
    for blk in sorted(summary):
        if blocks is not None and blk not in blocks:
            continue
        for cell in sorted(summary[blk]):
            entry = summary[blk][cell]
            if arm not in entry:
                continue
            if "knative" not in entry:
                raise SystemExit(f"FAIL LOUD: {blk}/{cell} has {arm} but no knative")
            kn = entry["knative"]["total_rtt"]
            pcts.append(100.0 * (entry[arm]["total_rtt"] - kn) / kn)
    if not pcts:
        raise SystemExit(f"FAIL LOUD: arm {arm} has no cells for blocks={blocks}")
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


def wilcoxon_exact_two_sided(diffs):
    """Exact two-sided signed-rank p. n<=16 so the 2^n null is enumerated, not sampled."""
    w_obs, ranks = _signed_ranks(diffs)
    n = len(ranks)
    if n == 0:
        return 1.0, w_obs, 0
    total = sum(ranks)
    centre = total / 2.0
    dev_obs = abs(w_obs - centre)
    hits = 0
    for signs in product((0, 1), repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s)
        if abs(w - centre) >= dev_obs - 1e-12:
            hits += 1
    return hits / (2 ** n), w_obs, n


def sign_test_exact(n_pos: int, n_eff: int) -> float:
    """One-sided P(X >= n_pos) for X ~ Binomial(n_eff, 1/2)."""
    if n_eff == 0:
        return 1.0
    return sum(comb(n_eff, i) for i in range(n_pos, n_eff + 1)) / 2 ** n_eff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path,
                    default=Path("simulation_data/gate_stats_summary.json"))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    summary = json.loads(args.summary.read_text())

    void = assert_mp_off_provenance(summary)
    if void:
        print("=== VOID: MP-OFF arms failed the pin (registration §Design/VOID) ===")
        for arm, why in void:
            print(f"  {arm}:")
            for key, ex in sorted(why.items()):
                print(f"    {key}: " + "; ".join(ex))
        print("\nFix and re-run the named arms. NO verdict computed; a VOID is not a FAIL.")
        return 2
    print(f"[PIN] all 16 MP-OFF arms verified: {PIN_COMMIT[:12]}, clean, "
          f"GNN_DISABLE_MESSAGE_PASSING=1 — proceeding")

    report = {"pin_commit": PIN_COMMIT, "alpha": PRIMARY_ALPHA, "n_pairs": len(SEEDS)}

    # ---- PRIMARY: paired exact Wilcoxon on mean margin vs Knative. ----------------
    on = {s: mean_margin(summary, MP_ON[s]) for s in SEEDS}
    off = {s: mean_margin(summary, MP_OFF[s]) for s in SEEDS}
    diffs = [off[s] - on[s] for s in SEEDS]          # >0 means MP-OFF is WORSE (higher RTT)
    p_w, w_obs, n_nz = wilcoxon_exact_two_sided(diffs)
    print("\nPRIMARY — paired exact Wilcoxon on per-seed mean margin vs Knative")
    print("  seed:   " + " ".join(f"s{s}" for s in SEEDS))
    print("  MP-ON:  " + " ".join(f"{on[s]:+.1f}" for s in SEEDS))
    print("  MP-OFF: " + " ".join(f"{off[s]:+.1f}" for s in SEEDS))
    print("  diff:   " + " ".join(f"{d:+.1f}" for d in diffs))
    print(f"  mean diff = {sum(diffs)/len(diffs):+.2f} pp "
          f"(positive = MP-OFF worse)   W+ = {w_obs:.1f}, n_nonzero = {n_nz}")
    print(f"  exact two-sided p = {p_w:.5f}  -> "
          f"{'DIFFERENCE DETECTED' if p_w <= PRIMARY_ALPHA else 'no difference detected'}")
    report["primary"] = {"mp_on_margin": on, "mp_off_margin": off,
                         "diffs": diffs, "mean_diff_pp": sum(diffs) / len(diffs),
                         "w_plus": w_obs, "n_nonzero": n_nz, "p_two_sided": p_w,
                         "detected": p_w <= PRIMARY_ALPHA}

    # ---- CO-PRIMARY: paired sign test on severe collapse, non-tied pairs only. ----
    on_arms = [MP_ON[s] for s in SEEDS]
    off_arms = [MP_OFF[s] for s in SEEDS]
    c_on, _ = collapse_counts(summary, on_arms, SEVERE_PCT)
    c_off, _ = collapse_counts(summary, off_arms, SEVERE_PCT)
    von = [c_on[MP_ON[s]] for s in SEEDS]
    voff = [c_off[MP_OFF[s]] for s in SEEDS]
    worse = sum(1 for a, b in zip(voff, von) if b > a)   # MP-OFF collapses more
    better = sum(1 for a, b in zip(voff, von) if b < a)
    ties = sum(1 for a, b in zip(voff, von) if a == b)
    n_eff = worse + better
    p_sign = sign_test_exact(worse, n_eff)
    print(f"\nCO-PRIMARY — paired sign test on severe collapse (>= +{SEVERE_PCT:.0f}%)")
    print(f"  MP-ON  counts: {von}")
    print(f"  MP-OFF counts: {voff}")
    print(f"  MP-OFF worse in {worse}, better in {better}, tied in {ties} of 16 pairs")
    print(f"  non-tied n = {n_eff}, one-sided exact p = {p_sign:.4g}"
          + ("   (ties expected by construction — see registration)" if ties else ""))
    report["co_primary"] = {"mp_on_counts": von, "mp_off_counts": voff, "worse": worse,
                            "better": better, "ties": ties, "n_effective": n_eff,
                            "p_one_sided": p_sign}

    # ---- Descriptive: other thresholds, and the backbone/flat split. --------------
    report["descriptive"] = {}
    for t in DESCRIPTIVE_PCT:
        a, _ = collapse_counts(summary, on_arms, t)
        b, _ = collapse_counts(summary, off_arms, t)
        va = [a[MP_ON[s]] for s in SEEDS]
        vb = [b[MP_OFF[s]] for s in SEEDS]
        print(f"\n+{t:.0f}% [descriptive]  MP-ON {va}\n"
              f"                   MP-OFF {vb}")
        report["descriptive"][f"+{t:.0f}%"] = {"mp_on": va, "mp_off": vb}

    print("\nBACKBONE vs FLAT (descriptive; the Phase 1 exploratory split)")
    split = {}
    for name, blocks in (("backbone", BACKBONE_BLOCKS), ("flat", FLAT_BLOCKS)):
        m_on = sum(mean_margin(summary, MP_ON[s], blocks) for s in SEEDS) / len(SEEDS)
        m_off = sum(mean_margin(summary, MP_OFF[s], blocks) for s in SEEDS) / len(SEEDS)
        print(f"  {name:8s}: MP-ON {m_on:+6.1f}%   MP-OFF {m_off:+6.1f}%   "
              f"delta {m_off - m_on:+.1f} pp")
        split[name] = {"mp_on": m_on, "mp_off": m_off, "delta_pp": m_off - m_on}
    report["backbone_flat_split"] = split

    # ---- Reading rule, applied mechanically. --------------------------------------
    mean_d = sum(diffs) / len(diffs)
    if p_w <= PRIMARY_ALPHA and mean_d > 0:
        verdict = "MP_LOAD_BEARING"
        msg = ("Removing message passing made placement materially WORSE. Message passing "
               "is load-bearing;\nthe Phase 1 claim may keep its 'graph-aware' wording.")
    elif p_w <= PRIMARY_ALPHA and mean_d < 0:
        verdict = "MP_HARMFUL"
        msg = ("Removing message passing made placement BETTER. Report as-is; the graph "
               "channel is\nactively harmful and the architecture work must be re-scoped.")
    else:
        verdict = "NO_DIFFERENCE_DETECTED"
        msg = ("No difference detected. On this evidence the reliability edge is NOT "
               "attributable to\nmessage passing, and the Phase 1 claim must be reworded "
               "away from 'graph-aware'.\nThis is a FAILURE TO DETECT, not proof of "
               "equivalence — that would need a registered TOST.")
    report["verdict"] = verdict
    print(f"\n=== MP ABLATION VERDICT: {verdict} ===\n{msg}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
