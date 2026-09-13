#!/usr/bin/env python3
"""Score objective_pivot_v1 Phase 1 against its signed registration.

Registration: docs/lineages/objective_pivot_v1/phase1-registration-draft.md, SIGNED OFF
2026-08-28. Like score_gnn_draw_study.py (whose collapse rule and Fisher arithmetic this
imports), the thresholds and statistics here ARE the registration — this script takes no
tuning arguments, and it refuses to compute a verdict until the venue/code identity of
every NEW arm is asserted.

The design, in brief (rationale and power tables live in the registration):
  - 16 seeded GNN draws (1-8 from gnn_draw_study_v1; 9-16 trained/gated at the pinned
    commit c08aa7e) vs the FROZEN 16-draw MLP group, on the same 30 cells.
  - PRIMARY: one-sided Mann-Whitney rank-sum (midranks, fixed-seed Monte-Carlo
    permutation) on per-draw collapse-count vectors at +50%; must ALSO clear at +100%.
  - +30% and the clean/unclean dichotomy are DESCRIPTIVE ONLY (the dichotomy's joint
    power at n=16 is ~0.135 — computed before sign-off — and the +30% line is
    unpowerable at affordable n; the claim is scoped to severe collapse).
  - SECONDARY (its own claim): sign test on per-draw mean total_rtt margin vs same-cell
    Knative; bar >= 13/16 draws negative (p = 0.0106 under the null).
  - VOID: any new arm whose recorded provenance mismatches the pin table.

Usage:
  score_objective_pivot_phase1.py --summary simulation_data/gate_stats_summary.json \\
      [--json-out simulation_data/objective_pivot_phase1_verdict.json]
"""

import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_gnn_draw_study import MLP_ARMS, collapse_counts, fisher_exact_greater  # noqa: E402

# ---- Registered constants. Do not tune. -------------------------------------------
PRIMARY_THRESHOLD_PCT = 50.0
SENSITIVITY_MUST_HOLD_PCT = (100.0,)   # verdict must also clear here
DESCRIPTIVE_PCT = (30.0,)              # reported, never part of the verdict
ALPHA = 0.05

GNN_SEEDS = tuple(range(1, 17))
NEW_SEEDS = tuple(range(9, 17))        # provenance-asserted against the pin
GNN_ARMS = {seed: f"gnndraws{seed}" for seed in GNN_SEEDS}

SIGN_TEST_MIN_NEG = 13                 # of 16 draws with mean margin < 0 vs Knative

# Fixed-seed Monte-Carlo permutation for the rank statistic. Exact enumeration of
# C(32,16) ~ 6.0e8 is out of reach in stdlib; 200k permutations gives an MC standard
# error < 0.001 at p ~ 0.05, far finer than any decision boundary here. The seed is part
# of the registration: re-running the scorer must reproduce the p-value bit-for-bit.
PERM_SEED = 20260828
N_PERM = 200_000

# The pin (verified 2026-08-28 from job 712389's run_provenance; see the registration's
# venue table). Every NEW arm must record exactly this, or it is VOID.
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
    "GNN_DISABLE_MESSAGE_PASSING": None,
}


def assert_new_arm_provenance(summary: dict) -> list:
    """Return the list of VOID arms (with reasons); empty means all new arms are clean."""
    void = []
    for seed in NEW_SEEDS:
        arm = GNN_ARMS[seed]
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
                        mismatches.setdefault(key, set()).add(
                            f"{blk}/{cell}: {got!r}")
        if n_cells == 0:
            void.append((arm, {"missing": {"no cells in summary"}}))
        elif mismatches:
            void.append((arm, {k: sorted(v)[:3] for k, v in mismatches.items()}))
    return void


def _midranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def rank_sum_p_less(gnn_vals, mlp_vals, n_perm=N_PERM, seed=PERM_SEED):
    """One-sided Monte-Carlo permutation p for 'GNN counts are stochastically smaller'.

    Statistic: sum of the GNN group's midranks in the pooled ranking; small = GNN lower.
    Add-one estimator so p can never be reported as exactly 0.
    """
    pooled = list(gnn_vals) + list(mlp_vals)
    n = len(gnn_vals)
    obs = sum(_midranks(pooled)[:n])
    rng = random.Random(seed)
    idx = list(range(len(pooled)))
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(idx)
        ranks = _midranks([pooled[i] for i in idx])
        if sum(ranks[:n]) <= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def sign_test_p_at_least(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2) — the registered secondary's null."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def mean_margins_vs_knative(summary: dict, arms):
    """{arm: mean per-cell % margin vs same-cell knative} over exactly 30 cells."""
    out = {}
    for arm in arms:
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
        if len(pcts) != 30:
            raise SystemExit(f"FAIL LOUD: arm {arm} has {len(pcts)} cells, expected 30")
        out[arm] = sum(pcts) / len(pcts)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path,
                    default=Path("simulation_data/gate_stats_summary.json"))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())

    # ---- VOID gate first: no verdict is computed over unverified arms. ------------
    void = assert_new_arm_provenance(summary)
    if void:
        print("=== VOID: venue/code identity mismatch on new arms (registration §Venue) ===")
        for arm, why in void:
            print(f"  {arm}:")
            for key, examples in sorted(why.items()):
                print(f"    {key}: " + "; ".join(examples))
        print("\nFix and re-run the affected arms at the pinned commit. NO verdict was "
              "computed; a VOID is not a FAIL.")
        return 2
    print(f"[PIN] all {len(NEW_SEEDS)} new arms verified: commit {PIN_COMMIT[:12]}, "
          f"clean tree, registered env — proceeding to statistics")

    arms = [GNN_ARMS[s] for s in GNN_SEEDS]
    report = {"alpha": ALPHA, "pin_commit": PIN_COMMIT, "n_gnn": len(arms),
              "n_mlp": len(MLP_ARMS), "perm_seed": PERM_SEED, "n_perm": N_PERM,
              "thresholds": {}}

    # ---- Primary + must-hold sensitivity: rank-sum on collapse-count vectors. -----
    verdict_ps = {}
    for t in (PRIMARY_THRESHOLD_PCT,) + SENSITIVITY_MUST_HOLD_PCT + DESCRIPTIVE_PCT:
        gc, _ = collapse_counts(summary, arms, t)
        mc, _ = collapse_counts(summary, MLP_ARMS, t)
        gnn_vals = [gc[a] for a in arms]
        mlp_vals = [mc[a] for a in MLP_ARMS]
        p_rank = rank_sum_p_less(gnn_vals, mlp_vals)
        gnn_clean = sum(1 for v in gnn_vals if v == 0)
        mlp_clean = sum(1 for v in mlp_vals if v == 0)
        p_fisher = fisher_exact_greater(
            gnn_clean, len(gnn_vals) - gnn_clean, mlp_clean, len(mlp_vals) - mlp_clean)
        role = ("PRIMARY" if t == PRIMARY_THRESHOLD_PCT
                else "SENSITIVITY" if t in SENSITIVITY_MUST_HOLD_PCT
                else "descriptive")
        if role != "descriptive":
            verdict_ps[t] = p_rank
        print(f"\n+{t:.0f}% [{role}]")
        print(f"  GNN counts (s1..s16): {gnn_vals}")
        print(f"  MLP counts (frozen):  {mlp_vals}")
        print(f"  rank-sum p = {p_rank:.5f}   (dichotomy, descriptive: "
              f"clean {gnn_clean}/16 vs {mlp_clean}/16, Fisher p = {p_fisher:.4g})")
        report["thresholds"][f"+{t:.0f}%"] = {
            "role": role, "gnn_counts": gnn_vals, "mlp_counts": mlp_vals,
            "rank_sum_p": p_rank, "gnn_clean": gnn_clean, "mlp_clean": mlp_clean,
            "fisher_p_descriptive": p_fisher,
        }

    primary_ok = verdict_ps[PRIMARY_THRESHOLD_PCT] <= ALPHA
    sens_ok = all(verdict_ps[t] <= ALPHA for t in SENSITIVITY_MUST_HOLD_PCT)

    # ---- Secondary: sign test on per-draw mean margins vs Knative. ----------------
    margins = mean_margins_vs_knative(summary, arms)
    neg = sum(1 for a in arms if margins[a] < 0)
    p_sign = sign_test_p_at_least(neg, len(arms))
    sec_ok = neg >= SIGN_TEST_MIN_NEG
    print(f"\nSECONDARY (its own claim): per-draw mean margin vs Knative")
    print("  " + " ".join(f"s{s}={margins[GNN_ARMS[s]]:+.1f}%" for s in GNN_SEEDS))
    print(f"  negative draws: {neg}/16 (bar >= {SIGN_TEST_MIN_NEG})  "
          f"sign-test p = {p_sign:.4g}  -> {'PASS' if sec_ok else 'FAIL'}")
    report["secondary"] = {
        "per_draw_mean_margin_pct": {f"s{s}": margins[GNN_ARMS[s]] for s in GNN_SEEDS},
        "negative_draws": neg, "bar": SIGN_TEST_MIN_NEG, "sign_test_p": p_sign,
        "verdict": "PASS" if sec_ok else "FAIL",
    }

    verdict = "PASS" if (primary_ok and sens_ok) else "FAIL"
    report["verdict"] = verdict
    print(f"\n=== OBJECTIVE PIVOT PHASE 1 VERDICT: {verdict} ===")
    if verdict == "PASS":
        print("The registered claim may be written: across seeded draws, the GNN's severe-"
              "collapse burden\n(cells at total_rtt >= +50% vs Knative) is stochastically "
              "smaller than the MLP's.\nScope: severe collapse only — the +30% row above is "
              "the mandatory reported limitation.")
    else:
        print("The reliability claim is dropped as registered; the publishable frame is P6\n"
              "(terminal separability negative + the reusable diagnostic), with the draw\n"
              "studies reported as-is.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
