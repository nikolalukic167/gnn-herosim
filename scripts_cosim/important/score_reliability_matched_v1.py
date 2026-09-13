#!/usr/bin/env python3
"""Score reliability_matched_v1 against its registration.

Registration: docs/lineages/reliability_matched_v1.md, committed before any gate episode
ran. Every threshold, statistic, alpha and reading rule below is inherited verbatim from
the SIGNED objective_pivot_v1 Phase 1 registration; this file adds no new statistical
decision and takes no tuning arguments.

The question: does the GNN's severe-collapse reliability edge over the pointwise MLP
survive training BOTH classes on the same corpus? Phase 1 established it across corpora;
the 2026-09-03 latency pilot showed the latency half of that comparison was entirely a
corpus effect.

  PRIMARY      one-sided Mann-Whitney rank-sum (midranks, fixed-seed permutation) on the
               per-draw collapse-count vectors at +50%, H1: MLP burden > GNN burden.
  MUST-HOLD    the same statistic at +100%.
  DESCRIPTIVE  +30%, the clean/unclean dichotomy, worst-cell magnitudes. No verdict role.
  VOID         the venue control fires, or any arm is missing cells.

Collapse rule (Phase 1's, unchanged): a cell counts as collapsed when the arm's total_rtt
is at least THRESHOLD percent above the SAME-CELL Knative arm from the same run.

Usage:
  score_reliability_matched_v1.py --eval-dir simulation_data/reliability_matched_v1 \\
      [--json-out simulation_data/reliability_matched_v1_verdict.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---- Registered constants. Do not tune. -------------------------------------------
PRIMARY_PCT = 50.0
MUST_HOLD_PCT = (100.0,)
DESCRIPTIVE_PCT = (30.0,)
ALPHA = 0.05

SEEDS = tuple(range(1, 17))
MLP_ARMS = {s: f"mlp_matched_s{s}" for s in SEEDS}
VENUE_CONTROL_ARMS = {s: f"gnn_venuectl_s{s}" for s in (1, 5, 9, 13)}
KNATIVE_ARM = "knative"

N_CELLS = 20
BLOCK_FILES = (
    "eval_drawgate_backbone.json",
    "eval_promo175_backbone.json",
    "eval_bbrob_bb_core8_bw1p5.json",
    "eval_bbrob_bb_core4_bw0p5.json",
)

# The FROZEN GNN comparison group: link_mp_v1's 16 lgon arms, gated at pin 8aef27a on
# these same 20 cells. Recorded outcome, quoted from that lineage's node: zero severe
# collapses in all 48 arms x 20 cells at every threshold. Written as data, not recomputed,
# because the registration freezes this group — and the venue control below is what makes
# reusing it legitimate.
FROZEN_GNN_COUNTS = {pct: [0] * 16 for pct in (30.0, 50.0, 100.0)}

PERM_SEED = 20260903
N_PERM = 200_000


# ---- Statistics --------------------------------------------------------------------


def _midranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def ranksum_greater(treatment: List[float], control: List[float]) -> Tuple[float, float]:
    """One-sided rank-sum, H1: `treatment` is stochastically GREATER. Returns (W, p).

    Fixed-seed Monte-Carlo permutation over the pooled labels — the same instrument and
    the same 200k draws Phase 1 registered. Written out rather than imported from scipy:
    the cluster env has been found carrying two disagreeing installs of scipy
    (datalab-pitfalls #10), and a verdict script is the last place that should depend on
    which one loads.
    """
    pooled = list(treatment) + list(control)
    n_t = len(treatment)
    ranks = _midranks(pooled)
    w_obs = sum(ranks[:n_t])
    # Permute over the rank multiset in CANONICAL (sorted) order. The null is "n_t ranks
    # drawn at random from the pooled multiset", which does not depend on the order the
    # inputs arrived in — but permuting the unsorted array does, because the seeded
    # shuffle then indexes a differently-ordered list. Measured before any real data:
    # re-ordering an identical input moved p by 4e-5, well inside Monte-Carlo error and
    # still a reproducibility hole in a registered instrument. Sorting closes it.
    pool_ranks = sorted(ranks)
    rng = random.Random(PERM_SEED)
    idx = list(range(len(pool_ranks)))
    ge = 0
    for _ in range(N_PERM):
        rng.shuffle(idx)
        if sum(pool_ranks[i] for i in idx[:n_t]) >= w_obs - 1e-12:
            ge += 1
    # (ge + 1) / (B + 1): the observed labelling is itself a member of the null set.
    return w_obs, (ge + 1) / (N_PERM + 1)


# ---- Loading -----------------------------------------------------------------------


def load_cells(eval_dir: Path) -> Dict[str, Dict[str, float]]:
    """{(block/cell): {arm: total_rtt}} across all four block files."""
    out: Dict[str, Dict[str, float]] = {}
    for fname in BLOCK_FILES:
        path = eval_dir / fname
        if not path.is_file():
            raise SystemExit(
                f"FAIL LOUD: missing {path} — a partially-run gate cannot be scored; "
                "the registration fixes 20 cells and all four blocks are required."
            )
        payload = json.loads(path.read_text())
        block = fname[len("eval_"):-len(".json")]
        for row in payload["rows"]:
            out.setdefault(f"{block}/{row['cell']}", {})[row["label"]] = row["total_rtt"]
    if len(out) != N_CELLS:
        raise SystemExit(
            f"FAIL LOUD: {len(out)} cells found, registration fixes {N_CELLS}"
        )
    return out


def collapse_counts(cells, arms: Dict[int, str], threshold: float) -> Dict[int, int]:
    counts = {}
    for seed, arm in sorted(arms.items()):
        n = bad = 0
        for cell_id, entry in sorted(cells.items()):
            if arm not in entry:
                continue
            if KNATIVE_ARM not in entry:
                raise SystemExit(f"FAIL LOUD: {cell_id} has {arm} but no {KNATIVE_ARM}")
            kn = entry[KNATIVE_ARM]
            pct = 100.0 * (entry[arm] - kn) / kn
            n += 1
            bad += pct >= threshold
        if n != N_CELLS:
            raise SystemExit(
                f"FAIL LOUD: arm {arm} has {n} cells, expected {N_CELLS} — "
                "a partially-evaluated arm cannot be ranked"
            )
        counts[seed] = bad
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    cells = load_cells(args.eval_dir)

    # ---- VOID gate: the venue control (registration §Venue control) ------------------
    venue: Dict[str, Dict[str, int]] = {}
    venue_fired = []
    for pct in (30.0, 50.0, 100.0):
        counts = collapse_counts(cells, VENUE_CONTROL_ARMS, pct)
        venue[f"{pct:.0f}"] = {VENUE_CONTROL_ARMS[s]: c for s, c in counts.items()}
        for seed, c in counts.items():
            if c > 0:
                venue_fired.append(
                    f"{VENUE_CONTROL_ARMS[seed]} has {c} collapsed cells at +{pct:.0f}% "
                    "where the frozen table records zero"
                )

    summary = {
        "registration": "docs/lineages/reliability_matched_v1.md",
        "eval_dir": str(args.eval_dir),
        "n_cells": N_CELLS,
        "venue_control": venue,
        "frozen_gnn_counts": FROZEN_GNN_COUNTS,
    }

    print("\n=== reliability_matched_v1 — registered readout ===")
    print(f"  cells: {N_CELLS} backbone cells across 4 blocks")
    print("\n  VENUE CONTROL (4 frozen GNN arms re-gated at HEAD; must be all zero):")
    for pct, row in venue.items():
        print(f"    +{pct}%: {row}")

    if venue_fired:
        summary["verdict"] = "VOID (venue control fired)"
        summary["venue_failures"] = venue_fired
        print("\n  VERDICT: VOID — the venue control fired:")
        for line in venue_fired:
            print(f"    {line}")
        print("  The frozen GNN group cannot be compared against arms from this tree.")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(summary, indent=2))
        return 0

    # ---- The comparison --------------------------------------------------------------
    rows = {}
    for pct in (PRIMARY_PCT,) + MUST_HOLD_PCT + DESCRIPTIVE_PCT:
        mlp = collapse_counts(cells, MLP_ARMS, pct)
        mlp_vec = [mlp[s] for s in SEEDS]
        gnn_vec = FROZEN_GNN_COUNTS[pct]
        if set(mlp_vec) == {0} and set(gnn_vec) == {0}:
            w, p = 0.0, 1.0
            tie_at_zero = True
        else:
            w, p = ranksum_greater([float(x) for x in mlp_vec],
                                   [float(x) for x in gnn_vec])
            tie_at_zero = False
        rows[f"{pct:.0f}"] = {
            "mlp_counts": mlp_vec, "gnn_counts": gnn_vec,
            "mlp_clean": sum(1 for x in mlp_vec if x == 0),
            "gnn_clean": sum(1 for x in gnn_vec if x == 0),
            "ranksum_w": w, "p_one_sided": p, "tie_at_zero": tie_at_zero,
            "role": ("PRIMARY" if pct == PRIMARY_PCT
                     else "MUST-HOLD" if pct in MUST_HOLD_PCT else "DESCRIPTIVE"),
        }
    summary["thresholds"] = rows

    print("\n  %-6s %-11s %-38s %-38s %9s" % ("thr", "role", "MLP counts (matched corpus)",
                                              "GNN counts (frozen)", "p"))
    for pct in (30.0, 50.0, 100.0):
        r = rows[f"{pct:.0f}"]
        print("  +%-5s %-11s %-38s %-38s %9.5f" % (
            f"{pct:.0f}%", r["role"], str(r["mlp_counts"]), str(r["gnn_counts"]),
            r["p_one_sided"]))

    primary = rows[f"{PRIMARY_PCT:.0f}"]
    holds = all(rows[f"{p:.0f}"]["p_one_sided"] < ALPHA for p in MUST_HOLD_PCT)
    if primary["tie_at_zero"]:
        verdict = ("TIE-AT-ZERO — no reliability difference is detectable once the corpus "
                   "is matched, because neither class collapses on this corpus")
    elif primary["p_one_sided"] < ALPHA and holds:
        verdict = "PASS — the reliability edge survives corpus matching"
    else:
        verdict = ("FAIL — the MLP's collapse burden is not stochastically greater; the "
                   "reliability claim does not survive corpus matching")
    summary["verdict"] = verdict
    summary["alpha"] = ALPHA

    print(f"\n  VERDICT: {verdict}")
    if primary["tie_at_zero"]:
        print("  Registered limitation: at n=16 a floor effect cannot be distinguished")
        print("  from 'both classes are genuinely robust'. Outcome 2 of the registration.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
