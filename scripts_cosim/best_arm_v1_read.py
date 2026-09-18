"""best_arm_v1 — the registered bars for F1/F2/F3, as module constants.

Committed BEFORE either missing arm-set is served. This lineage tests **half of CLAUDE.md's
standing answer** — *"the best scheduler the program has is still a pointwise one"* — against
the repaired bipartite arm, at the three unsaturated client rungs.

The number that motivated it is POST-HOC and is not evidence: at 40 clients `gnnedge0` beats
`mpoff_516` by −12.96 % (15/16), computed after the fact on a contrast nothing had signed,
inside a lineage whose registered question was about aggregation. These bars are what turn it
into evidence or refute it.

Bars reused UNCHANGED from `peer_only_v1`'s B4/B8 so this reads against the clause it is
testing rather than against a bar of its own choosing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_only_v1_read import (  # noqa: E402
    V_UNREADABLE, V_A_FASTER, V_B_FASTER, V_PAIR_TIE, collapse_to_seed, read_pair_pct,
)

__all__ = [
    "F_SEPARATE_PCT", "F_ALPHA", "F_MIN_SEEDS", "F_CLIENTS", "F_SATURATION_SHARE",
    "V_GRAPH_FASTER", "V_POINTWISE_FASTER", "V_F1_NOT_SEP",
    "V_GRAPH_IS_BEST", "V_POINTWISE_STILL_BEST", "V_STILL_NOT_ESTABLISHED",
    "V_MARGIN_GROWS", "V_MARGIN_NOT_MONOTONE",
    "read_f1", "read_f2", "read_f3", "collapse_to_seed",
]

# --- the bars, signed 2026-09-18 -------------------------------------------------------
F_SEPARATE_PCT = 5.0        # B4_TIE_PCT, unchanged
F_ALPHA = 0.05              # B4_ALPHA, unchanged
F_MIN_SEEDS = 16            # B4_MIN_SEEDS, unchanged

# The whole client ladder, and only its UNSATURATED part. 20 clients IS the R0 server cell
# (6 servers, 20 clients) and is read from the server gate's results -- the same cells, not a
# parallel measurement. Saturated rungs are deliberately out of scope: B8 already scoped B4's
# clause to them and this lineage does not revisit it.
F_CLIENTS = (20, 40, 80)

# A rung is saturated if reactive's queue is >= 90 % of its elapsed (peer_only_v1's registered
# bar). Measured at 63/73/73 % on these three, so all three are in scope -- but the share is
# printed beside every number rather than assumed, because "unknown is not a pass".
F_SATURATION_SHARE = 0.90

# --- verdicts --------------------------------------------------------------------------
V_GRAPH_FASTER = "GRAPH-ARM-FASTER"
V_POINTWISE_FASTER = "POINTWISE-FASTER"
V_F1_NOT_SEP = "NOT-SEPARATED"

V_GRAPH_IS_BEST = "GRAPH-ARM-IS-BEST"
V_POINTWISE_STILL_BEST = "POINTWISE-STILL-BEST"
V_STILL_NOT_ESTABLISHED = "BEST-ARM-STILL-NOT-ESTABLISHED"

V_MARGIN_GROWS = "MARGIN-GROWS-WITH-LOAD"
V_MARGIN_NOT_MONOTONE = "MARGIN-NOT-MONOTONE"


def read_f1(graph_by_seed: Mapping[int, float], pointwise_by_seed: Mapping[int, float],
            *, min_seeds: Optional[int] = None) -> dict:
    """F1 at one rung: `gnnedge0` vs `mpoff_516`. Negative median = the graph arm faster."""
    r = read_pair_pct(graph_by_seed, pointwise_by_seed, tol=F_SEPARATE_PCT, alpha=F_ALPHA,
                      min_seeds=F_MIN_SEEDS if min_seeds is None else min_seeds)
    if r["verdict"] == V_UNREADABLE:
        return r
    v = {V_A_FASTER: V_GRAPH_FASTER, V_B_FASTER: V_POINTWISE_FASTER,
         V_PAIR_TIE: V_F1_NOT_SEP}[r["verdict"]]
    return {**r, "verdict": v,
            "bar": {"separate_pct": F_SEPARATE_PCT, "alpha": F_ALPHA, "min_seeds": F_MIN_SEEDS}}


def read_f2(per_rung: Mapping[int, Mapping[str, object]]) -> dict:
    """F2, the composite, and the signed consequence for CLAUDE.md's standing answer.

    A rung that is UNREADABLE is not counted as either side -- it is counted as missing, and if
    fewer than all three rungs are readable the composite says so rather than deciding on two.
    An arm lost to a resource kill must never be able to move a clause about the programme.
    """
    missing = [c for c in F_CLIENTS if c not in per_rung
               or per_rung[c].get("verdict") == V_UNREADABLE]
    if missing:
        return {"verdict": V_UNREADABLE, "missing": missing,
                "why": f"rungs {missing} unreadable; a clause about the programme is not "
                       "decided on a partial ladder"}
    verdicts = [per_rung[c]["verdict"] for c in F_CLIENTS]
    graph = sum(v == V_GRAPH_FASTER for v in verdicts)
    point = sum(v == V_POINTWISE_FASTER for v in verdicts)
    if graph >= 2 and point == 0:
        return {"verdict": V_GRAPH_IS_BEST, "graph_wins": graph, "pointwise_wins": point,
                "why": "the graph arm is faster at >= 2 of 3 unsaturated rungs and behind at "
                       "none; CLAUDE.md's pointwise clause is OVERTURNED at unsaturated load"}
    if point >= 2:
        return {"verdict": V_POINTWISE_STILL_BEST, "graph_wins": graph, "pointwise_wins": point,
                "why": "the pointwise arm is faster at >= 2 of 3; the clause survives B8's "
                       "scoping and is restored to the unsaturated rungs"}
    return {"verdict": V_STILL_NOT_ESTABLISHED, "graph_wins": graph, "pointwise_wins": point,
            "why": "neither arm is shown better across the ladder; B8's reading stands and the "
                   "post-hoc -12.96 % is NOT reproduced under a registered bar"}


def read_f3(per_rung: Mapping[int, Mapping[str, object]]) -> dict:
    """F3: the shape of the margin across the ladder. DESCRIPTIVE -- no consequence is attached.

    Three rungs cannot establish a trend; this is recorded so a successor knows where to look,
    not to support a claim. Kept as its own reader so it cannot be mistaken for a bar.
    """
    meds = [per_rung.get(c, {}).get("median") for c in F_CLIENTS]
    if any(m is None for m in meds):
        return {"verdict": V_UNREADABLE, "why": "a rung has no median"}
    monotone = all(meds[i + 1] < meds[i] for i in range(len(meds) - 1))   # more negative = better
    return {"verdict": V_MARGIN_GROWS if monotone else V_MARGIN_NOT_MONOTONE,
            "medians": {c: meds[i] for i, c in enumerate(F_CLIENTS)},
            "note": "descriptive only; three rungs cannot establish a trend"}


def saturated(queue_share: Optional[float]) -> Optional[bool]:
    """None stays None: unknown is not a pass (PARITY.md)."""
    return None if queue_share is None else bool(queue_share >= F_SATURATION_SHARE)
