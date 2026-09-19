"""bipartite_aggr_v1 — the registered bars for E1/E2/E3, as module constants.

Committed BEFORE any arm is trained. This lineage differs from every other one in this
programme in a way worth stating: **it makes a prediction that can fail in a specific,
pre-named pattern**, rather than screening a change for an effect.

The claim. `bipartite_edge_v1` found that replacing the bipartite `GIN` with a
`BipartiteEdgeConv` is worth −20 to −29 % live, and that **edge-conditioning is not why** (the
zeroed-attribute control matches the treatment). The conv differs from the `GIN` in two
remaining ways: aggregation (`sum` → `mean`) and MLP shape. This tests the first, as a single
flag on the otherwise unchanged module.

Why aggregation is the suspect, and why the test is a prediction rather than a screen: a task
aggregates over its **candidate platforms**, and that set is not a fixed size. `cluster_scale_v1`
S0.d measured **3.55 candidates/task at 6 servers and 47.92 at 80** — a 13.5x ratio. A `sum`
over it scales with cluster size; a `mean` does not. And the bipartite penalty `peer_only_v1`
measured was **largest at 80 servers (−18.94 %) and ABSENT at 6 (p = 0.61)**.

So if aggregation is the mechanism, `sum` must cost **at 80 servers and not at 6**. That is
E3, and no other explanation this programme has offered predicts an existing null.

Run: pipenv run python3 -m pytest scripts_cosim/test_bipartite_aggr_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_only_v1_read import (  # noqa: E402
    V_UNREADABLE, V_A_FASTER, V_B_FASTER, V_PAIR_TIE, collapse_to_seed, read_pair_pct,
)

__all__ = [
    "E_SEPARATE_PCT", "E_ALPHA", "E_MIN_SEEDS", "E_RUNGS", "E_CANDIDATES_PER_TASK",
    "V_SUM_COSTS", "V_SUM_HELPS", "V_AGGR_NOT_SEP",
    "V_SCALE_DEPENDENT", "V_SCALE_FREE", "V_INVERTED", "V_AGGR_IS_NOT_THE_MECHANISM",
    "read_e1", "read_e2", "read_e3", "collapse_to_seed",
]

# --- the bars, signed 2026-09-18 -------------------------------------------------------
# Reused UNCHANGED from bipartite_edge_v1's D bars and peer_only_v1's C bars, so all three
# lineages' numbers are directly comparable rather than merely similar-looking.
E_SEPARATE_PCT = 5.0
E_ALPHA = 0.05
E_MIN_SEEDS = 16

# BOTH ends of the server ladder, and that is the point. R3 is where the bipartite penalty was
# largest; R0 is where it was ABSENT. A one-rung read cannot distinguish "sum is worse" from
# "sum is worse BECAUSE the candidate set is large", and only the second is a mechanism.
E_RUNGS = ("R3", "R0")

# The precondition, measured by cluster_scale_v1 S0.d on 12 minted cells and cited rather than
# re-measured (one fact, one home). Recorded HERE because the whole prediction rests on it: if
# these were equal, E1/E2 could not separate the hypothesis from any other story about scale.
E_CANDIDATES_PER_TASK = {"R0": 3.55, "R3": 47.92}     # 6 and 80 servers; ratio 13.5x

# --- verdicts --------------------------------------------------------------------------
V_SUM_COSTS = "SUM-COSTS"
V_SUM_HELPS = "SUM-HELPS"
V_AGGR_NOT_SEP = "AGGREGATION-NOT-SEPARATED"

# E3, the composite. These are the four patterns E1 x E2 can produce, all named in advance.
V_SCALE_DEPENDENT = "SUM-COSTS-ONLY-WHERE-CANDIDATES-ARE-MANY"
V_SCALE_FREE = "SUM-COSTS-EVERYWHERE"
V_INVERTED = "SUM-COSTS-ONLY-WHERE-CANDIDATES-ARE-FEW"
V_AGGR_IS_NOT_THE_MECHANISM = "AGGREGATION-IS-NOT-THE-MECHANISM"


def _pair(sum_by_seed, mean_by_seed, *, min_seeds: Optional[int]) -> dict:
    """Negative median = the SUM arm is faster. Orientation-neutral by construction; the
    caller names the arms and this only reports which won (docs/gates/gate-tools.md,
    2026-09-18: a named reader reused with swapped arguments inverts its verdict string)."""
    return read_pair_pct(sum_by_seed, mean_by_seed, tol=E_SEPARATE_PCT, alpha=E_ALPHA,
                         min_seeds=E_MIN_SEEDS if min_seeds is None else min_seeds)


def _aggr_verdict(r: Mapping[str, object]) -> dict:
    if r.get("verdict") == V_UNREADABLE:
        return dict(r)
    v = {V_A_FASTER: V_SUM_HELPS, V_B_FASTER: V_SUM_COSTS, V_PAIR_TIE: V_AGGR_NOT_SEP}[r["verdict"]]
    return {**r, "verdict": v,
            "bar": {"separate_pct": E_SEPARATE_PCT, "alpha": E_ALPHA, "min_seeds": E_MIN_SEEDS}}


def read_e1(sum_by_seed, mean_by_seed, *, min_seeds: Optional[int] = None) -> dict:
    """E1: `gnnedgesum` vs `gnnedge0` at R3 (80 servers, 47.92 candidates/task).
    Registered expectation: SUM-COSTS."""
    return _aggr_verdict(_pair(sum_by_seed, mean_by_seed, min_seeds=min_seeds))


def read_e2(sum_by_seed, mean_by_seed, *, min_seeds: Optional[int] = None) -> dict:
    """E2: the same contrast at R0 (6 servers, 3.55 candidates/task).
    Registered expectation: AGGREGATION-NOT-SEPARATED."""
    return _aggr_verdict(_pair(sum_by_seed, mean_by_seed, min_seeds=min_seeds))


def read_e3(e1: Mapping[str, object], e2: Mapping[str, object]) -> dict:
    """E3, the prediction. All four patterns named before any datum exists.

    The registered expectation is SCALE-DEPENDENT: sum costs at 80 servers and is not
    separated at 6. Any other pattern falsifies "the aggregator is the mechanism" -- including
    SCALE-FREE, which would say sum is simply a worse aggregator here and would leave the
    ABSENT-at-6-servers null still unexplained.
    """
    v1, v2 = e1.get("verdict"), e2.get("verdict")
    if V_UNREADABLE in (v1, v2):
        return {"verdict": V_UNREADABLE,
                "why": "a registered bar is unreadable; the prediction is not tested"}
    if v1 == V_SUM_COSTS and v2 == V_AGGR_NOT_SEP:
        return {"verdict": V_SCALE_DEPENDENT,
                "why": "sum costs where candidates are many (47.92/task) and not where they "
                       "are few (3.55/task) -- the predicted pattern, and it explains why the "
                       "bipartite penalty was absent at 6 servers"}
    if v1 == V_SUM_COSTS and v2 == V_SUM_COSTS:
        return {"verdict": V_SCALE_FREE,
                "why": "sum costs at BOTH rungs, so it is a worse aggregator here but NOT by "
                       "candidate-set scaling; the absent-at-6-servers null stays unexplained"}
    if v1 == V_AGGR_NOT_SEP and v2 == V_SUM_COSTS:
        return {"verdict": V_INVERTED,
                "why": "sum costs only where candidates are FEW -- the opposite of the "
                       "prediction, and no mechanism on offer accounts for it"}
    return {"verdict": V_AGGR_IS_NOT_THE_MECHANISM,
            "why": f"E1={v1}, E2={v2}: aggregation does not account for the bipartite penalty, "
                   "and the conv's remaining difference from the GIN is its MLP shape"}
