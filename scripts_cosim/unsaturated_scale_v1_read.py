"""unsaturated_scale_v1 -- the bars, signed 2026-09-19 before any datum exists.

The question: does a learned arm beat reactive Knative on BIG infrastructure where the baseline
is HEALTHY? Every 80-server number in the record is read against a drowning baseline: the
server ladder scales arrivals with servers (f4000 at 6 -> f300 at 80, both x13.3), yet
reactive's queue share goes 63 % -> 99 % and its elapsed 22 s -> 611 s. "-41 % vs reactive at
80 servers" therefore means "less drowned". No unsaturated big-infrastructure rung exists.

Two steps, both live (rule 6):

  S1  the baseline sweep -- reactive alone, 80 servers, 4 cells, the arrival ladder
      K_FACTORS. Its output is ONE selected factor, chosen by the rule in `select_factor`
      and nothing else. If no factor is unsaturated, that is the primary finding
      (`V_NO_UNSATURATED_RUNG`) and S2 is read as SECONDARY at the least-saturated one.
  S2  the learned arms at the selected factor, 16 checkpoints x 4 cells, against reactive
      (K1), the matched twin (K2) and each other (K3).

Bars are the chain's, unchanged: |median| >= 5 %, p < 0.05, n = 16 checkpoints, the unit is
the checkpoint (median over its cells), paired two-sided Wilcoxon. Every reader fixes its
orientation in its docstring because `read_pair_pct` is orientation-neutral.
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import median
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_only_v1_read import (  # noqa: E402
    V_UNREADABLE, V_A_FASTER, V_B_FASTER, V_PAIR_TIE, collapse_to_seed, read_pair_pct,
)

__all__ = [
    "K_SEPARATE_PCT", "K_ALPHA", "K_MIN_SEEDS", "K_SERVERS", "K_CELLS", "K_FACTORS",
    "K_SATURATION_SHARE", "K_TARGET_SHARE", "K_ARMS", "K_GRAPH", "K_TWIN", "K_GNN",
    "V_NO_UNSATURATED_RUNG", "V_UNSATURATED_RUNG", "V_UNSATURATED_AT_THE_EDGE",
    "V_ARM_BEATS_REACTIVE", "V_REACTIVE_FASTER", "V_NOT_SEP",
    "V_GRAPH_FASTER", "V_POINTWISE_FASTER",
    "V_SUM_COSTS", "V_SUM_HELPS", "V_SUM_NOT_SEP",
    "V_LEARNED_BEATS_REACTIVE", "V_NO_LEARNED_BEATS_REACTIVE",
    "read_s1", "select_factor", "read_k1", "read_k2", "read_k3", "read_k4",
    "saturated", "collapse_to_seed",
]

# --- the bars, signed 2026-09-19 -------------------------------------------------------
K_SEPARATE_PCT = 5.0        # B4_TIE_PCT / F / G / J, unchanged
K_ALPHA = 0.05
K_MIN_SEEDS = 16
K_SERVERS = 80
K_CELLS = ("cs80s9001", "cs80s9002", "cs80s9003", "cs80s9005")   # the R3 cells; 9004 hangs

# The arrival ladder. f is the rescale factor of `rescale_workload_arrivals.py`; the rate is
# ~1841/f arrivals/s (f300 = 6.139/s is R3, f4000 = 0.460/s is R0). 300 is REUSED from
# results/psv3_p3 (its reactive R3 arm IS this cell at this factor); the rest are minted.
K_FACTORS = (300, 500, 700, 1000, 2000, 4000, 8000)

# The chain's saturation bar (peer_only_v1 B7_SATURATED_QUEUE_SHARE). "Unknown is not a pass".
K_SATURATION_SHARE = 0.90
# The selection target. The three unsaturated rungs every standing win was measured at read
# 63 / 77 / 80 % (R0 / C40 / C80), so the selected rung must sit INSIDE that band, not at the
# edge of the bar: a rung at 89 % is "unsaturated" by the letter and not comparable to them.
K_TARGET_SHARE = 0.80

# S2 arms. Corpus tag -> results label as the po_v1 readers relabel them.
K_GRAPH = "be1670_gnnedge0"   # the repaired graph arm; the one that beats reactive at C40
K_TWIN = "1670_mpoff"         # its corpus-matched MP-OFF twin (corpus_matched_v1 G1)
K_GNN = "1670_gnn"            # the proper GNN, PeerConv + bipartite GIN with sum
K_ARMS = (K_GRAPH, K_TWIN, "1670_peeronly", K_GNN, "516_mpoff")

# --- verdicts: S1 ----------------------------------------------------------------------
V_UNSATURATED_RUNG = "UNSATURATED-80-SERVER-RUNG-EXISTS"
V_UNSATURATED_AT_THE_EDGE = "UNSATURATED-ONLY-AT-THE-EDGE-OF-THE-BAR"
V_NO_UNSATURATED_RUNG = "REACTIVE-SATURATES-AT-80-SERVERS-AT-EVERY-LOAD"

# --- verdicts: K1, one arm vs reactive -------------------------------------------------
V_ARM_BEATS_REACTIVE = "ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE"
V_REACTIVE_FASTER = "REACTIVE-FASTER-AT-UNSATURATED-SCALE"
V_NOT_SEP = "NOT-SEPARATED"

# --- verdicts: K2, graph vs matched twin -----------------------------------------------
V_GRAPH_FASTER = "GRAPH-FASTER-AT-UNSATURATED-SCALE"
V_POINTWISE_FASTER = "POINTWISE-FASTER-AT-UNSATURATED-SCALE"

# --- verdicts: K3, gnn (sum) vs gnnedge0 (mean) ----------------------------------------
V_SUM_COSTS = "SUM-COSTS-AT-UNSATURATED-SCALE"
V_SUM_HELPS = "SUM-HELPS-AT-UNSATURATED-SCALE"
V_SUM_NOT_SEP = "SUM-NOT-SEPARATED-AT-UNSATURATED-SCALE"

# --- verdicts: K4, the composite -------------------------------------------------------
V_LEARNED_BEATS_REACTIVE = "LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE"
V_NO_LEARNED_BEATS_REACTIVE = "NO-LEARNED-ARM-BEATS-REACTIVE-AT-UNSATURATED-SCALE"


def saturated(queue_share: Optional[float]) -> Optional[bool]:
    """None stays None: an unknown share renders UNKNOWN, never as a pass."""
    if queue_share is None:
        return None
    return float(queue_share) >= K_SATURATION_SHARE


def read_s1(share_by_factor: Mapping[int, Mapping[str, float]]) -> dict:
    """Per factor: reactive's queue share, median over K_CELLS. A factor missing a cell is
    UNREADABLE for selection -- a rung read on three cells is not the rung the arms run on.
    Also reports whether the ladder is monotone (share falls as f grows); descriptive only."""
    rows = {}
    for f in K_FACTORS:
        cells = share_by_factor.get(f) or {}
        missing = [c for c in K_CELLS if c not in cells]
        if missing:
            rows[f] = {"verdict": V_UNREADABLE, "missing": missing}
            continue
        vals = [float(cells[c]) for c in K_CELLS]
        qs = median(vals)
        rows[f] = {"queue_share": qs, "saturated": saturated(qs),
                   "in_band": qs <= K_TARGET_SHARE, "per_cell": dict(zip(K_CELLS, vals))}
    readable = [(f, r["queue_share"]) for f, r in rows.items() if "queue_share" in r]
    mono = all(b <= a for (_, a), (_, b) in zip(readable, readable[1:])) if len(readable) > 1 else None
    return {"factors": rows, "monotone": mono}


def select_factor(s1: Mapping[str, object]) -> dict:
    """THE selection rule. The smallest f (fastest arrivals, hardest rung) whose median
    reactive queue share is <= K_TARGET_SHARE. Failing that, the smallest f <= the saturation
    bar, labelled AT-THE-EDGE. Failing that, NO unsaturated rung: S2 runs at the least-saturated
    readable factor and is read as SECONDARY.

    Smallest f, not "closest to the target": picking the factor nearest 80 % would be choosing
    the load after seeing the ladder, and the arms' known strength grows with load."""
    rows = s1["factors"]
    readable = [(f, r) for f, r in rows.items() if "queue_share" in r]
    if not readable:
        return {"verdict": V_UNREADABLE, "why": "no factor carries all four cells"}
    in_band = [f for f, r in readable if r["in_band"]]
    if in_band:
        f = min(in_band)
        return {"verdict": V_UNSATURATED_RUNG, "factor": f, "queue_share": rows[f]["queue_share"],
                "primary": True}
    under_bar = [f for f, r in readable if not r["saturated"]]
    if under_bar:
        f = min(under_bar)
        return {"verdict": V_UNSATURATED_AT_THE_EDGE, "factor": f,
                "queue_share": rows[f]["queue_share"], "primary": True,
                "why": f"no factor reads <= {K_TARGET_SHARE:.0%}; the selected rung is under the "
                       f"{K_SATURATION_SHARE:.0%} bar but outside the band the standing wins were "
                       "measured in, and every quote must say so"}
    f = min(readable, key=lambda fr: fr[1]["queue_share"])[0]
    return {"verdict": V_NO_UNSATURATED_RUNG, "factor": f, "queue_share": rows[f]["queue_share"],
            "primary": False,
            "why": f"reactive's queue share is >= {K_SATURATION_SHARE:.0%} at every factor up to "
                   f"f{max(K_FACTORS)}; the baseline saturates at 80 servers independent of load. "
                   "S2 is SECONDARY at the least-saturated factor; the finding is about the baseline"}


def _oriented(a, b, *, min_seeds, faster_a, faster_b, tie):
    r = read_pair_pct(a, b, tol=K_SEPARATE_PCT, alpha=K_ALPHA,
                      min_seeds=K_MIN_SEEDS if min_seeds is None else min_seeds)
    if r["verdict"] == V_UNREADABLE:
        return r
    v = {V_A_FASTER: faster_a, V_B_FASTER: faster_b, V_PAIR_TIE: tie}[r["verdict"]]
    return {**r, "verdict": v,
            "bar": {"separate_pct": K_SEPARATE_PCT, "alpha": K_ALPHA, "min_seeds": K_MIN_SEEDS}}


def read_k1(arm_by_seed: Mapping[int, float], reactive_by_seed: Mapping[int, float],
            *, min_seeds: Optional[int] = None) -> dict:
    """One learned arm vs reactive at the selected rung. FIRST argument is the LEARNED arm;
    negative median = the arm is faster. Reactive is deterministic at seed 0 and is replicated
    across the arm's seeds by the caller (`_reactive_like`)."""
    return _oriented(arm_by_seed, reactive_by_seed, min_seeds=min_seeds,
                     faster_a=V_ARM_BEATS_REACTIVE, faster_b=V_REACTIVE_FASTER, tie=V_NOT_SEP)


def read_k2(graph_by_seed: Mapping[int, float], twin_by_seed: Mapping[int, float],
            *, min_seeds: Optional[int] = None) -> dict:
    """gnnedge0 (1,670) vs mpoff (1,670): the model-class contrast, corpus matched, at
    unsaturated scale. FIRST argument is the GRAPH arm; negative = graph faster."""
    return _oriented(graph_by_seed, twin_by_seed, min_seeds=min_seeds,
                     faster_a=V_GRAPH_FASTER, faster_b=V_POINTWISE_FASTER, tie=V_NOT_SEP)


def read_k3(gnn_by_seed: Mapping[int, float], gnnedge0_by_seed: Mapping[int, float],
            *, min_seeds: Optional[int] = None) -> dict:
    """gnn (GIN, sum) vs gnnedge0 (mean): does the sum defect still cost where the baseline is
    healthy? `bipartite_aggr_v1`'s mechanism predicts YES -- the candidate set is ~48/task at 80
    servers at any load. FIRST argument is gnn; a POSITIVE median means sum costs."""
    return _oriented(gnn_by_seed, gnnedge0_by_seed, min_seeds=min_seeds,
                     faster_a=V_SUM_HELPS, faster_b=V_SUM_COSTS, tie=V_SUM_NOT_SEP)


def read_k4(k1_by_arm: Mapping[str, Mapping[str, object]], *, primary: bool = True) -> dict:
    """The composite. Fires positive if ANY registered arm beats reactive under K1's bar. An
    UNREADABLE arm is reported as missing, never as a loss; if every arm is unreadable the
    composite is unreadable. `primary=False` (S1 found no unsaturated rung) keeps the verdict
    string but marks it SECONDARY, so it cannot be quoted as the primary result."""
    missing = [a for a in K_ARMS if a not in k1_by_arm or k1_by_arm[a].get("verdict") == V_UNREADABLE]
    readable = {a: r for a, r in k1_by_arm.items() if a in K_ARMS and r.get("verdict") != V_UNREADABLE}
    if not readable:
        return {"verdict": V_UNREADABLE, "missing": missing, "why": "no registered arm is readable"}
    winners = sorted(a for a, r in readable.items() if r["verdict"] == V_ARM_BEATS_REACTIVE)
    losers = sorted(a for a, r in readable.items() if r["verdict"] == V_REACTIVE_FASTER)
    out = {"winners": winners, "losers": losers, "missing": missing, "secondary": not primary}
    if winners:
        why = (f"{', '.join(winners)} beat{'s' if len(winners) == 1 else ''} reactive at 80 servers "
               f"under the chain's bar on a rung where reactive is not saturated")
        return {**out, "verdict": V_LEARNED_BEATS_REACTIVE, "why": why}
    why = ("no registered arm beats reactive at 80 servers where the baseline is healthy; "
           f"{len(losers)} of {len(readable)} readable arms are behind it")
    if missing:
        why += f"; {missing} unreadable, so this is not a verdict about them"
    return {**out, "verdict": V_NO_LEARNED_BEATS_REACTIVE, "why": why}
