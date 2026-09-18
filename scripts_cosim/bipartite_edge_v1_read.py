"""bipartite_edge_v1 — the registered bars for D1/D2/D3/D4, as module constants.

Committed BEFORE any arm is trained. Nothing in this module reads a result file; it turns
already-collapsed per-checkpoint numbers into the verdicts signed in
`docs/lineages/bipartite_edge_v1.md`. That separation is deliberate — a bar that lives in the
same function as the loader gets "adjusted" while looking at data.

The question: `peer_only_v1` closed with the peer-vs-bipartite contrast CONFOUNDED. `PeerConv`
is residual AND edge-aware; the bipartite `GIN` is neither. C4 tested the residual half
(RESIDUAL-DOES-NOT-TRANSFER). These bars read the other half — a bipartite conv that can see
the 5-column `[exec_time, latency, is_warm, energy, comm_time]` attribute the GIN is blind to.

Three things this module inherits from its parent lineage, each because it was got wrong once:

* The statistical unit is the CHECKPOINT (`collapse_to_seed`), never the (cell, seed) pair.
  peer_only_v1 B3 halved a headline by getting that wrong.
* Every verdict goes through `read_pair_pct`, which is ORIENTATION-NEUTRAL. A named reader
  reused with its arguments swapped inverts its verdict string while leaving its number
  correct (docs/gates/gate-tools.md, 2026-09-18).
* An arm lost to a resource kill NEVER relaxes a registered bar. The registered read reports
  UNREADABLE; a disclosed read may print beside it with the exclusion named.

Run: pipenv run python3 -m pytest scripts_cosim/test_bipartite_edge_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_only_v1_read import (  # noqa: E402
    V_UNREADABLE,
    V_A_FASTER,
    V_B_FASTER,
    V_PAIR_TIE,
    collapse_to_seed,
    read_pair_pct,
    read_vs_reactive,
)

__all__ = [
    "D_SEPARATE_PCT", "D_ALPHA", "D_MIN_SEEDS", "D_RUNGS", "D_CLIENTS",
    "V_EDGE_HELPS", "V_EDGE_COSTS", "V_EDGE_NOT_SEP",
    "V_ATTRS_ARE_LEVER", "V_CONV_IS_LEVER", "V_ATTRS_COST", "V_D2_NOT_SEP",
    "V_BIPARTITE_REPAIRED", "V_PENALTY_SURVIVES",
    "V_CLOSES_NEGATIVE",
    "read_d1", "read_d2", "read_d3", "read_d4", "read_lineage",
    "collapse_to_seed",
]

# --- the bars, signed 2026-09-18 -------------------------------------------------------
# Reused UNCHANGED from peer_only_v1's C1/C2/C4 so the two lineages' numbers are directly
# comparable. Changing one here would make "gnnedge vs gnn" and "peeronly vs gnn" bars that
# only look alike.
D_SEPARATE_PCT = 5.0
D_ALPHA = 0.05
D_MIN_SEEDS = 16

# The two rungs where the bipartite penalty was MEASURED (-18.94 % at 80 servers, -18.66 % at
# 40 clients). Deliberately NOT 6 servers or 80 clients, where the penalty is absent: C3 spent
# itself measuring a mechanism in the regime where the effect does not exist, and a negative
# from a rung with nothing to repair would be uninformative rather than evidence.
D_RUNGS = ("R3",)
D_CLIENTS = (40,)

# --- verdicts --------------------------------------------------------------------------
V_EDGE_HELPS = "EDGE-CONDITIONING-HELPS"
V_EDGE_COSTS = "EDGE-CONDITIONING-COSTS"
V_EDGE_NOT_SEP = "EDGE-CONDITIONING-NOT-SEPARATED"

V_ATTRS_ARE_LEVER = "ATTRIBUTES-ARE-THE-LEVER"
V_CONV_IS_LEVER = "CONV-IS-THE-LEVER"
V_ATTRS_COST = "ATTRIBUTES-COST"
V_D2_NOT_SEP = "ATTRIBUTES-NOT-SEPARATED"

V_BIPARTITE_REPAIRED = "BIPARTITE-REPAIRED"
V_PENALTY_SURVIVES = "PENALTY-SURVIVES"

# The kill condition, signed in advance (see the node). If D1 and D2 both read
# NOT-SEPARATED, BOTH halves of the peer_only_v1 confound have been tested and neither
# explains the penalty. There is no third half.
V_CLOSES_NEGATIVE = "EDGE-CONDITIONING-DOES-NOT-TRANSFER"


def _pair(a_by_seed, b_by_seed, *, min_seeds: Optional[int]) -> dict:
    return read_pair_pct(a_by_seed, b_by_seed, tol=D_SEPARATE_PCT, alpha=D_ALPHA,
                         min_seeds=D_MIN_SEEDS if min_seeds is None else min_seeds)


def read_d1(gnnedge_by_seed: Mapping[int, float], gnn_by_seed: Mapping[int, float],
            *, min_seeds: Optional[int] = None) -> dict:
    """D1: does edge-conditioning close the bipartite penalty? `gnnedge` vs `gnn`.

    Negative median = gnnedge faster = edge-conditioning helps. CONFOUNDED on its own with
    the GIN -> BipartiteEdgeConv swap; D2 is what separates them, and a D1 result must never
    be quoted without it.
    """
    r = _pair(gnnedge_by_seed, gnn_by_seed, min_seeds=min_seeds)
    if r["verdict"] == V_UNREADABLE:
        return r
    v = {V_A_FASTER: V_EDGE_HELPS, V_B_FASTER: V_EDGE_COSTS, V_PAIR_TIE: V_EDGE_NOT_SEP}[r["verdict"]]
    return {**r, "verdict": v,
            "bar": {"separate_pct": D_SEPARATE_PCT, "alpha": D_ALPHA, "min_seeds": D_MIN_SEEDS}}


def read_d2(gnnedge_by_seed: Mapping[int, float], gnnedge0_by_seed: Mapping[int, float],
            *, d1_verdict: Optional[str] = None, min_seeds: Optional[int] = None) -> dict:
    """D2, the load-bearing bar: is it the ATTRIBUTES, or the conv? `gnnedge` vs `gnnedge0`.

    `gnnedge0` is the same module with `edge_attr` zeroed — same parameter count, same depth,
    same aggregation — so this contrast isolates edge-conditioning exactly. `d1_verdict` is
    passed in because CONV-IS-THE-LEVER is only a meaningful reading when D1 found a gain in
    the first place: a tie here with a tie there is just two ties.
    """
    r = _pair(gnnedge_by_seed, gnnedge0_by_seed, min_seeds=min_seeds)
    if r["verdict"] == V_UNREADABLE:
        return r
    if r["verdict"] == V_A_FASTER:
        v = V_ATTRS_ARE_LEVER
    elif r["verdict"] == V_B_FASTER:
        v = V_ATTRS_COST
    elif d1_verdict == V_EDGE_HELPS:
        # D1 gained and the zeroed control matches it: the gain is mean-aggregation or MLP
        # shape, NOT edge-conditioning. Signed in advance so it cannot be reported otherwise.
        v = V_CONV_IS_LEVER
    else:
        v = V_D2_NOT_SEP
    return {**r, "verdict": v,
            "bar": {"separate_pct": D_SEPARATE_PCT, "alpha": D_ALPHA, "min_seeds": D_MIN_SEEDS}}


def read_d3(gnnedge_by_seed: Mapping[int, float], peeronly_by_seed: Mapping[int, float],
            *, d1_verdict: Optional[str] = None, min_seeds: Optional[int] = None) -> dict:
    """D3: does the new arm reach `peeronly`? The standing question is not "better than `gnn`"
    but "can a bipartite stage stop costing anything" — signed in advance so a within-family
    improvement cannot be quoted as one."""
    r = _pair(gnnedge_by_seed, peeronly_by_seed, min_seeds=min_seeds)
    if r["verdict"] == V_UNREADABLE:
        return r
    if r["verdict"] == V_B_FASTER:
        v = V_PENALTY_SURVIVES
    elif r["verdict"] == V_PAIR_TIE and d1_verdict == V_EDGE_HELPS:
        v = V_BIPARTITE_REPAIRED
    elif r["verdict"] == V_A_FASTER:
        # Not in the registered vocabulary because it was not anticipated: gnnedge beating
        # peeronly outright would be the strongest result the programme has. Reported under
        # the neutral label rather than squeezed into a bar that was signed for something
        # else -- and it would need its own registration before being quoted as a finding.
        v = V_A_FASTER
    else:
        v = V_PAIR_TIE
    return {**r, "verdict": v,
            "bar": {"separate_pct": D_SEPARATE_PCT, "alpha": D_ALPHA, "min_seeds": D_MIN_SEEDS}}


def read_d4(gnnedge_by_seed: Mapping[int, float], reactive_by_seed: Mapping[int, float],
            *, queue_share: Optional[float] = None, min_seeds: Optional[int] = None) -> dict:
    """D4: the live floor. `gnnedge` vs reactive Knative, with the SATURATION classification
    beside it — at 80 servers every arm beats reactive, so a win there is a ranking among
    arms and not a result. `queue_share` is reactive's queue as a fraction of its elapsed;
    >= 0.90 is the registered saturation bar."""
    r = read_vs_reactive(gnnedge_by_seed, reactive_by_seed,
                         min_seeds=D_MIN_SEEDS if min_seeds is None else min_seeds)
    saturated = None if queue_share is None else bool(queue_share >= 0.90)
    return {**r, "saturated": saturated, "queue_share": queue_share}


def read_lineage(d1: Mapping[str, object], d2: Mapping[str, object]) -> dict:
    """The kill condition, applied. Signed in the node before any arm was trained: if D1 and
    D2 both read NOT-SEPARATED, this lineage CLOSES negative and the standing answer gains
    the sentence that BOTH halves of the peer_only_v1 confound have now been tested."""
    if d1.get("verdict") == V_UNREADABLE or d2.get("verdict") == V_UNREADABLE:
        return {"verdict": V_UNREADABLE,
                "why": "a registered bar is unreadable; the kill condition does not apply"}
    if d1.get("verdict") == V_EDGE_NOT_SEP and d2.get("verdict") == V_D2_NOT_SEP:
        return {"verdict": V_CLOSES_NEGATIVE,
                "why": "both halves of the confound tested; neither explains the penalty"}
    return {"verdict": "OPEN", "why": "the kill condition did not fire"}


def rung_cells(cells: Sequence[str]) -> tuple:
    """Named so a caller cannot pass a bare string and collapse over its characters."""
    if isinstance(cells, str):
        raise ValueError("FAIL LOUD: rung_cells takes a sequence of cell names, not a string")
    return tuple(cells)
