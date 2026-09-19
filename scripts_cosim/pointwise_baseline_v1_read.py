"""pointwise_baseline_v1 — the registered bars for J0–J4, as module constants.

Committed BEFORE any arm is served. This lineage tests the sentence CLAUDE.md opens with:

    "Knative is the industry-standard reactive baseline, and the MLP is the pointwise
     control. **The MLP is a control, not a straw man** — the program's repeated finding is
     that it ties, so treat 'the MLP cannot match this' as a hypothesis to test, never an
     assumption to write from."

and the clause `best_arm_v1` had to carry:

    "`mpoff_516` is the MP-OFF twin of a GNN, **not the MLP**. The pointwise baseline this
     clause names has never been the tabular MLP."

**Every graph-vs-pointwise number in this programme compares the GNN against its own
message-passing-disabled twin.** That twin shares the GNN's encoder, scorer and decoder — it
is an ablation, not a model class. The actual pointwise model class has never been served on
these cells at all.

The comparison is fair by construction on the serving side: `MLPBatchScheduler` inherits graph
build, decode and roll-forward from `GNNScheduler` (via `XGBoostBatchScheduler`) and replaces
ONLY the scoring — one batched [N_edges, 22] → [N_edges] pass through a pointwise MLP. Same
decoder, same peer-group batching, same physics, same cells, same workload, same corpus.

Bars reused UNCHANGED from `peer_only_v1`'s B4/B8, `best_arm_v1`'s F and `corpus_matched_v1`'s
G, so this reads against the clauses it is testing rather than against a bar of its own.
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
    "J_SEPARATE_PCT", "J_ALPHA", "J_MIN_SEEDS", "J_CLIENTS", "J_CORPORA", "J_SATURATION_SHARE",
    "V_GRAPH_FASTER", "V_MLP_FASTER", "V_NOT_SEP",
    "V_GRAPH_BEATS_MLP", "V_MLP_BEATS_GRAPH", "V_GRAPH_VS_MLP_NOT_ESTABLISHED",
    "V_TWIN_IS_FAIR", "V_TWIN_IS_NOT_THE_MLP",
    "read_j1", "read_j2", "read_j3", "read_j3_composite", "saturated", "collapse_to_seed",
]

# --- the bars, signed 2026-09-19 -------------------------------------------------------
J_SEPARATE_PCT = 5.0        # B4_TIE_PCT / F_SEPARATE_PCT / G_SEPARATE_PCT, unchanged
J_ALPHA = 0.05
J_MIN_SEEDS = 16
J_CLIENTS = (20, 40, 80)    # the three UNSATURATED client rungs, as in F and G
J_CORPORA = ("1670", "516")
J_SATURATION_SHARE = 0.90   # "unknown is not a pass": a None share renders UNKNOWN

# --- verdicts: J1, one rung, one corpus -------------------------------------------------
V_GRAPH_FASTER = "GRAPH-FASTER-THAN-MLP"
V_MLP_FASTER = "MLP-FASTER-THAN-GRAPH"
V_NOT_SEP = "NOT-SEPARATED"

# --- verdicts: J2, the composite over one corpus's ladder --------------------------------
V_GRAPH_BEATS_MLP = "GRAPH-BEATS-THE-POINTWISE-MODEL-CLASS"
V_MLP_BEATS_GRAPH = "POINTWISE-MODEL-CLASS-BEATS-GRAPH"
V_GRAPH_VS_MLP_NOT_ESTABLISHED = "GRAPH-VS-MLP-NOT-ESTABLISHED"

# --- verdicts: J3, the load-bearing one -- is the MP-OFF twin a fair stand-in? ------------
V_TWIN_IS_FAIR = "MP-OFF-TWIN-STANDS-IN-FOR-THE-MLP"
V_TWIN_IS_NOT_THE_MLP = "MP-OFF-TWIN-IS-NOT-THE-MLP"


def read_j1(graph_by_seed: Mapping[int, float], mlp_by_seed: Mapping[int, float],
            *, min_seeds: Optional[int] = None) -> dict:
    """J1 at one rung of one corpus. Negative median = the GRAPH arm is faster.

    Orientation is fixed here once: the first argument is the graph arm. `read_pair_pct` is
    orientation-neutral and will invert its verdict string if the arguments are swapped, which
    is the defect this docstring exists to prevent.
    """
    r = read_pair_pct(graph_by_seed, mlp_by_seed, tol=J_SEPARATE_PCT, alpha=J_ALPHA,
                      min_seeds=J_MIN_SEEDS if min_seeds is None else min_seeds)
    if r["verdict"] == V_UNREADABLE:
        return r
    v = {V_A_FASTER: V_GRAPH_FASTER, V_B_FASTER: V_MLP_FASTER, V_PAIR_TIE: V_NOT_SEP}[r["verdict"]]
    return {**r, "verdict": v,
            "bar": {"separate_pct": J_SEPARATE_PCT, "alpha": J_ALPHA, "min_seeds": J_MIN_SEEDS}}


def read_j2(per_rung: Mapping[int, Mapping[str, object]]) -> dict:
    """J2 over one corpus's ladder, under F2/G2's rule UNCHANGED: two wins and no losses.

    The same rule as every earlier composite in this chain, so "does the answer change when
    the baseline becomes the real model class?" compares like with like. An UNREADABLE rung
    counts as MISSING, never as either side.
    """
    missing = [c for c in J_CLIENTS
               if c not in per_rung or per_rung[c].get("verdict") == V_UNREADABLE]
    if missing:
        return {"verdict": V_UNREADABLE, "missing": missing,
                "why": f"rungs {missing} unreadable; a model-class claim is not decided on a "
                       "partial ladder"}
    verdicts = [per_rung[c]["verdict"] for c in J_CLIENTS]
    graph = sum(v == V_GRAPH_FASTER for v in verdicts)
    mlp = sum(v == V_MLP_FASTER for v in verdicts)
    if graph >= 2 and mlp == 0:
        return {"verdict": V_GRAPH_BEATS_MLP, "graph_wins": graph, "mlp_wins": mlp,
                "why": "the graph arm is faster than the MLP at >= 2 of 3 unsaturated rungs "
                       "and behind at none, on a matched corpus and a shared decoder"}
    if mlp >= 2:
        return {"verdict": V_MLP_BEATS_GRAPH, "graph_wins": graph, "mlp_wins": mlp,
                "why": "the MLP is faster at >= 2 of 3; the pointwise model class wins this "
                       "comparison outright"}
    if graph:
        why = (f"the ladder is SPLIT: graph faster at {graph} of 3, behind at {mlp}. The "
               "per-rung win(s) are real under a registered bar; what is not established is a "
               "claim about the ladder.")
    else:
        why = ("the graph arm is faster at no rung; nothing here shows the graph model class "
               "ahead of the pointwise one")
    return {"verdict": V_GRAPH_VS_MLP_NOT_ESTABLISHED, "graph_wins": graph, "mlp_wins": mlp,
            "why": why}


def read_j3(mpoff_by_seed: Mapping[int, float], mlp_by_seed: Mapping[int, float],
            *, min_seeds: Optional[int] = None) -> dict:
    """J3 at one rung: the MP-OFF twin vs the MLP, SAME corpus. Negative = the twin faster.

    **This is the load-bearing read of the lineage.** Every "pointwise" number in this record
    is the twin's. If the two are not separated, the record's substitution was sound and its
    back-catalogue stands as written. If they are separated, the substitution was not sound,
    and every earlier pointwise claim is a claim about an ablation.
    """
    r = read_pair_pct(mpoff_by_seed, mlp_by_seed, tol=J_SEPARATE_PCT, alpha=J_ALPHA,
                      min_seeds=J_MIN_SEEDS if min_seeds is None else min_seeds)
    if r["verdict"] == V_UNREADABLE:
        return r
    v = {V_A_FASTER: "TWIN-FASTER", V_B_FASTER: "MLP-FASTER", V_PAIR_TIE: V_NOT_SEP}[r["verdict"]]
    return {**r, "verdict": v,
            "bar": {"separate_pct": J_SEPARATE_PCT, "alpha": J_ALPHA, "min_seeds": J_MIN_SEEDS}}


def read_j3_composite(per_rung: Mapping[int, Mapping[str, object]]) -> dict:
    """Is the MP-OFF twin a fair stand-in for the MLP anywhere it has been quoted?

    Signed consequence. Separation at ANY readable rung ⇒ `MP-OFF-TWIN-IS-NOT-THE-MLP`: the
    substitution is not sound, and every earlier pointwise claim in the record must be restated
    as being about the GNN's own ablation. No separation anywhere ⇒ `MP-OFF-TWIN-STANDS-IN-FOR-
    THE-MLP`, **on these cells only** — a non-separation is not proof of equality and is never a
    licence to substitute one arm for the other in general.
    """
    readable = {c: r for c, r in per_rung.items()
                if r.get("verdict") not in (None, V_UNREADABLE)}
    if not readable:
        return {"verdict": V_UNREADABLE, "why": "no rung carries both the twin and the MLP"}
    sep = {c: r["verdict"] for c, r in readable.items() if r["verdict"] != V_NOT_SEP}
    missing = [c for c in J_CLIENTS if c not in readable]
    if sep:
        return {"verdict": V_TWIN_IS_NOT_THE_MLP, "separated_at": sep, "unreadable": missing,
                "why": f"the twin and the MLP separate at {sorted(sep)} clients; every earlier "
                       "'pointwise' claim in this record is a claim about the GNN's own MP-OFF "
                       "ablation and must be restated as one"}
    return {"verdict": V_TWIN_IS_FAIR, "separated_at": {}, "unreadable": missing,
            "why": "the twin and the MLP are not separated at any readable rung, so the "
                   "substitution was sound ON THESE CELLS -- a non-separation, not proof of "
                   "equality, and not a licence to substitute in general"}


def saturated(queue_share: Optional[float]) -> Optional[bool]:
    """None stays None: unknown is not a pass (PARITY.md)."""
    return None if queue_share is None else bool(queue_share >= J_SATURATION_SHARE)
