"""corpus_matched_v1 — the registered bars for G1-G4, as module constants.

Committed BEFORE any number is read. `best_arm_v1` closed `BEST-ARM-STILL-NOT-ESTABLISHED`
and carried one clause that its own design could not discharge:

    "It is not a GNN-vs-MLP claim, and it is confounded with corpus. `mpoff_516` trains on
     516 datasets and `gnnedge0` on 1,670, and B1 measured corpus as the largest lever in
     this programme (`CORPUS-DOES-NOT-HELP` 6/6 -- more data made every arm *slower* live).
     Quote both arms' training caches or do not quote the comparison. A corpus-matched
     successor is what would make this a model-class result."

This is that successor. It asks a DIFFERENT question from `best_arm_v1`, and the difference
matters: F1 compared the *best arm of each kind*, which is inherently cross-corpus because
`mpoff_516` is the better pointwise arm. G1 holds the corpus fixed and varies only the model
class, so it can say "graph beats pointwise" or not without a corpus term in the contrast.
**Neither read supersedes the other.** F2's ladder is still the best-arm answer; G2 is the
model-class answer; G3 measures how big the difference between those two questions is.

The whole G ladder is served ALREADY -- `1670_mpoff` and `be1670_gnnedge0` are both on all
three client rungs at 16+ checkpoints, from `peer_only_v1` B7 and `best_arm_v1` F. Nothing new
is trained or run for it, which is exactly why these bars are committed before the read: a
contrast that is free to compute is a contrast that is easy to compute twice and report once.

Bars reused UNCHANGED from `peer_only_v1`'s B4/B8 and `best_arm_v1`'s F, so this reads against
the clause it is qualifying rather than against a bar of its own choosing.
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
    "G_SEPARATE_PCT", "G_ALPHA", "G_MIN_SEEDS", "G_CLIENTS", "G_SATURATION_SHARE",
    "V_GRAPH_FASTER", "V_POINTWISE_FASTER", "V_NOT_SEP",
    "V_EDGE_SURVIVES", "V_POINTWISE_AT_MATCHED", "V_MATCHED_NOT_ESTABLISHED",
    "V_C1670_FASTER", "V_C516_FASTER", "V_CORPUS_NOT_SEP",
    "V_CONFOUND_MATERIAL", "V_CONFOUND_IMMATERIAL",
    "V_CORPUS_INDEPENDENT", "V_CORPUS_CONTINGENT",
    "read_g1", "read_g2", "read_g3", "read_g3_composite", "read_g4",
    "saturated", "collapse_to_seed",
]

# --- the bars, signed 2026-09-18 -------------------------------------------------------
G_SEPARATE_PCT = 5.0        # B4_TIE_PCT / F_SEPARATE_PCT, unchanged
G_ALPHA = 0.05              # B4_ALPHA / F_ALPHA, unchanged
G_MIN_SEEDS = 16            # B4_MIN_SEEDS / F_MIN_SEEDS, unchanged

# The same three unsaturated client rungs `best_arm_v1` read, so G1 and F1 are comparable
# rung by rung. 20 clients IS the R0 server cell and is read from the server gate's results.
G_CLIENTS = (20, 40, 80)

# "Unknown is not a pass": a None share renders UNKNOWN, never False.
G_SATURATION_SHARE = 0.90

# --- verdicts: G1 / H1, one rung, corpus held fixed ------------------------------------
V_GRAPH_FASTER = "GRAPH-FASTER-AT-MATCHED-CORPUS"
V_POINTWISE_FASTER = "POINTWISE-FASTER-AT-MATCHED-CORPUS"
V_NOT_SEP = "NOT-SEPARATED"

# --- verdicts: G2 / H2, the composite over the ladder ----------------------------------
V_EDGE_SURVIVES = "MODEL-CLASS-EDGE-SURVIVES-MATCHING"
V_POINTWISE_AT_MATCHED = "POINTWISE-BETTER-AT-MATCHED-CORPUS"
V_MATCHED_NOT_ESTABLISHED = "MATCHED-EDGE-NOT-ESTABLISHED"

# --- verdicts: G3, the confound itself, same architecture / different corpus ------------
V_C1670_FASTER = "CORPUS-1670-FASTER"
V_C516_FASTER = "CORPUS-516-FASTER"
V_CORPUS_NOT_SEP = "CORPUS-NOT-SEPARATED"
V_CONFOUND_MATERIAL = "CONFOUND-IS-MATERIAL"
V_CONFOUND_IMMATERIAL = "CONFOUND-NOT-MATERIAL-HERE"

# --- verdicts: G4, the 2x2 -- does the model-class verdict depend on which corpus? -------
V_CORPUS_INDEPENDENT = "MODEL-CLASS-EDGE-IS-CORPUS-INDEPENDENT"
V_CORPUS_CONTINGENT = "MODEL-CLASS-EDGE-IS-CORPUS-CONTINGENT"


def read_g1(graph_by_seed: Mapping[int, float], pointwise_by_seed: Mapping[int, float],
            *, min_seeds: Optional[int] = None) -> dict:
    """One rung with the corpus HELD FIXED. Negative median = the graph arm is faster.

    Used for G1 (both arms at 1,670) and for H1 (both arms at 516). The reader does not know
    which corpus it was handed -- the caller pairs the arms and says so in the label -- so the
    orientation is fixed here once: first argument is the GRAPH arm. `read_pair_pct` is
    orientation-neutral and will happily invert its verdict string if the arguments are
    swapped, which is the defect this docstring exists to prevent.
    """
    r = read_pair_pct(graph_by_seed, pointwise_by_seed, tol=G_SEPARATE_PCT, alpha=G_ALPHA,
                      min_seeds=G_MIN_SEEDS if min_seeds is None else min_seeds)
    if r["verdict"] == V_UNREADABLE:
        return r
    v = {V_A_FASTER: V_GRAPH_FASTER, V_B_FASTER: V_POINTWISE_FASTER,
         V_PAIR_TIE: V_NOT_SEP}[r["verdict"]]
    return {**r, "verdict": v,
            "bar": {"separate_pct": G_SEPARATE_PCT, "alpha": G_ALPHA, "min_seeds": G_MIN_SEEDS}}


def read_g2(per_rung: Mapping[int, Mapping[str, object]]) -> dict:
    """The composite over the ladder, under `best_arm_v1`'s F2 rule UNCHANGED.

    Two wins and no losses, deliberately the same rule F2 used, so that "does the verdict move
    once corpus is matched?" is a comparison of like with like and not of two different bars.
    A rung that is UNREADABLE is counted as MISSING, never as either side: an arm lost to a
    resource kill must not be able to decide a model-class claim.
    """
    missing = [c for c in G_CLIENTS
               if c not in per_rung or per_rung[c].get("verdict") == V_UNREADABLE]
    if missing:
        return {"verdict": V_UNREADABLE, "missing": missing,
                "why": f"rungs {missing} unreadable; a model-class claim is not decided on a "
                       "partial ladder"}
    verdicts = [per_rung[c]["verdict"] for c in G_CLIENTS]
    graph = sum(v == V_GRAPH_FASTER for v in verdicts)
    point = sum(v == V_POINTWISE_FASTER for v in verdicts)
    if graph >= 2 and point == 0:
        return {"verdict": V_EDGE_SURVIVES, "graph_wins": graph, "pointwise_wins": point,
                "why": "with the corpus held fixed the graph arm is faster at >= 2 of 3 "
                       "unsaturated rungs and behind at none; the model-class edge is not a "
                       "corpus artifact"}
    if point >= 2:
        return {"verdict": V_POINTWISE_AT_MATCHED, "graph_wins": graph, "pointwise_wins": point,
                "why": "with the corpus held fixed the pointwise arm is faster at >= 2 of 3; "
                       "the pointwise clause is a model-class statement here, not an "
                       "artifact of the better pointwise arm having the smaller corpus"}
    if graph:
        why = (f"the matched ladder is SPLIT: graph faster at {graph} of 3, behind at {point}. "
               "The per-rung win(s) are real under a registered bar; what is not established "
               "is a claim about the ladder.")
    else:
        why = ("with the corpus held fixed the graph arm is faster at no rung; nothing here "
               "turns the split ladder into a model-class edge")
    return {"verdict": V_MATCHED_NOT_ESTABLISHED, "graph_wins": graph, "pointwise_wins": point,
            "why": why}


def read_g3(mpoff_1670_by_seed: Mapping[int, float], mpoff_516_by_seed: Mapping[int, float],
            *, min_seeds: Optional[int] = None) -> dict:
    """G3 at one rung: the SAME architecture (`mpoff`) at 1,670 vs 516 datasets.

    This is the confound `best_arm_v1` flagged, measured on the very cells it was flagged on.
    Negative median = the 1,670-dataset arm is faster. B1 read `CORPUS-DOES-NOT-HELP`, so the
    registered expectation is POSITIVE (1,670 slower).

    The two arms are different checkpoints of the same recipe, so the pairing is by checkpoint
    seed and the unit is a checkpoint, as everywhere else in this programme.
    """
    r = read_pair_pct(mpoff_1670_by_seed, mpoff_516_by_seed, tol=G_SEPARATE_PCT, alpha=G_ALPHA,
                      min_seeds=G_MIN_SEEDS if min_seeds is None else min_seeds)
    if r["verdict"] == V_UNREADABLE:
        return r
    v = {V_A_FASTER: V_C1670_FASTER, V_B_FASTER: V_C516_FASTER,
         V_PAIR_TIE: V_CORPUS_NOT_SEP}[r["verdict"]]
    return {**r, "verdict": v,
            "bar": {"separate_pct": G_SEPARATE_PCT, "alpha": G_ALPHA, "min_seeds": G_MIN_SEEDS}}


def read_g3_composite(per_rung: Mapping[int, Mapping[str, object]]) -> dict:
    """Was the flagged confound actually MATERIAL on these cells?

    Signed consequence. If the corpus separates the pointwise arm at ANY rung, then
    `best_arm_v1`'s F1 numbers mix a model-class term with a corpus term of measurable size,
    and F2 may not be quoted without G2 beside it. If it separates at NO rung, the confound was
    correctly flagged and is **not material here** -- F2's ladder may be quoted on its own, and
    that is a finding about this comparison only, never a general licence to mix corpora.
    """
    readable = {c: r for c, r in per_rung.items()
                if r.get("verdict") not in (None, V_UNREADABLE)}
    if not readable:
        return {"verdict": V_UNREADABLE, "why": "no rung carries both pointwise corpora"}
    sep = {c: r["verdict"] for c, r in readable.items() if r["verdict"] != V_CORPUS_NOT_SEP}
    missing = [c for c in G_CLIENTS if c not in readable]
    if sep:
        return {"verdict": V_CONFOUND_MATERIAL, "separated_at": sep, "unreadable": missing,
                "why": "corpus alone separates the pointwise arm at "
                       f"{sorted(sep)} clients; best_arm_v1's F1 mixes a model-class term with "
                       "a corpus term of measurable size and F2 must be quoted with G2"}
    return {"verdict": V_CONFOUND_IMMATERIAL, "separated_at": {}, "unreadable": missing,
            "why": "corpus alone does not separate the pointwise arm at any readable rung; the "
                   "confound was correctly flagged and is not material on THESE cells -- which "
                   "is not a licence to mix corpora anywhere else"}


def read_g4(g2: Mapping[str, object], h2: Optional[Mapping[str, object]]) -> dict:
    """G4, the 2x2: does the matched model-class verdict agree at 1,670 and at 516?

    `h2` is None until the 516 graph arm is trained and served; this returns UNREADABLE rather
    than treating an absent half as agreement. DESCRIPTIVE on two corpora -- two points cannot
    establish that a verdict is corpus-independent in general, only that it did not move here.
    """
    if not h2 or h2.get("verdict") == V_UNREADABLE:
        return {"verdict": V_UNREADABLE,
                "why": "the 516-corpus half is not served; a 2x2 is not read from one row"}
    if g2.get("verdict") == V_UNREADABLE:
        return {"verdict": V_UNREADABLE, "why": "the 1,670-corpus half is unreadable"}
    same = g2["verdict"] == h2["verdict"]
    return {"verdict": V_CORPUS_INDEPENDENT if same else V_CORPUS_CONTINGENT,
            "at_1670": g2["verdict"], "at_516": h2["verdict"],
            "note": "descriptive on two corpora; it did not move, which is not the same as "
                    "being corpus-independent in general"}


def saturated(queue_share: Optional[float]) -> Optional[bool]:
    """None stays None: unknown is not a pass (PARITY.md)."""
    return None if queue_share is None else bool(queue_share >= G_SATURATION_SHARE)
