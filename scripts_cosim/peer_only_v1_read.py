"""peer_only_v1 -- the registered read. Every bar below is a module constant.

Registered in docs/lineages/peer_only_v1.md before any arm was trained. Pure functions over
small summaries; the gate glue maps summary files onto them.

  A0  instrument   gnn/mpoff re-served reproduce partial_state_v3 P3 to 3 decimals
  A1  offline      peeronly vs mpoff / gnn held-out regret, paired by seed (ordering only)
  A2  PRIMARY      peeronly vs mpoff mean elapsed, live, paired by (cell, seed), R0 and R3
  A3  mechanism    peeronly queue vs gnn (lower by >= 5 %) and vs mpoff (within 5 %)
  A4  peer term    peeronly peer cost <= mpoff x 1.05
  B0  corpus       1,670 cache instrument
  B1  corpus lever each arm 1670 vs 516, paired by seed
  B2  headline     peeronly_1670 vs mpoff_1670 (A2 bars)
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from scripts_cosim.partial_state_v3_read import median, paired_tie

# --- registered bars -------------------------------------------------------------------
A0_TOL = 0.0005                # mean elapsed reproduced to three decimals
A1_TIE_PP, A1_ALPHA, A1_MIN_SEEDS = 1.0, 0.05, 12
A2_TIE_PCT, A2_ALPHA, A2_MIN_PAIRS = 5.0, 0.05, 12
A2_RUNGS = ("R0", "R3")
A3_QUEUE_IMPROVE_PCT = 5.0     # peeronly queue below gnn's by at least this, p < alpha
A3_TIE_PCT = 5.0               # ...and within this of mpoff's
A4_PEER_TOL = 0.05             # peer cost per task <= mpoff x (1 + tol)
B0_MAX_COLUMN_DIFF = 0.0
B0_MIN_DATASETS = 1500
B1_IMPROVE_PCT, B1_ALPHA = 5.0, 0.05
CHECKPOINT_SEEDS = (1, 2, 4, 5)

# --- verdict strings -------------------------------------------------------------------
V_A0_PASS, V_A0_FAIL = "INSTRUMENT-PASS", "MODEL-CHANGE-NOT-INERT"
V_BEATS, V_TIE, V_POINTWISE = "PEERONLY-BEATS-POINTWISE", "TIE", "POINTWISE-BETTER"
V_GIN_OVERREACTION, V_MECHANISM_NO = "GIN-IS-THE-OVERREACTION", "MECHANISM-NOT-CONFIRMED"
V_PEER_KEPT, V_PEER_LOST = "PEER-TERM-KEPT", "PEER-TERM-LOST"
V_CORPUS_HELPS, V_CORPUS_NO = "CORPUS-HELPS", "CORPUS-DOES-NOT-HELP"
V_UNREADABLE = "UNREADABLE"

PairKey = Tuple[str, int]      # (cell, checkpoint seed)


def read_a0(reproduced: Mapping[str, Tuple[float, float]]) -> dict:
    """arm -> (elapsed now, elapsed in partial_state_v3 P3). All must agree within A0_TOL."""
    rows = {k: (a, b, abs(a - b)) for k, (a, b) in reproduced.items()}
    ok = bool(rows) and all(d <= A0_TOL for _, _, d in rows.values())
    return {"verdict": V_A0_PASS if ok else V_A0_FAIL, "arms": rows, "tol": A0_TOL}


def _paired_rel(a: Mapping[PairKey, float], b: Mapping[PairKey, float], *, tol: float,
                alpha: float, min_pairs: int, lower_is_better: bool = True) -> dict:
    """paired_tie over (cell, seed) keys: relative % difference of a vs b."""
    keys = sorted(set(a) & set(b))
    ia = {i: a[k] for i, k in enumerate(keys)}
    ib = {i: b[k] for i, k in enumerate(keys)}
    r = paired_tie(ia, ib, tol=tol, alpha=alpha, min_seeds=min_pairs, relative=True,
                   lower_is_better=lower_is_better)
    r["pairs"] = keys
    return r


def read_a2_rung(peeronly: Mapping[PairKey, float], mpoff: Mapping[PairKey, float]) -> dict:
    """Mean elapsed, peeronly vs mpoff, paired by (cell, seed), one rung."""
    r = _paired_rel(peeronly, mpoff, tol=A2_TIE_PCT, alpha=A2_ALPHA, min_pairs=A2_MIN_PAIRS)
    if r["verdict"] == V_UNREADABLE:
        return r
    v = {"TIE": V_TIE, "ENCODING-HELPS": V_BEATS, "ENCODING-COSTS": V_POINTWISE}[r["verdict"]]
    return {**r, "verdict": v}


def headline(per_rung: Mapping[str, dict]) -> str:
    verdicts = [r["verdict"] for r in per_rung.values() if r.get("verdict") != V_UNREADABLE]
    if not verdicts:
        return V_UNREADABLE
    if V_POINTWISE in verdicts:
        return V_POINTWISE
    return V_BEATS if V_BEATS in verdicts else V_TIE


def read_a3_rung(peeronly_q: Mapping[PairKey, float], gnn_q: Mapping[PairKey, float],
                 mpoff_q: Mapping[PairKey, float]) -> dict:
    """Mean queue: peeronly below gnn by >= A3_QUEUE_IMPROVE_PCT (p < alpha) AND within
    A3_TIE_PCT of mpoff."""
    vs_gnn = _paired_rel(peeronly_q, gnn_q, tol=0.0, alpha=A3_ALPHA_DUMMY, min_pairs=A2_MIN_PAIRS)
    vs_mpoff = _paired_rel(peeronly_q, mpoff_q, tol=A3_TIE_PCT, alpha=A2_ALPHA, min_pairs=A2_MIN_PAIRS)
    if vs_gnn["verdict"] == V_UNREADABLE or vs_mpoff["verdict"] == V_UNREADABLE:
        return {"verdict": V_UNREADABLE, "vs_gnn": vs_gnn, "vs_mpoff": vs_mpoff}
    below_gnn = vs_gnn["median"] <= -A3_QUEUE_IMPROVE_PCT and vs_gnn["p"] < A2_ALPHA
    like_mpoff = vs_mpoff["verdict"] == "TIE"
    return {"verdict": V_GIN_OVERREACTION if (below_gnn and like_mpoff) else V_MECHANISM_NO,
            "below_gnn": below_gnn, "like_mpoff": like_mpoff, "vs_gnn": vs_gnn, "vs_mpoff": vs_mpoff}


A3_ALPHA_DUMMY = 1.0           # the vs-gnn read uses its own inequality, not paired_tie's verdict


def read_a4_rung(peeronly_peer: Mapping[PairKey, float], mpoff_peer: Mapping[PairKey, float]) -> dict:
    """Peer cost per task: peeronly <= mpoff x (1 + A4_PEER_TOL) on the paired median."""
    r = _paired_rel(peeronly_peer, mpoff_peer, tol=100.0 * A4_PEER_TOL, alpha=A2_ALPHA,
                    min_pairs=A2_MIN_PAIRS)
    if r["verdict"] == V_UNREADABLE:
        return r
    kept = r["median"] <= 100.0 * A4_PEER_TOL
    return {**r, "verdict": V_PEER_KEPT if kept else V_PEER_LOST}


def read_b1_rung(arm_1670: Mapping[PairKey, float], arm_516: Mapping[PairKey, float]) -> dict:
    """One arm, one rung: 1670-corpus checkpoints vs 516-corpus, paired by (cell, seed)."""
    r = _paired_rel(arm_1670, arm_516, tol=0.0, alpha=A3_ALPHA_DUMMY, min_pairs=A2_MIN_PAIRS)
    if r["verdict"] == V_UNREADABLE:
        return r
    helps = r["median"] <= -B1_IMPROVE_PCT and r["p"] < B1_ALPHA
    return {**r, "verdict": V_CORPUS_HELPS if helps else V_CORPUS_NO}


def read_b0(n_datasets: int, meta_agree: bool, ingredients_max_diff: float, test_ids_same: bool) -> dict:
    ok = (n_datasets >= B0_MIN_DATASETS and meta_agree and ingredients_max_diff <= B0_MAX_COLUMN_DIFF
          and test_ids_same)
    return {"verdict": "INSTRUMENT-PASS" if ok else "CORPUS-NOT-COMPARABLE", "n_datasets": n_datasets,
            "meta_agree": meta_agree, "ingredients_max_diff": ingredients_max_diff,
            "test_ids_same": test_ids_same}
