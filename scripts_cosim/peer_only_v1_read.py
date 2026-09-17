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
  B3  power        AMENDMENT 1 (2026-09-17): B2 again with the CHECKPOINT as the unit
  B4  power        AMENDMENT 2 (2026-09-17): clause 3 (vs the best pointwise arm) likewise
  B5  scale        AMENDMENT 3 (2026-09-17): the B3 contrast at 6 / 12 / 24 / 80 servers
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

# --- B3, AMENDMENT 1, registered 2026-09-17 BEFORE the extra arms were submitted ----------
# B2 pairs by (cell, checkpoint seed) and reports n = 16 from 4 checkpoints x 4 cells. The bar
# was signed that way and is read that way. But a claim about an ARCHITECTURE has the
# checkpoint as its independent unit, and four of them cannot produce a two-sided sign-test p
# below 0.125 however consistent they are. B3 re-reads the same contrast with one value per
# checkpoint -- the median over that checkpoint's four cells -- on all 16 trained seeds.
B3_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
B3_MIN_SEEDS = 16              # every trained checkpoint, or the read is UNREADABLE
B3_IMPROVE_PCT, B3_ALPHA = 5.0, 0.05
B3_RUNG = "R3"                 # the rung B2's headline is claimed at
# Registered expectation: POSITIVE in direction (4/4 checkpoints already agree), UNCERTAIN on
# significance. If B3 does not clear, the headline is recorded as UNDERPOWERED and CLAUDE.md's
# standing answer reverts to its pre-2026-09-17 wording. That consequence is signed here.

# --- B4, AMENDMENT 2, registered 2026-09-17 BEFORE the extra arms were submitted -----------
# The same defect on the other side of the ledger. Clause 3 of the head -- "the winning arm
# still loses to the best pointwise arm the program has", peeronly_1670 vs mpoff_516 at
# +10.40 % -- is also a 4-checkpoint read, and B3 has just shown a 4-checkpoint read
# overstating an effect by more than 2x. mpoff_516 is partial_state_v3's mpoff, whose 16 seeds
# all exist; A0 proves the re-serve is bit-identical, so the 12 unused ones extend the SAME arm.
# Paired by training seed across the two corpora, exactly as B1 pairs.
B4_MIN_SEEDS = 16
B4_TIE_PCT, B4_ALPHA = 5.0, 0.05
B4_RUNG = "R3"
# Registered expectation: the 4-seed read says peeronly_1670 LOSES by +10.40 %. If B4 confirms
# (median > +5 %, p < alpha) clause 3 stands as written. If it ties or reverses, clause 3 is
# WITHDRAWN and CLAUDE.md's second half ("the best scheduler is still a pointwise one") is
# rewritten to match. That consequence is signed here, before the data.

# --- B5, AMENDMENT 3, registered 2026-09-17 BEFORE the extra arms were submitted -----------
# B2/B3 measure the contrast at two rungs only: 6 servers (TIE) and 80 (peeronly ahead). The
# node's clause 6 carries "the winning rung is saturated" as a caveat with no measurement
# between the two ends. partial_state_v3 P3 already minted and served 12- and 24-server cells
# WITH their reactive baselines, so the middle of the curve costs only the two learned arms.
# B5 reads the B3 statistic (one value per checkpoint, 16 seeds) at each of the four rungs.
B5_RUNGS = ("R0", "R1", "R2", "R3")
B5_SERVERS = {"R0": 6, "R1": 12, "R2": 24, "R3": 80}
B5_MIN_SEEDS = 16
B5_IMPROVE_PCT, B5_ALPHA = 5.0, 0.05     # same bar as B3, applied per rung
B5_DISCLOSED_MIN_SEEDS = 12              # the DISCLOSED read only; never the registered bar
# Registered expectation: MONOTONE -- the margin grows with cluster size, because the measured
# advantage is queue-driven (99 % of it) and queue pressure grows with the rung. Falsified if
# the per-rung medians are not ordered R0 >= R1 >= R2 >= R3, which is a real prediction: a
# non-monotone curve would mean the advantage is a property of one operating point, not of
# load. The crossover rung (the first at which the bar clears) is reported either way.
#
# Arms lost to a RESOURCE KILL are read by cause, never by count (gate-tools 2026-09-16, the
# partial_state_v3 P3-a correction). B5_MIN_SEEDS is NOT relaxed when an arm dies: the
# registered read says UNREADABLE, and a second DISCLOSED read on the checkpoints complete at
# every rung is printed beneath it, with the excluded checkpoint and its cause named. Moving
# the bar after seeing which arm died would be tuning on the data; printing nothing would
# throw away 15 good checkpoints. Both numbers, clearly labelled, is the honest answer.

# --- verdict strings -------------------------------------------------------------------
V_A0_PASS, V_A0_FAIL = "INSTRUMENT-PASS", "MODEL-CHANGE-NOT-INERT"
V_BEATS, V_TIE, V_POINTWISE = "PEERONLY-BEATS-POINTWISE", "TIE", "POINTWISE-BETTER"
V_GIN_OVERREACTION, V_MECHANISM_NO = "GIN-IS-THE-OVERREACTION", "MECHANISM-NOT-CONFIRMED"
V_PEER_KEPT, V_PEER_LOST = "PEER-TERM-KEPT", "PEER-TERM-LOST"
V_CORPUS_HELPS, V_CORPUS_NO = "CORPUS-HELPS", "CORPUS-DOES-NOT-HELP"
V_UNDERPOWERED = "PEERONLY-BEATS-POINTWISE-UNDERPOWERED"
V_BEST_IS_POINTWISE, V_BEST_NOT_ESTABLISHED = "POINTWISE-STILL-BEST", "BEST-ARM-NOT-ESTABLISHED"
V_MONOTONE, V_NOT_MONOTONE = "MARGIN-GROWS-WITH-SCALE", "MARGIN-NOT-MONOTONE-IN-SCALE"
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


def collapse_to_seed(arm: Mapping[PairKey, float], *, rung_cells: Sequence[str]) -> Dict[int, float]:
    """One value per checkpoint seed: the median over that checkpoint's cells.

    Fails loud rather than averaging over a ragged set -- every seed must carry every cell,
    otherwise two checkpoints are being compared on different clusters.
    """
    by_seed: Dict[int, Dict[str, float]] = {}
    for (cell, seed), v in arm.items():
        by_seed.setdefault(int(seed), {})[cell] = float(v)
    out: Dict[int, float] = {}
    for seed, cells in by_seed.items():
        missing = [c for c in rung_cells if c not in cells]
        if missing:
            raise ValueError(f"FAIL LOUD: checkpoint seed {seed} is missing cells {missing}")
        out[seed] = median([cells[c] for c in rung_cells])
    return out


def read_b3(peeronly_by_seed: Mapping[int, float], mpoff_by_seed: Mapping[int, float],
            *, min_seeds: Optional[int] = None) -> dict:
    """B2's contrast with the CHECKPOINT as the unit -- one value per seed, all 16 required.

    `min_seeds` is ONLY for the disclosed read of a ladder that lost an arm to a resource kill
    (B5). It never relaxes the registered bar: the caller must label such a result disclosed.
    """
    r = paired_tie(dict(peeronly_by_seed), dict(mpoff_by_seed), tol=B3_IMPROVE_PCT,
                   alpha=B3_ALPHA, min_seeds=B3_MIN_SEEDS if min_seeds is None else min_seeds,
                   relative=True)
    if r["verdict"] == V_UNREADABLE:
        return r
    beats = r["median"] <= -B3_IMPROVE_PCT and r["p"] < B3_ALPHA
    return {**r, "verdict": V_BEATS if beats else V_UNDERPOWERED,
            "bar": {"improve_pct": B3_IMPROVE_PCT, "alpha": B3_ALPHA, "min_seeds": B3_MIN_SEEDS}}


def read_b4(peeronly_1670_by_seed: Mapping[int, float], mpoff_516_by_seed: Mapping[int, float]) -> dict:
    """Clause 3 with the checkpoint as the unit: is the best pointwise arm still ahead?"""
    r = paired_tie(dict(peeronly_1670_by_seed), dict(mpoff_516_by_seed), tol=B4_TIE_PCT,
                   alpha=B4_ALPHA, min_seeds=B4_MIN_SEEDS, relative=True)
    if r["verdict"] == V_UNREADABLE:
        return r
    pointwise_ahead = r["median"] >= B4_TIE_PCT and r["p"] < B4_ALPHA
    return {**r, "verdict": V_BEST_IS_POINTWISE if pointwise_ahead else V_BEST_NOT_ESTABLISHED,
            "bar": {"tie_pct": B4_TIE_PCT, "alpha": B4_ALPHA, "min_seeds": B4_MIN_SEEDS}}


def read_b5(per_rung: Mapping[str, dict]) -> dict:
    """rung -> read_b3 result. Reports the crossover rung and whether the margin is monotone."""
    readable = {r: v for r, v in per_rung.items() if v.get("verdict") != V_UNREADABLE}
    if len(readable) < len(B5_RUNGS):
        return {"verdict": V_UNREADABLE, "reason": f"only {sorted(readable)} readable",
                "per_rung": dict(per_rung)}
    meds = [readable[r]["median"] for r in B5_RUNGS]
    monotone = all(a >= b for a, b in zip(meds, meds[1:]))
    crossover = next((r for r in B5_RUNGS
                      if readable[r]["median"] <= -B5_IMPROVE_PCT and readable[r]["p"] < B5_ALPHA), None)
    return {"verdict": V_MONOTONE if monotone else V_NOT_MONOTONE,
            "medians": {r: readable[r]["median"] for r in B5_RUNGS},
            "servers": dict(B5_SERVERS), "crossover_rung": crossover,
            "crossover_servers": B5_SERVERS.get(crossover) if crossover else None,
            "per_rung": dict(per_rung)}


def read_b0(n_datasets: int, meta_agree: bool, ingredients_max_diff: float, test_ids_same: bool) -> dict:
    ok = (n_datasets >= B0_MIN_DATASETS and meta_agree and ingredients_max_diff <= B0_MAX_COLUMN_DIFF
          and test_ids_same)
    return {"verdict": "INSTRUMENT-PASS" if ok else "CORPUS-NOT-COMPARABLE", "n_datasets": n_datasets,
            "meta_agree": meta_agree, "ingredients_max_diff": ingredients_max_diff,
            "test_ids_same": test_ids_same}
