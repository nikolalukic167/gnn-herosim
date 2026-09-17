"""Tests for peer_only_v1's registered read."""
from scripts_cosim.peer_only_v1_read import (
    A0_TOL, A2_ALPHA, A2_MIN_PAIRS, A2_RUNGS, A2_TIE_PCT, A3_QUEUE_IMPROVE_PCT, A4_PEER_TOL,
    B0_MIN_DATASETS, B1_IMPROVE_PCT, CHECKPOINT_SEEDS, V_A0_FAIL, V_A0_PASS, V_BEATS,
    V_CORPUS_HELPS, V_CORPUS_NO, V_GIN_OVERREACTION, V_MECHANISM_NO, V_PEER_KEPT, V_PEER_LOST,
    V_POINTWISE, V_TIE, V_UNREADABLE, headline, read_a0, read_a2_rung, read_a3_rung,
    read_a4_rung, read_b0, read_b1_rung, B3_ALPHA, B3_IMPROVE_PCT, B3_MIN_SEEDS, B3_RUNG,
    B3_SEEDS, V_UNDERPOWERED, collapse_to_seed, read_b3,
)

CELLS = ("cs6s9001", "cs6s9002", "cs6s9003", "cs6s9005")


def _pairs(base=20.0, scale=1.0, spread=0.0):
    out = {}
    for i, c in enumerate(CELLS):
        for j, s in enumerate(CHECKPOINT_SEEDS):
            out[(c, s)] = (base + i + 0.1 * j) * (scale + spread * ((i + j) % 3 - 1) * 0.01)
    return out


def test_registered_constants():
    assert A0_TOL == 0.0005
    assert (A2_TIE_PCT, A2_ALPHA, A2_MIN_PAIRS) == (5.0, 0.05, 12)
    assert A2_RUNGS == ("R0", "R3") and CHECKPOINT_SEEDS == (1, 2, 4, 5)
    assert (A3_QUEUE_IMPROVE_PCT, A4_PEER_TOL, B0_MIN_DATASETS, B1_IMPROVE_PCT) == (5.0, 0.05, 1500, 5.0)


def test_a0_reproduction_to_three_decimals():
    assert read_a0({"gnn": (26.5551, 26.5553), "mpoff": (35.2700, 35.2700)})["verdict"] == V_A0_PASS
    assert read_a0({"gnn": (26.5551, 26.5563)})["verdict"] == V_A0_FAIL
    assert read_a0({})["verdict"] == V_A0_FAIL


def test_a2_beats_tie_pointwise_and_headline():
    mpoff = _pairs()
    assert read_a2_rung(_pairs(scale=0.85), mpoff)["verdict"] == V_BEATS
    assert read_a2_rung(_pairs(scale=1.01), mpoff)["verdict"] == V_TIE
    assert read_a2_rung(_pairs(scale=1.20), mpoff)["verdict"] == V_POINTWISE
    assert headline({"R0": {"verdict": V_TIE}, "R3": {"verdict": V_BEATS}}) == V_BEATS
    assert headline({"R0": {"verdict": V_BEATS}, "R3": {"verdict": V_POINTWISE}}) == V_POINTWISE
    assert headline({"R0": {"verdict": V_TIE}, "R3": {"verdict": V_TIE}}) == V_TIE
    assert headline({"R0": {"verdict": V_UNREADABLE}}) == V_UNREADABLE


def test_a2_needs_twelve_pairs():
    few = {k: v for k, v in _pairs().items() if k[0] in CELLS[:2]}     # 8 pairs
    assert read_a2_rung(few, _pairs())["verdict"] == V_UNREADABLE


def test_a3_mechanism_needs_both_halves():
    gnn_q, mpoff_q = _pairs(base=30.0), _pairs(base=20.0)
    # peeronly queue like mpoff (10 % below gnn's scale-equivalent) -> fires
    assert read_a3_rung(_pairs(base=20.0, scale=1.01), gnn_q, mpoff_q)["verdict"] == V_GIN_OVERREACTION
    # peeronly queue like gnn -> not below gnn -> no
    assert read_a3_rung(_pairs(base=30.0), gnn_q, mpoff_q)["verdict"] == V_MECHANISM_NO
    # below gnn but far below mpoff too (not "within 5 %") -> no
    assert read_a3_rung(_pairs(base=12.0), gnn_q, mpoff_q)["verdict"] == V_MECHANISM_NO


def test_a4_peer_term_kept_or_lost():
    mpoff_peer = _pairs(base=5.0)
    assert read_a4_rung(_pairs(base=5.0, scale=0.97), mpoff_peer)["verdict"] == V_PEER_KEPT
    assert read_a4_rung(_pairs(base=5.0, scale=1.04), mpoff_peer)["verdict"] == V_PEER_KEPT
    assert read_a4_rung(_pairs(base=5.0, scale=1.15), mpoff_peer)["verdict"] == V_PEER_LOST


def test_b1_corpus_lever():
    a516 = _pairs()
    assert read_b1_rung(_pairs(scale=0.90), a516)["verdict"] == V_CORPUS_HELPS
    assert read_b1_rung(_pairs(scale=0.97), a516)["verdict"] == V_CORPUS_NO


def test_b0_instrument():
    assert read_b0(1670, True, 0.0, True)["verdict"] == "INSTRUMENT-PASS"
    assert read_b0(1499, True, 0.0, True)["verdict"] == "CORPUS-NOT-COMPARABLE"
    assert read_b0(1670, True, 1e-9, True)["verdict"] == "CORPUS-NOT-COMPARABLE"
    assert read_b0(1670, True, 0.0, False)["verdict"] == "CORPUS-NOT-COMPARABLE"


def _seed_pairs(seeds, base=20.0, scale=1.0, jitter=0.0):
    """One (cell, seed) entry per cell for each seed, so collapse_to_seed has a full row."""
    out = {}
    for i, c in enumerate(CELLS):
        for s in seeds:
            out[(c, s)] = (base + i) * scale * (1.0 + jitter * ((s % 3) - 1))
    return out


def test_collapse_to_seed_takes_the_median_over_cells():
    arm = {("cs6s9001", 1): 10.0, ("cs6s9002", 1): 20.0, ("cs6s9003", 1): 30.0,
           ("cs6s9005", 1): 40.0}
    assert collapse_to_seed(arm, rung_cells=CELLS) == {1: 25.0}


def test_collapse_to_seed_fails_loud_on_a_ragged_seed():
    arm = {("cs6s9001", 1): 10.0, ("cs6s9002", 1): 20.0}
    try:
        collapse_to_seed(arm, rung_cells=CELLS)
    except ValueError as exc:
        assert "cs6s9003" in str(exc)
    else:
        raise AssertionError("a seed missing two cells must fail loud, not average over what it has")


def test_b3_requires_every_trained_checkpoint():
    few = tuple(range(1, 13))
    po = collapse_to_seed(_seed_pairs(few, scale=0.80), rung_cells=CELLS)
    mp = collapse_to_seed(_seed_pairs(few), rung_cells=CELLS)
    assert read_b3(po, mp)["verdict"] == V_UNREADABLE
    assert read_b3(po, mp)["n"] == len(few) < B3_MIN_SEEDS


def test_b3_confirms_only_a_consistent_and_large_seed_level_margin():
    po = collapse_to_seed(_seed_pairs(B3_SEEDS, scale=0.80), rung_cells=CELLS)
    mp = collapse_to_seed(_seed_pairs(B3_SEEDS), rung_cells=CELLS)
    r = read_b3(po, mp)
    assert r["verdict"] == V_BEATS and r["n"] == B3_MIN_SEEDS and r["median"] < -B3_IMPROVE_PCT
    # inside the band -> underpowered, not a win
    near = collapse_to_seed(_seed_pairs(B3_SEEDS, scale=0.98), rung_cells=CELLS)
    assert read_b3(near, mp)["verdict"] == V_UNDERPOWERED
    # large but the wrong way -> also not a win
    worse = collapse_to_seed(_seed_pairs(B3_SEEDS, scale=1.20), rung_cells=CELLS)
    assert read_b3(worse, mp)["verdict"] == V_UNDERPOWERED


def test_b3_bar_constants_are_registered_values():
    assert (B3_IMPROVE_PCT, B3_ALPHA, B3_MIN_SEEDS, B3_RUNG) == (5.0, 0.05, 16, "R3")
    assert len(B3_SEEDS) == 16 and set(CHECKPOINT_SEEDS) <= set(B3_SEEDS)
