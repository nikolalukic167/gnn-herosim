"""Tests for peer_only_v1's registered read."""
from scripts_cosim.peer_only_v1_read import (
    A0_TOL, A2_ALPHA, A2_MIN_PAIRS, A2_RUNGS, A2_TIE_PCT, A3_QUEUE_IMPROVE_PCT, A4_PEER_TOL,
    B0_MIN_DATASETS, B1_IMPROVE_PCT, CHECKPOINT_SEEDS, V_A0_FAIL, V_A0_PASS, V_BEATS,
    V_CORPUS_HELPS, V_CORPUS_NO, V_GIN_OVERREACTION, V_MECHANISM_NO, V_PEER_KEPT, V_PEER_LOST,
    V_POINTWISE, V_TIE, V_UNREADABLE, headline, read_a0, read_a2_rung, read_a3_rung,
    read_a4_rung, read_b0, read_b1_rung,
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
