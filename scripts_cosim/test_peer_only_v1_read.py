"""Tests for peer_only_v1's registered read."""
from scripts_cosim.peer_only_v1_read import (
    A0_TOL, A2_ALPHA, A2_MIN_PAIRS, A2_RUNGS, A2_TIE_PCT, A3_QUEUE_IMPROVE_PCT, A4_PEER_TOL,
    B0_MIN_DATASETS, B1_IMPROVE_PCT, CHECKPOINT_SEEDS, V_A0_FAIL, V_A0_PASS, V_BEATS,
    V_CORPUS_HELPS, V_CORPUS_NO, V_GIN_OVERREACTION, V_MECHANISM_NO, V_PEER_KEPT, V_PEER_LOST,
    V_POINTWISE, V_TIE, V_UNREADABLE, headline, read_a0, read_a2_rung, read_a3_rung,
    read_a4_rung, read_b0, read_b1_rung, B3_ALPHA, B3_IMPROVE_PCT, B3_MIN_SEEDS, B3_RUNG,
    B3_SEEDS, V_UNDERPOWERED, collapse_to_seed, read_b3, B5_RUNGS, B5_SERVERS,
    V_MONOTONE, V_NOT_MONOTONE, read_b5, B7_CLIENTS, B7_LOAD_FLAT_PCT, B7_MIN_SEEDS,
    B7_SATURATED_QUEUE_SHARE, B7_SERVERS, V_DISPERSION, V_LOAD_SWEEP, classify_b7_rungs,
    B8_CLIENTS, read_b8,
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


def _b3_like(median_pct, p=0.001):
    return {"verdict": V_BEATS if (median_pct <= -5.0 and p < 0.05) else V_UNDERPOWERED,
            "median": median_pct, "p": p, "n": 16}


def test_b5_reports_monotone_and_the_crossover_rung():
    r = read_b5({"R0": _b3_like(+2.0, p=0.6), "R1": _b3_like(-1.0, p=0.4),
                 "R2": _b3_like(-6.0), "R3": _b3_like(-9.0)})
    assert r["verdict"] == V_MONOTONE
    assert r["crossover_rung"] == "R2" and r["crossover_servers"] == 24


def test_b5_falsifies_its_own_monotone_expectation():
    r = read_b5({"R0": _b3_like(-9.0), "R1": _b3_like(-1.0, p=0.4),
                 "R2": _b3_like(-6.0), "R3": _b3_like(-2.0, p=0.3)})
    assert r["verdict"] == V_NOT_MONOTONE and r["crossover_rung"] == "R0"


def test_b5_refuses_a_partial_ladder():
    r = read_b5({"R0": _b3_like(-9.0), "R1": {"verdict": V_UNREADABLE, "reason": "no summaries"},
                 "R2": _b3_like(-6.0), "R3": _b3_like(-2.0)})
    assert r["verdict"] == V_UNREADABLE and "R1" not in r["reason"]


def test_b5_rung_table_matches_the_gate():
    assert B5_RUNGS == ("R0", "R1", "R2", "R3")
    assert B5_SERVERS == {"R0": 6, "R1": 12, "R2": 24, "R3": 80}


def test_b7_saturation_bar_splits_the_rungs_on_reactive_alone():
    r = classify_b7_rungs({5: {"elapsed": 22.0, "queue": 13.9},      # 63 % -> unsaturated
                           10: {"elapsed": 23.0, "queue": 15.0},     # 65 % -> unsaturated
                           40: {"elapsed": 23.5, "queue": 22.0}})    # 94 % -> saturated
    assert r["saturated"] == [40] and r["unsaturated"] == [5, 10]
    assert r["primary_rung"] == 10, "the primary read is the LARGEST unsaturated rung"


def test_b7_calls_a_rising_baseline_a_load_sweep():
    r = classify_b7_rungs({5: {"elapsed": 22.0, "queue": 13.0},
                           80: {"elapsed": 300.0, "queue": 295.0}})
    assert r["verdict"] == V_LOAD_SWEEP


def test_b7_calls_a_flat_baseline_a_dispersion_sweep():
    r = classify_b7_rungs({5: {"elapsed": 22.0, "queue": 13.0},
                           80: {"elapsed": 23.5, "queue": 14.0}})
    assert r["verdict"] == V_DISPERSION and r["reactive_elapsed_spread_pct"] < 10.0


def test_b7_with_every_rung_saturated_has_no_primary():
    r = classify_b7_rungs({5: {"elapsed": 100.0, "queue": 99.0},
                           80: {"elapsed": 105.0, "queue": 104.0}})
    assert r["saturated"] == [5, 80] and r["primary_rung"] is None


def test_b7_constants_are_the_registered_values():
    assert B7_CLIENTS == (5, 10, 20, 40, 80) and B7_SERVERS == 6
    assert (B7_SATURATED_QUEUE_SHARE, B7_LOAD_FLAT_PCT, B7_MIN_SEEDS) == (0.90, 10.0, 16)


def _b4_like(median_pct, p=0.001):
    from scripts_cosim.peer_only_v1_read import V_BEST_IS_POINTWISE, V_BEST_NOT_ESTABLISHED
    v = V_BEST_IS_POINTWISE if (median_pct >= 5.0 and p < 0.05) else V_BEST_NOT_ESTABLISHED
    return {"verdict": v, "median": median_pct, "p": p, "n": 16}


def test_b8_holds_only_when_it_holds_at_both_unsaturated_rungs():
    from scripts_cosim.peer_only_v1_read import V_BEST_IS_POINTWISE, V_BEST_NOT_ESTABLISHED
    both = read_b8({40: _b4_like(+9.0), 80: _b4_like(+7.0)})
    assert both["verdict"] == V_BEST_IS_POINTWISE and both["scoped_to_saturated"] is False
    one = read_b8({40: _b4_like(+9.0), 80: _b4_like(-3.0, p=0.2)})
    assert one["verdict"] == V_BEST_NOT_ESTABLISHED and one["scoped_to_saturated"] is True


def test_b8_refuses_a_partial_ladder():
    assert read_b8({40: _b4_like(+9.0)})["verdict"] == V_UNREADABLE


def test_b8_rungs_are_the_unsaturated_ones_b7_measured():
    assert B8_CLIENTS == (40, 80) and set(B8_CLIENTS) < set(B7_CLIENTS)


# --- C1 / C2: the bipartite arm at power (AMENDMENTS 6 and 7) -----------------------------

def _seeds(base, delta_pct, n=16, jitter=0.0):
    """16 checkpoint values for an arm `delta_pct` from `base`, with a deterministic wobble."""
    out = {}
    for s in range(1, n + 1):
        w = jitter * ((s % 5) - 2)
        out[s] = base * (1.0 + delta_pct / 100.0 + w / 100.0)
    return out


def test_c1_calls_a_large_consistent_peeronly_lead_bipartite_costs():
    from scripts_cosim.peer_only_v1_read import read_c1, V_BIPARTITE_COSTS
    r = read_c1(_seeds(100.0, -20.0, jitter=1.0), _seeds(100.0, 0.0))
    assert r["verdict"] == V_BIPARTITE_COSTS and r["n"] == 16


def test_c1_calls_a_small_lead_not_separated():
    from scripts_cosim.peer_only_v1_read import read_c1, V_BIPARTITE_NOT_SEP
    r = read_c1(_seeds(100.0, -1.0, jitter=1.0), _seeds(100.0, 0.0))
    assert r["verdict"] == V_BIPARTITE_NOT_SEP


def test_c1_can_read_the_gin_as_a_help_that_direction_is_reachable():
    from scripts_cosim.peer_only_v1_read import read_c1, V_BIPARTITE_HELPS
    r = read_c1(_seeds(100.0, +20.0, jitter=1.0), _seeds(100.0, 0.0))
    assert r["verdict"] == V_BIPARTITE_HELPS


def test_c1_refuses_fewer_than_sixteen_checkpoints():
    """The whole point of C1: a 4-checkpoint read is not a read."""
    from scripts_cosim.peer_only_v1_read import read_c1
    r = read_c1(_seeds(100.0, -20.0, n=4), _seeds(100.0, 0.0, n=4))
    assert r["verdict"] == V_UNREADABLE


def test_c1_ladder_needs_both_rungs_to_state_the_clause_flatly():
    from scripts_cosim.peer_only_v1_read import (read_c1, read_c1_ladder, V_BIPARTITE_COSTS,
                                                 V_BIPARTITE_NOT_SEP)
    costs = read_c1(_seeds(100.0, -20.0, jitter=1.0), _seeds(100.0, 0.0))
    flat = read_c1(_seeds(100.0, -1.0, jitter=1.0), _seeds(100.0, 0.0))
    assert read_c1_ladder({"R3": costs, "R0": costs})["verdict"] == V_BIPARTITE_COSTS
    assert read_c1_ladder({"R3": costs, "R0": flat})["verdict"] == V_BIPARTITE_NOT_SEP


def test_c1_ladder_refuses_a_partial_ladder():
    from scripts_cosim.peer_only_v1_read import read_c1, read_c1_ladder
    costs = read_c1(_seeds(100.0, -20.0, jitter=1.0), _seeds(100.0, 0.0))
    assert read_c1_ladder({"R3": costs})["verdict"] == V_UNREADABLE


def test_c2_one_rung_reads_the_bipartite_arm_against_reactive():
    from scripts_cosim.peer_only_v1_read import (read_c2_rung, V_BIPARTITE_BEATS_REACTIVE,
                                                 V_BIPARTITE_LOSES_REACTIVE)
    beats = read_c2_rung(_seeds(100.0, -15.0, jitter=1.0), _seeds(100.0, 0.0))
    assert beats["verdict"] == V_BIPARTITE_BEATS_REACTIVE
    loses = read_c2_rung(_seeds(100.0, +15.0, jitter=1.0), _seeds(100.0, 0.0))
    assert loses["verdict"] == V_BIPARTITE_LOSES_REACTIVE


def test_c2_works_if_it_works_at_either_unsaturated_rung():
    from scripts_cosim.peer_only_v1_read import (read_c2, read_c2_rung, read_c1,
                                                 V_BIPARTITE_BEATS_REACTIVE,
                                                 V_BIPARTITE_LOSES_REACTIVE)
    beats = read_c2_rung(_seeds(100.0, -15.0, jitter=1.0), _seeds(100.0, 0.0))
    loses = read_c2_rung(_seeds(100.0, +15.0, jitter=1.0), _seeds(100.0, 0.0))
    sib = read_c1(_seeds(100.0, -8.0, jitter=1.0), _seeds(100.0, 0.0))
    r = read_c2({40: loses, 80: beats}, {40: sib, 80: sib})
    assert r["verdict"] == V_BIPARTITE_BEATS_REACTIVE and r["rungs_where_it_works"] == [80]
    r2 = read_c2({40: loses, 80: loses}, {40: sib, 80: sib})
    assert r2["verdict"] == V_BIPARTITE_LOSES_REACTIVE and r2["rungs_where_it_works"] == []


def test_c2_refuses_a_partial_ladder():
    from scripts_cosim.peer_only_v1_read import read_c2, read_c2_rung, read_c1
    beats = read_c2_rung(_seeds(100.0, -15.0, jitter=1.0), _seeds(100.0, 0.0))
    sib = read_c1(_seeds(100.0, -8.0, jitter=1.0), _seeds(100.0, 0.0))
    assert read_c2({40: beats}, {40: sib})["verdict"] == V_UNREADABLE


def test_c1_c2_constants_are_the_registered_values():
    from scripts_cosim.peer_only_v1_read import (C1_RUNGS, C1_MIN_SEEDS, C1_SEPARATE_PCT,
                                                 C1_ALPHA, C2_CLIENTS, C2_MIN_SEEDS,
                                                 C2_SEPARATE_PCT, C2_ALPHA)
    assert C1_RUNGS == ("R3", "R0") and (C1_MIN_SEEDS, C1_SEPARATE_PCT, C1_ALPHA) == (16, 5.0, 0.05)
    assert C2_CLIENTS == (40, 80) and (C2_MIN_SEEDS, C2_SEPARATE_PCT, C2_ALPHA) == (16, 5.0, 0.05)
    assert C2_CLIENTS == B8_CLIENTS      # C2 reads the same rungs B8 did, deliberately
