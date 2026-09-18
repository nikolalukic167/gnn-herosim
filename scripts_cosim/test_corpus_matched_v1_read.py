"""corpus_matched_v1: the registered bars behave as signed, before a single number is read.

The whole G ladder is already on disk, so nothing stops this read from being run twice and
reported once. These tests are what make the bars binding: every pattern that could discharge
`best_arm_v1`'s corpus clause -- and every pattern that must NOT -- is named here first.

Run: pipenv run python3 -m pytest scripts_cosim/test_corpus_matched_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.corpus_matched_v1_read import (  # noqa: E402
    G_ALPHA, G_CLIENTS, G_MIN_SEEDS, G_SEPARATE_PCT,
    V_C1670_FASTER, V_C516_FASTER, V_CONFOUND_IMMATERIAL, V_CONFOUND_MATERIAL,
    V_CORPUS_CONTINGENT, V_CORPUS_INDEPENDENT, V_CORPUS_NOT_SEP,
    V_EDGE_SURVIVES, V_GRAPH_FASTER, V_MATCHED_NOT_ESTABLISHED, V_NOT_SEP,
    V_POINTWISE_AT_MATCHED, V_POINTWISE_FASTER,
    read_g1, read_g2, read_g3, read_g3_composite, read_g4, saturated,
)
from scripts_cosim.best_arm_v1_read import (  # noqa: E402
    F_ALPHA, F_MIN_SEEDS, F_SEPARATE_PCT,
)
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

N = G_MIN_SEEDS


def _by(base, step=0.1):
    return {s: base + step * s for s in range(N)}


def test_the_bars_are_f_and_b4s_bars_unchanged():
    """G must be readable against F rung by rung, which it is not if the bar moved."""
    assert (G_SEPARATE_PCT, G_ALPHA, G_MIN_SEEDS) == (F_SEPARATE_PCT, F_ALPHA, F_MIN_SEEDS)
    assert (G_SEPARATE_PCT, G_ALPHA, G_MIN_SEEDS) == (5.0, 0.05, 16)
    assert G_CLIENTS == (20, 40, 80)


# --- G1: one rung, corpus held fixed ------------------------------------------------------

def test_g1_is_oriented_so_a_faster_graph_arm_reads_graph_faster():
    assert read_g1(_by(80.0), _by(100.0))["verdict"] == V_GRAPH_FASTER
    assert read_g1(_by(120.0), _by(100.0))["verdict"] == V_POINTWISE_FASTER
    assert read_g1(_by(101.0), _by(100.0))["verdict"] == V_NOT_SEP
    # swapping the arguments must FLIP it -- read_pair_pct is orientation-neutral
    a, b = _by(80.0), _by(100.0)
    assert read_g1(b, a)["verdict"] == V_POINTWISE_FASTER


def test_g1_refuses_a_short_read_rather_than_relaxing_the_bar():
    short_a, short_b = {s: 80.0 for s in range(5)}, {s: 100.0 for s in range(5)}
    assert read_g1(short_a, short_b)["verdict"] == V_UNREADABLE
    long_a, long_b = {s: 80.0 for s in range(10)}, {s: 100.0 for s in range(10)}
    assert read_g1(long_a, long_b, min_seeds=10)["verdict"] == V_GRAPH_FASTER


# --- G2: the composite that can discharge best_arm_v1's corpus clause ----------------------

def _rungs(*verdicts):
    return {c: {"verdict": v, "median": -1.0 * (i + 1)}
            for i, (c, v) in enumerate(zip(G_CLIENTS, verdicts))}


def test_g2_uses_f2s_rule_unchanged_two_wins_and_no_losses():
    r = read_g2(_rungs(V_NOT_SEP, V_GRAPH_FASTER, V_GRAPH_FASTER))
    assert r["verdict"] == V_EDGE_SURVIVES and "not a corpus artifact" in r["why"]
    # two wins AND a loss is exactly best_arm_v1's actual F ladder shape -- it must not survive
    assert read_g2(_rungs(V_POINTWISE_FASTER, V_GRAPH_FASTER, V_GRAPH_FASTER))["verdict"] \
        == V_MATCHED_NOT_ESTABLISHED


def test_g2_reads_the_pointwise_side_when_the_pointwise_arm_wins_twice():
    assert read_g2(_rungs(V_POINTWISE_FASTER, V_POINTWISE_FASTER, V_NOT_SEP))["verdict"] \
        == V_POINTWISE_AT_MATCHED


def test_g2_does_not_call_a_split_ladder_a_failure():
    """The same false sentence best_arm_v1's F2 had to be fixed for: a split ladder contains a
    real win and must not be described as the graph arm failing everywhere."""
    r = read_g2(_rungs(V_POINTWISE_FASTER, V_NOT_SEP, V_GRAPH_FASTER))
    assert r["verdict"] == V_MATCHED_NOT_ESTABLISHED
    assert "SPLIT" in r["why"] and "faster at no rung" not in r["why"]
    r0 = read_g2(_rungs(V_POINTWISE_FASTER, V_NOT_SEP, V_NOT_SEP))
    assert "faster at no rung" in r0["why"]


def test_g2_will_not_decide_a_model_class_claim_on_a_partial_ladder():
    rungs = _rungs(V_GRAPH_FASTER, V_GRAPH_FASTER, V_NOT_SEP)
    rungs[80] = {"verdict": V_UNREADABLE}
    r = read_g2(rungs)
    assert r["verdict"] == V_UNREADABLE and r["missing"] == [80]
    del rungs[20]
    assert read_g2(rungs)["verdict"] == V_UNREADABLE


# --- G3: the confound itself ---------------------------------------------------------------

def test_g3_is_oriented_so_a_faster_1670_arm_reads_1670_faster():
    assert read_g3(_by(80.0), _by(100.0))["verdict"] == V_C1670_FASTER
    # B1's CORPUS-DOES-NOT-HELP shape: more data, slower live
    assert read_g3(_by(120.0), _by(100.0))["verdict"] == V_C516_FASTER
    assert read_g3(_by(101.0), _by(100.0))["verdict"] == V_CORPUS_NOT_SEP


def test_g3_composite_calls_the_confound_material_on_a_single_separated_rung():
    r = read_g3_composite({20: {"verdict": V_CORPUS_NOT_SEP}, 40: {"verdict": V_C516_FASTER},
                           80: {"verdict": V_CORPUS_NOT_SEP}})
    assert r["verdict"] == V_CONFOUND_MATERIAL and sorted(r["separated_at"]) == [40]
    assert "must be quoted with G2" in r["why"]


def test_g3_composite_says_immaterial_HERE_and_refuses_to_generalise():
    r = read_g3_composite({c: {"verdict": V_CORPUS_NOT_SEP} for c in G_CLIENTS})
    assert r["verdict"] == V_CONFOUND_IMMATERIAL
    assert "THESE cells" in r["why"] and "not a licence" in r["why"]


def test_g3_composite_is_unreadable_with_nothing_readable_and_reports_gaps():
    assert read_g3_composite({})["verdict"] == V_UNREADABLE
    assert read_g3_composite({c: {"verdict": V_UNREADABLE} for c in G_CLIENTS})["verdict"] \
        == V_UNREADABLE
    # a partially served G3 still decides, but it names what it could not read
    r = read_g3_composite({20: {"verdict": V_CORPUS_NOT_SEP}})
    assert r["verdict"] == V_CONFOUND_IMMATERIAL and r["unreadable"] == [40, 80]


# --- G4: the 2x2 ---------------------------------------------------------------------------

def test_g4_refuses_to_read_a_2x2_from_one_row():
    g2 = {"verdict": V_EDGE_SURVIVES}
    assert read_g4(g2, None)["verdict"] == V_UNREADABLE
    assert read_g4(g2, {"verdict": V_UNREADABLE})["verdict"] == V_UNREADABLE
    assert read_g4({"verdict": V_UNREADABLE}, {"verdict": V_EDGE_SURVIVES})["verdict"] \
        == V_UNREADABLE


def test_g4_agrees_only_when_both_corpora_give_the_same_verdict():
    same = read_g4({"verdict": V_EDGE_SURVIVES}, {"verdict": V_EDGE_SURVIVES})
    assert same["verdict"] == V_CORPUS_INDEPENDENT and "not the same as" in same["note"]
    diff = read_g4({"verdict": V_EDGE_SURVIVES}, {"verdict": V_MATCHED_NOT_ESTABLISHED})
    assert diff["verdict"] == V_CORPUS_CONTINGENT
    assert (diff["at_1670"], diff["at_516"]) == (V_EDGE_SURVIVES, V_MATCHED_NOT_ESTABLISHED)


def test_unknown_saturation_is_not_a_pass():
    assert saturated(None) is None
    assert saturated(0.73) is False
    assert saturated(0.95) is True
