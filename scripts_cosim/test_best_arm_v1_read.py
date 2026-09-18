"""best_arm_v1: the registered bars behave as signed, before either missing arm-set is served.

This lineage can overturn a clause in CLAUDE.md, so the reader is exercised on every pattern
that could do so — and on the ones that must NOT, in particular a partial ladder.

Run: pipenv run python3 -m pytest scripts_cosim/test_best_arm_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.best_arm_v1_read import (  # noqa: E402
    F_ALPHA, F_CLIENTS, F_MIN_SEEDS, F_SEPARATE_PCT,
    V_F1_NOT_SEP, V_GRAPH_FASTER, V_GRAPH_IS_BEST, V_MARGIN_GROWS, V_MARGIN_NOT_MONOTONE,
    V_POINTWISE_FASTER, V_POINTWISE_STILL_BEST, V_STILL_NOT_ESTABLISHED,
    read_f1, read_f2, read_f3, saturated,
)
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

N = F_MIN_SEEDS


def _by(base, step=0.1):
    return {s: base + step * s for s in range(N)}


def test_the_bars_are_b4s_bars_unchanged():
    assert (F_SEPARATE_PCT, F_ALPHA, F_MIN_SEEDS) == (5.0, 0.05, 16)
    assert F_CLIENTS == (20, 40, 80)


def test_f1_is_oriented_so_a_faster_graph_arm_reads_graph_arm_faster():
    assert read_f1(_by(80.0), _by(100.0))["verdict"] == V_GRAPH_FASTER
    assert read_f1(_by(120.0), _by(100.0))["verdict"] == V_POINTWISE_FASTER
    assert read_f1(_by(101.0), _by(100.0))["verdict"] == V_F1_NOT_SEP
    # swapping the arguments must flip it, not preserve it
    a, b = _by(80.0), _by(100.0)
    assert read_f1(b, a)["verdict"] == V_POINTWISE_FASTER


def test_f1_refuses_a_short_read_rather_than_relaxing_b4s_bar():
    """The registered read refuses; a DISCLOSED read is possible only because the caller asked
    for it by name. Note what the disclosed read then says at n = 5: NOT-SEPARATED, because a
    two-sided Wilcoxon on 5 pairs cannot reach p < 0.05 at all (its minimum is 0.0625). That is
    the reader being honest about power, and it is why the registered bar is 16."""
    short_a, short_b = {s: 80.0 for s in range(5)}, {s: 100.0 for s in range(5)}
    assert read_f1(short_a, short_b)["verdict"] == V_UNREADABLE
    disclosed5 = read_f1(short_a, short_b, min_seeds=5)
    assert disclosed5["verdict"] == V_F1_NOT_SEP and disclosed5["median"] < -F_SEPARATE_PCT

    # At n = 10 the same 20 % gap does clear, so the refusal above is about power, not a bug.
    long_a, long_b = {s: 80.0 for s in range(10)}, {s: 100.0 for s in range(10)}
    assert read_f1(long_a, long_b, min_seeds=10)["verdict"] == V_GRAPH_FASTER


# --- F2: the composite that can move CLAUDE.md -------------------------------------------

def _rungs(*verdicts):
    return {c: {"verdict": v, "median": -1.0 * (i + 1)} for i, (c, v) in enumerate(zip(F_CLIENTS, verdicts))}


def test_f2_overturns_the_clause_only_on_two_wins_and_no_losses():
    r = read_f2(_rungs(V_F1_NOT_SEP, V_GRAPH_FASTER, V_GRAPH_FASTER))
    assert r["verdict"] == V_GRAPH_IS_BEST and "OVERTURNED" in r["why"]
    # two wins but a loss elsewhere is NOT enough to overturn a programme-level clause
    assert read_f2(_rungs(V_POINTWISE_FASTER, V_GRAPH_FASTER, V_GRAPH_FASTER))["verdict"] \
        == V_STILL_NOT_ESTABLISHED


def test_f2_restores_the_clause_when_the_pointwise_arm_wins_twice():
    assert read_f2(_rungs(V_POINTWISE_FASTER, V_POINTWISE_FASTER, V_F1_NOT_SEP))["verdict"] \
        == V_POINTWISE_STILL_BEST


def test_f2_falls_through_to_not_established_on_a_split_or_all_ties():
    assert read_f2(_rungs(V_F1_NOT_SEP, V_F1_NOT_SEP, V_F1_NOT_SEP))["verdict"] \
        == V_STILL_NOT_ESTABLISHED
    # A split ladder must NOT be described as "not reproduced" -- the win at one rung is real.
    r = read_f2(_rungs(V_POINTWISE_FASTER, V_F1_NOT_SEP, V_GRAPH_FASTER))
    assert r["verdict"] == V_STILL_NOT_ESTABLISHED
    assert "SPLIT" in r["why"] and "NOT reproduced" not in r["why"]
    # Only a ladder with NO graph win may say the margin did not reproduce.
    r0 = read_f2(_rungs(V_POINTWISE_FASTER, V_F1_NOT_SEP, V_F1_NOT_SEP))
    assert "NOT reproduced" in r0["why"]


def test_f2_will_not_decide_a_programme_level_clause_on_a_partial_ladder():
    """An arm lost to a resource kill must never be able to move CLAUDE.md."""
    rungs = _rungs(V_GRAPH_FASTER, V_GRAPH_FASTER, V_F1_NOT_SEP)
    rungs[80] = {"verdict": V_UNREADABLE}
    r = read_f2(rungs)
    assert r["verdict"] == V_UNREADABLE and r["missing"] == [80]
    del rungs[20]
    assert read_f2(rungs)["verdict"] == V_UNREADABLE


# --- F3 and saturation --------------------------------------------------------------------

def test_f3_is_descriptive_and_says_so():
    grows = {20: {"median": -1.0}, 40: {"median": -8.0}, 80: {"median": -15.0}}
    r = read_f3(grows)
    assert r["verdict"] == V_MARGIN_GROWS and "descriptive only" in r["note"]
    flat = {20: {"median": -8.0}, 40: {"median": -1.0}, 80: {"median": -15.0}}
    assert read_f3(flat)["verdict"] == V_MARGIN_NOT_MONOTONE


def test_unknown_saturation_is_not_a_pass():
    assert saturated(None) is None
    assert saturated(0.73) is False
    assert saturated(0.95) is True
