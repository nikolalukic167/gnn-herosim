"""bipartite_aggr_v1: the registered bars behave as signed, before any arm exists.

The point of exercising them now is that this lineage's value is its PREDICTION, and a
prediction is only a prediction if every outcome is named in advance and the reader cannot
be talked into the one you wanted. All four E1 x E2 patterns are asserted here.

Run: pipenv run python3 -m pytest scripts_cosim/test_bipartite_aggr_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.bipartite_aggr_v1_read import (  # noqa: E402
    E_ALPHA, E_CANDIDATES_PER_TASK, E_MIN_SEEDS, E_RUNGS, E_SEPARATE_PCT,
    V_AGGR_IS_NOT_THE_MECHANISM, V_AGGR_NOT_SEP, V_INVERTED, V_SCALE_DEPENDENT,
    V_SCALE_FREE, V_SUM_COSTS, V_SUM_HELPS,
    read_e1, read_e2, read_e3,
)
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

N = E_MIN_SEEDS


def _by_seed(base, step=0.1, n=N):
    return {s: base + step * s for s in range(n)}


def test_the_registered_constants_match_the_signed_node():
    assert (E_SEPARATE_PCT, E_ALPHA, E_MIN_SEEDS) == (5.0, 0.05, 16)
    assert E_RUNGS == ("R3", "R0")
    # The precondition the whole prediction rests on. If these were equal the lineage could
    # not separate its hypothesis from any other story about cluster scale.
    assert E_CANDIDATES_PER_TASK["R3"] / E_CANDIDATES_PER_TASK["R0"] > 10


def test_e1_is_oriented_so_slower_sum_reads_as_sum_costs():
    costs = read_e1(_by_seed(130.0), _by_seed(100.0))
    assert costs["verdict"] == V_SUM_COSTS and costs["median"] > 0
    helps = read_e1(_by_seed(70.0), _by_seed(100.0))
    assert helps["verdict"] == V_SUM_HELPS
    assert read_e1(_by_seed(101.0), _by_seed(100.0))["verdict"] == V_AGGR_NOT_SEP


def test_swapping_the_arguments_flips_the_verdict():
    a, b = _by_seed(130.0), _by_seed(100.0)
    assert read_e1(a, b)["verdict"] == V_SUM_COSTS
    assert read_e1(b, a)["verdict"] == V_SUM_HELPS


def test_a_short_read_refuses_rather_than_relaxing_the_bar():
    short_sum = {s: 130.0 for s in range(N - 2)}
    short_mean = {s: 100.0 for s in range(N - 2)}
    assert read_e1(short_sum, short_mean)["verdict"] == V_UNREADABLE
    assert read_e2(short_sum, short_mean, min_seeds=N - 2)["verdict"] == V_SUM_COSTS


# --- E3: all four patterns, named before any datum ---------------------------------------

def test_e3_fires_scale_dependent_only_on_the_predicted_pattern():
    e1 = read_e1(_by_seed(130.0), _by_seed(100.0))        # sum costs at 80 servers
    e2 = read_e2(_by_seed(100.5), _by_seed(100.0))        # not separated at 6
    r = read_e3(e1, e2)
    assert r["verdict"] == V_SCALE_DEPENDENT
    assert "absent at 6 servers" in r["why"]


def test_e3_names_sum_costing_everywhere_as_a_failure_of_the_mechanism():
    both = read_e1(_by_seed(130.0), _by_seed(100.0))
    r = read_e3(both, both)
    assert r["verdict"] == V_SCALE_FREE
    assert "unexplained" in r["why"]


def test_e3_names_the_inverted_pattern():
    e1 = read_e1(_by_seed(100.5), _by_seed(100.0))        # not separated at 80
    e2 = read_e2(_by_seed(130.0), _by_seed(100.0))        # costs at 6
    assert read_e3(e1, e2)["verdict"] == V_INVERTED


def test_e3_falls_through_to_not_the_mechanism_on_every_other_pattern():
    tie = read_e1(_by_seed(100.5), _by_seed(100.0))
    helps = read_e1(_by_seed(70.0), _by_seed(100.0))
    assert read_e3(tie, tie)["verdict"] == V_AGGR_IS_NOT_THE_MECHANISM
    assert read_e3(helps, tie)["verdict"] == V_AGGR_IS_NOT_THE_MECHANISM
    assert read_e3(helps, helps)["verdict"] == V_AGGR_IS_NOT_THE_MECHANISM


def test_e3_refuses_when_a_bar_is_unreadable_rather_than_reading_it_as_a_null():
    e1 = read_e1({s: 130.0 for s in range(3)}, {s: 100.0 for s in range(3)})
    assert e1["verdict"] == V_UNREADABLE
    r = read_e3(e1, read_e2(_by_seed(100.5), _by_seed(100.0)))
    assert r["verdict"] == V_UNREADABLE and "not tested" in r["why"]
