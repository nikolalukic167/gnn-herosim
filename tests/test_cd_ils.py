"""cd_ib0_ils (HEROSIM_PG_ILS_PER): the label engine's ILS run live on the rule's own S after the CD passes. Off by default; every score goes through
`_pg_plan_cost` (so through the decision-cost meter); deterministic; one implementation shared with scripts_cosim/uncut_ils.py. The toy is
test_cd_expansion's: the pair that single-task moves cannot move and a joint move can."""
import os
import sys

import pytest

from src.placement.decision_cost import meter
from src.policy.peer_greedy_network import scheduler as S
from test_cd_expansion import _shell


def _decide(monkeypatch, per, **kw):
    if per is not None:
        monkeypatch.setenv(S.PG_ILS_PER_ENV, str(per))
    s, tasks, state = _shell(monkeypatch, **kw)
    return s, s._pg_decide(tasks, state)


def test_off_is_the_registered_cd(monkeypatch):
    monkeypatch.delenv(S.PG_ILS_PER_ENV, raising=False)
    s, plan = _decide(monkeypatch, None)
    assert plan == {0: (1, 11), 1: (2, 22)} and s.pg_ils_per == 0 and s.pg_ils_batches == 0 and s.pg_ils_evals == 0


def test_ils_finds_the_pair_move_icm_cannot_and_books_it(monkeypatch):
    s, plan = _decide(monkeypatch, 20)
    assert plan == {0: (3, 33), 1: (3, 33)}
    assert s.pg_ils_batches == 1 and s.pg_ils_improved == 1 and s.pg_ils_tasks_moved == 2
    assert s.pg_ils_gain_seconds == pytest.approx(16.0)       # split pair 10 + 10 -> both on C: (1 + stacking 1) x 2
    assert s.pg_ils_evals <= s.pg_ils_budget and s.pg_ils_stop_other == 0
    assert s.pg_expand_batches == 0                            # the expansion is the ILS's own, never the CD refine's


def test_every_score_is_metered(monkeypatch):
    s, _ = _decide(monkeypatch, 20)
    before_ils = s.sched_plan_evals
    monkeypatch.delenv(S.PG_ILS_PER_ENV)
    s2, _ = _decide(monkeypatch, None)
    # the ILS arm's extra metered plans are exactly the plans it scored plus the one write-back score of the accepted plan
    assert before_ils - s2.sched_plan_evals == s.pg_ils_evals + 1


def test_budget_caps_the_distinct_plans_scored(monkeypatch):
    s, _ = _decide(monkeypatch, 1)
    n, mean_cands = 2, 3
    assert s.pg_ils_budget == 1 * n * mean_cands
    assert s.pg_ils_evals <= s.pg_ils_budget


def test_two_runs_are_identical(monkeypatch):
    a, pa = _decide(monkeypatch, 5)
    b, pb = _decide(monkeypatch, 5)
    assert pa == pb and (a.pg_ils_evals, a.pg_ils_gain_seconds, a.pg_ils_stop_evals) == (b.pg_ils_evals, b.pg_ils_gain_seconds, b.pg_ils_stop_evals)


def test_no_gain_leaves_the_cd_plan(monkeypatch):
    s, plan = _decide(monkeypatch, 20, exec_times=[{"pa": 0.0, "pb": 100.0, "pc": 20.0}, {"pa": 100.0, "pb": 0.0, "pc": 20.0}])
    assert plan == {0: (1, 11), 1: (2, 22)} and s.pg_ils_batches == 1 and s.pg_ils_improved == 0 and s.pg_ils_tasks_moved == 0


def test_bad_values_and_combinations_fail_loudly(monkeypatch):
    monkeypatch.setenv(S.PG_ILS_PER_ENV, "x")
    with pytest.raises(ValueError, match=S.PG_ILS_PER_ENV):
        _shell(monkeypatch)
    monkeypatch.setenv(S.PG_ILS_PER_ENV, "-1")
    with pytest.raises(ValueError, match=S.PG_ILS_PER_ENV):
        _shell(monkeypatch)
    monkeypatch.setenv(S.PG_ILS_PER_ENV, "3")
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    with pytest.raises(RuntimeError, match="replaces the expansion"):
        _shell(monkeypatch)
    monkeypatch.delenv(S.PG_CD_EXPANSION_ENV)
    with pytest.raises(RuntimeError, match="peer_greedy_network_cd only"):
        _shell(monkeypatch, cls=S.PeerGreedyNetworkBatchScheduler)


def test_one_search_for_the_labels_and_the_arm():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import uncut_ils
    from src.policy.peer_greedy_network import ils

    assert uncut_ils.ils is ils.ils and uncut_ils.CAP == ils.CAP


def test_counters_are_exported_and_the_gate_serves_the_arm():
    import inspect

    from src.placement import orchestrator

    exporter = inspect.getsource(orchestrator.Orchestrator._scheduler_counters)
    for k in S.PG_ILS_COUNTERS:
        assert k in S.PEER_GREEDY_COUNTERS and f'"{k}"' in exporter
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G

    for kind, per in G.ILS_PER.items():
        assert G.RULE_POLICY[kind] == "peer_greedy_network_cd" and kind in G.R1A_DIAG and kind in G.R1A_ARMS
        assert G.ib0_alpha(kind) == {"HEROSIM_PG_BATCH_ALPHA": "0"} and G.check_alias(kind) == "cd"
        assert G.split_dc(kind + "_k100") == (kind, 1.0)
        for rung in ("moderate", "heavy"):
            assert G.dc_cost_ms(kind, 1.0, rung)[0] == G.DC_ENV
