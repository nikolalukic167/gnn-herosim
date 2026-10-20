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
    monkeypatch.setattr(S, "ILS_MIN_TASKS", 2)       # the toy batch has 2 tasks; the registered floor (11) is tested separately
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
    assert s.pg_ils_evals <= s.pg_ils_budget == 20 * 2 * 3 and s.pg_ils_stop_other == 0
    assert s.pg_expand_batches == 0                            # the expansion is the ILS's own, never the CD refine's


def test_ils_plans_are_booked_apart_from_the_bars(monkeypatch):
    s, _ = _decide(monkeypatch, 20)
    monkeypatch.delenv(S.PG_ILS_PER_ENV)
    s2, _ = _decide(monkeypatch, None)
    # the bar's own plans stay on sched_plan_evals, identical to the ILS-off run; the ILS phase's plans (searched + the accepted plan's write-back) are the S counter
    assert s.sched_plan_evals == s2.sched_plan_evals
    assert s.sched_plan_evals_s == s.pg_ils_evals + 1 and s2.sched_plan_evals_s == 0


def test_a_batch_below_the_label_floor_is_the_bars_plan(monkeypatch):
    monkeypatch.setenv(S.PG_ILS_PER_ENV, "20")
    s, tasks, state = _shell(monkeypatch)
    assert S.ILS_MIN_TASKS == 11
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)} and s.pg_ils_small_batches == 1 and s.pg_ils_batches == 0 and s.sched_plan_evals_s == 0


def test_the_size_cap_keeps_the_bars_plan_above_g(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    monkeypatch.setenv(S.PG_EXACT_G_ENV, "1")
    s, tasks, state = _shell(monkeypatch)
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)} and s.pg_exact_g_skipped == 1 and s.pg_exact_batches == 0
    monkeypatch.setenv(S.PG_EXACT_G_ENV, "2")
    s, tasks, state = _shell(monkeypatch)
    assert s._pg_decide(tasks, state) == {0: (3, 33), 1: (3, 33)} and s.pg_exact_g_skipped == 0 and s.pg_exact_batches == 1


def test_g_needs_the_exact_search_and_a_positive_integer(monkeypatch):
    monkeypatch.setenv(S.PG_EXACT_G_ENV, "3")
    with pytest.raises(RuntimeError, match="needs peer_greedy_network_cd"):
        _shell(monkeypatch)
    monkeypatch.setenv(S.PG_EXACT_G_ENV, "0")
    with pytest.raises(ValueError, match=S.PG_EXACT_G_ENV):
        _shell(monkeypatch)


def test_the_s_rate_charges_only_the_ils_plans(monkeypatch):
    from src.placement import decision_cost as dc

    monkeypatch.delenv(dc.ENV, raising=False)
    monkeypatch.setenv(dc.BUCKET_ENV, '{"le10": 1, "11_50": 2, "gt50": 3}')
    monkeypatch.setenv(dc.S_BUCKET_ENV, '{"le10": 10, "11_50": 20, "gt50": 30}')
    assert dc.rate_ms(20) == 2 and dc.s_rate_ms(20) == 20 and dc.s_rate_ms(5) == 10 and dc.s_rate_ms(80) == 30
    monkeypatch.setenv(dc.S_BUCKET_ENV, '{"le10": 10}')
    with pytest.raises(ValueError, match=dc.S_BUCKET_ENV):
        dc.s_rate_ms(20)
    monkeypatch.setenv(dc.S_BUCKET_ENV, '{"le10": 1, "11_50": 1, "gt50": 1}')
    monkeypatch.delenv(dc.BUCKET_ENV)
    with pytest.raises(ValueError, match="ordinary decision charge is off"):
        dc.s_rate_ms(20)
    monkeypatch.delenv(dc.S_BUCKET_ENV)
    assert dc.s_rate_ms(20) == 0.0


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


def test_counters_are_exported_and_the_gate_serves_the_arms():
    import inspect

    from src.placement import orchestrator

    exporter = inspect.getsource(orchestrator.Orchestrator._scheduler_counters)
    for k in (*S.PG_ILS_COUNTERS, "pg_exact_g_skipped", "sched_plan_evals_s"):
        assert k in S.PEER_GREEDY_COUNTERS and f'"{k}"' in exporter
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G

    assert G.ILS_PER == {"cd_ib0_ils": 200}       # the label engine's own default, not a study choice
    for kind in (*G.ILS_PER, G.EXACT_LIVE):
        assert G.RULE_POLICY[kind] == "peer_greedy_network_cd" and kind in G.R1A_DIAG and kind in G.R1A_ARMS
        assert G.ib0_alpha(kind) == {"HEROSIM_PG_BATCH_ALPHA": "0"}
        assert G.split_dc(kind + "_k100") == (kind, 1.0)
    assert G.check_alias("cd_ib0_ils") == "cd" and G.check_alias(G.EXACT_LIVE) == "cd_exactS_ib0"
    for rung in ("moderate", "heavy"):
        assert G.dc_cost_ms("cd_ib0_ils", 1.0, rung) == G.dc_cost_ms("cd_ib0_inflightfull", 1.0, rung)    # the bar's own charge for the bar's part
        sr = __import__("json").loads(G.dc_s_cost_ms("cd_ib0_ils", 1.0, rung))
        assert sorted(sr) == ["11_50", "gt50", "le10"]
        assert G.dc_cost_ms(G.EXACT_LIVE, 1.0, rung)[0] == G.DC_BUCKET_ENV
        assert G.dc_s_cost_ms(G.EXACT_LIVE, 1.0, rung) is None and G.dc_s_cost_ms("cd_ib0_inflightfull", 1.0, rung) is None


def test_exact_live_fails_loud_until_g_is_filed(monkeypatch, tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import json

    import fresh_topo_burst_v1_gate as G

    f = tmp_path / "w.json"
    f.write_text(json.dumps({"w_ms_per_plan": {}}))
    monkeypatch.setattr(G, "DC_W_FILE", str(f))
    with pytest.raises(SystemExit, match="exact_live_G"):
        G.exact_live_g()
    f.write_text(json.dumps({"exact_live_G": 6}))
    assert G.exact_live_g() == 6
