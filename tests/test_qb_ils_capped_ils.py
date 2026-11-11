"""The two ILS baselines: the budget-capped ILS (HEROSIM_ILS_CAP_S_BY_BUCKET) and the ILS started from the served GNN's plan (HEROSIM_QBILS). Both off by default; the cap
never lets a group charge more than its seconds; the "cd" seed sends the CD passes' own plan through the seed path and must equal the stock ILS rule. Toy of test_cd_expansion."""
import json

import pytest

from src.policy.peer_greedy_network import scheduler as S
from test_cd_expansion import _shell

BAR_MS, S_MS = 1000.0, 100.0
OPT = {0: (3, 33), 1: (3, 33)}


def _rates(monkeypatch):
    monkeypatch.setenv("HEROSIM_DECISION_COST_MS_BY_BUCKET", json.dumps({"le10": BAR_MS, "11_50": BAR_MS, "gt50": BAR_MS}))
    monkeypatch.setenv("HEROSIM_DECISION_COST_MS_S_BY_BUCKET", json.dumps({"le10": S_MS, "11_50": S_MS, "gt50": S_MS}))


def _run(monkeypatch, per=20, cap=None, seed=None, cls=None):
    monkeypatch.setattr(S, "ILS_MIN_TASKS", 2)
    monkeypatch.setenv(S.PG_ILS_PER_ENV, str(per))
    _rates(monkeypatch)
    if cap is not None:
        monkeypatch.setenv(S.ILS_CAP_ENV, json.dumps({"le10": cap, "11_50": cap, "gt50": cap}))
    s, tasks, state = _shell(monkeypatch, **({"cls": cls} if cls else {}))
    return s, s._pg_decide(tasks, state, seed=seed), tasks, state


def _charged(s):
    return (BAR_MS * s.sched_plan_evals + S_MS * s.sched_plan_evals_s) * 1e-3


def test_cap_off_is_the_stock_ils(monkeypatch):
    s, plan, *_ = _run(monkeypatch)
    assert plan == OPT and s.ils_cap_s is None and s.pg_ils_cap_binding == s.pg_ils_cap_skipped == s.pg_ils_cap_plans == 0


def test_a_cap_above_the_ils_budget_changes_nothing(monkeypatch):
    s0, plan0, *_ = _run(monkeypatch)
    s1, plan1, *_ = _run(monkeypatch, cap=1e6)
    assert plan1 == plan0 and s1.pg_ils_cap_binding == 0
    assert (s1.sched_plan_evals, s1.sched_plan_evals_s, s1.pg_ils_evals) == (s0.sched_plan_evals, s0.sched_plan_evals_s, s0.pg_ils_evals)


def test_a_binding_cap_stops_the_ils_and_the_group_never_charges_more_than_it(monkeypatch):
    s0, _, *_ = _run(monkeypatch)
    bar_s = BAR_MS * s0.sched_plan_evals * 1e-3
    for k in (3, 5, 8):
        cap = bar_s + k * S_MS * 1e-3
        s, plan, *_ = _run(monkeypatch, cap=cap)
        assert s.pg_ils_cap_binding == 1 and s.pg_ils_evals <= k - 1 and _charged(s) <= cap + 1e-9


def test_a_cap_below_two_plans_serves_the_bars_plan_uncharged_for_search(monkeypatch):
    s0, _, *_ = _run(monkeypatch)
    s, plan, *_ = _run(monkeypatch, cap=BAR_MS * s0.sched_plan_evals * 1e-3 + 1 * S_MS * 1e-3)   # room for one plan, one is kept for the write-back
    assert plan == {0: (1, 11), 1: (2, 22)} and s.pg_ils_cap_skipped == 1 and s.sched_plan_evals_s == 0 and s.pg_ils_batches == 0
    s, plan, *_ = _run(monkeypatch, cap=0.0)
    assert plan == {0: (1, 11), 1: (2, 22)} and s.pg_ils_cap_skipped == 1


def test_cap_needs_the_ils_and_its_s_rate(monkeypatch):
    monkeypatch.delenv(S.PG_ILS_PER_ENV, raising=False)
    monkeypatch.setenv(S.ILS_CAP_ENV, json.dumps({"le10": 1.0, "11_50": 1.0, "gt50": 1.0}))
    with pytest.raises(ValueError, match="FAIL LOUD"):
        _shell(monkeypatch)
    monkeypatch.setenv(S.PG_ILS_PER_ENV, "20")
    monkeypatch.setenv(S.ILS_CAP_ENV, json.dumps({"le10": 1.0}))
    with pytest.raises(ValueError, match="FAIL LOUD"):
        _shell(monkeypatch)
    monkeypatch.setenv(S.ILS_CAP_ENV, json.dumps({"le10": 1.0, "11_50": 1.0, "gt50": 1.0}))
    monkeypatch.setattr(S, "ILS_MIN_TASKS", 2)
    monkeypatch.delenv("HEROSIM_DECISION_COST_MS_S_BY_BUCKET", raising=False)
    monkeypatch.delenv("HEROSIM_DECISION_COST_MS_BY_BUCKET", raising=False)
    s, tasks, state = _shell(monkeypatch)
    with pytest.raises(RuntimeError, match="FAIL LOUD"):
        s._pg_decide(tasks, state)


def test_the_cd_seed_is_the_stock_ils_rule(monkeypatch):
    s0, plan0, *_ = _run(monkeypatch)
    s1, plan1, *_ = _run(monkeypatch, seed="cd")
    assert plan1 == plan0
    for name in ("sched_plan_evals", "sched_plan_evals_s", "pg_ils_batches", "pg_ils_evals", "pg_ils_budget", "pg_ils_improved", "pg_ils_tasks_moved", "pg_cd_passes", "pg_cd_moves", "pg_batches"):
        assert getattr(s1, name) == getattr(s0, name), name
    assert s1.pg_ils_gain_seconds == pytest.approx(s0.pg_ils_gain_seconds)


def test_the_cd_seed_with_a_cap_equals_the_capped_stock_rule(monkeypatch):
    s0, _, *_ = _run(monkeypatch)
    cap = BAR_MS * s0.sched_plan_evals * 1e-3 + 4 * S_MS * 1e-3
    a, plan_a, *_ = _run(monkeypatch, cap=cap)
    b, plan_b, *_ = _run(monkeypatch, cap=cap, seed="cd")
    assert plan_a == plan_b and (a.sched_plan_evals, a.sched_plan_evals_s) == (b.sched_plan_evals, b.sched_plan_evals_s)


def test_a_given_plan_is_the_ils_start_in_place_of_the_cd_plan(monkeypatch):
    s, plan, *_ = _run(monkeypatch, seed={0: (3, 33), 1: (1, 11)})
    assert plan == OPT and s.pg_ils_improved == 1 and s.pg_cd_passes == 0
    assert s.sched_plan_evals == 0 and s.sched_plan_evals_s == s.pg_ils_evals + 1    # no bar plans; the ILS's own scores and the write-back are the S channel
    s, plan, *_ = _run(monkeypatch, seed=dict(OPT))
    assert plan == OPT and s.pg_ils_improved == 0


def _hybrid(monkeypatch, plan, qbils="1"):
    monkeypatch.setenv(S.QBILS_ENV, qbils)
    monkeypatch.setenv(S.PG_ILS_PER_ENV, "20")
    monkeypatch.setattr(S, "ILS_MIN_TASKS", 2)
    _rates(monkeypatch)
    s, tasks, state = _shell(monkeypatch, cls=S.HybridGnnCdScheduler)
    s.qbils, s.hyb_g, s._ladder, s._ladder_big, s._ladder_nopass = qbils, 0, None, None, False
    for name in S.HYB_COUNTERS:
        setattr(s, name, 0)
    monkeypatch.setattr(S.GNNScheduler, "_prefix_inference", lambda self, *a: dict(plan))
    return s, tasks, state


def test_qbils_runs_the_ils_from_the_gnn_plan_on_every_group(monkeypatch):
    s, tasks, state = _hybrid(monkeypatch, {0: (3, 33), 1: (1, 11)})
    assert s._prefix_inference(tasks, state, None, None) == OPT
    assert s.hyb_large_batches == 1 and s.hyb_small_batches == 0 and s.pg_ils_batches == 1 and s.qbils_unseeded == 0


def test_qbils_keeps_the_gnn_plan_when_a_task_has_no_couple(monkeypatch):
    s, tasks, state = _hybrid(monkeypatch, {0: (3, 33)})
    assert s._prefix_inference(tasks, state, None, None) == {0: (3, 33)} and s.qbils_unseeded == 1 and s.pg_ils_batches == 0


def test_qbils_cd_is_the_stock_hybrid_small_path_plus_ils(monkeypatch):
    s, tasks, state = _hybrid(monkeypatch, {0: (9, 9), 1: (9, 9)}, qbils="cd")   # the GNN is never called
    assert s._prefix_inference(tasks, state, None, None) == OPT and s.pg_cd_passes == 1 and s.pg_ils_batches == 1


def test_qbils_needs_the_ils_budget_and_a_known_value(monkeypatch):
    src = open(S.__file__).read()
    assert 'raise ValueError(f"FAIL LOUD: {QBILS_ENV} needs {PG_ILS_PER_ENV}' in src and 'raise ValueError(f"FAIL LOUD: {QBILS_ENV} must be unset, 1 or cd' in src


def _gate():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G

    return G


def test_the_gate_names_both_arms_as_scratch_and_splits_their_suffixes():
    G = _gate()
    for k in (*G.QBILS_KINDS, *G.ILSCAP_KINDS):
        assert k in G.R1A_ARMS and k in G.R1A_SCRATCH
    base = f"{G.COMB_STATIC_E40[0]}_nosplit_sr2"
    assert G.split_qbils(base + "_qbils") == (base, "1") and G.split_qbils(base + "_qbilscd") == (base, "cd") and G.split_qbils(base) == (base, None)
    assert G.ILSCAP_KINDS == ("cd_ib0_inflightfull_fsaware_sym_ils10szw_g0_cap",) and G.ILSCAP_KINDS[0][:-4] in G.ILS_PER


def test_the_cap_table_row_is_scaled_by_k_and_fails_loud_without_a_row(monkeypatch, tmp_path):
    G = _gate()
    f = tmp_path / "caps.json"
    f.write_text(json.dumps({"rand_s20_p48c50|moderate": {"le10": 0.2, "11_50": 1.5, "gt50": 9.0, "_cells": 96}}))
    monkeypatch.setattr(G, "ILS_CAP_FILE", str(f))
    assert json.loads(G.ils_cap_s("rand_s20_p48c50", "moderate", 0.5)) == {"le10": 0.1, "11_50": 0.75, "gt50": 4.5}
    with pytest.raises(SystemExit, match="FAIL LOUD"):
        G.ils_cap_s("rand_s20_p48c50", "heavy", 1.0)
    with pytest.raises(SystemExit, match="FAIL LOUD"):
        G.ils_cap_s("rand_s20_p48c50", "moderate", None)


def test_a_seed_on_couples_outside_the_candidate_set_still_gets_the_papers_budget(monkeypatch):
    """The GNN may pick a replica the CD candidate set does not offer (uninitialized): that task is pinned, the budget still counts its candidates, no zero-budget crash."""
    s, _plan, tasks, state = _run(monkeypatch, per=20)
    for r in state.replicas["cnn"]:
        r[1].initialized.triggered = r[0].id == 3          # only C is initialized: the candidate set is {C} for every task
    s2, tasks2, state2 = _shell(monkeypatch)
    for r in state2.replicas["cnn"]:
        r[1].initialized.triggered = r[0].id == 3
    plan = s2._pg_decide(tasks2, state2, seed={0: (1, 11), 1: (2, 22)})   # both seed couples are outside {C}: both pinned
    assert plan == {0: (1, 11), 1: (2, 22)} and s2.pg_ils_batches == 1 and s2.pg_ils_budget == 20 * 2 * 1
