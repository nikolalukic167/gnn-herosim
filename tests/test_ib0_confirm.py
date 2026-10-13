"""ib0_confirm_v1: the ib0 surrogate (HEROSIM_PG_BATCH_ALPHA=0) on the expansion refines -- plain CD (cdx_ib0) and the GNN-seeded refiner (cdxapply_ib0) --
the hand-seed control (GNN_CD_SEED=hand) and the flag-off identity of the relaxed alpha validation."""
import os
import sys
from types import SimpleNamespace

import pytest

from src.policy.peer_greedy_network import scheduler as S
from test_cd_expansion import _refiner, _shell

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
import fresh_topo_burst_v1_gate as G  # noqa: E402

SEED = {0: (1, 11), 1: (2, 22)}


def _clear(monkeypatch):
    for name in (S.PG_DRAIN_ALPHA_ENV, S.PG_BATCH_ALPHA_ENV, S.PG_CD_EXACT_ENV, S.PG_CD_EXPANSION_ENV):
        monkeypatch.delenv(name, raising=False)


def test_alpha_is_accepted_by_plain_cd_and_the_refiner_without_the_exact_search(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    monkeypatch.setenv(S.PG_BATCH_ALPHA_ENV, "0")
    s, _t, _st = _shell(monkeypatch)
    assert s.pg_alpha_on and s.pg_batch_alpha == 0.0 and not s.pg_cd_exact
    r, _t, _st = _refiner(monkeypatch)
    assert r.pg_alpha_on and r.pg_batch_alpha == 0.0 and r._policy_label == "gnn_cd_refine"


def test_alpha_is_refused_on_the_refiner_with_the_exact_search(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(S.PG_BATCH_ALPHA_ENV, "0")
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    with pytest.raises(RuntimeError, match="FAIL LOUD"):
        _refiner(monkeypatch)


def _flag_off_run(monkeypatch, alpha_env):
    _clear(monkeypatch)
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    for k, v in alpha_env.items():
        monkeypatch.setenv(k, v)
    s, tasks, state = _shell(monkeypatch)
    assert not s.pg_alpha_on
    plan = s._pg_decide(tasks, state)
    r, rtasks, rstate = _refiner(monkeypatch)
    assert not r.pg_alpha_on
    rplan, info = r.refine(rtasks, rstate, dict(SEED), passes=3)
    counters = lambda o: {n: getattr(o, n) for n in S.PEER_GREEDY_COUNTERS if hasattr(o, n)}  # noqa: E731
    return plan, counters(s), rplan, info, counters(r)


def test_flag_off_identity_plain_cd_and_refiner(monkeypatch):
    """alphas unset and alphas at 1 give the same plan, the same move info and the same counters for plain CD (cd_expand) and for the refiner (cdxapply)."""
    off = _flag_off_run(monkeypatch, {})
    one = _flag_off_run(monkeypatch, {S.PG_DRAIN_ALPHA_ENV: "1", S.PG_BATCH_ALPHA_ENV: "1"})
    assert off == one
    assert off[0] == {0: (3, 33), 1: (3, 33)} and off[2] == {0: (3, 33), 1: (3, 33)} and off[3]["moved"] == 2   # and the expansion did run


def test_ib0_zeroes_only_the_in_batch_stacking_part_on_the_refiner(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr(S, "platform_queue_drain_seconds", lambda platform, orch, memo, exec_scale=1.0: 4.0)
    r1, tasks, state = _refiner(monkeypatch)
    nodes = r1.nodes.items
    plan = [(nodes[2], state.replicas["cnn"][2][1])] * 2
    full, _a, _b = r1._pg_plan_cost(tasks, plan, r1._pg_orchestrator(), {}, nodes)
    monkeypatch.setenv(S.PG_BATCH_ALPHA_ENV, "0")
    monkeypatch.setattr(S, "platform_queue_drain_seconds", lambda platform, orch, memo, exec_scale=1.0: 4.0)
    r2, tasks2, state2 = _refiner(monkeypatch)
    ib0, _c, _d = r2._pg_plan_cost(tasks2, plan, r2._pg_orchestrator(), {}, nodes)
    assert ib0 == pytest.approx(full - 2 * 1.0)   # each of the two co-located tasks drops the other's 1 s of service


def test_hand_seed_is_the_cd_first_pass_and_restores_the_books(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    s, tasks, state = _shell(monkeypatch)
    first = {}
    s._pg_batch_pass(tasks, state, s._pg_orchestrator(), memo={}, committed_service={}, planned={}, placements=first, service_of={}, refine=False)
    r, rtasks, rstate = _refiner(monkeypatch)
    before = {n: getattr(r, n) for n in S.PEER_GREEDY_COUNTERS if hasattr(r, n)}
    hand = r.hand_seed(rtasks, rstate)
    assert hand == {i: tuple(v) for i, v in first.items()}
    assert {n: getattr(r, n) for n in S.PEER_GREEDY_COUNTERS if hasattr(r, n)} == before
    # the refiner started from the hand seed reaches what cd_expand reaches on the toy
    plan, _ = r.refine(rtasks, rstate, hand, passes=3)
    assert plan == s._pg_decide(tasks, state) == {0: (3, 33), 1: (3, 33)}


def test_gnn_scheduler_takes_a_hand_seed_only_when_asked():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src/policy/gnn/scheduler.py")).read()
    assert 'os.environ.get("GNN_CD_SEED", "").strip() == "hand"' in src and "placements = self._cd_refiner.hand_seed(tasks, system_state)" in src
    assert "GNN_CD_SEED must be unset or 'hand'" in src


def test_harness_names_the_ib0_expansion_arms():
    assert G.RULE_POLICY["cdx_ib0"] == "peer_greedy_network_cd" and "cdx_ib0" in G.R1A_DIAG and "cdx_ib0" in G.R1A_ARMS
    assert G.RULE_POLICY["cd_exactS_a05"] == "peer_greedy_network_cd" and G.EXACT_ALPHA["cd_exactS_a05"] == {"HEROSIM_PG_BATCH_ALPHA": "0.5"}
    assert G.RULE_POLICY["cd_exactS_ib0_d05"] == "peer_greedy_network_cd" and G.EXACT_ALPHA["cd_exactS_ib0_d05"] == {"HEROSIM_PG_DRAIN_ALPHA": "0.5", "HEROSIM_PG_BATCH_ALPHA": "0"}
    assert "cd_exactS_ib0_d05" in G.R1A_DIAG
    assert "_cdxib0" in G.SUFFIXES and "_cdxhib0" in G.SUFFIXES
    assert {"ra_gnn_eng_cdxib0", "ra_gnn_eng_cdxhib0", "ra_twin_eng_cdxib0"} <= set(G.R1A_CDX)
    assert G.ib0_alpha("cdx_ib0") == G.ib0_alpha("ra_gnn_eng_cdxib0") == G.ib0_alpha("ra_gnn_eng_cdxhib0") == {"HEROSIM_PG_BATCH_ALPHA": "0"}
    assert G.ib0_alpha("cd_expand") == G.ib0_alpha("ra_gnn_eng_cdxapply") == {}
    assert G.check_alias("cdx_ib0") == "cd_expand" and G.check_alias("ra_twin_eng_cdxib0") == "ra_twin_eng_cdxapply"
    assert G.check_alias("ra_gnn_eng_cdxhib0") == "ra_gnn_eng_cdxapply" and G.check_alias("cd_exactS") == "cd_exactS"
    assert "GNN_CD_SEED" in G.ORACLE_ENV
    src = open(G.__file__).read()
    assert 'env["GNN_CD_SEED"] = "hand"' in src and "HEROSIM_PG_CD_EXPANSION=\"1\", **ib0_alpha(kind))" in src
