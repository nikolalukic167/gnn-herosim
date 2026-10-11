"""plan_rerank_v1 Phase 0 (HEROSIM_PG_ORACLE=1): cd_exactS whose exact search is scored by the real single-batch RTT instead of S.
The replay pool is stubbed here (tests/test_cd_expansion.py's toy: two partners that each run free on their own node, 10 s exchange, both at
1 s on node C); the labeller itself is covered by the fidelity replay's own tests and by the cost smoke."""
import os
import sys

import pytest

from src.policy.peer_greedy_network import scheduler as S
from test_cd_expansion import _shell

SPLIT = ((1, 11), (2, 22))
TOGETHER_C = ((3, 33), (3, 33))


class StubOracle:
    def __init__(self, labels):
        self.labels_by_plan, self.calls = labels, 0

    def labels(self, snapshot, types, plans):
        self.calls += 1
        return [self.labels_by_plan.get(tuple(tuple(c) for c in p), 50.0) for p in plans]


def _oracle_shell(monkeypatch, labels, max_plans=None):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    monkeypatch.setenv(S.PG_ORACLE_ENV, "1")
    if max_plans is not None:
        monkeypatch.setenv(S.PG_ORACLE_MAX_PLANS_ENV, str(max_plans))
    monkeypatch.setattr("src.placement.live_audit.capture_batch_snapshot", lambda *a, **k: {})
    s, tasks, state = _shell(monkeypatch)
    s._live_audit_policy_name = "peer_greedy_network_cd"
    s._plan_oracle = StubOracle(labels)
    return s, tasks, state


def test_oracle_applies_a_plan_the_surrogate_rejects(monkeypatch):
    # S prefers the co-located plan on C; the real RTT says the split plan is better
    s, tasks, state = _oracle_shell(monkeypatch, {SPLIT: 5.0, TOGETHER_C: 9.0})
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)}
    assert (s.pg_oracle_batches, s.pg_oracle_plans, s.pg_oracle_changed, s.pg_oracle_tasks) == (1, 9, 1, 2)
    assert s.pg_oracle_gain_seconds == pytest.approx(4.0)   # S's plan (co-located on C) costs 9 s, the oracle's 5 s
    assert s._plan_oracle.calls == 1


def test_oracle_replay_tie_keeps_the_surrogates_choice(monkeypatch):
    s, tasks, state = _oracle_shell(monkeypatch, {})
    assert s._pg_decide(tasks, state) == {0: (3, 33), 1: (3, 33)}   # what cd_exactS applies
    assert (s.pg_oracle_batches, s.pg_oracle_changed) == (1, 0) and s.pg_oracle_gain_seconds == 0.0


def test_oracle_cap_sends_larger_batches_to_the_surrogate_search_and_counts_them(monkeypatch):
    s, tasks, state = _oracle_shell(monkeypatch, {SPLIT: 0.0}, max_plans=4)
    assert s._pg_decide(tasks, state) == {0: (3, 33), 1: (3, 33)}
    assert s._plan_oracle.calls == 0
    assert (s.pg_oracle_batches, s.pg_oracle_seen_batches, s.pg_oracle_over_cap_batches, s.pg_oracle_over_cap_tasks) == (0, 1, 1, 2)
    assert s.pg_exact_batches == 1


def test_oracle_flag_off_never_builds_the_pool(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    monkeypatch.delenv(S.PG_ORACLE_ENV, raising=False)
    s, tasks, state = _shell(monkeypatch)
    assert s._pg_decide(tasks, state) == {0: (3, 33), 1: (3, 33)}
    assert not hasattr(s, "_plan_oracle") and all(getattr(s, k) == 0 for k in S.PG_EXACT_COUNTERS if k.startswith("pg_oracle"))


def test_oracle_needs_the_exact_search_and_the_plain_cd_flavour(monkeypatch):
    monkeypatch.delenv(S.PG_CD_EXACT_ENV, raising=False)
    monkeypatch.setenv(S.PG_ORACLE_ENV, "1")
    with pytest.raises(RuntimeError, match="FAIL LOUD"):
        _shell(monkeypatch)


def test_oracle_counters_are_exported():
    import inspect

    from src.placement import orchestrator

    exporter = inspect.getsource(orchestrator.Orchestrator._scheduler_counters)
    for k in S.PG_EXACT_COUNTERS:
        if k.startswith("pg_oracle"):
            assert k in S.PEER_GREEDY_COUNTERS and f'"{k}"' in exporter


def test_pool_refuses_to_start_without_its_inputs(monkeypatch):
    from src.placement import plan_oracle

    for name in (plan_oracle.CELL_CONFIG_ENV, plan_oracle.SIM_INPUT_ENV):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="FAIL LOUD"):
        plan_oracle.PlanOracle()
    monkeypatch.setenv(plan_oracle.CELL_CONFIG_ENV, "c.json")
    monkeypatch.setenv(plan_oracle.SIM_INPUT_ENV, "d")
    monkeypatch.setenv("HEROSIM_SNAPSHOT_FIDELITY", "0")
    with pytest.raises(RuntimeError, match="HEROSIM_SNAPSHOT_FIDELITY=1"):
        plan_oracle.PlanOracle()


def test_harness_serves_cd_oracle_as_a_named_measurement_arm():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G

    assert G.RULE_POLICY["cd_oracle"] == "peer_greedy_network_cd"
    assert "cd_oracle" in G.R1A_DIAG and "cd_oracle" in G.R1A_ARMS
    src = open(G.__file__).read()
    assert 'if kind == "cd_oracle":\n            env.update(HEROSIM_PG_CD_EXACT="1"' in src and "HEROSIM_PG_ORACLE=\"1\"" in src
    assert "*KEEPWARM_ENV, *ORACLE_ENV):" in src
    for name in ("HEROSIM_PG_ORACLE", "HEROSIM_SNAPSHOT_FIDELITY", "HEROSIM_ORACLE_CELL_CONFIG"):
        assert name in G.ORACLE_ENV
    assert "a non-oracle arm ran the live oracle" in src
