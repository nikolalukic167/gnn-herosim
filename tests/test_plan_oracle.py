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


SNAP = {"fidelity": {"peers": {}, "future": {}}}


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
    monkeypatch.setattr("src.placement.live_audit.capture_batch_snapshot", lambda *a, **k: dict(SNAP))
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


def test_unlabellable_batch_keeps_cd_exactS_and_is_counted(monkeypatch):
    # a partner outside the batch that the live run had not placed yet: no snapshot gives the rendezvous time, the replay cannot label the batch
    s, tasks, state = _oracle_shell(monkeypatch, {SPLIT: 0.0})
    monkeypatch.setattr("src.placement.live_audit.capture_batch_snapshot",
                        lambda *a, **k: {"fidelity": {"peers": {"0": [[99, None, 1.0]]}, "future": {}}})
    assert s._pg_decide(tasks, state) == {0: (3, 33), 1: (3, 33)}
    assert s._plan_oracle.calls == 0
    assert (s.pg_oracle_batches, s.pg_oracle_unplaced_batches, s.pg_oracle_unplaced_tasks, s.pg_oracle_seen_batches) == (0, 1, 2, 1)
    monkeypatch.setattr("src.placement.live_audit.capture_batch_snapshot",
                        lambda *a, **k: {"fidelity": {"peers": {"0": [[99, None, 1.0]]}, "future": {"99": [3.0, "A"]}}})
    s.pg_oracle_unplaced_batches = 0
    s._pg_decide(tasks, state)
    assert s._plan_oracle.calls == 1 and s.pg_oracle_unplaced_batches == 0   # a partner with a known placement time is labellable


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


def _alpha_shell(monkeypatch, **env):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return _shell(monkeypatch)


def test_alpha_one_is_the_registered_s_and_scaling_prices_the_queue_term(monkeypatch):
    # queue term only: a platform that already holds 4 s of drain and a plan that stacks both tasks on one platform
    monkeypatch.setattr(S, "platform_queue_drain_seconds", lambda platform, orch, memo, exec_scale=1.0: 4.0)
    s1, tasks, state = _alpha_shell(monkeypatch)
    assert s1.pg_alpha_on is False
    nodes = s1.nodes.items
    plan = [(nodes[2], state.replicas["cnn"][2][1])] * 2
    full, _svc, _sc = s1._pg_plan_cost(tasks, plan, s1._pg_orchestrator(), {}, nodes)
    monkeypatch.setattr(S, "platform_queue_drain_seconds", lambda platform, orch, memo, exec_scale=1.0: 4.0)
    for env, want in (({S.PG_DRAIN_ALPHA_ENV: "0"}, 0.0), ({S.PG_BATCH_ALPHA_ENV: "0"}, None)):
        monkeypatch.delenv(S.PG_DRAIN_ALPHA_ENV, raising=False)
        monkeypatch.delenv(S.PG_BATCH_ALPHA_ENV, raising=False)
        s2, tasks2, state2 = _alpha_shell(monkeypatch, **env)
        cost, _svc2, _sc2 = s2._pg_plan_cost(tasks2, plan, s2._pg_orchestrator(), {}, nodes)
        if want == 0.0:
            assert cost == pytest.approx(2.0), (full, cost)   # exec only: 1 s each, co-located, no exchange
        else:
            assert cost == pytest.approx(full - 2 * 1.0)         # only the in-batch stacking part (each task sees the other's 1 s of service) is zeroed
    monkeypatch.delenv(S.PG_BATCH_ALPHA_ENV, raising=False)


def test_alpha_is_refused_outside_cd_and_the_cd_refiner(monkeypatch):
    from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkBatchScheduler

    monkeypatch.delenv(S.PG_CD_EXACT_ENV, raising=False)
    monkeypatch.setenv(S.PG_DRAIN_ALPHA_ENV, "0")
    with pytest.raises(RuntimeError, match="FAIL LOUD"):
        _shell(monkeypatch, cls=PeerGreedyNetworkBatchScheduler)   # the plain batch greedy has no CD refine
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    monkeypatch.setenv(S.PG_DRAIN_ALPHA_ENV, "1.5")
    with pytest.raises(ValueError, match="FAIL LOUD"):
        _shell(monkeypatch)


def test_harness_serves_the_alpha_arms(monkeypatch):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G

    for kind, env in (("cd_exactS_a0", {"HEROSIM_PG_DRAIN_ALPHA": "0"}), ("cd_exactS_ib0", {"HEROSIM_PG_BATCH_ALPHA": "0"})):
        assert G.RULE_POLICY[kind] == "peer_greedy_network_cd" and kind in G.R1A_DIAG and kind in G.R1A_ARMS and G.EXACT_ALPHA[kind] == env
    assert {"HEROSIM_PG_DRAIN_ALPHA", "HEROSIM_PG_BATCH_ALPHA"} <= set(G.ORACLE_ENV)


def test_worker_builds_the_cell_infrastructure_once_per_config(monkeypatch):
    import src.executesimulation as ES
    from src.placement import plan_oracle

    calls = []
    monkeypatch.setattr(ES, "prepare_infrastructure_for_real_simulation", lambda cfg, seed=None, sim_input_path=None: calls.append(("p", seed)) or {"cfg": cfg})
    monkeypatch.setattr(ES, "load_simulation_inputs", lambda path: calls.append(("l", str(path))) or {"path": str(path)})
    plan_oracle._memoize_static_inputs()
    a = ES.prepare_infrastructure_for_real_simulation({"x": 1}, seed=None, sim_input_path="d")
    b = ES.prepare_infrastructure_for_real_simulation({"x": 1}, seed=None, sim_input_path="d")
    c = ES.prepare_infrastructure_for_real_simulation({"x": 2}, seed=None, sim_input_path="d")
    assert a is b and a != c and ES.load_simulation_inputs("d") is ES.load_simulation_inputs("d")
    assert calls == [("p", None), ("p", None), ("l", "d")]
