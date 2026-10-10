"""cd_expand (HEROSIM_PG_CD_EXPANSION=1): exact-move alpha-expansion inside the batched CD refine. Off by default: the registered
peer_greedy_network_cd never calls it. The toy below is the case single-task moves cannot solve: two partners that each sit on the
node that executes them for free, paying a 10 s exchange, while a third node runs both at 1 s with no exchange. Moving one task alone
costs more than it saves (ICM is stuck); moving the pair is the improving move."""
from types import SimpleNamespace

import pytest

from src.policy.peer_greedy_network import scheduler as S
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkBatchScheduler, PeerGreedyNetworkCDScheduler

EXCHANGE_S = 10.0


def _platform(pid, short):
    return SimpleNamespace(id=pid, type={"shortName": short}, initialized=SimpleNamespace(triggered=True),
                           _payload_transfer_time=lambda peer_node_name, payload: EXCHANGE_S,
                           peer_link_latency=lambda peer_node_name, context="": 0.0)


def _shell(monkeypatch, cls=PeerGreedyNetworkCDScheduler, exec_times=None):
    """A CD scheduler over three one-platform nodes A, B, C; drain, cold start, latency and I/O are zero, so S is exec + exchange
    (+ in-batch stacking). Task 0 runs free on A, task 1 free on B, both at 1 s on C; the two are partners."""
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    monkeypatch.setattr(S, "platform_queue_drain_seconds", lambda platform, orch, memo, exec_scale=1.0: 0.0)
    monkeypatch.setattr(S, "incoming_cold_start_time", lambda task, platform: 0.0)
    monkeypatch.setattr(S, "network_latency_between", lambda src, node, nodes: 0.0)
    monkeypatch.setattr(S, "_approx_comm", lambda task_type: 0.0)
    nodes = [SimpleNamespace(id=1, node_name="A"), SimpleNamespace(id=2, node_name="B"), SimpleNamespace(id=3, node_name="C")]
    reps = [(nodes[0], _platform(11, "pa")), (nodes[1], _platform(22, "pb")), (nodes[2], _platform(33, "pc"))]
    exec_times = exec_times or [{"pa": 0.0, "pb": 100.0, "pc": 1.0}, {"pa": 100.0, "pb": 0.0, "pc": 1.0}]
    tasks = [SimpleNamespace(id=i, type={"name": "cnn", "executionTime": et}, node_name="client0") for i, et in enumerate(exec_times)]
    s = object.__new__(cls)
    s._pg_init()
    s._get_valid_replicas = lambda replicas, task: sorted(replicas, key=lambda r: (r[0].id, r[1].id))
    s._pg_orchestrator = lambda: SimpleNamespace(peer_exchange={0: {1: 1.0}, 1: {0: 1.0}}, task_by_id={})
    s.nodes = SimpleNamespace(items=nodes)
    s.prefix_pairs_in_batch = 0
    s.prefix_peers_outside_batch = 0
    state = SimpleNamespace(replicas={"cnn": reps})
    return s, tasks, state


def test_flag_off_is_the_registered_cd_and_never_expands(monkeypatch):
    monkeypatch.delenv(S.PG_CD_EXPANSION_ENV, raising=False)
    s, tasks, state = _shell(monkeypatch)

    def never(*a, **k):
        raise AssertionError("expansion ran with the flag off")

    monkeypatch.setattr(s, "_pg_expand", never)
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)}   # ICM is stuck on the split pair
    assert s.pg_cd_passes == 1 and s.pg_cd_moves == 0
    assert (s.pg_expand_batches, s.pg_expand_sweeps, s.pg_expand_moves, s.pg_expand_tasks_moved, s.pg_expand_evals,
            s.pg_expand_gain_seconds, s.pg_expand_labels_skipped) == (0, 0, 0, 0, 0, 0.0, 0)


def test_flag_on_moves_the_pair_icm_cannot(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    s, tasks, state = _shell(monkeypatch)
    assert s._pg_decide(tasks, state) == {0: (3, 33), 1: (3, 33)}
    assert (s.pg_expand_batches, s.pg_expand_moves, s.pg_expand_tasks_moved) == (1, 1, 2)
    # the start plan, then sweep 1: labels A (1 subset), B (1), C (3: {0}, {1}, {0,1}); sweep 2 finds nothing (A: 3, B: 3, C: none)
    assert s.pg_expand_sweeps == 2 and s.pg_expand_evals == 1 + 5 + 6
    # S summed: split pair 10 + 10 = 20 -> both on C: (1 + stacking 1) x 2 = 4
    assert s.pg_expand_gain_seconds == pytest.approx(16.0)
    # the single-task passes ran again on the new plan and kept it
    assert s.pg_cd_passes == 2 and s.pg_cd_moves == 0
    # the partner books count decisions only, not the scored trial plans: the first pass + two refine passes
    assert s.pg_partners_known == 1 + 2 + 2
    # the batch books describe the new plan
    assert s.pg_expand_labels_skipped == 0


def test_flag_on_leaves_a_plan_without_an_improving_subset_untouched(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    # C is now worse than the split: 20 s each, so neither a single move nor the pair improves on 10 + 10
    s, tasks, state = _shell(monkeypatch, exec_times=[{"pa": 0.0, "pb": 100.0, "pc": 20.0}, {"pa": 100.0, "pb": 0.0, "pc": 20.0}])
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)}
    assert (s.pg_expand_batches, s.pg_expand_sweeps, s.pg_expand_moves, s.pg_expand_tasks_moved) == (1, 1, 0, 0)
    assert s.pg_expand_gain_seconds == 0.0 and s.pg_cd_passes == 1


def test_a_label_shared_by_too_many_tasks_is_skipped_and_counted(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    monkeypatch.setenv(S.PG_CD_EXPANSION_MAX_MOVERS_ENV, "1")
    s, tasks, state = _shell(monkeypatch)
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)}   # the pair move to C needs 2 movers
    assert s.pg_expand_labels_skipped == 1 and s.pg_expand_moves == 0


def test_a_task_held_outside_its_candidates_is_pinned(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    s, tasks, state = _shell(monkeypatch, exec_times=[{"pa": 0.0, "pb": 0.0, "pc": 1.0}, {"pa": 100.0, "pb": 0.0, "pc": 1.0}])
    # task 0 may only take A at refine time but was placed on B before: it is scored where it is and never moved; with both
    # free on B and co-located, no subset of task 1 improves, so the plan stands
    only_a = {(1, 11)}
    s._pg_allowed = {0: only_a, 1: {(1, 11), (2, 22), (3, 33)}}
    placements, planned, service_of, committed = {0: (2, 22), 1: (2, 22)}, {0: "B", 1: "B"}, {0: ("B:22", 0.0), 1: ("B:22", 0.0)}, {"B:22": 0.0}
    moved = s._pg_expand(tasks, state, s._pg_orchestrator(), {}, committed, planned, placements, service_of)
    assert moved == 0 and placements == {0: (2, 22), 1: (2, 22)} and s.pg_expand_batches == 1


def test_bad_values_and_the_wrong_flavour_fail_loudly(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "2")
    with pytest.raises(ValueError, match=S.PG_CD_EXPANSION_ENV):
        _shell(monkeypatch)
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    monkeypatch.setenv(S.PG_CD_EXPANSION_SWEEPS_ENV, "0")
    with pytest.raises(ValueError, match=S.PG_CD_EXPANSION_SWEEPS_ENV):
        _shell(monkeypatch)
    monkeypatch.delenv(S.PG_CD_EXPANSION_SWEEPS_ENV, raising=False)
    with pytest.raises(RuntimeError, match="peer_greedy_network_cd only"):
        _shell(monkeypatch, cls=PeerGreedyNetworkBatchScheduler)


def test_counters_are_exported():
    import inspect

    from src.placement import orchestrator

    exporter = inspect.getsource(orchestrator.Orchestrator._scheduler_counters)
    for k in ("pg_expand_batches", "pg_expand_sweeps", "pg_expand_moves", "pg_expand_tasks_moved", "pg_expand_evals",
              "pg_expand_gain_seconds", "pg_expand_labels_skipped"):
        assert k in S.PEER_GREEDY_COUNTERS
        assert f'"{k}"' in exporter   # the orchestrator's schedulerCounters whitelist, which the gate summaries read


def test_harness_serves_cd_expand_as_a_named_diagnostic_arm():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G

    assert G.RULE_POLICY["cd_expand"] == "peer_greedy_network_cd"
    assert "cd_expand" in G.R1A_DIAG and "cd_expand" in G.R1A_ARMS
    src = open(G.__file__).read()
    assert 'if kind == "cd_expand":\n            env["HEROSIM_PG_CD_EXPANSION"] = "1"' in src
    assert '"HEROSIM_PG_CD_EXPANSION", "GNN_SLATE_NO_SPLIT", *KEEPWARM_ENV' in src   # scrubbed per cell before the kind sets it
