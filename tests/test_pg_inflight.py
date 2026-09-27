"""burst_ladder_v1: HEROSIM_PG_INFLIGHT adds the in-flight task's remaining service to CD's drain.

Default off must leave the rule's choice and score unchanged; on, a platform serving a long task
reads busy and the rule moves to the idle one.
"""
from types import SimpleNamespace

import pytest

import src.policy.peer_greedy_network.scheduler as S


class _Rule(S._PeerGreedyCore):
    exchange_on = False
    _policy_label = "test_rule"


def _platform(pid, node, now, inflight_end=None):
    return SimpleNamespace(id=pid, node=node, type={"shortName": "rpiCpu"},
                           current_task=object() if inflight_end is not None else None,
                           inflight_service_end=inflight_end, env=SimpleNamespace(now=now))


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(S, "platform_queue_drain_seconds", lambda p, orch, memo: 0.0)
    monkeypatch.setattr(S, "incoming_cold_start_time", lambda task, p: 0.0)
    monkeypatch.setattr(S, "network_latency_between", lambda a, n, nodes: 0.0)
    n0 = SimpleNamespace(id=0, node_name="node0")
    n1 = SimpleNamespace(id=1, node_name="node1")
    busy = _platform(10, n0, now=100.0, inflight_end=104.0)   # 4 s of service left
    idle = _platform(20, n1, now=100.0)
    task = SimpleNamespace(id=1, node_name="client", type={"executionTime": {"rpiCpu": 1.0}, "name": "dnn2"})
    return task, [(n0, busy), (n1, idle)]


def _choose(task, cands):
    rule = _Rule()
    rule._pg_init()
    node, plat, _svc = rule._pg_choose(task, cands, None, memo={}, committed_service={}, planned={}, nodes=[])
    return rule, node, plat


def test_default_ignores_the_inflight_task(monkeypatch, world):
    monkeypatch.delenv(S.PG_INFLIGHT_ENV, raising=False)
    task, cands = world
    rule, node, plat = _choose(task, cands)
    assert (node.id, plat.id) == (0, 10)  # tie on score, lowest node id wins: the busy one
    assert rule.pg_inflight_charged == 0 and rule.pg_inflight_seconds == 0.0


def test_flag_charges_remaining_service_and_moves_off_the_busy_platform(monkeypatch, world):
    monkeypatch.setenv(S.PG_INFLIGHT_ENV, "1")
    task, cands = world
    rule, node, plat = _choose(task, cands)
    assert (node.id, plat.id) == (1, 20)
    assert rule.pg_inflight_charged == 1
    assert rule.pg_inflight_seconds == pytest.approx(4.0)


def test_flag_rejects_garbage(monkeypatch):
    monkeypatch.setenv(S.PG_INFLIGHT_ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _Rule()._pg_init()
