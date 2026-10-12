"""cd_declared (r1_attribution_v1): CD over exactly the declared slate the learned arms are served; inert unless GNN_SERVE_CANDIDATE_SLATE is set."""
from types import SimpleNamespace

import pytest

from src.placement import declared_slate, live_audit
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkCDScheduler


def _sched(monkeypatch, n_tasks=3, n_replicas=8):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    nodes = [SimpleNamespace(id=10 + i, node_name=f"node{i}") for i in range(n_replicas)]
    reps = [(n, SimpleNamespace(id=100 + i, initialized=SimpleNamespace(triggered=True))) for i, n in enumerate(nodes)]
    tasks = [SimpleNamespace(id=i, type={"name": "cnn"}, node_name="client0") for i in range(n_tasks)]
    s = object.__new__(PeerGreedyNetworkCDScheduler)
    s._pg_init()
    s._get_valid_replicas = lambda replicas, task: sorted(replicas, key=lambda r: (r[0].id, r[1].id))
    s._pg_orchestrator = lambda: SimpleNamespace(peer_exchange={0: {1: 1.0}, 1: {0: 1.0, 2: 1.0}, 2: {1: 1.0}})
    calls = []

    def decide(batch, state):
        calls.append(([int(t.id) for t in batch], None if s._pg_allowed is None else {k: set(v) for k, v in s._pg_allowed.items()}))
        return {i: (10, 100) for i in range(len(batch))}

    s._pg_decide = decide

    def payload(sched, task, node, platform):
        # cost grows with the platform id, so the cheapest five are platforms 100..104 for every task
        return {"node_id": node.id, "platform_id": platform.id, "initialized": True, "queue_drain_seconds": float(platform.id - 100),
                "execution_time": 1.0, "network_latency": 0.0}

    monkeypatch.setattr(live_audit, "_candidate_payload", payload)
    state = SimpleNamespace(replicas={"cnn": reps})
    return s, tasks, state, calls, payload


def test_inert_when_the_slate_is_unset(monkeypatch):
    monkeypatch.delenv(declared_slate.ENV, raising=False)
    s, tasks, state, calls, _ = _sched(monkeypatch)
    s._prefix_inference(tasks, state, {}, None)
    assert calls == [([0, 1, 2], None)] and s._pg_allowed is None and s.pg_declared_batches == 0


def test_uses_exactly_declared_slate_slate(monkeypatch):
    monkeypatch.setenv(declared_slate.ENV, declared_slate.RULE)
    monkeypatch.setenv(declared_slate.MAX_PLANS_ENV, "10")  # 5^3 plans > 10: sub-batches of <= 4 tasks
    s, tasks, state, calls, payload = _sched(monkeypatch, n_tasks=6)
    monkeypatch.setattr(s, "_pg_orchestrator", lambda: SimpleNamespace(peer_exchange={i: {i + 1: 1.0} for i in range(5)}))
    placements = s._prefix_inference(tasks, state, {}, None)

    payloads = [{"task_id": int(t.id), "candidates": [payload(s, t, n, p) for n, p in s._get_valid_replicas(state.replicas["cnn"], t)]} for t in tasks]
    pairs = [(i, i + 1) for i in range(5)] + [(i + 1, i) for i in range(5)]
    want = declared_slate.slate(payloads, pairs)
    assert want.sub_batched and len(want.groups) == 2
    assert [c[0] for c in calls] == [[int(tasks[i].id) for i in g] for g in want.groups]
    for (ids, allowed), group in zip(calls, want.groups):
        assert allowed == {int(tasks[i].id): {(int(c["node_id"]), int(c["platform_id"])) for c in want.kept[i]} for i in group}
        assert all(len(a) == declared_slate.TOP_K for a in allowed.values())
    assert sorted(placements) == list(range(6)) and s._pg_allowed is None
    assert (s.pg_declared_batches, s.pg_declared_pruned, s.pg_declared_sub_batched, s.pg_declared_groups) == (1, 1, 1, 2)


def test_pg_restrict_filters_only_while_a_slate_is_active(monkeypatch):
    s, tasks, state, _, _ = _sched(monkeypatch)
    valid = s._get_valid_replicas(state.replicas["cnn"], tasks[0])
    assert s._pg_restrict(tasks[0], valid) is valid
    s._pg_allowed = {0: {(valid[0][0].id, valid[0][1].id)}}
    assert s._pg_restrict(tasks[0], valid) == [valid[0]]


def test_a_bad_slate_value_fails_loudly(monkeypatch):
    monkeypatch.setenv(declared_slate.ENV, "top3")
    s, tasks, state, _, _ = _sched(monkeypatch)
    with pytest.raises(ValueError, match=declared_slate.ENV):
        s._prefix_inference(tasks, state, {}, None)
