from types import SimpleNamespace

import pytest

from src.policy.knative_network.scheduler import KnativeInFlightScheduler, KnativeScheduler


def _platform(pid, queue=0, inflight=0, current=False, admitted=False, initialized=True):
    p = SimpleNamespace(id=pid, queue=SimpleNamespace(items=list(range(queue))), virtual_warmup_count=0,
                        inflight=[object() for _ in range(inflight)], current_task=object() if current else None,
                        admitted=object() if admitted else None, initialized=SimpleNamespace(triggered=initialized))
    p.queue_length = lambda: len(p.queue.items) + p.virtual_warmup_count
    return p


def _place(cls, couples):
    s = object.__new__(cls)
    s._get_valid_replicas = lambda replicas, task: list(replicas)
    state = SimpleNamespace(replicas={"T": couples})
    gen = s.placement(state, SimpleNamespace(id=1, type={"name": "T"}))
    try:
        next(gen)
    except StopIteration as e:
        return e.value
    raise AssertionError("placement must return")


def _couples():
    # replica 1 (lowest id): empty queue but five tasks waiting on its compute lock; replica 2: idle
    busy, idle = _platform(1, queue=0, inflight=5, current=True), _platform(2)
    return [(SimpleNamespace(id=1), busy), (SimpleNamespace(id=2), idle)], busy, idle


def test_release_counts_the_tasks_waiting_on_the_compute_lock(monkeypatch):
    import src.placement.infrastructure as infra

    monkeypatch.setattr(infra, "REPLICA_RELEASE", True)
    couples, busy, idle = _couples()
    assert _place(KnativeInFlightScheduler, couples)[1] is idle
    assert _place(KnativeScheduler, couples)[1] is busy  # the queue alone is blind to them: the id tie-break picks replica 1


def test_without_release_it_is_the_queue_as_before(monkeypatch):
    import src.placement.infrastructure as infra

    monkeypatch.setattr(infra, "REPLICA_RELEASE", False)
    couples, busy, idle = _couples()
    assert _place(KnativeInFlightScheduler, couples)[1] is busy
    couples[0][1].queue.items.extend([1, 2])
    assert _place(KnativeInFlightScheduler, couples)[1] is idle


def test_ties_keep_the_id_tie_break_and_queued_tasks_still_count(monkeypatch):
    import src.placement.infrastructure as infra

    monkeypatch.setattr(infra, "REPLICA_RELEASE", True)
    a, b = _platform(1), _platform(2)
    assert _place(KnativeInFlightScheduler, [(SimpleNamespace(id=2), b), (SimpleNamespace(id=1), a)])[1] is a
    a.queue.items.append(1)
    assert _place(KnativeInFlightScheduler, [(SimpleNamespace(id=1), a), (SimpleNamespace(id=2), b)])[1] is b


def test_only_the_knative_network_policy_is_switched():
    import os

    src = open(os.path.join(os.path.dirname(__file__), "..", "src/placement/simulation.py")).read()
    assert '"kn_network_kn_network": (KnativeNetworkOrchestrator, KnativeNetworkAutoscaler, KnativeInFlightScheduler)' in src
    assert KnativeScheduler._connections(_platform(1, queue=3, inflight=9)) == 3
