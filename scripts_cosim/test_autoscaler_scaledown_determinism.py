"""
Determinism of the autoscaler scale-down tie-break.

`eb6d131` ("Fix nondeterministic placement tie-breaks") sorted by (node.id, platform.id)
at every *scheduler* site, and at the autoscaler's `available_hardware` loops -- but it
missed `remove_replica`'s `sorted_replicas` in all three autoscalers, where the key stayed
`len(couple[1].queue.items)` alone. That key is non-total: scale-down only ever removes a
replica whose queue is *empty*, so every eligible candidate ties at 0 and `sorted` (being
stable) just returns them in input order. The input is a `Set[Tuple[Node, Platform]]`
whose iteration order follows id()-based object hashes, which `PYTHONHASHSEED` does NOT
pin (it randomizes str/bytes hashing only). Measured residual spread on the siv1 live
gate: 0.05% of total_rtt -- small, but enough to swamp a sub-percent gate margin.

These tests drive the real `remove_replica` with the same logical candidates presented in
different orders and assert the same replica is chosen every time.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy.gnn.autoscaler import KnativeAutoscaler as GnnAutoscaler
from src.policy.knative.autoscaler import KnativeAutoscaler as KnativeAutoscalerImpl
from src.policy.knative_network.autoscaler import (
    KnativeAutoscaler as KnativeNetworkAutoscaler,
)

AUTOSCALERS = [
    pytest.param(KnativeAutoscalerImpl, id="knative"),
    pytest.param(GnnAutoscaler, id="gnn"),
    pytest.param(KnativeNetworkAutoscaler, id="knative_network"),
]

TASK_TYPE = {"name": "task_a"}


def _couple(node_id: int, platform_id: int, queue_len: int = 0):
    """A duck-typed (Node, Platform) couple, eligible for scale-down by default."""
    node = SimpleNamespace(id=node_id)
    platform = SimpleNamespace(
        id=platform_id,
        queue=SimpleNamespace(items=[object()] * queue_len),
        current_task=None,
        idle_since=0.0,
    )
    return (node, platform)


def _make_autoscaler(cls, now=1000.0, keep_alive=10.0):
    """Build the autoscaler without running __init__ (which needs a full SimPy env)."""
    autoscaler = object.__new__(cls)
    autoscaler.env = SimpleNamespace(now=now)
    autoscaler.policy = SimpleNamespace(keep_alive=keep_alive)
    return autoscaler


def _run(autoscaler, candidates):
    """`remove_replica` is a generator function; its result is StopIteration.value."""
    system_state = SimpleNamespace(
        scheduler_state=SimpleNamespace(average_contention={TASK_TYPE["name"]: {}})
    )
    gen = autoscaler.remove_replica(candidates, TASK_TYPE, system_state)
    try:
        next(gen)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("remove_replica yielded; expected it to return immediately")


@pytest.mark.parametrize("cls", AUTOSCALERS)
def test_scaledown_choice_is_independent_of_candidate_order(cls):
    """All candidates tie at queue_len=0, so only a total sort key makes this stable."""
    couples = [_couple(node_id=n, platform_id=p) for n, p in [(3, 7), (1, 2), (5, 1), (1, 9)]]

    expected = _run(_make_autoscaler(cls), list(couples))
    assert expected is not None, "no replica was eligible for scale-down; test is vacuous"

    # The lowest (node.id, platform.id) must win regardless of presentation order.
    assert (expected[0].id, expected[1].id) == (1, 2)

    for order in ([*reversed(couples)], couples[2:] + couples[:2], couples[1:] + couples[:1]):
        chosen = _run(_make_autoscaler(cls), list(order))
        assert (chosen[0].id, chosen[1].id) == (1, 2), (
            f"{cls.__module__}: scale-down picked {(chosen[0].id, chosen[1].id)} for one "
            f"input order and {(expected[0].id, expected[1].id)} for another -- the sort "
            f"key is not total, so set iteration order decides which replica dies"
        )


@pytest.mark.parametrize("cls", AUTOSCALERS)
def test_scaledown_still_prefers_shortest_queue(cls):
    """The id tie-break must not override the primary key."""
    # (1, 2) has the lowest ids but a non-empty queue, so it is not even eligible;
    # (4, 4) is the only empty-queue candidate and must be the one chosen.
    couples = [_couple(1, 2, queue_len=3), _couple(4, 4, queue_len=0), _couple(2, 1, queue_len=5)]
    chosen = _run(_make_autoscaler(cls), list(couples))
    assert chosen is not None
    assert (chosen[0].id, chosen[1].id) == (4, 4)


@pytest.mark.parametrize("cls", AUTOSCALERS)
def test_scaledown_returns_none_when_nothing_eligible(cls):
    """Busy or recently-idle replicas are never removed."""
    busy = _couple(1, 1, queue_len=2)
    recently_idle = _couple(2, 2, queue_len=0)
    recently_idle[1].idle_since = 995.0  # now=1000, keep_alive=10 -> only 5s idle
    running = _couple(3, 3, queue_len=0)
    running[1].current_task = object()

    assert _run(_make_autoscaler(cls), [busy, recently_idle, running]) is None
