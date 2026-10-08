"""Request timeout (R1.1): a placed task still waiting for an unplaced peer 300 s after placement fails (at the next
autoscaler tick), frees its replica and enters latency at its elapsed time. The peer rendezvous is the one placed-task wait with no bound of its
own, so a hold-and-wait cycle between draining replicas and starved peers ends here."""
import simpy

from src.placement import infrastructure as infra
from src.placement.infrastructure import Platform, REQUEST_TIMEOUT_S, expire_requests, interrupt_if_waiting
from simpy.resources.store import FilterStore


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _platform(env, peers_ready_at=None):
    """A Platform whose rendezvous waits on an event that fires at `peers_ready_at` (never when None)."""
    storage = NS(type={"remote": True, "throughput": {"read": 1.0, "write": 1.0}, "latency": {"read": 0.0, "write": 0.0}})
    node = NS(storage=FilterStore(env), network={"bandwidth": 1.0}, local_dependencies=0,
              orchestrator_ref=NS(request_failures=0, peer_exchange={}, task_by_id={}), node_name="node0")
    node.storage.items.append(storage)

    class P(Platform):
        def platform_process(self):
            yield env.event()

        def _dependency_transfer_time(self, task):
            return 0.0

        def _peer_exchange_time(self, task):
            return 0.0

        def _peer_rendezvous_events(self, task):
            ready = env.event()
            if peers_ready_at is not None:
                env.process(self._fire(ready, peers_ready_at))
            return [ready]

        def _fire(self, ready, at):
            yield env.timeout(at)
            ready.succeed()

    return P(env, 1, {"shortName": "rpi", "name": "Raspberry Pi"}, node), node


def _task(env, scheduled_time=0.0):
    app = NS(type={"name": "app"})
    return NS(id=7, env=env, type={"name": "dnn1", "executionTime": {"rpi": 1.0}, "stateSize": {"app": {"input": 1e12, "output": 0}}},
              application=app, dependencies=[], node_name="client_node2", scheduled_time=scheduled_time,
              started=env.event(), done=env.event(), arrived=env.event(), storage={"input": None, "output": None},
              failed=False, failure_reason=None, peer_rendezvous_wait=0.0, peer_exchange_time=0.0, is_internal=True,
              cold_start_time=0.0, local_dependencies=None)


def _tick(env):
    """The autoscaler's reconcile loop, which sweeps the request deadlines once per tick."""
    while True:
        expire_requests(env)
        yield env.timeout(1.0)


def _serve(env, platform, task):
    env.process(_tick(env))
    platform.inflight.append(task)
    done_at = []
    task.done.callbacks.append(lambda _e: done_at.append(env.now))
    env.process(platform._serve_task(task, 0.0, release=True))
    return done_at


def test_a_task_whose_peer_never_arrives_fails_300_s_after_placement(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    env = simpy.Environment()
    platform, node = _platform(env)
    task = _task(env, scheduled_time=20.0)
    env.run(until=20.0)
    done_at = _serve(env, platform, task)
    env.run(until=1000)
    assert task.failed and task.failure_reason == infra.REQUEST_TIMEOUT
    assert len(done_at) == 1 and 20.0 + REQUEST_TIMEOUT_S <= done_at[0] < 20.0 + REQUEST_TIMEOUT_S + 1.0 + 1e-6
    assert platform.inflight == [] and platform.idle_since == done_at[0]
    assert node.orchestrator_ref.request_failures == 1
    assert task.started.triggered and not platform.rendezvous_procs


def test_a_wait_that_ends_in_time_is_untouched_and_the_stale_deadline_does_nothing(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    env = simpy.Environment()
    platform, node = _platform(env, peers_ready_at=100.0)
    task = _task(env)
    _serve(env, platform, task)
    env.run(until=1000)  # the deadline passes at 300 s while the task sleeps in its input stage
    assert not task.failed and task.started.triggered and not task.done.triggered
    assert node.orchestrator_ref.request_failures == 0 and platform.inflight == [task]


def test_a_second_interrupt_in_the_same_instant_is_not_delivered():
    env = simpy.Environment()
    seen = []

    def waiter():
        try:
            yield env.timeout(100)
        except simpy.Interrupt as i:
            seen.append(i.cause)
        yield env.timeout(5)  # a second interrupt delivered here would raise

    proc = env.process(waiter())
    env.run(until=1)
    assert interrupt_if_waiting(proc, "a") is True
    assert interrupt_if_waiting(proc, "b") is False
    env.run()
    assert seen == ["a"]


def test_a_drain_held_by_a_starved_peer_completes_when_the_request_times_out(monkeypatch):
    """The W4 hold-and-wait: a draining replica's only task waits for a peer that needs a platform this drain must free
    first, and nothing can release it. The timeout ends the wait, the replica empties, the drain completes."""
    from src.policy.gnn.autoscaler import KnativeAutoscaler

    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    env = simpy.Environment()
    platform, node = _platform(env)
    platform.queue = NS(items=[])
    platform.current_task = None
    task = _task(env)
    _serve(env, platform, task)

    a = object.__new__(KnativeAutoscaler)
    a.env, a._draining, a.scale_events = env, {("dnn2", "client_node2")}, []
    a.data = NS(task_types={"dnn2": {}})
    released_at, created = [], []
    a._release_replica = lambda state, fn, replica, already_removed=False: released_at.append(env.now) or replica

    def create_first_replica(state, task_type, source_node_name=None):
        created.append(source_node_name)
        yield env.timeout(0)

    a.create_first_replica = create_first_replica
    env.process(a._release_when_drained(NS(), "dnn1", (node, platform), ("dnn2", "client_node2")))
    env.run(until=1000)
    assert len(released_at) == 1 and REQUEST_TIMEOUT_S <= released_at[0] < REQUEST_TIMEOUT_S + 1.2
    assert created == ["client_node2"] and not a._draining
