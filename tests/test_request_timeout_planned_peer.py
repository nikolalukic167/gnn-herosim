"""A partner the batch path planned and then deferred keeps `planned_node_name` with no platform. It is waiting for hardware, so its
partners' request timeout must see it as unplaced (found on CD 9607 g1 x5: four platforms held by a group's own members waiting on
its starved rf task, which counted as placed, so the 300 s clock never started). A batch-mate planned but not yet enqueued is
still placed, so completed runs gain no rendezvous."""
import simpy
from simpy.resources.store import FilterStore

from src.placement import infrastructure as infra
from src.placement.infrastructure import Platform, REQUEST_TIMEOUT_S, expire_requests, peer_is_placed
from src.placement.request_timeout import request_deadline


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _partner(**kw):
    base = dict(id=9, platform=None, planned_node_name=None, postponed_count=0, dispatched_time=50.0)
    base.update(kw)
    return NS(**base)


def _platform(env, partner):
    storage = NS(type={"remote": True, "throughput": {"read": 1.0, "write": 1.0}, "latency": {"read": 0.0, "write": 0.0}})
    orch = NS(request_failures=0, peer_exchange={7: {9: 1.0}}, task_by_id={9: partner}, peer_ready_event=lambda pid: env.event())
    node = NS(storage=FilterStore(env), network={"bandwidth": 1.0}, local_dependencies=0, orchestrator_ref=orch, node_name="node0")
    node.storage.items.append(storage)

    class P(Platform):
        def platform_process(self):
            yield env.event()

        def _dependency_transfer_time(self, task):
            return 0.0

        def _peer_exchange_time(self, task):
            return 0.0

    return P(env, 1, {"shortName": "rpi", "name": "Raspberry Pi"}, node), node


def _holder(env):
    app = NS(type={"name": "app"})
    return NS(id=7, env=env, type={"name": "dnn1", "executionTime": {"rpi": 1.0}, "stateSize": {"app": {"input": 1e12, "output": 0}}},
              application=app, dependencies=[], node_name="client_node0", scheduled_time=20.0,
              started=env.event(), done=env.event(), arrived=env.event(), storage={"input": None, "output": None},
              failed=False, failure_reason=None, peer_rendezvous_wait=0.0, peer_exchange_time=0.0, is_internal=True,
              cold_start_time=0.0, local_dependencies=None)


def test_a_planned_then_deferred_partner_is_unplaced_for_the_deadline_and_the_rendezvous(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    env = simpy.Environment()
    partner = _partner(planned_node_name="node0", postponed_count=3)
    platform, _ = _platform(env, partner)
    holder = _holder(env)
    assert not peer_is_placed(partner)
    assert platform._unplaced_peers(holder) == [partner]
    assert request_deadline(20.0, platform._unplaced_peers(holder)) == 50.0 + REQUEST_TIMEOUT_S
    assert len(platform._peer_rendezvous_events(holder)) == 1


def test_a_batch_mate_planned_but_not_yet_enqueued_is_placed(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    env = simpy.Environment()
    mate = _partner(planned_node_name="node0", postponed_count=0)
    platform, _ = _platform(env, mate)
    holder = _holder(env)
    assert peer_is_placed(mate)
    assert platform._unplaced_peers(holder) == []
    assert platform._peer_rendezvous_events(holder) == []


def test_a_peer_on_a_platform_is_placed_even_after_a_deferral(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    env = simpy.Environment()
    peer = _partner(platform=NS(id=3), planned_node_name="node0", postponed_count=5)
    platform, _ = _platform(env, peer)
    assert peer_is_placed(peer)
    assert platform._unplaced_peers(_holder(env)) == [] and platform._peer_rendezvous_events(_holder(env)) == []


def test_the_holder_of_a_planned_then_deferred_partner_times_out_300_s_after_the_partner_arrives(monkeypatch):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    env = simpy.Environment()
    partner = _partner(planned_node_name="node0", postponed_count=37026)  # arrived at 50 s, deferred ever since
    platform, node = _platform(env, partner)
    task = _holder(env)

    def tick():
        while True:
            expire_requests(env)
            yield env.timeout(1.0)

    env.run(until=20.0)
    env.process(tick())
    platform.inflight.append(task)
    done_at = []
    task.done.callbacks.append(lambda _e: done_at.append(env.now))
    env.process(platform._serve_task(task, 0.0, release=True))
    env.run(until=1000)
    assert task.failed and task.failure_reason == infra.REQUEST_TIMEOUT
    assert len(done_at) == 1 and 50.0 + REQUEST_TIMEOUT_S <= done_at[0] < 50.0 + REQUEST_TIMEOUT_S + 1.0 + 1e-6
    assert platform.inflight == [] and node.orchestrator_ref.request_failures == 1
