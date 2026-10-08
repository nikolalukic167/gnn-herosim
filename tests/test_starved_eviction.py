"""evict_idle_for: a starved type may take the platform of one idle (or drained) replica of another type (client_local_v1).

Under single-origin groups a type could starve forever: a new replica needs a free platform and nothing frees one
held by another type's replica. The scheduler calls evict_idle_for only for a task it found starved.
"""
import simpy

from src.policy.gnn.autoscaler import KnativeAutoscaler

class NS:
    """Hashable attribute bag (replicas live in sets)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


TYPES = {
    "dnn1": {"name": "dnn1", "platforms": ["rpi"], "memoryRequirements": {"rpi": 1.0}},
    "dnn2": {"name": "dnn2", "platforms": ["rpi"], "memoryRequirements": {"rpi": 1.0}},
}


def _platform(env, pid, idle_since, busy=False):
    p = NS(id=pid, env=env, type={"shortName": "rpi"}, queue=NS(items=[1] if busy else []), current_task=None,
           initialized=env.event(), idle_since=idle_since, last_removed=None, previous_task=None)
    p.initialized.succeed()
    p.queue_length = lambda: len(p.queue.items)
    return p


def _setup(n_dnn1=3, busy=(), reach=("client_node2",)):
    env = simpy.Environment()
    nodes, replicas = [], set()
    for i in range(n_dnn1):
        node = NS(id=40 + i, node_name=f"node{i}", available_memory=0.0, available_platforms=0,
                  network_map={c: {"latency": 0.02} for c in reach})
        nodes.append(node)
        replicas.add((node, _platform(env, 200 + i, idle_since=10.0 * (i + 1), busy=i in busy)))
    a = object.__new__(KnativeAutoscaler)
    a.env, a.data, a.scale_events = env, NS(task_types=TYPES), []
    a.created = []

    def create_first_replica(system_state, task_type, source_node_name=None):
        a.created.append((task_type["name"], source_node_name))
        yield env.timeout(0)

    a.create_first_replica = create_first_replica
    state = NS(replicas={"dnn1": replicas, "dnn2": set()}, available_resources={n: set() for n in nodes},
               scheduler_state=NS(average_contention={"dnn1": {}}))
    return a, state, nodes


def test_evicts_the_longest_idle_reachable_replica_of_another_type():
    a, state, nodes = _setup(busy=(0,))
    node, platform = a.evict_idle_for(state, TYPES["dnn2"], "client_node2")
    assert (node.id, platform.id) == (41, 201)  # 40 is busy; 41 idle since 20 s beats 42 (30 s)
    assert platform in state.available_resources[node] and len(state.replicas["dnn1"]) == 2
    assert node.available_memory == 1.0 and a.scale_events[-1]["action"] == "down"


def test_never_takes_a_functions_last_replica_or_an_unreachable_one():
    a, state, _ = _setup(n_dnn1=1)
    assert a.evict_idle_for(state, TYPES["dnn2"], "client_node2") is None
    a, state, _ = _setup(reach=("client_node7",))
    assert a.evict_idle_for(state, TYPES["dnn2"], "client_node2") is None


def test_drains_the_least_loaded_busy_replica_when_none_is_idle():
    a, state, nodes = _setup(busy=(0, 1, 2))
    for n, p in state.replicas["dnn1"]:
        p.queue.items = [1, 1]
    lightest = next(p for n, p in state.replicas["dnn1"] if n.id == 41)
    lightest.queue.items = []
    lightest.current_task = "running"
    assert a.evict_idle_for(state, TYPES["dnn2"], "client_node2") is None
    assert len(state.replicas["dnn1"]) == 2 and all(p is not lightest for _, p in state.replicas["dnn1"])
    assert lightest not in state.available_resources[nodes[1]]  # held until it drains
    assert a.evict_idle_for(state, TYPES["dnn2"], "client_node2") is None  # one drain per (type, source)
    assert len(state.replicas["dnn1"]) == 2
    a.env.run(until=1.0)
    assert lightest not in state.available_resources[nodes[1]]
    lightest.current_task = None
    a.env.run(until=2.0)
    assert lightest in state.available_resources[nodes[1]] and nodes[1].available_memory == 1.0
    assert a.scale_events[-1]["action"] == "down" and not a._draining
    assert a.created == [("dnn2", "client_node2")]


def test_drain_releases_a_rendezvous_on_the_starved_peer_and_plans_it_onto_the_drained_node():
    """The drain deadlock: the draining platform's task waits for a peer that is the starved task itself."""
    from simpy.exceptions import Interrupt
    from src.placement.infrastructure import STARVED_RENDEZVOUS

    a, state, nodes = _setup(busy=(0, 1, 2))
    for _, p in state.replicas["dnn1"]:
        p.queue.items = [1, 1]
    node = nodes[1]
    plat = next(p for n, p in state.replicas["dnn1"] if n is node)
    plat.queue.items = []
    waiter = NS(id=7, type=TYPES["dnn1"], node_name="client_node2", platform=plat, planned_node_name=None)
    starved = NS(id=8, type=TYPES["dnn2"], node_name="client_node2", platform=None, planned_node_name=None)
    node.orchestrator_ref = NS(peer_exchange={7: {8: 1e6}}, task_by_id={7: waiter, 8: starved})
    log = []

    def worker():
        plat.current_task, plat.rendezvous_task = waiter, waiter
        try:
            yield a.env.event()  # the starved peer is never scheduled
        except Interrupt as i:
            log.append(i.cause)
        plat.rendezvous_task = None
        yield a.env.timeout(0.5)
        plat.current_task = None

    plat.run = a.env.process(worker())
    a.env.run(until=0.01)
    assert a.evict_idle_for(state, TYPES["dnn2"], "client_node2") is None
    a.env.run(until=3.0)
    assert log == [STARVED_RENDEZVOUS] and starved.planned_node_name == node.node_name
    assert plat in state.available_resources[node] and a.created == [("dnn2", "client_node2")]


# --- starve_v1: the starved-replica hang (cross-source drain cycle, free-pool leak, spin, log volume) -------------------


def _blocked_platform(a, state, nodes, idx, waiter_id, peer, waiter_type="dnn1"):
    """Make replica `idx` busy with a task parked in rendezvous on `peer` (an unplaced task)."""
    node = nodes[idx]
    plat = next(p for n, p in state.replicas["dnn1"] if n is node)
    plat.queue.items = []
    waiter = NS(id=waiter_id, type=TYPES[waiter_type], node_name="client_node2", platform=plat, planned_node_name=None)
    plat.current_task, plat.rendezvous_task = waiter, waiter
    orch = getattr(node, "orchestrator_ref", None) or NS(peer_exchange={}, task_by_id={})
    orch.peer_exchange[waiter_id] = {peer.id: 1e6}
    orch.task_by_id.update({waiter_id: waiter, peer.id: peer})
    node.orchestrator_ref = orch
    return plat


def test_a_drain_for_one_source_releases_a_starved_same_type_peer_from_another_source():
    """The cycle seen on 9565 g2: platforms drained for dnn2 from client A hold tasks waiting on dnn2 from client B."""
    from simpy.exceptions import Interrupt
    from src.placement.infrastructure import STARVED_RENDEZVOUS

    a, state, nodes = _setup(busy=(0, 1, 2), reach=("client_node2", "client_node9"))
    for _, p in state.replicas["dnn1"]:
        p.queue.items = [1, 1]
    peer = NS(id=8, type=TYPES["dnn2"], node_name="client_node9", platform=None, planned_node_name=None,
              postponed_count=3)
    plat = _blocked_platform(a, state, nodes, 1, 7, peer)
    log = []

    def worker():
        try:
            yield a.env.event()
        except Interrupt as i:
            log.append(i.cause)
        plat.rendezvous_task = None
        plat.current_task = None

    plat.run = a.env.process(worker())
    a.env.run(until=0.01)
    assert a.evict_idle_for(state, TYPES["dnn2"], "client_node2") is None  # drain starts for (dnn2, client_node2)
    a.env.run(until=3.0)
    assert log == [STARVED_RENDEZVOUS] and peer.planned_node_name == nodes[1].node_name
    assert a.created == [("dnn2", "client_node2")]


def test_a_peer_not_yet_deferred_or_out_of_reach_is_not_released():
    a, state, nodes = _setup(busy=(0, 1, 2), reach=("client_node2",))
    for _, p in state.replicas["dnn1"]:
        p.queue.items = [1, 1]
    arriving = NS(id=8, type=TYPES["dnn2"], node_name="client_node9", platform=None, planned_node_name=None,
                  postponed_count=0)
    _blocked_platform(a, state, nodes, 1, 7, arriving)
    assert a._releasable_peers(nodes[1], nodes[1].orchestrator_ref.task_by_id[7], ("dnn2", "client_node2")) is None
    arriving.postponed_count = 2  # starved, but client_node9 does not reach node1
    assert a._releasable_peers(nodes[1], nodes[1].orchestrator_ref.task_by_id[7], ("dnn2", "client_node2")) is None


def test_a_victim_blocked_on_a_peer_the_drain_cannot_release_is_skipped():
    """Draining it would hold the platform forever; the other busy replica is drained instead."""
    a, state, nodes = _setup(busy=(0, 1, 2), reach=("client_node2",))
    for _, p in state.replicas["dnn1"]:
        p.queue.items = [1, 1, 1]
    foreign = NS(id=8, type=TYPES["dnn1"], node_name="client_node5", platform=None, planned_node_name=None,
                 postponed_count=4)
    stuck = _blocked_platform(a, state, nodes, 0, 7, foreign)
    stuck.queue.items = []  # least loaded of the three, yet not drainable
    assert a.evict_idle_for(state, TYPES["dnn2"], "client_node2") is None
    drained = [n for n in nodes if all(n is not m for m, _ in state.replicas["dnn1"])]
    assert len(drained) == 1 and drained[0] is not nodes[0]


def test_a_decoded_but_not_yet_enqueued_target_is_never_evicted():
    a, state, nodes = _setup()
    target = next((n.id, p.id) for n, p in state.replicas["dnn1"] if n.id == 41)
    a.reserve_target(target)
    chosen = a.evict_idle_for(state, TYPES["dnn2"], "client_node2")
    assert chosen is not None and (chosen[0].id, chosen[1].id) != target
    a.unreserve_target(target)
    assert not a._reserved_targets


def test_create_first_replica_never_swaps_the_shared_free_pool():
    """Two overlapping calls used to restore each other's filtered dict, dropping nodes from the pool for good."""
    a, state, nodes = _setup()
    a.data = NS(task_types={"dnn2": {"name": "dnn2", "platforms": ["rpi"]}}, platform_types={"rpi": {"shortName": "rpi"}})
    pool = {n: {_platform(a.env, 900 + n.id, 0.0)} for n in nodes}
    state.available_resources = pool
    seen = []

    def scale_up(count, system_state, function_name, hardware_target, cause="reachability", reachable_nodes=None):
        seen.append((system_state.available_resources is pool, reachable_nodes))
        yield a.env.timeout(1.0)
        return None

    a.scale_up = scale_up
    del a.create_first_replica  # use the real method, not the _setup stub
    task_type = {"name": "dnn2", "platforms": ["rpi"]}
    a.env.process(a.create_first_replica(state, task_type, source_node_name="client_node2"))
    a.env.run(until=0.5)
    a.env.process(a.create_first_replica(state, task_type, source_node_name="client_node2"))
    a.env.run(until=5.0)
    assert state.available_resources is pool and all(identical for identical, _ in seen) and len(seen) == 2
    assert all(r == set(nodes) for _, r in seen)


def test_starved_log_is_rate_limited_and_counts_what_it_dropped(caplog):
    import logging
    from src.placement.starved_defer import STARVED_LOG_INTERVAL_S, log_starved

    owner = NS()
    with caplog.at_level(logging.ERROR):
        emitted = [log_starved(owner, t * 0.01, ("k",), "No compatible hardware") for t in range(5000)]
        assert sum(emitted) == 1 and len(caplog.records) == 1
        assert log_starved(owner, STARVED_LOG_INTERVAL_S + 1.0, ("k",), "No compatible hardware")
        assert "4999 identical lines suppressed" in caplog.records[-1].getMessage()
        assert log_starved(owner, 0.0, ("other",), "No compatible hardware")  # a different key is not suppressed


class _Harness:
    """A scheduler reduced to StarvedDeferMixin: one task, an autoscaler that creates nothing until `free_at`."""

    def __init__(self, free_at=None, nodes=None):
        from src.placement.starved_defer import StarvedDeferMixin

        class S(StarvedDeferMixin):
            pass

        self.env = simpy.Environment()
        self.s = S()
        self.s.env = self.env
        self.s.tasks = simpy.Store(self.env)
        self.s.nodes = nodes
        self.s._init_starved_defer()
        self.creates, self.evicts, self.free_at = [], 0, free_at
        h = self

        class A:
            def create_first_replica(self, system_state, task_type, source_node_name=None):
                h.creates.append(h.env.now)
                yield h.env.timeout(0)
                if h.free_at is not None and h.env.now >= h.free_at:
                    h.placed = True
                    return None
                return StopIteration("none")

            def evict_idle_for(self, *a):
                h.evicts += 1
                return None

        self.s.autoscaler = A()
        self.placed = False
        self.task = NS(id=1, type={"name": "dnn2", "platforms": ["rpi"]}, node_name="client_node2", postponed_count=0)

    def run(self, until):
        def driver():
            while not self.placed:
                task = yield self.s.tasks.get()
                yield from self.s._defer(task, NS())

        self.s.tasks.put(self.task)
        self.env.process(driver())
        self.env.run(until=until)


def test_a_starved_task_does_not_respin_on_every_retry_and_still_places_when_capacity_frees():
    from src.placement.starved_defer import DEFER_RETRY_S, DEFER_SPIN_LIMIT

    h = _Harness(free_at=30.0)
    h.run(until=60.0)
    assert h.placed
    # the first burst spins DEFER_SPIN_LIMIT times at t=0; each later second costs one creation attempt and one evict
    assert len(h.creates) <= DEFER_SPIN_LIMIT + int(30 / DEFER_RETRY_S) + 3
    assert h.evicts <= 1 + int(30 / DEFER_RETRY_S) + 1


def test_a_task_no_reachable_node_can_ever_serve_fails_loudly():
    import pytest
    from src.placement.starved_defer import StarvedForeverError

    pynq = NS(node_name="node2", network_map={"client_node2": 1}, platforms=NS(items=[NS(type={"shortName": "pynq"})]))
    xavier = NS(node_name="node4", network_map={"client_node9": 1}, platforms=NS(items=[NS(type={"shortName": "rpi"})]))
    h = _Harness(nodes=NS(items=[pynq, xavier]))
    with pytest.raises(StarvedForeverError, match="no capacity will ever free"):
        h.run(until=10.0)
