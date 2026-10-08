"""A decoded target is reserved against eviction until its task is on the platform's queue. When the replica vanishes before
the task's turn (S7's stale-placement branch), the ORIGINAL reserved target must be released, whether the task is re-decoded
onto another replica or deferred; releasing the re-decoded target instead left the original reserved for good."""
import simpy
from simpy.resources.store import FilterStore, Store

from src.policy.gnn import scheduler as sched_mod
from src.policy.gnn.autoscaler import KnativeAutoscaler
from src.policy.gnn.scheduler import GNNScheduler


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


TYPES = {
    "dnn1": {"name": "dnn1", "platforms": ["rpi"], "memoryRequirements": {"rpi": 1.0}},
    "dnn2": {"name": "dnn2", "platforms": ["rpi"], "memoryRequirements": {"rpi": 1.0}},
    "dnn3": {"name": "dnn3", "platforms": ["rpi"], "memoryRequirements": {"rpi": 1.0}},
}


def _platform(env, pid):
    p = NS(id=pid, env=env, type={"shortName": "rpi"}, queue=Store(env), current_task=None, initialized=env.event(),
           idle_since=1.0, last_removed=None, previous_task=None)
    p.initialized.succeed()
    p.queue_length = lambda: len(p.queue.items)
    return p


def _run(monkeypatch, other_replica_survives):
    env = simpy.Environment()
    node = NS(id=41, node_name="node1", available_memory=0.0, available_platforms=0, unused=True,
              wall_clock_scheduling_time=0.0, network_map={"client_node2": {}}, platforms=FilterStore(env))
    p_a, p_b, p_c, p_d = (_platform(env, i) for i in (10, 11, 12, 13))
    for p in (p_a, p_b, p_c, p_d):
        node.platforms.items.append(p)
    # dnn1 owns p_a (decoded) and, in the re-decode case, p_b; dnn3 owns two idle replicas p_c, p_d
    dnn1 = {(node, p_a)} | ({(node, p_b)} if other_replica_survives else set())
    state = NS(replicas={"dnn1": dnn1, "dnn2": set(), "dnn3": {(node, p_c), (node, p_d)}},
               available_resources={node: set()}, scheduler_state=NS(average_contention={}))

    a = object.__new__(KnativeAutoscaler)
    a.env, a.data, a.scale_events = env, NS(task_types=TYPES), []
    s = object.__new__(GNNScheduler)
    s.env, s.autoscaler, s.nodes = env, a, FilterStore(env)
    s.nodes.items.append(node)
    s.mutex = Store(env)
    s.mutex.items.append(state)
    s.prefix_batches = s.prefix_tasks_decoded = s.prefix_tasks_deferred = s.gnn_pure_decisions = 0
    s._live_audit_policy_name = "test"
    s._get_valid_replicas = lambda replicas, task: sorted(replicas, key=lambda r: r[1].id)
    s._capture_full_queue_snapshot = lambda: {}
    s._capture_temporal_state_snapshot = lambda: {}
    s._record_residence_placed = lambda task: None
    decode_calls = []

    def prefix_inference(tasks, st, qs, ts):
        decode_calls.append([t.id for t in tasks])
        target = next(iter(sorted(st.replicas[tasks[0].type["name"]], key=lambda r: r[1].id)))
        return {i: (target[0].id, target[1].id) for i in range(len(tasks))}

    s._prefix_inference = prefix_inference

    def defer(task, st):
        # the other deferred task's turn: the decoded replica goes away (a KPA scale-down) before the decoded task's turn
        if task.type["name"] == "dnn2":
            st.replicas["dnn1"].discard((node, p_a))
        yield env.timeout(0)

    s._defer = defer
    monkeypatch.setattr(sched_mod, "maybe_capture_batch_live_audit_snapshot", lambda *a, **k: None)
    monkeypatch.setattr("src.placement.replica_seeding.start_deferred_cold_init", lambda *a, **k: None)

    def task(i, name):
        return NS(id=i, type=TYPES[name], node_name="client_node2", planned_node_name=None, postponed_count=0,
                  scheduled=env.event(), dependencies=[])

    t1, t2 = task(1, "dnn1"), task(2, "dnn2")
    env.process(s._process_task_batch_prefix([t1, t2]))
    env.run(until=10)
    return a, state, node, p_a, p_b, p_c, t1, decode_calls


def test_redecode_releases_the_original_reservation(monkeypatch):
    a, state, node, p_a, p_b, p_c, t1, calls = _run(monkeypatch, other_replica_survives=True)
    assert calls == [[1], [1]]  # decoded in the batch, then re-decoded alone: the branch fired
    assert t1.platform is p_b  # placed on the re-decoded replica
    assert not a._reserved_targets
    # the original target is evictable again: hand p_a to dnn3 as an idle replica and ask for it
    state.replicas["dnn3"] = {(node, p_a), (node, p_c)}
    state.replicas["dnn1"].discard((node, p_a))
    chosen = a.evict_idle_for(state, TYPES["dnn2"], "client_node2")
    assert chosen is not None and chosen[1].id in (p_a.id, p_c.id)
    assert (node.id, p_a.id) not in a._reserved_targets


def test_defer_path_releases_the_original_reservation(monkeypatch):
    a, state, node, p_a, p_b, p_c, t1, calls = _run(monkeypatch, other_replica_survives=False)
    assert calls == [[1]]  # no replica left on re-decode: the task is deferred, not placed
    assert getattr(t1, "platform", None) is None
    assert not a._reserved_targets
    state.replicas["dnn3"] = {(node, p_a), (node, p_c)}
    chosen = a.evict_idle_for(state, TYPES["dnn2"], "client_node2")
    assert chosen is not None


def test_the_old_release_leaves_the_original_reserved(monkeypatch):
    """Control: releasing placements[idx] after a re-decode (the code before the fix) strands the original reservation."""
    a = object.__new__(KnativeAutoscaler)
    a.reserve_target((41, 10))
    a.unreserve_target((41, 11))  # what the old code released after re-decoding onto (41, 11)
    assert (41, 10) in a._reserved_targets
