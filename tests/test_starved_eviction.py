"""evict_idle_for: a starved type may take the platform of one idle replica of another type (client_local_v1).

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
