"""GNN_SEQ_GROUP_LOAD (src/policy/gnn/scheduler.py): under the declared slate, a sub-batch decoded after another sees the earlier
one's placements as queued load -- one more task in the queue count and its drain seconds in the v5 backlog -- and with the flag
off every sub-batch is decoded on the batch's own snapshot, exactly as before."""
from types import SimpleNamespace

import pytest

from src.placement import declared_slate as D
from src.placement import live_audit
from src.policy.gnn.scheduler import GNNScheduler


class FakePlatform:
    def __init__(self, pid, node, short="xavierGpu"):
        self.id = pid
        self.node = node
        self.type = {"shortName": short}
        self.queue = SimpleNamespace(items=[])
        self.virtual_warmup_total_time = 0.0

    def _payload_transfer_time(self, other, payload):
        return 0.01 * payload


class FakeNode:
    def __init__(self, nid, name, network_map):
        self.id, self.node_name, self.network_map = nid, name, network_map


def make_node(nid, name, pids, network_map=None):
    node = FakeNode(nid, name, network_map or {})
    node.platforms = SimpleNamespace(items=[FakePlatform(p, node) for p in pids])
    return node


def make_task(tid, src="client0", exec_s=2.0):
    return SimpleNamespace(id=tid, node_name=src, platform=None, planned_node_name=None,
                           type={"name": "cnn", "executionTime": {"xavierGpu": exec_s},
                                 "stateSize": {"app": {"input": 100 * 1024 * 1024, "output": 0}}})


def scheduler_with_two_groups(monkeypatch, plan_of_core):
    """A GNNScheduler shell whose batch of 2 tasks is cut into two sub-batches of one task each; `_prefix_inference_core` is
    recorded and answers from `plan_of_core`."""
    nodes = [make_node(1, "server1", [10, 11]), make_node(2, "server2", [20])]
    s = object.__new__(GNNScheduler)
    s.nodes = SimpleNamespace(items=nodes)
    orch = SimpleNamespace(peer_exchange={}, task_by_id={})
    s._orchestrator = lambda: orch
    s._get_valid_replicas = lambda replicas, task: [(nodes[0], nodes[0].platforms.items[0]), (nodes[1], nodes[1].platforms.items[0])]
    monkeypatch.setattr(live_audit, "_candidate_payload",
                        lambda sched, task, n, p: {"node_id": n.id, "platform_id": p.id, "queue_key": f"{n.node_name}:{p.id}"})
    tasks = [make_task(100), make_task(101)]
    kept = [[{"node_id": 1, "platform_id": 10}, {"node_id": 2, "platform_id": 20}]] * 2
    monkeypatch.setattr(D, "slate", lambda payloads, pairs: D.Slate(kept, [2, 2], 4, [[0], [1]], False, True))
    calls = []

    def core(batch, state, queue, temporal, allowed=None, backlog_extra=None):
        calls.append({"ids": [t.id for t in batch], "queue": dict(queue), "queue_obj": queue, "backlog_extra": None if backlog_extra is None else dict(backlog_extra)})
        return {0: plan_of_core[batch[0].id]}

    s._prefix_inference_core = core
    state = SimpleNamespace(replicas={"cnn": set()})
    return s, tasks, state, calls


def test_flag_off_decodes_every_sub_batch_on_the_batch_snapshot(monkeypatch):
    monkeypatch.setenv(D.ENV, D.RULE)
    monkeypatch.delenv("GNN_SEQ_GROUP_LOAD", raising=False)
    s, tasks, state, calls = scheduler_with_two_groups(monkeypatch, {100: (1, 10), 101: (1, 10)})
    snap = {"server1:10": 3, "server2:20": 0}
    out = s._prefix_inference(tasks, state, snap, None)
    assert out == {0: (1, 10), 1: (1, 10)}
    assert [c["queue_obj"] is snap for c in calls] == [True, True]
    assert [c["backlog_extra"] for c in calls] == [None, None]
    assert getattr(s, "seq_group_loaded_groups", 0) == 0


def test_flag_on_a_later_sub_batch_sees_the_earlier_ones_load(monkeypatch):
    monkeypatch.setenv(D.ENV, D.RULE)
    monkeypatch.setenv("GNN_SEQ_GROUP_LOAD", "1")
    monkeypatch.setattr(live_audit, "pending_task_seconds", lambda task, platform, orch, planned: 2.5)
    s, tasks, state, calls = scheduler_with_two_groups(monkeypatch, {100: (1, 10), 101: (2, 20)})
    snap = {"server1:10": 3, "server2:20": 0}
    s._prefix_inference(tasks, state, snap, None)
    assert calls[0]["queue"] == snap and calls[0]["backlog_extra"] is None
    assert calls[1]["queue"] == {"server1:10": 4, "server2:20": 0}        # task 100 now queued on (1, 10)
    assert calls[1]["backlog_extra"] == {(1, 10): 2.5}
    assert snap == {"server1:10": 3, "server2:20": 0}                      # the batch snapshot itself is untouched
    assert (s.seq_group_loaded_groups, s.seq_group_loaded_tasks) == (1, 2)


def test_flag_on_without_the_declared_slate_fails_loud(monkeypatch):
    monkeypatch.delenv(D.ENV, raising=False)
    monkeypatch.setenv("GNN_SEQ_GROUP_LOAD", "1")
    s = object.__new__(GNNScheduler)
    with pytest.raises(RuntimeError, match="GNN_SEQ_GROUP_LOAD"):
        s._prefix_inference([], SimpleNamespace(replicas={}), {}, None)
    monkeypatch.setenv("GNN_SEQ_GROUP_LOAD", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        s._prefix_inference([], SimpleNamespace(replicas={}), {}, None)


@pytest.mark.parametrize("peer_on", ["0", "1"])
def test_pending_seconds_is_what_the_drain_charges_once_queued(monkeypatch, peer_on):
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", peer_on)
    node = make_node(1, "server1", [10], network_map={"client0": {"latency": 0.004}, "server2": {"latency": 0.002}})
    platform = node.platforms.items[0]
    task, peer = make_task(100), make_task(101)
    orch = SimpleNamespace(peer_exchange={100: {101: 5e6}}, task_by_id={100: task, 101: peer})
    planned = {101: "server2"}
    pending = live_audit.pending_task_seconds(task, platform, orch, planned)
    before = live_audit.platform_queue_drain_seconds(platform, orch)
    peer.planned_node_name = "server2"
    platform.queue.items.append(task)
    after = live_audit.platform_queue_drain_seconds(platform, orch)
    assert pending == pytest.approx(after - before, rel=1e-12)
    assert pending > 2.0 + (5e4 if peer_on == "1" else 0.0)


def test_backlog_extra_adds_to_the_v5_backlog(monkeypatch):
    from src.policy.tabular import reduced_features

    monkeypatch.setattr(reduced_features, "resolve_partial_state_contract", lambda: "partial_state_v5")
    monkeypatch.setattr(live_audit, "candidate_backlog_seconds", lambda sched, n, p, memo: 1.0)
    nodes = [make_node(1, "server1", [10]), make_node(2, "server2", [20])]
    s = object.__new__(GNNScheduler)
    s.v4_backlog_batches = s.v4_backlog_nonzero = 0
    state = SimpleNamespace(replicas={"cnn": {(nodes[0], nodes[0].platforms.items[0]), (nodes[1], nodes[1].platforms.items[0])}})
    tl = {0: [(1, 10), (2, 20)]}
    assert s._v4_backlog_seconds(tl, state) == {(1, 10): 1.0, (2, 20): 1.0}
    assert s._v4_backlog_seconds(tl, state, {(1, 10): 2.5}) == {(1, 10): 3.5, (2, 20): 1.0}
