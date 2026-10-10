"""GNN_SLATE_NO_SPLIT (src/placement/declared_slate.py, GNNScheduler._prefix_inference): the declared serving variant that keeps
the top-5 slate but decodes an over-100k batch whole. Off by default: the sub-batches are decoded exactly as before."""
from types import SimpleNamespace

import pytest

from src.placement import declared_slate as D
from src.placement import live_audit
from src.policy.gnn.scheduler import GNNScheduler


def shell(monkeypatch, groups, sub_batched):
    """A GNNScheduler with a 4-task batch; the slate is pinned to `groups`; `_prefix_inference_core` is recorded."""
    s = object.__new__(GNNScheduler)
    orch = SimpleNamespace(peer_exchange={})
    s._orchestrator = lambda: orch
    s._get_valid_replicas = lambda replicas, task: []
    monkeypatch.setattr(live_audit, "_candidate_payload", lambda *a: {})
    kept = [[{"node_id": t, "platform_id": 10 + t}, {"node_id": 9, "platform_id": 90}] for t in range(4)]
    monkeypatch.setattr(D, "slate", lambda payloads, pairs: D.Slate(kept, [7] * 4, 7 ** 4, groups, True, sub_batched))
    calls = []

    def core(batch, state, queue, temporal, allowed=None):
        calls.append({"ids": [t.id for t in batch], "allowed": allowed, "queue": queue})
        return {k: (int(t.id), 10 + int(t.id)) for k, t in enumerate(batch)}

    s._prefix_inference_core = core
    tasks = [SimpleNamespace(id=i, type={"name": "cnn"}) for i in range(4)]
    return s, tasks, calls


def run(s, tasks):
    snap = {"q": 1}
    return s._prefix_inference(tasks, SimpleNamespace(replicas={}), snap, None), snap


def test_flag_off_decodes_the_sub_batches_as_before(monkeypatch):
    monkeypatch.setenv(D.ENV, D.RULE)
    monkeypatch.delenv(D.NO_SPLIT_ENV, raising=False)
    s, tasks, calls = shell(monkeypatch, [[0, 1], [2, 3]], True)
    out, snap = run(s, tasks)
    assert [c["ids"] for c in calls] == [[0, 1], [2, 3]]
    assert calls[1]["allowed"] == {0: {(2, 12), (9, 90)}, 1: {(3, 13), (9, 90)}}
    assert all(c["queue"] is snap for c in calls)
    assert out == {i: (i, 10 + i) for i in range(4)}
    assert getattr(s, "slate_declared_unsplit", 0) == 0


def test_flag_on_decodes_an_over_cap_batch_whole_on_the_same_slate(monkeypatch):
    monkeypatch.setenv(D.ENV, D.RULE)
    monkeypatch.setenv(D.NO_SPLIT_ENV, "1")
    s, tasks, calls = shell(monkeypatch, [[0, 1], [2, 3]], True)
    out, snap = run(s, tasks)
    assert [c["ids"] for c in calls] == [[0, 1, 2, 3]]
    assert calls[0]["allowed"] == {t: {(t, 10 + t), (9, 90)} for t in range(4)}   # every task keeps its own top-K
    assert out == {i: (i, 10 + i) for i in range(4)}
    assert s.slate_declared_unsplit == 1 and s.slate_declared_sub_batched == 1


def test_flag_on_leaves_an_unsplit_batch_alone(monkeypatch):
    monkeypatch.setenv(D.ENV, D.RULE)
    monkeypatch.setenv(D.NO_SPLIT_ENV, "1")
    s, tasks, calls = shell(monkeypatch, [[0, 1, 2, 3]], False)
    run(s, tasks)
    assert [c["ids"] for c in calls] == [[0, 1, 2, 3]] and getattr(s, "slate_declared_unsplit", 0) == 0


def test_the_variant_is_validated_and_needs_the_declared_slate(monkeypatch):
    monkeypatch.delenv(D.NO_SPLIT_ENV, raising=False)
    monkeypatch.delenv(D.ENV, raising=False)
    assert D.serving_no_split() is False
    monkeypatch.setenv(D.NO_SPLIT_ENV, "1")
    with pytest.raises(ValueError, match="is unset"):
        D.serving_no_split()
    with pytest.raises(ValueError, match="is unset"):
        D.require_matching_slate(None, what="m")              # an unpruned checkpoint cannot take the variant
    monkeypatch.setenv(D.ENV, D.RULE)
    assert D.serving_no_split() is True
    D.require_matching_slate(D.RULE, what="m")                 # accepted by name for a declared-slate checkpoint
    monkeypatch.setenv(D.NO_SPLIT_ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        D.require_matching_slate(D.RULE, what="m")


def test_counter_and_provenance_are_recorded():
    import inspect

    from src import executesimulation
    from src.placement.orchestrator import Orchestrator

    assert '"slate_declared_unsplit"' in inspect.getsource(Orchestrator._scheduler_counters)
    assert '"GNN_SLATE_NO_SPLIT"' in inspect.getsource(executesimulation)
