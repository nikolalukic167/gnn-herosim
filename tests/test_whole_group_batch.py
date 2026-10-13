"""uncut_groups_s0_v1 ruling C: whole-group batches, and an out-of-range cell batch_size fails loud.

gate-tools 2026-10-13: until this change GNNOrchestrator printed and ignored a cell batch_size outside the
scheduler's range and served GNN_BATCH_SIZE, so snapshot replays of > 16-task batches were served 4 at a time.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import simpy

from src.policy.gnn import orchestrator as orch_mod
from src.policy.gnn.orchestrator import WHOLE_GROUP, apply_cfg_batch_size
from src.policy.gnn.scheduler import GNNScheduler, PG_BATCH_COUNTERS, PREFIX_DECODE_MODE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))


def _sched(decode=PREFIX_DECODE_MODE, by_group=True, size=4, lo=1, hi=16):
    s = GNNScheduler.__new__(GNNScheduler)
    s._decode_mode, s.batch_by_peer_group, s.whole_group = decode, by_group, False
    s.batch_size, s.batch_min, s.batch_max = size, lo, hi
    for name in PG_BATCH_COUNTERS:
        setattr(s, name, 0)
    return s


@pytest.fixture(autouse=True)
def _no_env_batch_size(monkeypatch):
    monkeypatch.setenv("GNN_BATCH_SIZE", "0")
    monkeypatch.delenv("GNN_BATCH_SIZE")


def test_in_range_config_is_applied_as_before():
    s = _sched()
    apply_cfg_batch_size(s, 10)
    assert s.batch_size == 10 and not s.whole_group
    apply_cfg_batch_size(s, "10")  # int() of the config value, as before
    assert s.batch_size == 10


@pytest.mark.parametrize("bad", [17, 18, 140, 3000, 0])
def test_out_of_range_config_fails_loud(bad):
    with pytest.raises(ValueError, match="FAIL LOUD"):
        apply_cfg_batch_size(_sched(), bad)


@pytest.mark.parametrize("bad", [10.5, True, "ten", None])
def test_non_integer_config_fails_loud(bad):
    with pytest.raises(ValueError, match="FAIL LOUD"):
        apply_cfg_batch_size(_sched(), bad)


def test_argmax_range_config_fails_loud():
    with pytest.raises(ValueError, match="FAIL LOUD"):
        apply_cfg_batch_size(_sched(decode="argmax", by_group=False, lo=2, hi=4), 10)


def test_disagreeing_env_fails_loud(monkeypatch):
    monkeypatch.setenv("GNN_BATCH_SIZE", "4")
    with pytest.raises(ValueError, match="FAIL LOUD"):
        apply_cfg_batch_size(_sched(), 10)
    with pytest.raises(ValueError, match="FAIL LOUD"):
        apply_cfg_batch_size(_sched(), WHOLE_GROUP)


def test_whole_group_lifts_the_cap_only_on_the_peer_group_path():
    s = _sched()
    apply_cfg_batch_size(s, WHOLE_GROUP)
    assert s.whole_group and s.batch_size > 10 ** 9 and s.batch_max > 10 ** 9
    for decode, by_group in (("argmax", True), (PREFIX_DECODE_MODE, False)):
        with pytest.raises(ValueError, match="FAIL LOUD"):
            apply_cfg_batch_size(_sched(decode=decode, by_group=by_group), WHOLE_GROUP)


def test_the_orchestrator_calls_the_rule():
    text = open(orch_mod.__file__.replace(".pyc", ".py")).read()
    assert "apply_cfg_batch_size(self.scheduler, self.scheduler_config[\"batch_size\"])" in text
    assert "Ignoring infrastructure scheduler.batch_size" not in text


def _collect(n_members, cfg_bs, late_at=None):
    """Run the real peer-group collector on one group of n_members, all queued at t = 0 (or one member
    arriving at `late_at`), and return the first batch."""
    env = simpy.Environment()
    s = _sched()
    apply_cfg_batch_size(s, cfg_bs)
    s.env, s.batch_timeout, s.peer_group_incomplete_batches = env, 1.0, 0
    s.tasks = simpy.FilterStore(env)
    s._residence_batches = []
    tasks = [SimpleNamespace(id=i, dependencies=[], scheduled=env.event()) for i in range(n_members)]
    orch = SimpleNamespace(task_by_id={t.id: t for t in tasks})
    s._orchestrator = lambda: orch
    s._peer_group = lambda task: set(range(n_members))
    for t in tasks:
        if late_at is None or t.id != n_members - 1:
            s.tasks.put(t)

    def late():
        yield env.timeout(late_at)
        s.tasks.put(tasks[-1])

    if late_at is not None:
        env.process(late())
    out = {}

    def run():
        out["batch"] = yield from s._collect_task_batch()

    env.process(run())
    env.run(until=5.0)
    return out["batch"], s


def test_a_3000_task_group_is_served_whole_only_in_whole_group_mode():
    batch, s = _collect(3000, 10)
    assert len(batch) == 10 and s.pgb_split_gt100 == 10 and s.pgb_whole_gt100 == 0 and s.pgb_max_batch == 10
    batch, s = _collect(3000, WHOLE_GROUP)
    assert len(batch) == 3000 and s.peer_group_incomplete_batches == 0
    assert s.pgb_whole_gt100 == 3000 and s.pgb_split_gt100 == 0 and s.pgb_max_batch == 3000


def test_late_member_rule_is_unchanged_in_whole_group_mode():
    batch, s = _collect(50, WHOLE_GROUP, late_at=2.0)  # arrives after the 1 s window
    assert len(batch) == 49 and s.peer_group_incomplete_batches == 1
    assert s.pgb_split_11_50 == 49 and s.pgb_whole_11_50 == 0  # bucketed by the group's original size, 50
    batch, s = _collect(50, WHOLE_GROUP, late_at=0.5)  # arrives inside it
    assert len(batch) == 50 and s.peer_group_incomplete_batches == 0 and s.pgb_whole_11_50 == 50
    batch, s = _collect(4, 10)
    assert s.pgb_whole_le10 == 4 and s.pgb_max_batch == 4


def test_members_carry_group_and_served_size():
    batch, _ = _collect(50, WHOLE_GROUP, late_at=2.0)  # 49 served, original group 50
    assert {(t.pgb_group_size, t.pgb_served) for t in batch} == {(50, 49)}
    batch, _ = _collect(4, 10)
    assert {(t.pgb_group_size, t.pgb_served) for t in batch} == {(4, 4)}


def test_task_bsz_sidecar_rows_follow_tasks_npy(tmp_path):
    import numpy as np
    from scripts_cosim.fresh_topo_burst_v1_gate import write_task_bsz, write_task_times

    trs = [{"taskId": 7, "dispatchedTime": 2.0, "doneTime": 5.0, "groupSize": 3, "servedBatchSize": 2},
           {"taskId": 3, "dispatchedTime": 1.0, "doneTime": 4.0},
           {"taskId": -1, "dispatchedTime": 0.5, "doneTime": 0.6}]
    assert write_task_bsz(trs, str(tmp_path / "a.tasks_bsz.npy")) == write_task_times(trs, str(tmp_path / "a.tasks.npy")) == 2
    b, t = np.load(tmp_path / "a.tasks_bsz.npy"), np.load(tmp_path / "a.tasks.npy")
    assert (b[:, 1:3] == t).all() and b[:, 0].tolist() == [3, 7] and b[:, 3:].tolist() == [[-1, -1], [3, 2]]
