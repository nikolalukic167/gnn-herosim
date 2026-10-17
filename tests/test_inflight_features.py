"""partial_state_v6: the in-flight columns (placement/inflight_features.py)."""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))

from src.placement import inflight_features as IF  # noqa: E402
from src.policy.tabular import reduced_features as RF  # noqa: E402

OUT_BYTES = 10 * 1024 * 1024  # output estimate: 10 MiB / (100 MiB/s) + 1 ms = 0.101 s
OUT_S = OUT_BYTES / (100.0 * 1024.0 * 1024.0) + 0.001
TYPE = {"name": "t", "executionTime": {"cpu": 2.0}, "stateSize": {"app": {"input": 100.0 * 1024.0 * 1024.0, "output": OUT_BYTES}}}
TYPES = {"t": TYPE}


def task(start=None, io_end=None):
    return SimpleNamespace(type=TYPE, application=SimpleNamespace(type={"name": "app"}), compute_start_time=start, io_end_time=io_end)


def platform(inflight=(), queue=0, admitted=False, now=100.0, current=None):
    p = SimpleNamespace(env=SimpleNamespace(now=now), type={"shortName": "cpu"}, inflight=list(inflight), admitted=object() if admitted else None, current_task=current)
    p.queue_length = lambda: queue
    return p


@pytest.fixture
def release(monkeypatch):
    import src.placement.infrastructure as infra

    monkeypatch.setattr(infra, "REPLICA_RELEASE", True)


def test_contract_family_and_defaults_unchanged(monkeypatch):
    v6 = RF.PARTIAL_STATE_CONTRACT_V6
    assert v6 == "partial_state_v6" and RF.validate_partial_state_contract(v6) == v6
    for group in (RF.PEER_BLOCK_CONTRACTS, RF.LOAD_SECONDS_CONTRACTS, RF.BOTH_DIRECTION_CONTRACTS, RF.FOUR_TYPE_CONTRACTS):
        assert v6 in group and RF.PARTIAL_STATE_CONTRACT_V5 in group
    assert RF.INFLIGHT_CONTRACTS == frozenset({v6})
    assert RF.partial_state_feature_dim(v6) == RF.partial_state_feature_dim(RF.PARTIAL_STATE_CONTRACT_V5) == 27
    assert RF.DEFAULT_PARTIAL_STATE_CONTRACT == RF.PARTIAL_STATE_CONTRACT_V1
    monkeypatch.delenv("PARTIAL_STATE_CONTRACT", raising=False)
    assert not IF.inflight_enabled()
    for c in ("partial_state_v1", "partial_state_v3", "partial_state_v4", "partial_state_v5"):
        assert not IF.inflight_enabled(c)
    assert IF.inflight_enabled(v6)
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", v6)
    assert IF.inflight_enabled()


def test_a_v5_checkpoint_is_refused_under_v6_and_the_reverse():
    with pytest.raises(RF.PartialStateContractMismatchError):
        RF.require_matching_partial_state_contract("partial_state_v5", "partial_state_v6", model_label="m")
    with pytest.raises(RF.PartialStateContractMismatchError):
        RF.require_matching_partial_state_contract("partial_state_v6", "partial_state_v5", model_label="m")
    with pytest.raises(RF.PartialStateContractMismatchError):
        RF.require_matching_partial_state_contract(None, "partial_state_v6", model_label="m")
    RF.require_matching_partial_state_contract("partial_state_v6", "partial_state_v6", model_label="m")


def test_an_empty_queue_with_three_lock_waiters_is_not_zero(release):
    idle = platform()
    waiters = platform(inflight=[task(io_end=90.0), task(io_end=95.0), task(io_end=99.0)], queue=0)
    assert IF.live_state(idle) == (0, 0.0)
    count, work = IF.live_state(waiters)
    assert count == 3 and work == pytest.approx(3 * (2.0 + OUT_S)) and work > 0.0
    cols = IF.columns(count, work, 1.0)
    assert cols[0] == 3.0 and cols[1] == pytest.approx(math.log1p(work)) and cols[1] > 0.0
    assert not np.allclose(IF.columns(*IF.live_state(idle), 1.0), cols)


def test_the_stages_of_a_task_and_the_count(release):
    running = task(start=99.0, io_end=95.0)          # 99 + 2 - 100 = 1 s of execution left, plus the output
    in_output = task(start=90.0, io_end=80.0)        # execution over: owes the whole output estimate
    waiting = task(io_end=96.0)                      # execution + output
    in_input = task(io_end=None)                     # execution + _approx_comm (input and output at the storage rate, plus the two 1 ms latencies)
    p = platform(inflight=[running, in_output, waiting, in_input], queue=4, admitted=True, current=running)
    count, work = IF.live_state(p)
    approx_comm = (100.0 / 100.0 + 0.001) + (OUT_S - 0.001 + 0.001)
    assert count == 4 + 1 + 4 and work == pytest.approx((1.0 + OUT_S) + OUT_S + (2.0 + OUT_S) + (2.0 + approx_comm))


def test_a_task_in_cold_start_counts_but_owes_no_work_yet(release):
    p = platform(queue=0, admitted=True)
    assert IF.live_state(p) == (1, 0.0)


def test_live_capture_requires_the_release_physics(monkeypatch):
    import src.placement.infrastructure as infra

    monkeypatch.setattr(infra, "REPLICA_RELEASE", False)
    with pytest.raises(RuntimeError, match="REPLICA_RELEASE"):
        IF.live_state(platform())


def ghost(stage, **kw):
    g = {"q": "n:1", "tid": 1, "fn": "t", "src": "c", "exec": 2.0, "output": OUT_S, "input_bytes": 0.0, "pop": -1.0, "stage": stage}
    g.update(kw)
    return g


def test_ghost_records_give_the_same_numbers_as_the_live_stages(release):
    # one scenario, two representations: now = 100, queue 4, one task admitted (cold)
    running = task(start=99.0, io_end=95.0)
    in_output = task(start=90.0, io_end=80.0)
    waiting = task(io_end=96.0)
    in_input = task(io_end=None)
    live = IF.live_state(platform(inflight=[running, in_output, waiting, in_input], queue=4, admitted=True, current=running))
    ghosts = [ghost("compute", compute_remaining=1.0), ghost("compute", compute_remaining=0.0), ghost("lock_wait", order=-4.0),
              ghost("input_io", io_remaining=3.0), ghost("cold", cold_remaining=0.5)]
    cap = IF.state_from_ghosts(ghosts, 4, TYPES, "cpu")
    assert cap[0] == live[0]
    assert cap[1] == pytest.approx(live[1], abs=1e-12)
    # a rendezvous task is priced like an input-stage one
    assert IF.state_from_ghosts([ghost("rendezvous", io_remaining=0.0)], 0, TYPES, "cpu") == IF.state_from_ghosts([ghost("input_io", io_remaining=9.0)], 0, TYPES, "cpu")


def test_the_stored_input_progress_is_not_used(release):
    a = IF.state_from_ghosts([ghost("input_io", io_remaining=0.1)], 0, TYPES, "cpu")
    b = IF.state_from_ghosts([ghost("input_io", io_remaining=15.0)], 0, TYPES, "cpu")
    assert a == b


def test_ghosts_fail_loud_on_an_unknown_stage_or_an_execution_that_is_not_the_table_value():
    with pytest.raises(ValueError, match="unknown in-flight stage"):
        IF.state_from_ghosts([ghost("teleport")], 0, TYPES, "cpu")
    with pytest.raises(ValueError, match="execution physics"):
        IF.state_from_ghosts([ghost("lock_wait", exec=2.5)], 0, TYPES, "cpu")


def test_columns_scale_the_count_like_the_queue_column():
    assert IF.columns(8, 0.0, 4.0).tolist() == [2.0, 0.0]
    assert IF.columns(5, math.e - 1.0, 1.0)[1] == pytest.approx(1.0)
    assert IF.columns(0, -3.0, 1.0)[1] == 0.0


def test_widen_graph_appends_and_removes_the_held_columns():
    torch = pytest.importorskip("torch")
    g = SimpleNamespace(platform_features=torch.zeros(3, 18), inflight_platform_extra=torch.ones(3, 2))
    IF.widen_graph(g)
    assert g.platform_features.shape == (3, 20) and not hasattr(g, "inflight_platform_extra") and g.platform_features[:, 18:].sum() == 6
    with pytest.raises(ValueError, match="no in-flight columns"):
        IF.widen_graph(SimpleNamespace(platform_features=torch.zeros(1, 2)))


# ---- the loader enforces the contract, and the platform width is the v5 width + 2 -------------------------------------------

def _checkpoint(tmp_path, contract, platform_dim, task_dim=5):
    import json

    import torch

    from src.policy.gnn.gnn_model import TaskPlacementGNN

    model = TaskPlacementGNN(task_feature_dim=task_dim, platform_feature_dim=platform_dim, embedding_dim=64, hidden_dim=64, num_layers=3, edge_dim=5)
    path = tmp_path / f"m_{contract}.pt"
    torch.save(model.state_dict(), path)
    path.with_suffix(".contract.json").write_text(json.dumps({
        "partial_state_contract": contract, "inference_feature_layout": "dim22",
        "queue_feature_contract": "legacy_v0", "queue_norm_mode": "scheduler_adaptive", "topology_feature_contract": "src_index_v0"}))
    return path


def test_loader_serves_a_v6_checkpoint_only_under_a_v6_run(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from src.executesimulation import load_gnn_model

    path = _checkpoint(tmp_path, "partial_state_v6", 16 + 2)      # 14 legacy + has_rf, has_cnn + the two in-flight columns
    monkeypatch.delenv("PARTIAL_STATE_CONTRACT", raising=False)
    with pytest.raises(Exception, match="partial_state_v6"):
        load_gnn_model(path)
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", "partial_state_v5")
    with pytest.raises(Exception, match="partial_state_v6"):
        load_gnn_model(path)
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", "partial_state_v6")
    model, _device = load_gnn_model(path)
    assert model.platform_encoder.net[0].in_features == 18 and model.task_encoder.net[0].in_features == 5


def test_loader_refuses_v5_and_legacy_checkpoints_under_a_v6_run(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from src.executesimulation import load_gnn_model

    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", "partial_state_v6")
    with pytest.raises(Exception, match="partial_state_v5"):
        load_gnn_model(_checkpoint(tmp_path, "partial_state_v5", 16))
    with pytest.raises(ValueError, match="four-type feature blocks"):
        load_gnn_model(_checkpoint(tmp_path, "partial_state_v3", 14, task_dim=3))
