"""The physics environment a cache is built under is recorded and refused on mismatch (trainer, prefix serving, loader),
and make_warm_corpus rejects, and counts, a batch whose offered candidates sit on one node."""
import json

import pytest

from scripts_cosim.make_warm_corpus import SINGLE_NODE_REASON, batch_peer_components, candidate_nodes
from src.placement.cache_physics import current_physics_env, require_matching_physics_env


def test_current_physics_env_follows_the_environment(monkeypatch):
    for k in ("HEROSIM_TRANSFER_MODEL", "HEROSIM_REPLICA_RELEASE", "HEROSIM_SCALEOUT", "HEROSIM_INFLIGHT_CAPTURE"):
        monkeypatch.delenv(k, raising=False)
    assert current_physics_env() == {"transfer_model": "store_forward", "replica_release": "0", "scaleout": "legacy",
                                     "inflight_capture": "legacy"}
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    monkeypatch.setenv("HEROSIM_REPLICA_RELEASE", "1")
    monkeypatch.setenv("HEROSIM_SCALEOUT", "kpa")
    assert current_physics_env() == {"transfer_model": "pipelined", "replica_release": "1", "scaleout": "kpa", "inflight_capture": "legacy"}


def test_mismatch_is_refused_and_names_every_difference(monkeypatch):
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    monkeypatch.setenv("HEROSIM_REPLICA_RELEASE", "1")
    monkeypatch.setenv("HEROSIM_SCALEOUT", "kpa")
    require_matching_physics_env(current_physics_env(), what="cache x")
    built = {"transfer_model": "store_forward", "replica_release": "1", "scaleout": "legacy"}
    with pytest.raises(ValueError, match="transfer_model.*scaleout"):
        require_matching_physics_env(built, what="cache x")


def _r1(monkeypatch):
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    monkeypatch.setenv("HEROSIM_REPLICA_RELEASE", "1")
    monkeypatch.setenv("HEROSIM_SCALEOUT", "kpa")
    monkeypatch.delenv("HEROSIM_INFLIGHT_CAPTURE", raising=False)


def test_inflight_capture_mismatch_is_refused_both_ways(monkeypatch):
    _r1(monkeypatch)
    legacy = current_physics_env()
    monkeypatch.setenv("HEROSIM_INFLIGHT_CAPTURE", "service_end_v1")
    with pytest.raises(ValueError, match="inflight_capture: built/trained under 'legacy', this run has 'service_end_v1'"):
        require_matching_physics_env(legacy, what="legacy cache served service_end_v1")
    service_end = current_physics_env()
    monkeypatch.delenv("HEROSIM_INFLIGHT_CAPTURE")
    with pytest.raises(ValueError, match="inflight_capture: built/trained under 'service_end_v1', this run has 'legacy'"):
        require_matching_physics_env(service_end, what="service_end cache served legacy")
    require_matching_physics_env(legacy, what="matching")


def test_a_record_without_inflight_capture_means_legacy(monkeypatch):
    _r1(monkeypatch)
    old = {"transfer_model": "pipelined", "replica_release": "1", "scaleout": "kpa"}
    require_matching_physics_env(old, what="r1a smoke cache")
    monkeypatch.setenv("HEROSIM_INFLIGHT_CAPTURE", "service_end_v1")
    with pytest.raises(ValueError, match="inflight_capture"):
        require_matching_physics_env(old, what="r1a smoke cache")


def test_cache_records_the_capture_mode_of_its_snapshots_not_the_build_shell(monkeypatch):
    import pytest as _p
    pgc = _p.importorskip("src.notebooks.prepare_graphs_cache")
    legacy_seed = {"fidelity_replay": {"snapshot": {"replicas_by_type": {"dnn1": [{"node_name": "n0", "platform_id": 0}]}}}}
    service_seed = {"fidelity_replay": {"snapshot": {"replicas_by_type": {"dnn1": [{"node_name": "n0", "platform_id": 0, "current_task_remaining": 0.0}]}}}}
    assert pgc._inflight_capture_of_seed(legacy_seed) == "legacy"
    assert pgc._inflight_capture_of_seed(service_seed) == "service_end_v1"
    assert pgc._inflight_capture_of_seed({}) == "legacy"
    assert pgc._single_inflight_capture({"a": {"inflight_capture": "legacy"}, "b": {}}) == "legacy"
    with pytest.raises(RuntimeError, match="different in-flight modes"):
        pgc._single_inflight_capture({"a": {"inflight_capture": "legacy"}, "b": {"inflight_capture": "service_end_v1"}})


def test_a_cache_without_the_field_is_tolerated_unless_required(monkeypatch):
    monkeypatch.delenv("HEROSIM_TRANSFER_MODEL", raising=False)
    require_matching_physics_env(None, what="old cache")
    with pytest.raises(ValueError, match="records no physics_env"):
        require_matching_physics_env(None, what="old cache", require=True)


def _ckpt(tmp_path, physics_env):
    import torch

    from src.policy.gnn.gnn_model import TaskPlacementGNN

    model = TaskPlacementGNN(task_feature_dim=3, platform_feature_dim=14, embedding_dim=64, hidden_dim=64,
                             num_layers=3, edge_dim=5)
    path = tmp_path / "m.pt"
    torch.save(model.state_dict(), path)
    side = {"inference_feature_layout": "dim22", "queue_feature_contract": "legacy_v0",
            "queue_norm_mode": "scheduler_adaptive", "topology_feature_contract": "src_index_v0"}
    if physics_env is not None:
        side["physics_env"] = physics_env
    path.with_suffix(".contract.json").write_text(json.dumps(side))
    return path


def test_loader_refuses_a_checkpoint_trained_under_other_physics(tmp_path, monkeypatch):
    from src.executesimulation import load_gnn_model

    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    monkeypatch.setenv("HEROSIM_REPLICA_RELEASE", "1")
    monkeypatch.setenv("HEROSIM_SCALEOUT", "kpa")
    bad = _ckpt(tmp_path, {"transfer_model": "store_forward", "replica_release": "1", "scaleout": "kpa"})
    with pytest.raises(ValueError, match="physics environment mismatch"):
        load_gnn_model(bad)
    good = _ckpt(tmp_path, current_physics_env())
    model, _ = load_gnn_model(good)
    assert model is not None


def test_prefix_loader_refuses_a_checkpoint_trained_under_other_physics(tmp_path, monkeypatch):
    from src.policy.gnn.prefix_serving import PrefixServingError, load_prefix_conditioned_gnn

    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    path = _ckpt(tmp_path, {"transfer_model": "store_forward", "replica_release": "0", "scaleout": "legacy"})
    side = json.loads(path.with_suffix(".contract.json").read_text())
    side["partial_state_edge_features"] = True
    path.with_suffix(".contract.json").write_text(json.dumps(side))
    with pytest.raises(PrefixServingError, match="physics environment mismatch"):
        load_prefix_conditioned_gnn(path)


def _snap(rows):
    """rows: [(task_type, [queue_key, ...])]"""
    return {"tasks": [{"task_type": t, "candidates": [{"queue_key": k} for k in keys]} for t, keys in rows]}


def test_candidate_nodes_counts_only_what_the_sweep_offers():
    snap = _snap([("rf", ["node0:1", "node1:2"]), ("cnn", ["node0:3", "node4:5"])])
    assert candidate_nodes(snap, {"rf": {"node0:1"}, "cnn": {"node0:3"}}) == {"node0"}
    assert candidate_nodes(snap, {"rf": {"node0:1", "node1:2"}, "cnn": {"node0:3"}}) == {"node0", "node1"}
    assert SINGLE_NODE_REASON == "single_candidate_node"


def test_trainer_refuses_a_cache_built_under_other_physics(tmp_path):
    """train_near_rtt reads metadata.json first and stops before touching any graph."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("torch_geometric")
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"version": "5.7", "physics_env": {"transfer_model": "store_forward", "replica_release": "0", "scaleout": "legacy"}}))
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=f"{repo}:{repo}/src/notebooks", WANDB_MODE="disabled",
               HEROSIM_TRANSFER_MODEL="pipelined", HEROSIM_REPLICA_RELEASE="1", HEROSIM_SCALEOUT="kpa")
    proc = subprocess.run([sys.executable, str(repo / "src/notebooks/train_near_rtt.py"), "--cache-dir", str(tmp_path)],
                          capture_output=True, text=True, env=env, timeout=240, cwd=repo)
    assert proc.returncode != 0
    assert "physics environment mismatch" in proc.stderr


def test_batch_peer_components():
    ev = [{}] * 4
    assert batch_peer_components({"events": ev, "peer_exchange": [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0]]}) == 1
    assert batch_peer_components({"events": ev, "peer_exchange": [[0, 1, 1.0], [2, 3, 1.0]]}) == 2
    assert batch_peer_components({"events": ev}) == 4
