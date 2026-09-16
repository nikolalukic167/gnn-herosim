"""peer_only_v1: the `peeronly` arm = PeerConv on, bipartite GIN off (mp_platform_edges=False).

Weight-invisible, so the guarantees are behavioural: the sidecar key survives the serving
whitelist, the flag's default is the old behaviour, and on a real peer-corpus graph the
peeronly encoding is exactly "the MP-OFF encoding with PeerConv applied to the task block".
"""
import json
import pickle
from pathlib import Path

import pytest
import torch

from src.executesimulation import checkpoint_mp_config
from src.policy.gnn.gnn_model import TaskPlacementGNN

SMOKE = Path("simulation_data/graphs_cache_peer_affinity_v1_r2_smoke")


def test_sidecar_key_survives_the_serving_whitelist(tmp_path):
    pt = tmp_path / "m.pt"
    pt.write_bytes(b"")
    (tmp_path / "m.contract.json").write_text(json.dumps(
        {"mp_peer_edges": True, "mp_platform_edges": False, "disable_message_passing": False,
         "task_type_onehot_dim": 4, "partial_state_contract": "partial_state_v3"}))
    cfg = checkpoint_mp_config(pt)
    assert cfg["mp_platform_edges"] is False and cfg["mp_peer_edges"] is True


def _model(monkeypatch, *, platform_edges, disable_mp=False, task_dim=8, plat_dim=14, peer_dim=1, seed=0):
    monkeypatch.setenv("GNN_DISABLE_MESSAGE_PASSING", "1" if disable_mp else "0")
    monkeypatch.delenv("GNN_MP_PLATFORM_EDGES_OFF", raising=False)
    torch.manual_seed(seed)
    m = TaskPlacementGNN(task_feature_dim=task_dim, platform_feature_dim=plat_dim, embedding_dim=16,
                         hidden_dim=32, num_layers=2, edge_dim=5, task_type_onehot_dim=4,
                         mp_peer_edges=True, peer_edge_dim=peer_dim, mp_platform_edges=platform_edges)
    return m.eval()


def test_flag_default_is_the_old_behaviour_and_env_off_flips_it(monkeypatch):
    m = _model(monkeypatch, platform_edges=None)
    assert m.mp_platform_edges is True
    monkeypatch.setenv("GNN_MP_PLATFORM_EDGES_OFF", "1")
    torch.manual_seed(0)
    m2 = TaskPlacementGNN(task_feature_dim=8, platform_feature_dim=14, task_type_onehot_dim=4,
                          mp_peer_edges=True, mp_platform_edges=None)
    assert m2.mp_platform_edges is False
    assert _model(monkeypatch, platform_edges=False).mp_platform_edges is False


@pytest.mark.skipif(not SMOKE.is_dir(), reason=f"cache not present at {SMOKE}")
def test_peeronly_encoding_is_mpoff_plus_peerconv_on_a_real_graph(monkeypatch):
    g = pickle.load(open(SMOKE / "graphs.pkl", "rb"))[0]
    task_dim = int(g.task_features.shape[1])
    plat_dim = int(g.platform_features.shape[1])
    peer_dim = int(g.peer_edge_attr.shape[1])
    gnn = _model(monkeypatch, platform_edges=True, task_dim=task_dim, plat_dim=plat_dim, peer_dim=peer_dim)
    peeronly = _model(monkeypatch, platform_edges=False, task_dim=task_dim, plat_dim=plat_dim, peer_dim=peer_dim)
    mpoff = _model(monkeypatch, platform_edges=True, disable_mp=True, task_dim=task_dim, plat_dim=plat_dim, peer_dim=peer_dim)
    # identical weights everywhere: the arms differ only in which message passing RUNS
    peeronly.load_state_dict(gnn.state_dict(), strict=True)
    mpoff.load_state_dict(gnn.state_dict(), strict=True)
    with torch.no_grad():
        t_gnn, p_gnn = gnn._encode(g)
        t_po, p_po = peeronly._encode(g)
        t_off, p_off = mpoff._encode(g)
        # platform side: peeronly == mpoff (the encoder, untouched), != gnn (smoothed by the GIN)
        assert torch.allclose(p_po, p_off)
        assert not torch.allclose(p_po, p_gnn)
        # task side: peeronly == PeerConv(mpoff's task embeddings) exactly, != gnn
        expected = peeronly.peer_conv(t_off, g.peer_edge_index, g.peer_edge_attr)
        assert torch.allclose(t_po, expected)
        assert not torch.allclose(t_po, t_gnn)
        assert not torch.allclose(t_po, t_off)      # PeerConv actually did something
