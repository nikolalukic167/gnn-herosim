import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import mp_twin_parity as MT  # noqa: E402


def _model(monkeypatch, mp_off):
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    if mp_off:
        monkeypatch.setenv("GNN_DISABLE_MESSAGE_PASSING", "1")
    else:
        monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)
    torch.manual_seed(1)
    m = TaskPlacementGNN(task_feature_dim=7, platform_feature_dim=9, embedding_dim=16, hidden_dim=16, edge_dim=5, task_type_onehot_dim=4,
                         partial_state_edge_dim=27, mp_peer_edges=True, peer_edge_dim=1, mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True)
    return m.eval()


def test_the_mp_off_model_is_blind_to_permuted_neighbours_and_the_mp_model_is_not(monkeypatch):
    off, on = MT.probe(_model(monkeypatch, True)), MT.probe(_model(monkeypatch, False))
    assert off["disable_mp"] and off["task0_max_abs_diff"] == 0.0 and off["platform_max_abs_diff"] == 0.0
    assert not on["disable_mp"] and on["task0_max_abs_diff"] > 0.0 and on["platform_max_abs_diff"] > 0.0
    assert MT.check("off", off, True) and MT.check("on", on, False)


def test_a_twin_served_with_message_passing_on_fails_the_check(monkeypatch):
    # the loader hole: MP-off weights served without the flag run the (never-trained) message passing
    served_on = MT.probe(_model(monkeypatch, False))
    assert not MT.check("twin served MP on", served_on, True)
    served_off = MT.probe(_model(monkeypatch, True))
    assert not MT.check("FL served MP off", served_off, False)


def test_the_permutation_fixes_task_0_and_moves_every_other_task():
    d = type("D", (), {})()
    import torch_geometric.data as tgd
    d = tgd.Data(task_features=torch.arange(MT.N_TASKS).float().unsqueeze(1), task_type_onehot4=torch.eye(4)[torch.arange(MT.N_TASKS) % 4])
    e = MT.permuted(d)
    col = e.task_features.squeeze(1).tolist()
    assert col[0] == 0.0 and all(col[i] != float(i) for i in range(1, MT.N_TASKS)) and sorted(col) == list(map(float, range(MT.N_TASKS)))
