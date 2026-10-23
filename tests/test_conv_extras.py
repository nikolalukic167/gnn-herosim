"""live_link_features_v1: the generic hook for extra bipartite-conv edge columns (gnn_model.conv_edge_extras): mp_link_edge_conv (re-encode per placement) and mp_link_static_attr (static block)."""
import os
import sys

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.executesimulation import checkpoint_mp_config  # noqa: E402
from src.policy.gnn.arm_models import GRAPH_FLAGS  # noqa: E402
from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402

TASK_DIM, PLAT_DIM, EMB, HID = 5, 21, 8, 16


def model(**kw):
    torch.manual_seed(0)
    base = dict(mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True, mp_peer_edges=False)
    base.update(kw)
    return TaskPlacementGNN(task_feature_dim=TASK_DIM, platform_feature_dim=PLAT_DIM, embedding_dim=EMB, hidden_dim=HID, num_layers=2, edge_dim=5, task_type_onehot_dim=4,
                            partial_state_edge_dim=30, **base).eval()


def graph(seed=0, extras=()):
    g = torch.Generator().manual_seed(seed)
    n_t, n_p = 3, 4
    src = [t for t in range(n_t) for _ in range(n_p)]
    dst = [n_t + p for _ in range(n_t) for p in range(n_p)]
    ei, ea = to_undirected(torch.tensor([src, dst]), torch.rand(len(src), 5, generator=g), num_nodes=n_t + n_p)
    d = Data(edge_index=ei, edge_attr=ea, n_tasks=n_t, n_platforms=n_p, task_features=torch.rand(n_t, TASK_DIM, generator=g),
             platform_features=torch.rand(n_p, PLAT_DIM, generator=g), task_type_onehot4=torch.eye(4)[torch.arange(n_t) % 4])
    for name in extras:
        setattr(d, name, torch.rand(ei.size(1), 3, generator=g))
    return d


def test_default_model_has_no_extras_and_its_conv_is_unchanged():
    m = model()
    assert m.conv_edge_extras == [] and m._conv_extras_dim == 0 and not m.mp_link_edge_conv and not m.mp_link_static_attr
    base_in = m.bip_convs[0].message_mlp[0].in_features
    assert model(mp_link_edge_conv=True).bip_convs[0].message_mlp[0].in_features == base_in + 3
    assert model(mp_link_static_attr=True).bip_convs[0].message_mlp[0].in_features == base_in + 3
    both = model(mp_link_edge_conv=True, mp_link_static_attr=True)
    assert both.conv_edge_extras == [("link_edge_conv_attr", 3), ("link_static_attr", 3)] and both.bip_convs[0].message_mlp[0].in_features == base_in + 6


def test_the_extras_are_weight_visible_so_a_strict_load_across_the_boundary_fails():
    plain, flagged = model(), model(mp_link_edge_conv=True)
    with pytest.raises(RuntimeError):
        flagged.load_state_dict(plain.state_dict())
    with pytest.raises(RuntimeError):
        plain.load_state_dict(flagged.state_dict())
    # the two flags widen the conv by the same 3 columns: weights cannot tell them apart, so the sidecar is the only record of which one a checkpoint is
    model(mp_link_static_attr=True).load_state_dict(flagged.state_dict())


def test_encode_reads_the_attribute_row_aligned_and_fails_loud_otherwise():
    m = model(mp_link_edge_conv=True)
    g = graph(extras=("link_edge_conv_attr",))
    with torch.no_grad():
        te, pe = m._encode(g)
        g2 = graph(extras=("link_edge_conv_attr",))
        g2.link_edge_conv_attr = g2.link_edge_conv_attr + 1.0
        te2, pe2 = m._encode(g2)
    assert not torch.equal(pe, pe2), "the conv must read the extra columns"
    for bad in (None, torch.zeros(g.edge_index.size(1) - 1, 3), torch.zeros(g.edge_index.size(1), 2)):
        g3 = graph()
        if bad is not None:
            g3.link_edge_conv_attr = bad
        with pytest.raises(ValueError, match="link_edge_conv_attr"):
            m._encode(g3)


def test_a_model_without_the_flag_ignores_the_attribute_and_is_bit_identical_to_the_old_path():
    plain = model()
    g = graph()
    g_extra = graph(extras=("link_edge_conv_attr", "link_static_attr"))
    with torch.no_grad():
        a = plain._encode(g)
        b = plain._encode(g_extra)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_the_two_flags_are_independent_and_ordered():
    m = model(mp_link_edge_conv=True, mp_link_static_attr=True)
    g = graph(extras=("link_edge_conv_attr", "link_static_attr"))
    with torch.no_grad():
        base = m._encode(g)
        g.link_static_attr = g.link_static_attr + 1.0
        moved = m._encode(g)
    assert not torch.equal(base[1], moved[1])


def test_the_flags_need_the_edge_conv_and_refuse_plan_raw():
    with pytest.raises(ValueError, match="need mp_bipartite_edge_conv"):
        model(mp_bipartite_edge_conv=False, mp_bipartite_edge_attr_zero=False, mp_link_edge_conv=True)
    with pytest.raises(ValueError, match="plan_raw"):
        TaskPlacementGNN(task_feature_dim=TASK_DIM, platform_feature_dim=PLAT_DIM, embedding_dim=EMB, hidden_dim=HID, num_layers=2, edge_dim=5, task_type_onehot_dim=4,
                         partial_state_edge_dim=2, mp_bipartite_edge_conv=True, mp_peer_edges=False, plan_raw=True, mp_link_edge_conv=True)


def test_env_switches_and_the_serving_whitelist_and_graph_free_refusal(monkeypatch, tmp_path):
    import json

    monkeypatch.setenv("GNN_MP_LINK_EDGE_CONV", "1")
    assert model().mp_link_edge_conv is True
    monkeypatch.delenv("GNN_MP_LINK_EDGE_CONV")
    monkeypatch.setenv("GNN_MP_LINK_STATIC_ATTR", "1")
    assert model().mp_link_static_attr is True
    assert "mp_link_edge_conv" in GRAPH_FLAGS and "mp_link_static_attr" in GRAPH_FLAGS
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "executesimulation.py")).read()
    assert src.count("mp_link_edge_conv") >= 2 and src.count("mp_link_static_attr") >= 2
    pt = tmp_path / "m.pt"
    pt.write_bytes(b"")
    (tmp_path / "m.contract.json").write_text(json.dumps({"mp_link_edge_conv": True, "mp_link_static_attr": False, "mp_bipartite_edge_conv": True, "mp_peer_edges": True,
                                                          "mp_platform_edges": True, "disable_message_passing": False, "task_type_onehot_dim": 4, "partial_state_contract": "partial_state_v7"}))
    cfg = checkpoint_mp_config(pt)
    assert cfg["mp_link_edge_conv"] is True and cfg["mp_link_static_attr"] is False
