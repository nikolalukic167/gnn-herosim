"""bipartite_edge_v1: the edge-conditioned bipartite conv and its zeroed-attribute control.

`peer_only_v1` closed with the bipartite penalty CONFOUNDED: `PeerConv` is residual AND
edge-aware, the bipartite `GIN` is neither. C4 tested the residual half and it did not
transfer. These two arms test the other half, and the whole point of the pair is that the
ONLY difference between them is whether the conv can see `data.edge_attr`.

What is pinned, in the order it can break:

1. `mp_bipartite_edge_conv` is weight-VISIBLE, so a strict load across the boundary fails
   by itself, in both directions.
2. `mp_bipartite_edge_attr_zero` is NOT. The two arms are byte-compatible checkpoints, so
   the sidecar is the only thing that distinguishes them — it must survive the serving
   whitelist (`checkpoint_mp_config` silently drops keys it does not list) AND reach the
   constructor on the prefix-conditioned serving path the gates actually run.
3. The control is EXACTLY the treatment with zeroed attributes: on a graph whose edge_attr
   is already all zeros the two produce bit-identical embeddings, and on a real graph they
   do not. That equality is what makes "GIN -> EdgeConv" a controlled change rather than a
   confounded one.
4. Every combination the conv cannot honour fails loud rather than padding a zero: extra
   message-passing edges (which carry no attribute), an absent attribute, a misaligned one,
   and the two nonsense flag combinations.

Run: pipenv run python3 -m pytest tests/test_bipartite_edge_arm.py -q
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.executesimulation import checkpoint_mp_config          # noqa: E402
from src.policy.gnn.gnn_model import TaskPlacementGNN           # noqa: E402

SMOKE = REPO_ROOT / "simulation_data/graphs_cache_peer_affinity_v1_r2_smoke"

TASK_DIM, PLAT_DIM, EMB, HID, LAYERS = 7, 22, 8, 16, 3


def _model(*, seed=0, **kw):
    torch.manual_seed(seed)
    m = TaskPlacementGNN(
        task_feature_dim=TASK_DIM, platform_feature_dim=PLAT_DIM,
        embedding_dim=EMB, hidden_dim=HID, num_layers=LAYERS, edge_dim=5, **kw,
    )
    return m.eval()


def _graph(n_tasks=3, n_platforms=4, seed=0, zero_attr=False):
    g = torch.Generator().manual_seed(seed)
    src, dst, attrs = [], [], []
    for t in range(n_tasks):
        for p in range(n_platforms):
            src.append(t)
            dst.append(n_tasks + p)
            attrs.append(torch.rand(5, generator=g).tolist())
    ei = torch.tensor([src, dst], dtype=torch.long)
    ea = torch.zeros(len(attrs), 5) if zero_attr else torch.tensor(attrs, dtype=torch.float32)
    ei, ea = to_undirected(ei, ea, num_nodes=n_tasks + n_platforms)
    data = Data(
        edge_index=ei, n_tasks=n_tasks, n_platforms=n_platforms,
        task_features=torch.rand(n_tasks, TASK_DIM, generator=g),
        platform_features=torch.rand(n_platforms, PLAT_DIM, generator=g),
    )
    data.edge_attr = ea
    return data


# 1. weight visibility -----------------------------------------------------------------

def test_the_conv_is_weight_visible_so_a_strict_load_refuses_both_directions():
    gin = _model()
    edge = _model(mp_bipartite_edge_conv=True)
    sd_gin, sd_edge = gin.state_dict(), edge.state_dict()

    assert not any(k.startswith("bip_convs.") for k in sd_gin)
    assert sum(k.startswith("bip_convs.") for k in sd_edge) > 0
    # The GIN module is still CONSTRUCTED in the edge arm, so `gin.convs.*` keys remain in
    # the checkpoint. prefix_serving infers num_layers from exactly those keys; if this
    # ever stops being true, that inference silently returns 0 and refuses every load.
    assert sum(k.startswith("gin.convs.") for k in sd_edge) > 0

    with pytest.raises(RuntimeError):
        gin.load_state_dict(sd_edge)
    with pytest.raises(RuntimeError):
        edge.load_state_dict(sd_gin)


def test_the_zero_control_is_weight_invisible_so_the_sidecar_is_its_only_record():
    treat = _model(mp_bipartite_edge_conv=True)
    ctrl = _model(mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True)
    # Byte-compatible in both directions: identical keys, identical shapes. That is the
    # whole hazard — nothing about the weights says which arm a checkpoint came from.
    assert set(treat.state_dict()) == set(ctrl.state_dict())
    ctrl.load_state_dict(treat.state_dict(), strict=True)
    assert ctrl.mp_bipartite_edge_attr_zero is True


# 2. the sidecar keys reach serving ----------------------------------------------------

def test_both_sidecar_keys_survive_the_serving_whitelist(tmp_path):
    pt = tmp_path / "m.pt"
    pt.write_bytes(b"")
    (tmp_path / "m.contract.json").write_text(json.dumps({
        "mp_peer_edges": True, "mp_platform_edges": True,
        "mp_bipartite_edge_conv": True, "mp_bipartite_edge_attr_zero": True,
        "disable_message_passing": False, "task_type_onehot_dim": 4,
        "partial_state_contract": "partial_state_v3",
    }))
    cfg = checkpoint_mp_config(pt)
    assert cfg["mp_bipartite_edge_conv"] is True
    assert cfg["mp_bipartite_edge_attr_zero"] is True


def test_the_control_flag_reaches_the_model_on_the_prefix_serving_path(tmp_path, monkeypatch):
    """The gates decode via masked_topo, which loads through prefix_serving — NOT through
    load_gnn_model's own constructor. A key wired into one and not the other serves the
    control as the treatment, and the two arms become the same measurement."""
    from src.policy.gnn.prefix_serving import load_prefix_conditioned_gnn

    snapshot = dict(os.environ)
    try:
        for name in ("GNN_DECODE_MODE", "PARTIAL_STATE_CONTRACT", "PARTIAL_STATE_PEER_MASS",
                     "GNN_DISABLE_MESSAGE_PASSING", "GNN_MP_PLATFORM_EDGES_OFF",
                     "GNN_PREFIX_ALPHA_KEY"):
            monkeypatch.delenv(name, raising=False)

        trained = _model(mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True,
                         task_type_onehot_dim=4, mp_peer_edges=True, partial_state_edge_dim=22)
        ckpt = tmp_path / "gnnedge0-seed1.pt"
        torch.save(trained.state_dict(), ckpt)
        from src.notebooks.prepare_graphs_cache import DAG_TASK_TYPE_VOCAB
        (tmp_path / "gnnedge0-seed1.contract.json").write_text(json.dumps({
            "mp_peer_edges": True, "mp_platform_edges": True,
            "mp_bipartite_edge_conv": True, "mp_bipartite_edge_attr_zero": True,
            "disable_message_passing": False, "partial_state_edge_features": True,
            "task_type_onehot_dim": 4, "dag_task_type_vocab": list(DAG_TASK_TYPE_VOCAB),
            "partial_state_contract": "partial_state_v3", "partial_state_feature_dim": 22,
            "peer_mass": True, "dag_alpha_key": "2.5",
            "decode_replica_reuse": True, "decode_relax_on_stuck": True,
        }))

        model, _opts, sidecar = load_prefix_conditioned_gnn(ckpt, device=None, adopt_env=True)
        assert sidecar["mp_bipartite_edge_attr_zero"] is True
        assert model.mp_bipartite_edge_conv is True
        assert model.mp_bipartite_edge_attr_zero is True
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


# 3. the control is exactly the treatment with zeroed attributes ------------------------

def test_the_control_equals_the_treatment_iff_the_attributes_are_already_zero():
    treat = _model(mp_bipartite_edge_conv=True)
    ctrl = _model(mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True)
    ctrl.load_state_dict(treat.state_dict(), strict=True)   # identical weights

    real = _graph()
    zeros = _graph(zero_attr=True)
    with torch.no_grad():
        t_real, p_real = treat._encode(real)
        c_real, cp_real = ctrl._encode(real)
        t_zero, p_zero = treat._encode(zeros)
        c_zero, cp_zero = ctrl._encode(zeros)

    # On an already-zero graph the flag cannot do anything: bit-identical, not merely close.
    assert torch.equal(t_zero, c_zero) and torch.equal(p_zero, cp_zero)
    # And the control reproduces exactly what the treatment does on the zeroed graph, which
    # is the definition the arm is registered under.
    assert torch.equal(c_real, t_zero) and torch.equal(cp_real, p_zero)
    # On a real graph the attributes change the embedding, or there is nothing to measure.
    assert not torch.equal(t_real, c_real)
    assert not torch.equal(p_real, cp_real)


def test_the_edge_conv_actually_changes_the_platform_block_against_the_gin():
    data = _graph()
    gin = _model()
    edge = _model(mp_bipartite_edge_conv=True)
    with torch.no_grad():
        _t_gin, p_gin = gin._encode(data)
        _t_edge, p_edge = edge._encode(data)
    assert p_gin.shape == p_edge.shape
    assert not torch.allclose(p_gin, p_edge)


@pytest.mark.skipif(not SMOKE.is_dir(), reason=f"cache not present at {SMOKE}")
def test_on_a_real_cached_graph_the_two_arms_separate():
    graph = pickle.load(open(SMOKE / "graphs.pkl", "rb"))[0]
    torch.manual_seed(0)
    kw = dict(task_feature_dim=int(graph.task_features.shape[1]),
              platform_feature_dim=int(graph.platform_features.shape[1]),
              embedding_dim=16, hidden_dim=32, num_layers=2, edge_dim=5)
    treat = TaskPlacementGNN(mp_bipartite_edge_conv=True, **kw).eval()
    torch.manual_seed(0)
    ctrl = TaskPlacementGNN(mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True, **kw).eval()
    ctrl.load_state_dict(treat.state_dict(), strict=True)
    assert int(graph.edge_attr.shape[1]) == 5, "the conv is conditioned on the 5-column attr"
    with torch.no_grad():
        _t, p_treat = treat._encode(graph)
        _c, p_ctrl = ctrl._encode(graph)
    assert not torch.allclose(p_treat, p_ctrl)


# 4. everything the conv cannot honour fails loud ---------------------------------------

def test_refuses_the_two_nonsense_flag_combinations():
    with pytest.raises(ValueError, match="meaningless without"):
        _model(mp_bipartite_edge_attr_zero=True)
    with pytest.raises(ValueError, match="mp_platform_edges=False"):
        _model(mp_bipartite_edge_conv=True, mp_platform_edges=False)


def test_refuses_extra_message_passing_edges_rather_than_padding_a_zero_attribute():
    """node/dag/net edges carry no edge_attr. Padding them with zeros would make "this
    edge has no attribute" indistinguishable from the zeroed-attribute CONTROL, which is
    the one distinction this lineage exists to make."""
    data = _graph()
    # Same-node platform<->platform edges: the cheapest extra-edge option to construct,
    # and it stands in for dag/net edges, which reach the identical code path.
    data.node_edge_index = torch.tensor([[3, 4], [4, 3]], dtype=torch.long)
    model = _model(mp_bipartite_edge_conv=True, mp_node_edges=True,
                   mp_node_edges_candidates_only=False)
    with pytest.raises(ValueError, match="no edge_attr"):
        model._encode(data)


def test_refuses_an_absent_or_misaligned_edge_attr():
    model = _model(mp_bipartite_edge_conv=True)

    missing = _graph()
    missing.edge_attr = torch.empty((0, 5))
    with pytest.raises(ValueError, match="carries no edge_attr"):
        model._encode(missing)

    misaligned = _graph()
    misaligned.edge_attr = misaligned.edge_attr[:-1]
    with pytest.raises(ValueError, match="aligned row-for-row"):
        model._encode(misaligned)


def test_the_flags_default_off_and_the_env_switches_flip_them(monkeypatch):
    monkeypatch.delenv("GNN_MP_BIPARTITE_EDGE_CONV", raising=False)
    monkeypatch.delenv("GNN_MP_BIPARTITE_EDGE_ATTR_ZERO", raising=False)
    plain = _model()
    assert plain.mp_bipartite_edge_conv is False and plain.mp_bipartite_edge_attr_zero is False

    monkeypatch.setenv("GNN_MP_BIPARTITE_EDGE_CONV", "1")
    monkeypatch.setenv("GNN_MP_BIPARTITE_EDGE_ATTR_ZERO", "1")
    from_env = _model()
    assert from_env.mp_bipartite_edge_conv is True and from_env.mp_bipartite_edge_attr_zero is True
    # An explicit argument still wins over the environment, as it does for every other flag.
    assert _model(mp_bipartite_edge_conv=True,
                  mp_bipartite_edge_attr_zero=False).mp_bipartite_edge_attr_zero is False


# 5. bipartite_aggr_v1: sum vs mean over a task's candidate platforms ---------------------

def test_the_aggregation_flag_is_weight_invisible_and_changes_the_embedding():
    """`aggr` changes no parameter, so a sum checkpoint and a mean checkpoint are byte-
    compatible and load into each other in silence -- the sidecar is the only record. But it
    must actually change the output, or the arm is a no-op that would read as a clean null."""
    mean = _model(mp_bipartite_edge_conv=True)
    summ = _model(mp_bipartite_edge_conv=True, mp_bipartite_aggr="sum")
    assert mean.mp_bipartite_aggr == "mean" and summ.mp_bipartite_aggr == "sum"
    assert all(c.aggr_kind == "sum" for c in summ.bip_convs)
    # byte-compatible in both directions
    assert set(mean.state_dict()) == set(summ.state_dict())
    summ.load_state_dict(mean.state_dict(), strict=True)

    data = _graph()
    with torch.no_grad():
        _t_m, p_mean = mean._encode(data)
        _t_s, p_sum = summ._encode(data)
    assert not torch.allclose(p_mean, p_sum), "sum and mean must differ, or the arm is a no-op"


def test_sum_scales_with_candidate_count_and_mean_does_not():
    """The hypothesis in one assertion. A task's candidate set grows 3.55 -> 47.92 platforms
    from 6 to 80 servers (cluster_scale_v1 S0.d); a sum over it scales with that count and a
    mean does not. Measured on the conv's own aggregation, holding weights fixed."""
    mean = _model(mp_bipartite_edge_conv=True)
    summ = _model(mp_bipartite_edge_conv=True, mp_bipartite_aggr="sum")
    summ.load_state_dict(mean.state_dict(), strict=True)

    small, large = _graph(n_tasks=2, n_platforms=3), _graph(n_tasks=2, n_platforms=24)
    with torch.no_grad():
        t_small_m, _ = mean._encode(small)
        t_large_m, _ = mean._encode(large)
        t_small_s, _ = summ._encode(small)
        t_large_s, _ = summ._encode(large)
    # The task block is what aggregates over candidate platforms.
    grow_sum = float(t_large_s.abs().mean() / t_small_s.abs().mean())
    grow_mean = float(t_large_m.abs().mean() / t_small_m.abs().mean())
    assert grow_sum > grow_mean, (
        f"sum should grow faster with candidate count than mean; got sum x{grow_sum:.2f} "
        f"vs mean x{grow_mean:.2f}"
    )


def test_the_aggregation_flag_survives_the_whitelist_as_a_string_not_a_bool(tmp_path):
    """bool('sum') and bool('mean') are both True. Read in the bool block this key would
    serve every arm as the same one while looking correctly whitelisted."""
    pt = tmp_path / "m.pt"
    pt.write_bytes(b"")
    (tmp_path / "m.contract.json").write_text(json.dumps({
        "mp_bipartite_edge_conv": True, "mp_bipartite_aggr": "sum",
        "partial_state_contract": "partial_state_v3",
    }))
    assert checkpoint_mp_config(pt)["mp_bipartite_aggr"] == "sum"

    (tmp_path / "m.contract.json").write_text(json.dumps({
        "mp_bipartite_edge_conv": True, "mp_bipartite_aggr": "summ",
        "partial_state_contract": "partial_state_v3",
    }))
    with pytest.raises(ValueError, match="neither 'mean' nor 'sum'"):
        checkpoint_mp_config(pt)


def test_refuses_a_bad_aggregation_and_one_set_without_the_conv():
    with pytest.raises(ValueError, match="must be 'mean' or 'sum'"):
        _model(mp_bipartite_edge_conv=True, mp_bipartite_aggr="max")
    with pytest.raises(ValueError, match="meaningless"):
        _model(mp_bipartite_aggr="sum")
