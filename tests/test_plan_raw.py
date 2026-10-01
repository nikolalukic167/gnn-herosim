"""raw_plan_v1 (src/policy/gnn/plan_raw.py): the plan as raw graph facts.

Pins what the arms depend on:
- the block marks the committed edge in both directions and the committed count per platform;
- with message passing, a partner's placement reaches another task's logits, including through a
  same-node platform the task can use;
- the MP-OFF twin sees only its own edges: a partner's placement on another platform leaves its
  logits unchanged, and a commitment onto one of its own candidates changes them (the count column);
- the shared scorer dispatches to the raw plan, and the constructor refuses a mis-sized block.
"""
from __future__ import annotations

import os

import pytest
import torch
from torch_geometric.data import Data

from src.policy.gnn.gnn_model import TaskPlacementGNN
from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
from src.policy.gnn.plan_raw import PLAN_RAW_DIM, plan_raw_edge_attr

ENVS = ("GNN_DISABLE_MESSAGE_PASSING", "GNN_PLAN_RAW", "GNN_PLAN_RAW_SUM", "GNN_MP_NODE_EDGES")

# platform positions 0,1 on node 0; 2,3 on node 1. (node_id, platform_id) placements.
PLACEMENT = {0: (0, 10), 1: (0, 11), 2: (1, 20), 3: (1, 21)}
CANDS = {0: [0, 2], 1: [1, 3], 2: [0, 3]}


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.get(k) for k in ENVS}
    for k in ENVS:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _graph() -> Data:
    torch.manual_seed(0)
    n_tasks, n_platforms = 3, 4
    fwd = [(t, n_tasks + p) for t in range(n_tasks) for p in CANDS[t]]
    rows = fwd + [(d, s) for s, d in fwd]
    data = Data()
    data.n_tasks, data.n_platforms = n_tasks, n_platforms
    data.task_features = torch.randn(n_tasks, 3)
    data.task_type_onehot4 = torch.eye(4)[:n_tasks]
    data.platform_features = torch.randn(n_platforms, 14)
    data.edge_index = torch.tensor(rows, dtype=torch.long).t().contiguous()
    fwd_attr = torch.randn(len(fwd), 5)
    data.edge_attr = torch.cat([fwd_attr, fwd_attr])
    data.node_edge_index = torch.tensor([[3, 4, 5, 6], [4, 3, 6, 5]])
    data.peer_edge_index = torch.tensor([[0, 1], [1, 0]])
    data.peer_edge_attr = torch.ones(2, 1)
    keys = {}
    meta = {}
    for t in range(n_tasks):
        keys[t] = []
        for p in CANDS[t]:
            k = f"k{t}_{p}"
            keys[t].append(k)
            meta[k] = {"platform_pos": p}
    data.task_logit_to_queue_key = keys
    data.queue_key_to_platform_meta = meta
    # keyed by task index, as the cache and the live builder store it
    data.task_logit_to_placement = {t: [PLACEMENT[p] for p in CANDS[t]] for t in range(n_tasks)}
    return data


def _model(*, mp: bool) -> TaskPlacementGNN:
    if not mp:
        os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1"
    torch.manual_seed(1)
    model = TaskPlacementGNN(
        task_feature_dim=3, platform_feature_dim=14, task_type_onehot_dim=4,
        mp_peer_edges=True, mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True,
        partial_state_edge_dim=PLAN_RAW_DIM, plan_raw=True,
    )
    model.eval()
    return model


def _logits(model, data, committed, task):
    with torch.no_grad():
        return make_partial_state_score_fn(model, data, None)(task, committed).clone()


def test_block_marks_both_directions_and_counts():
    data = _graph()
    attr = plan_raw_edge_attr(data, {0: PLACEMENT[0], 2: PLACEMENT[0]})
    ei = data.edge_index.t().tolist()
    for (s, d), (flag, count) in zip(ei, attr.tolist()):
        pair = (s, d) if s < 3 else (d, s)
        assert flag == (1.0 if pair in ((0, 3), (2, 3)) else 0.0)
        plat = (d if s < 3 else s) - 3
        assert count == (2.0 if plat == 0 else 0.0)


def test_committed_placement_must_be_a_candidate():
    with pytest.raises(KeyError):
        plan_raw_edge_attr(_graph(), {1: PLACEMENT[0]})


def test_gnn_sees_where_a_partner_went():
    data, model = _graph(), _model(mp=True)
    a = _logits(model, data, {0: PLACEMENT[0]}, 1)
    b = _logits(model, data, {0: PLACEMENT[2]}, 1)
    assert not torch.allclose(a, b)


def test_mlp_twin_is_blind_to_a_partner_elsewhere_but_reads_its_own_count():
    data, model = _graph(), _model(mp=False)
    base = _logits(model, data, {}, 1)
    assert torch.allclose(base, _logits(model, data, {0: PLACEMENT[0]}, 1))
    assert torch.allclose(base, _logits(model, data, {0: PLACEMENT[2]}, 1))
    assert not torch.allclose(base, _logits(model, data, {2: PLACEMENT[3]}, 1))


def test_constructor_refuses_a_mis_sized_block():
    with pytest.raises(ValueError, match="partial_state_edge_dim"):
        TaskPlacementGNN(task_feature_dim=3, platform_feature_dim=14, task_type_onehot_dim=4,
                         mp_peer_edges=True, mp_bipartite_edge_conv=True,
                         partial_state_edge_dim=25, plan_raw=True)


# raw_plan_v2: the committed-load channel (TaskPlacementGNN._committed_load)

def _sum_model(*, convs: bool) -> TaskPlacementGNN:
    if not convs:
        os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1"
    torch.manual_seed(1)
    model = TaskPlacementGNN(
        task_feature_dim=3, platform_feature_dim=14, task_type_onehot_dim=4,
        mp_peer_edges=True, mp_bipartite_edge_conv=convs, mp_bipartite_edge_attr_zero=False,
        partial_state_edge_dim=PLAN_RAW_DIM, plan_raw=True, plan_raw_sum=True,
    )
    model.eval()
    return model


def _platform_emb(model, data, committed):
    data.partial_state_edge_attr = plan_raw_edge_attr(data, committed)
    with torch.no_grad():
        return model._encode(data)[1].clone()


def test_load_channel_moves_only_the_committed_platform():
    data, model = _graph(), _sum_model(convs=False)
    base = _platform_emb(model, data, {})
    moved = _platform_emb(model, data, {0: PLACEMENT[0]})
    assert not torch.allclose(base[0], moved[0])
    assert torch.equal(base[1:], moved[1:])


def test_load_channel_is_a_sum_not_a_mean():
    data, model = _graph(), _sum_model(convs=False)
    base = _platform_emb(model, data, {})
    one = _platform_emb(model, data, {0: PLACEMENT[0]}) - base
    other = _platform_emb(model, data, {2: PLACEMENT[0]}) - base
    both = _platform_emb(model, data, {0: PLACEMENT[0], 2: PLACEMENT[0]}) - base
    assert torch.allclose(both[0], one[0] + other[0], atol=1e-6)
    assert not torch.allclose(both[0], one[0], atol=1e-4)


def test_no_conv_twin_keeps_the_channel_and_drops_the_convs():
    keys = _sum_model(convs=False).state_dict().keys()
    assert any(k.startswith("load_mlp.") for k in keys)
    assert not any(k.startswith("bip_convs.") for k in keys)


def test_twin_reads_load_committed_onto_its_own_candidate_only():
    data, model = _graph(), _sum_model(convs=False)
    base = _logits(model, data, {}, 1)
    assert torch.allclose(base, _logits(model, data, {0: PLACEMENT[0]}, 1))
    assert not torch.allclose(base, _logits(model, data, {2: PLACEMENT[3]}, 1))


def test_sum_checkpoint_does_not_load_into_a_model_without_the_channel():
    sd = _sum_model(convs=True).state_dict()
    torch.manual_seed(1)
    plain = TaskPlacementGNN(
        task_feature_dim=3, platform_feature_dim=14, task_type_onehot_dim=4,
        mp_peer_edges=True, mp_bipartite_edge_conv=True,
        partial_state_edge_dim=PLAN_RAW_DIM, plan_raw=True,
    )
    with pytest.raises(RuntimeError, match="load_mlp"):
        plain.load_state_dict(sd)


def test_plan_raw_sum_requires_plan_raw():
    with pytest.raises(ValueError, match="needs plan_raw"):
        TaskPlacementGNN(task_feature_dim=3, platform_feature_dim=14, task_type_onehot_dim=4,
                         mp_peer_edges=True, mp_bipartite_edge_conv=True,
                         partial_state_edge_dim=PLAN_RAW_DIM, plan_raw_sum=True)
