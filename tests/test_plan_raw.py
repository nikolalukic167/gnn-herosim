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

ENVS = ("GNN_DISABLE_MESSAGE_PASSING", "GNN_PLAN_RAW", "GNN_MP_NODE_EDGES")

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
