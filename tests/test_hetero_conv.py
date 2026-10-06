"""hetero_conv_v1 (HeteroBipartiteEdgeConv): per-relation and per-node-type weights in the bipartite stack.

Pins:
- each relation's message weights reach only the node type it points at (t2p and p2p -> platforms, p2t -> tasks),
  and each update MLP touches only its own node type;
- on the raw plan, the hetero GNN still sees where a partner went;
- the hetero and plain checkpoints never load into each other, and the flag refuses meaningless combinations.
"""
from __future__ import annotations

import os

import pytest
import torch

from src.policy.gnn.gnn_model import HeteroBipartiteEdgeConv, TaskPlacementGNN
from src.policy.gnn.plan_raw import PLAN_RAW_DIM
from test_plan_raw import PLACEMENT, _clean_env, _graph, _logits  # noqa: F401  (fixture re-export)

ENVS = ("GNN_MP_BIPARTITE_HETERO",)


@pytest.fixture(autouse=True)
def _clean_hetero_env():
    saved = {k: os.environ.get(k) for k in ENVS}
    for k in ENVS:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _model(*, hetero: bool) -> TaskPlacementGNN:
    torch.manual_seed(1)
    model = TaskPlacementGNN(
        task_feature_dim=3, platform_feature_dim=14, task_type_onehot_dim=4,
        mp_peer_edges=True, mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True,
        partial_state_edge_dim=PLAN_RAW_DIM, plan_raw=True, mp_bipartite_hetero=hetero,
    )
    return model.eval()


def _conv_inputs():
    torch.manual_seed(0)
    n_tasks = 3
    # t2p, p2t and p2p edges over 3 tasks (rows 0-2) and 4 platforms (rows 3-6)
    ei = torch.tensor([[0, 1, 2, 3, 4, 6, 3, 4],
                       [3, 4, 6, 0, 1, 2, 4, 3]])
    return torch.randn(7, 8), ei, torch.randn(ei.size(1), 2), n_tasks


def _bump(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.5)


@pytest.mark.parametrize("part,moves", [
    ("t2p", "platforms"), ("p2p", "platforms"), ("p2t", "tasks"),
    ("update_task", "tasks"), ("update_platform", "platforms"),
])
def test_each_weight_block_reaches_only_its_own_node_type(part, moves):
    x, ei, ea, n_tasks = _conv_inputs()
    torch.manual_seed(2)
    conv = HeteroBipartiteEdgeConv(8, 16, edge_dim=2, dropout_p=0.0).eval()
    with torch.no_grad():
        before = conv(x, ei, ea, n_tasks)
        _bump(conv.message_mlps[part] if part in conv.message_mlps else getattr(conv, part))
        after = conv(x, ei, ea, n_tasks)
    changed_tasks = not torch.allclose(before[:n_tasks], after[:n_tasks])
    changed_plats = not torch.allclose(before[n_tasks:], after[n_tasks:])
    assert (changed_tasks, changed_plats) == ((True, False) if moves == "tasks" else (False, True))


def test_refuses_task_to_task_edges():
    x, ei, ea, n_tasks = _conv_inputs()
    conv = HeteroBipartiteEdgeConv(8, 16, edge_dim=2)
    with pytest.raises(ValueError, match="task->task"):
        conv(x, torch.cat([ei, torch.tensor([[0], [1]])], dim=1), torch.cat([ea, ea[:1]]), n_tasks)


def test_hetero_gnn_sees_where_a_partner_went():
    data, model = _graph(), _model(hetero=True)
    assert not torch.allclose(_logits(model, data, {0: PLACEMENT[0]}, 1), _logits(model, data, {0: PLACEMENT[2]}, 1))


def test_hetero_and_plain_checkpoints_do_not_load_into_each_other():
    hetero, plain = _model(hetero=True), _model(hetero=False)
    assert any(".message_mlps.p2t." in k for k in hetero.state_dict())
    with pytest.raises(RuntimeError):
        plain.load_state_dict(hetero.state_dict())
    with pytest.raises(RuntimeError):
        hetero.load_state_dict(plain.state_dict())


def test_flag_refuses_meaningless_combinations():
    kw = dict(task_feature_dim=3, platform_feature_dim=14, task_type_onehot_dim=4, mp_peer_edges=True,
              partial_state_edge_dim=PLAN_RAW_DIM, plan_raw=True, mp_bipartite_hetero=True)
    with pytest.raises(ValueError, match="needs mp_bipartite_edge_conv"):
        TaskPlacementGNN(**kw)
    with pytest.raises(ValueError, match="by mean per relation"):
        TaskPlacementGNN(mp_bipartite_edge_conv=True, mp_bipartite_aggr="sum", **kw)
