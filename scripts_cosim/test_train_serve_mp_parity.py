#!/usr/bin/env python3
"""Guard: the served GNN must message-pass over the SAME graph it was trained on.

`src/notebooks/train_near_rtt.py` fits weights with ``self.gin(x, data.edge_index)`` —
the bipartite task<->platform graph only; it never references ``node_edge_index``. The
inference copy in ``src/policy/gnn/gnn_model.py`` used to concatenate every same-node
platform<->platform edge into the message-passing index, so the served model ran on a
graph its weights had never seen.

Measured cost of that drift on the deployed checkpoint (2026-08-16):
  * ~1428 same-node edges vs ~57 bipartite per graph — a 25:1 flood
  * 87.5% of argmax decisions changed vs the training graph
  * live sparse_p35/s42 total_rtt 276.0M (10.93x Knative) -> 22.3M (0.88x) once dropped

The trainer runs training at import time, so it cannot be imported to diff the two
classes directly. These tests instead pin the invariant the trainer relies on: by
default the inference model must ignore ``node_edge_index``, and the opt-in flag must
demonstrably change the result (so it cannot rot into a no-op).

Run:
    pipenv run python3 -m pytest scripts_cosim/test_train_serve_mp_parity.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MP_NODE_EDGES_ENV = "GNN_MP_NODE_EDGES"
DISABLE_MP_ENV = "GNN_DISABLE_MESSAGE_PASSING"


def _graph_with_same_node_edges(seed: int = 0) -> Data:
    """2 tasks, 4 platforms; platforms 0 and 1 share a physical node."""
    torch.manual_seed(seed)  # features must be identical across variants being compared
    data = Data()
    data.n_tasks = 2
    data.n_platforms = 4
    data.task_features = torch.randn(2, 3)
    data.platform_features = torch.randn(4, 14)
    src = torch.tensor([0, 0, 1, 1])
    dst = torch.tensor([2, 3, 2, 4])  # platform global index = n_tasks + pos
    data.edge_index = torch.stack([src, dst])
    data.edge_attr = torch.randn(4, 5)
    data.node_edge_index = torch.tensor([[2, 3], [3, 2]])
    return data


def _model(**env):
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    for key in (MP_NODE_EDGES_ENV, DISABLE_MP_ENV):
        os.environ.pop(key, None)
    os.environ.update(env)
    torch.manual_seed(0)
    model = TaskPlacementGNN(task_feature_dim=3, platform_feature_dim=14)
    model.eval()
    return model


def _logits(model, data) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat(model(data))


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.get(k) for k in (MP_NODE_EDGES_ENV, DISABLE_MP_ENV)}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_default_ignores_same_node_edges():
    """Default serving must equal a graph carrying no same-node edges at all."""
    data = _graph_with_same_node_edges()
    bare = _graph_with_same_node_edges()
    bare.node_edge_index = torch.empty((2, 0), dtype=torch.long)

    model = _model()
    assert torch.allclose(_logits(model, data), _logits(model, bare)), (
        "Serving default message-passes over same-node edges that train_near_rtt.py "
        "never uses. This is the 12.4x live-RTT regression; keep them opt-in."
    )


def test_opt_in_flag_actually_changes_message_passing():
    """The escape hatch must still work, or the ablation silently becomes a no-op."""
    data = _graph_with_same_node_edges()
    default = _logits(_model(), data)
    opted_in = _logits(_model(**{MP_NODE_EDGES_ENV: "1"}), data)
    assert not torch.allclose(default, opted_in), (
        f"{MP_NODE_EDGES_ENV}=1 did not change the output — the flag is dead code."
    )


def test_disable_message_passing_still_available():
    """Rung C of the ablation ladder: encoder embeddings bypass GIN entirely."""
    data = _graph_with_same_node_edges()
    default = _logits(_model(), data)
    no_mp = _logits(_model(**{DISABLE_MP_ENV: "1"}), data)
    assert not torch.allclose(default, no_mp)


def test_boolean_flags_fail_loud():
    from src.policy.gnn.gnn_model import _env_flag

    os.environ[MP_NODE_EDGES_ENV] = "maybe"
    with pytest.raises(ValueError, match="not a boolean"):
        _env_flag(MP_NODE_EDGES_ENV)


def test_same_node_edges_would_flood_message_passing():
    """Document the magnitude: the flood is why this drift was catastrophic, not subtle."""
    from src.policy.gnn.gnn_model import build_same_node_edge_index

    # 8 physical nodes x 4 platforms each, a realistic sparse-cell shape.
    node_to_positions = {n: list(range(n * 4, n * 4 + 4)) for n in range(8)}
    same_node = build_same_node_edge_index(node_to_positions, n_tasks=4)
    bipartite_edges = 4 * 8  # 4 tasks x ~8 reachable platforms
    assert same_node.shape[1] > 2 * bipartite_edges, (
        "Same-node edges are expected to dominate the bipartite graph; if this no "
        "longer holds the flood rationale above needs re-measuring."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
