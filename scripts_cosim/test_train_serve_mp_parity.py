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


# ---------------------------------------------------------------------------------------
# Restoring USEFUL message passing: gated residual + candidate-restricted same-node edges.
# ---------------------------------------------------------------------------------------
def test_candidate_restriction_keeps_only_reachable_pairs():
    """Same-node edges are only meaningful between platforms a task could actually use."""
    from src.policy.gnn.gnn_model import restrict_node_edges_to_candidates

    data = _graph_with_same_node_edges()
    # Platforms 0,1 are reachable (edge_index dsts 2,3); add a same-node pair 2<->3 that
    # no task can reach, plus the reachable 0<->1 pair.
    node_ei = torch.tensor([[2, 3, 4, 5], [3, 2, 5, 4]])
    kept = restrict_node_edges_to_candidates(
        node_ei, data.edge_index, n_tasks=data.n_tasks, n_platforms=data.n_platforms
    )
    assert kept.tolist() == [[2, 3], [3, 2]], (
        "Unreachable same-node pairs must be dropped; they add aggregation mass with no "
        "decision-relevant signal."
    )


def test_candidate_restriction_survives_no_candidates():
    """36% of real cached graphs have zero candidate pairs — must degenerate, not crash."""
    from src.policy.gnn.gnn_model import restrict_node_edges_to_candidates

    data = _graph_with_same_node_edges()
    node_ei = torch.tensor([[4, 5], [5, 4]])  # neither platform is reachable
    kept = restrict_node_edges_to_candidates(
        node_ei, data.edge_index, n_tasks=data.n_tasks, n_platforms=data.n_platforms
    )
    assert kept.numel() == 0


def test_mp_gate_makes_the_residual_self_describing():
    """A residual checkpoint must not load into a non-residual model, or vice versa.

    The residual adds no shape change to any existing weight, so without `mp_gate` a
    residual checkpoint would strict-load into a plain model and silently serve a
    different architecture — the exact failure mode this whole file exists to prevent.
    """
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    plain = TaskPlacementGNN(task_feature_dim=3, platform_feature_dim=14)
    residual = TaskPlacementGNN(task_feature_dim=3, platform_feature_dim=14, mp_residual=True)

    assert "mp_gate" not in plain.state_dict()
    assert "mp_gate" in residual.state_dict()

    with pytest.raises(RuntimeError):
        plain.load_state_dict(residual.state_dict())
    with pytest.raises(RuntimeError):
        residual.load_state_dict(plain.state_dict())


def test_residual_changes_output_with_identical_weights():
    """The residual must actually alter scoring, or it is dead code."""
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    data = _graph_with_same_node_edges()
    plain = _model()
    residual = TaskPlacementGNN(
        task_feature_dim=3, platform_feature_dim=14, mp_residual=True
    )
    state = dict(plain.state_dict())
    state["mp_gate"] = torch.ones(1)
    residual.load_state_dict(state)  # ONLY the residual differs
    residual.eval()

    assert not torch.allclose(_logits(plain, data), _logits(residual, data))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------------------
# route_b stage 2: workload-DAG message passing + per-step prefix conditioning.
#
# Same rule as same-node edges, one lineage later: both blocks are opt-in, and a default
# checkpoint must be bit-identical whether or not a graph happens to carry them. The
# cache now ships dag_edge_index / task_type_onehot4 on every --dag-partial-state graph,
# so "the graph doesn't have it" no longer protects an older checkpoint — only the flag
# defaults do.
# ---------------------------------------------------------------------------------------
DAG_EDGES_ENV = "GNN_MP_DAG_EDGES"

_PARTIAL_STATE_DIM = 38


def _graph_with_dag_and_prefix(seed: int = 0) -> Data:
    """The same 2-task/4-platform graph, plus the stage-2 DAG and prefix blocks."""
    data = _graph_with_same_node_edges(seed)
    # task 0 -> task 1, stored parent->child; the model emits both directions itself.
    data.dag_edge_index = torch.tensor([[0], [1]])
    data.task_type_onehot4 = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    torch.manual_seed(seed + 991)
    data.partial_state_edge_attr = torch.randn(4, _PARTIAL_STATE_DIM)
    return data


def _dag_model(*, dag: bool, onehot: int = 4, prefix: int = 0, seed: int = 0):
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    os.environ.pop(DAG_EDGES_ENV, None)
    torch.manual_seed(seed)
    model = TaskPlacementGNN(
        task_feature_dim=3,
        platform_feature_dim=14,
        mp_dag_edges=dag,
        task_type_onehot_dim=onehot,
        partial_state_edge_dim=prefix,
    )
    model.eval()
    return model


def test_default_ignores_dag_edges():
    """A default checkpoint must not message-pass over DAG edges it never trained on."""
    data = _graph_with_dag_and_prefix()
    bare = _graph_with_dag_and_prefix()
    bare.dag_edge_index = torch.empty((2, 0), dtype=torch.long)

    model = _model()
    assert torch.allclose(_logits(model, data), _logits(model, bare)), (
        "Serving default message-passes over workload-DAG edges. Same failure mode as "
        "same-node edges in 2026-08-16: keep them opt-in."
    )


def test_default_ignores_partial_state_edge_attr():
    """A default checkpoint must never be silently prefix-conditioned."""
    data = _graph_with_dag_and_prefix()
    bare = _graph_with_dag_and_prefix()
    del bare.partial_state_edge_attr

    model = _model()
    assert torch.allclose(_logits(model, data), _logits(model, bare)), (
        "A checkpoint with partial_state_edge_dim=0 read the prefix block. Prefix "
        "conditioning must be a declared, weight-visible property."
    )


def test_dag_flag_actually_changes_message_passing():
    """Kills flag rot: an opt-in that changes nothing is worse than no flag."""
    data = _graph_with_dag_and_prefix()
    on = _dag_model(dag=True)
    off = _dag_model(dag=False)
    assert not torch.allclose(_logits(on, data), _logits(off, data)), (
        "mp_dag_edges changed no logits, so the arm registered as DAG-aware is not."
    )


def test_prefix_block_actually_changes_scores():
    """Same, for the prefix half: the 38 columns must reach the scorer."""
    from src.policy.tabular.reduced_features import PARTIAL_STATE_FEATURE_DIM

    data = _graph_with_dag_and_prefix()
    zeroed = _graph_with_dag_and_prefix()
    zeroed.partial_state_edge_attr = torch.zeros(4, PARTIAL_STATE_FEATURE_DIM)

    model = _dag_model(dag=True, prefix=PARTIAL_STATE_FEATURE_DIM)
    assert not torch.allclose(_logits(model, data), _logits(model, zeroed)), (
        "The partial-state block did not move any score; the T2 arm would be scoring "
        "as if no prefix existed."
    )


def test_dag_flag_without_type_onehot_fails_loud():
    """Undirected DAG mixing without the 4-way one-hot makes cnn/rf interchangeable."""
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    with pytest.raises(ValueError, match="task_type_onehot_dim"):
        TaskPlacementGNN(
            task_feature_dim=3, platform_feature_dim=14, mp_dag_edges=True
        )


def test_mp_dag_edges_missing_attr_fails_loud():
    """A DAG model handed a non-DAG graph must fail, not degrade to bipartite."""
    data = _graph_with_dag_and_prefix()
    del data.dag_edge_index
    model = _dag_model(dag=True)
    with pytest.raises(ValueError, match="dag_edge_index"):
        _logits(model, data)


def test_task_type_onehot_missing_fails_loud():
    data = _graph_with_dag_and_prefix()
    del data.task_type_onehot4
    model = _dag_model(dag=True)
    with pytest.raises(ValueError, match="task_type_onehot4"):
        _logits(model, data)


def test_partial_state_attr_missing_fails_loud():
    """Never score an all-zero prefix by accident — that is arm A3, not A1."""
    from src.policy.tabular.reduced_features import PARTIAL_STATE_FEATURE_DIM

    data = _graph_with_dag_and_prefix()
    del data.partial_state_edge_attr
    model = _dag_model(dag=True, prefix=PARTIAL_STATE_FEATURE_DIM)
    with pytest.raises(ValueError, match="partial_state_edge_attr"):
        _logits(model, data)


def test_dag_edges_out_of_range_fail_loud():
    """DAG edges index the task block only; a platform index would corrupt the graph."""
    data = _graph_with_dag_and_prefix()
    data.dag_edge_index = torch.tensor([[0], [5]])  # 5 is a platform node
    model = _dag_model(dag=True)
    with pytest.raises(ValueError, match="outside the task block"):
        _logits(model, data)


def test_dag_edges_are_a_minority_term():
    """The 2026-08-16 flood guard, rewritten for DAG edges.

    Measured on gnn_datasets_dag4_route_b_smoke_s/ds_00000: candidates per task are
    6/6/4/4, so the bipartite graph has 40 columns after to_undirected, while diamond4's
    4 DAG edges become 8 — 0.20x, versus the 25.9x-29.9x same-node flood. Platform
    in-degree is also untouched (DAG edges are task<->task), so the queue-erasure
    mechanism that made the same-node flood catastrophic cannot fire here at all.

    Pinned as a fact rather than left in a comment: if a future grid makes DAG edges
    dominant, that rationale needs re-measuring before the arm is trusted.
    """
    dag_edges_undirected = 2 * 4  # diamond4, both directions
    bipartite_edges = 6 + 6 + 4 + 4  # the measured route_b candidate counts
    assert dag_edges_undirected <= 0.5 * bipartite_edges, (
        "Workload-DAG edges are expected to be a minority term against the bipartite "
        "graph; if this no longer holds, re-measure the flood rationale."
    )
