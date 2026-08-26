"""The two properties that make the route_b stage-2 A1-vs-A2/A3 comparison valid.

1. **T1/T2 column parity.** The 38 partial-state columns the GNN sees on a candidate
   edge must be, bit for bit, the last 38 columns of the MLP's dim63crk row for the same
   (graph, plan, step, candidate). §2's fairness rule is that both tiers read ONE
   definition of occupancy — ``reduced_features.partial_state_columns``. If anyone ever
   clones that formula "just for the GNN", this test fails, which is the entire point.

2. **Train/decode prefix parity.** The sequence of (task, committed-prefix) pairs the
   trainer teacher-forces along a tied plan must equal the sequence the §4 masked decoder
   visits when it reproduces that same plan. A T2 arm trained on prefixes it never
   decodes under is measuring nothing; here the two come from one closure, and this pins
   that they actually agree.

Both run against the real --dag-partial-state smoke cache, not a synthetic graph, so a
cache-layout drift shows up here too.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402
from src.policy.gnn.partial_state_edges import (  # noqa: E402
    candidate_edge_rows,
    make_partial_state_score_fn,
    refresh_partial_state_edge_attr,
)
from src.policy.gnn.seq_decode import (  # noqa: E402
    decode_masked_topo_placement,
    topological_task_order,
)
from src.policy.tabular.reduced_features import (  # noqa: E402
    DIM63CRK_FEATURE_DIM,
    PARTIAL_STATE_FEATURE_DIM,
    build_partial_state_context_from_graph,
    extract_rows_dim63crk_from_batch_graph,
    partial_state_columns,
)

CACHE = REPO_ROOT / "simulation_data" / "graphs_cache_route_b_smoke_s_dag" / "graphs.pkl"
ALPHA = "2.0"


@pytest.fixture(scope="module")
def graphs():
    if not CACHE.exists():
        pytest.skip(f"no --dag-partial-state cache at {CACHE}")
    with open(CACHE, "rb") as fh:
        return pickle.load(fh)


def _ctx(graph):
    ctx = build_partial_state_context_from_graph(graph)
    ctx.node_caps = graph.partial_state_ctx["node_caps_by_alpha"][ALPHA]
    return ctx


def test_gnn_prefix_rows_equal_the_single_definition(graphs):
    """The GNN's edge rows ARE partial_state_columns — not a re-derivation of it."""
    for graph in graphs:
        rows = candidate_edge_rows(graph)
        ctx = _ctx(graph)
        order = topological_task_order(int(graph.n_tasks), graph.dag_parents)
        plan = graph.tied_optimal_logit_plans[ALPHA][0]
        committed = {}
        for task_idx in order:
            attr = refresh_partial_state_edge_attr(graph, ctx, task_idx, committed)
            candidates = [tuple(c) for c in graph.task_logit_to_placement[task_idx]]
            expected = partial_state_columns(ctx, task_idx, candidates, committed)
            got = attr[torch.as_tensor(rows[task_idx], dtype=torch.long)].numpy()
            assert np.array_equal(got, expected.astype(np.float32)), (
                f"task {task_idx}: the GNN's prefix block diverged from "
                "partial_state_columns — §2's one-definition rule is broken"
            )
            committed[task_idx] = tuple(
                graph.task_logit_to_placement[task_idx][plan[task_idx]]
            )


def test_t1_and_t2_read_the_same_38_columns(graphs):
    """The GNN's block equals the tail of the MLP's dim63crk row, exactly.

    This is the fairness claim made executable: the two tiers differ in architecture and
    in what message passing adds, never in what occupancy means.
    """
    graph = graphs[0]
    mlp_rows, skip = extract_rows_dim63crk_from_batch_graph(
        graph, "t2-parity", alpha_key=ALPHA
    )
    assert not skip, skip
    assert mlp_rows, "the dim63crk extractor produced no rows"

    rows = candidate_edge_rows(graph)
    ctx = _ctx(graph)
    order = topological_task_order(int(graph.n_tasks), graph.dag_parents)

    # Rebuild plan 0's prefixes and compare against the extractor's rows for that plan.
    plan = graph.tied_optimal_logit_plans[ALPHA][0]
    # Keyed on the row's own logit_idx, so this compares the SAME candidate rather
    # than trusting two enumerations to line up.
    by_task = {}
    for row in mlp_rows:
        if "plan0" in row.graph_id:
            by_task.setdefault(row.task_idx, []).append(row)
    assert by_task, "no plan-0 rows — the dim63crk graph_id convention changed"

    committed = {}
    compared = 0
    for task_idx in order:
        attr = refresh_partial_state_edge_attr(graph, ctx, task_idx, committed)
        gnn_block = attr[torch.as_tensor(rows[task_idx], dtype=torch.long)].numpy()
        for mlp_row in by_task.get(task_idx, []):
            feats = np.asarray(mlp_row.features, dtype=np.float32)
            assert feats.shape[-1] == DIM63CRK_FEATURE_DIM
            mlp_tail = feats[-PARTIAL_STATE_FEATURE_DIM:]
            assert np.array_equal(gnn_block[int(mlp_row.logit_idx)], mlp_tail), (
                f"task {task_idx} logit {mlp_row.logit_idx}: T1 and T2 disagree about "
                "the partial-state columns; someone cloned the formula"
            )
            compared += 1
        committed[task_idx] = tuple(
            graph.task_logit_to_placement[task_idx][plan[task_idx]]
        )
    assert compared > 0, "compared no rows — the graph_id convention changed"


def test_train_and_decode_visit_the_same_prefixes(graphs):
    """Teacher-forced training and masked_topo decoding walk one identical sequence."""
    graph = graphs[0]
    ctx = _ctx(graph)
    n_tasks = int(graph.n_tasks)
    order = topological_task_order(n_tasks, graph.dag_parents)
    plan = graph.tied_optimal_logit_plans[ALPHA][0]

    # What the trainer teacher-forces along plan 0.
    train_seen = []
    committed = {}
    for task_idx in order:
        train_seen.append((task_idx, dict(committed)))
        committed[task_idx] = tuple(
            int(v) for v in graph.task_logit_to_placement[task_idx][plan[task_idx]]
        )

    # What the decoder visits when its scores force that same plan.
    decode_seen = []

    def forcing_score_fn(task_idx, prefix):
        decode_seen.append((task_idx, dict(prefix)))
        n = len(graph.task_logit_to_placement[task_idx])
        scores = [0.0] * n
        scores[int(plan[task_idx])] = 1.0
        return scores

    demands = {
        t: [
            float(ctx.demand[(t, tuple(int(v) for v in c))])
            for c in graph.task_logit_to_placement[t]
        ]
        for t in range(n_tasks)
    }
    combo = decode_masked_topo_placement(
        [torch.empty(0)] * n_tasks,
        graph.task_logit_to_placement,
        n_tasks,
        dag_parents=graph.dag_parents,
        node_caps=ctx.node_caps,
        demands=demands,
        score_fn=forcing_score_fn,
    )
    assert combo is not None, "the tied-optimal plan failed to decode under its own mask"
    assert train_seen == decode_seen, (
        "train-time and decode-time prefixes diverge; the T2 arm would be trained on "
        "states it never decodes under"
    )


def test_one_encode_is_reused_across_steps(graphs):
    """The cost lever: the GIN is prefix-independent, so it runs once per graph.

    If a future change routes prefix state into a node feature this reuse silently stops
    being correct — so the cache is asserted, not assumed.
    """
    graph = graphs[0]
    model = TaskPlacementGNN(
        task_feature_dim=int(graph.task_features.size(-1)),
        platform_feature_dim=int(graph.platform_features.size(-1)),
        edge_dim=int(graph.edge_attr.size(-1)),
        mp_dag_edges=True,
        task_type_onehot_dim=4,
        partial_state_edge_dim=PARTIAL_STATE_FEATURE_DIM,
    ).eval()

    calls = {"n": 0}
    real_encode = model._encode

    def counting_encode(data):
        calls["n"] += 1
        return real_encode(data)

    model._encode = counting_encode
    ctx = _ctx(graph)
    score = make_partial_state_score_fn(model, graph, ctx)
    order = topological_task_order(int(graph.n_tasks), graph.dag_parents)
    plan = graph.tied_optimal_logit_plans[ALPHA][0]

    committed = {}
    with torch.no_grad():
        for task_idx in order:
            score(task_idx, committed)
            committed[task_idx] = tuple(
                graph.task_logit_to_placement[task_idx][plan[task_idx]]
            )
    assert calls["n"] == 1, f"expected 1 GIN pass per graph, got {calls['n']}"


def test_score_fn_refuses_a_model_that_would_discard_the_prefix(graphs):
    """Building the columns and then dropping them is the silent-A3 failure."""
    graph = graphs[0]
    model = TaskPlacementGNN(
        task_feature_dim=int(graph.task_features.size(-1)),
        platform_feature_dim=int(graph.platform_features.size(-1)),
        edge_dim=int(graph.edge_attr.size(-1)),
        mp_dag_edges=True,
        task_type_onehot_dim=4,
    ).eval()
    with pytest.raises(ValueError, match="partial_state_edge_dim"):
        make_partial_state_score_fn(model, graph, _ctx(graph))


def test_candidate_edge_rows_detects_a_reordered_logit_map(graphs):
    """The ordering is guaranteed by the cache builder — and verified anyway."""
    graph = graphs[0]
    candidate_edge_rows(graph)  # populates the memo on a good graph

    tampered = graphs[1]
    keys = dict(tampered.task_logit_to_queue_key)
    keys[0] = list(reversed(list(keys[0])))
    tampered.task_logit_to_queue_key = keys
    if hasattr(tampered, "_candidate_edge_rows"):
        del tampered._candidate_edge_rows
    with pytest.raises(ValueError, match="misalignment"):
        candidate_edge_rows(tampered)
