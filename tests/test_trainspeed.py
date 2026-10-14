"""uncut_arm1 trainer speed (rp/uncut-trainspeed): the rows-only score and the partial_state_columns partner lists.

Runs on a real partial_state_v5 cache: TRAINSPEED_CACHE (a graphs.pkl, default the Arm 1 21-graph mini-cache on datalab); skipped when absent.
  * the rows-only score equals model._score(...)[t] to 1e-5 relative, in eval and in train mode, and consumes the same RNG stream;
  * the flag is wired only into the training loss closure: the decode (train_near_rtt._masked_topo_regret_for_graph) and the serving
    callers build the score closure without it.
"""
from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CACHE = Path(os.environ.get("TRAINSPEED_CACHE", "/share/nikola.lukic/uncut_arm1/cache_run/cache/graphs.pkl"))
ENV = {"PARTIAL_STATE_CONTRACT": "partial_state_v5", "PARTIAL_STATE_LOAD_SECONDS": "1", "PARTIAL_STATE_EXCHANGE_SECONDS": "1",
       "PARTIAL_STATE_PEER_MASS": "1"}


@pytest.fixture(scope="module")
def graphs():
    if not CACHE.exists():
        pytest.skip(f"no partial_state_v5 cache at {CACHE}")
    with open(CACHE, "rb") as fh:
        return pickle.load(fh)


@pytest.fixture(autouse=True)
def _contract(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def _walk(graph, model, rows_only):
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
    from src.policy.gnn.seq_decode import topological_task_order
    from src.policy.tabular.reduced_features import build_partial_state_context_from_graph

    alpha = "inf"
    ctx = build_partial_state_context_from_graph(graph)
    ctx.node_caps = graph.partial_state_ctx["node_caps_by_alpha"][alpha]
    score = make_partial_state_score_fn(model, graph, ctx, task_rows_only=rows_only)
    plan = graph.tied_optimal_logit_plans[alpha][0]
    committed, out = {}, []
    for t in topological_task_order(int(graph.n_tasks), graph.dag_parents):
        out.append(score(t, committed).detach().clone())
        committed[t] = tuple(int(v) for v in graph.task_logit_to_placement[t][int(plan[t])])
    return out


def _model(graph):
    from src.policy.gnn.gnn_model import TaskPlacementGNN
    from src.policy.tabular.reduced_features import partial_state_feature_dim

    return TaskPlacementGNN(
        task_feature_dim=int(graph.task_features.size(-1)),
        platform_feature_dim=int(graph.platform_features.size(-1)),
        edge_dim=int(graph.edge_attr.size(-1)),
        task_type_onehot_dim=4,
        partial_state_edge_dim=partial_state_feature_dim("partial_state_v5"),
    )


@pytest.mark.parametrize("train", [False, True])
def test_rows_only_equals_the_full_path(graphs, train):
    graph = max(graphs, key=lambda g: int(g.n_tasks))
    torch.manual_seed(0)
    model = _model(graph)
    model.train(train)
    torch.manual_seed(7)
    full = _walk(graph, model, False)
    rng_full = torch.get_rng_state()
    torch.manual_seed(7)
    rows = _walk(graph, model, True)
    assert torch.equal(rng_full, torch.get_rng_state()), "rows-only must consume the dropout RNG stream the full path did"
    assert len(full) == len(rows)
    for a, b in zip(full, rows):
        assert a.shape == b.shape
        # relative to the logit vector's scale: a near-zero logit has no meaningful per-element relative error
        assert float((a - b).abs().max()) <= 1e-5 * max(float(a.abs().max()), 1.0), float((a - b).abs().max())


def test_rows_only_is_wired_into_the_training_closure_only():
    src = (REPO_ROOT / "src/notebooks/train_near_rtt.py").read_text()
    uses = re.findall(r"make_partial_state_score_fn\([^)]*\)", src)
    flagged = [u for u in uses if "task_rows_only" in u]
    assert len(uses) == 2 and len(flagged) == 1, uses
    loss_body = src[src.index("def loss_tied_teacher_forced_ce"):src.index("class NearRttRankingLoss")]
    assert flagged[0] in loss_body
    for path in ("src/policy/gnn/prefix_serving.py", "src/policy/gnn/scheduler.py"):
        assert "task_rows_only" not in (REPO_ROOT / path).read_text()
