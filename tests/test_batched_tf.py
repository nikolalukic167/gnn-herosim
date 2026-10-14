"""uncut_groups_s0_v1 Arm 1b: the batched teacher-forced loss (NEAR_RTT_BATCHED_TF, src/policy/gnn/batched_tf.py).

Ground truth is the sequential walk in train_near_rtt.loss_tied_teacher_forced_ce (Arm 1), executed from the trainer's own source
(the trainer trains at import and cannot be imported). Runs on a real Arm 1 cache, BATCHED_TF_CACHE (default
/share/nikola.lukic/uncut_arm1/cache_run/cache/graphs.pkl); every test skips when it is absent. Graphs used: a typical one, the largest
101-256-task group, and a graph with more than one tied plan (the Arm 1 corpus has none, so a real graph is given extra plans).

  (a) every per-step log-prob of the batched path equals the sequential one within 1e-5, dropout off, prefix and full-context states;
  (b) causality: perturbing later placements leaves step <= t bit-unchanged, and the gradient of step <= t with respect to the later
      steps' feature rows is exactly 0;
  L5  the encode is prefix-free (identical whatever partial_state_edge_attr holds);
  loss  the batched loss equals the sequential loss in eval mode; in train mode it differs only by the dropout stream;
  off   with the flag off the loss function is the unchanged sequential one (bit-equal to the function with the branch removed);
  det   two runs of a few optimiser steps are bit-equal, on CPU and on CUDA when available.
"""
from __future__ import annotations

import ast
import math
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CACHE = Path(os.environ.get("BATCHED_TF_CACHE", "/share/nikola.lukic/uncut_arm1/cache_run/cache/graphs.pkl"))
ENV = {"PARTIAL_STATE_CONTRACT": "partial_state_v5", "PARTIAL_STATE_LOAD_SECONDS": "1", "PARTIAL_STATE_EXCHANGE_SECONDS": "1",
       "PARTIAL_STATE_PEER_MASS": "1"}
ALPHA = "inf"


@pytest.fixture(autouse=True)
def _contract(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


@pytest.fixture(scope="module")
def graphs():
    if not CACHE.exists():
        pytest.skip(f"no partial_state_v5 cache at {CACHE}")
    with open(CACHE, "rb") as fh:
        gs = pickle.load(fh)
    return [g for g in gs if (getattr(g, "tied_optimal_logit_plans", {}) or {}).get(ALPHA)]


def _with_extra_plans(graph, order):
    """The Arm 1 corpus labels exactly one plan per state (no ties), so a multi-plan graph is built from a real one: its label plus
    two plans that share the label's prefix up to the middle of the commit order and differ after it (and a third that differs
    from the start), which exercises the per-plan prefix, the dedupe and the logsumexp over plans."""
    import copy

    g = copy.copy(graph)
    base = [int(v) for v in graph.tied_optimal_logit_plans[ALPHA][0]]
    n = int(graph.n_tasks)
    rng = np.random.RandomState(1)

    def variant(first_changed):
        pl = list(base)
        for t in order[first_changed:]:
            c = len(graph.task_logit_to_placement[t])
            if c > 1:
                pl[t] = int((base[t] + 1 + rng.randint(c - 1)) % c)
        return pl

    plans = [base, variant(n // 2), variant(n // 2 + 1), variant(0)]
    g.tied_optimal_logit_plans = dict(graph.tied_optimal_logit_plans)
    g.tied_optimal_logit_plans[ALPHA] = plans
    assert len({tuple(p) for p in plans}) > 1
    return g


def _pick(graphs):
    typical = sorted(graphs, key=lambda g: int(g.n_tasks))[len(graphs) // 2]
    big = [g for g in graphs if 101 <= int(g.n_tasks) <= 256]
    assert big, "the cache has no 101-256-task group"
    small = [g for g in graphs if 8 <= int(g.n_tasks) <= 30 and max(len(g.task_logit_to_placement[t]) for t in range(int(g.n_tasks))) > 1]
    return {"typical": typical, "group101_256": min(big, key=lambda g: int(g.n_tasks)),
            "multi_plan": _with_extra_plans(small[len(small) // 2], _order(small[len(small) // 2]))}


def _model(graph, seed=0):
    from src.policy.gnn.gnn_model import TaskPlacementGNN
    from src.policy.tabular.reduced_features import partial_state_feature_dim

    torch.manual_seed(seed)
    return TaskPlacementGNN(
        task_feature_dim=int(graph.task_features.shape[1]), platform_feature_dim=int(graph.platform_features.shape[1]),
        embedding_dim=128, hidden_dim=128, num_layers=3, dropout=0.1, post_gin_dropout=0.1,
        normalize_platform_inputs=int(graph.platform_features.shape[1]) == 21,
        mp_peer_edges=True, mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True,
        task_type_onehot_dim=4, partial_state_edge_dim=partial_state_feature_dim("partial_state_v5"),
    )


def _ctx(graph, alpha=ALPHA):
    from src.policy.tabular.reduced_features import build_partial_state_context_from_graph

    ctx = build_partial_state_context_from_graph(graph)
    ctx.node_caps = graph.partial_state_ctx["node_caps_by_alpha"][alpha]
    return ctx


def _order(graph):
    from src.policy.gnn.seq_decode import topological_task_order

    return topological_task_order(int(graph.n_tasks), graph.dag_parents)


def _loss_fn(batched: bool, strip_branch: bool = False):
    """loss_tied_teacher_forced_ce from the trainer's source, with NEAR_CFG.batched_tf = batched."""
    from src.policy.gnn import arm2 as _arm2
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
    from src.policy.gnn.seq_decode import topological_task_order
    from src.policy.tabular.reduced_features import build_partial_state_context_from_graph

    tree = ast.parse((REPO_ROOT / "src/notebooks/train_near_rtt.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "loss_tied_teacher_forced_ce")
    if strip_branch:
        fn.body = [s for s in fn.body if not (isinstance(s, ast.If) and "batched_tf" in ast.unparse(s.test))]
        assert not any("batched_tf" in ast.unparse(s) for s in fn.body), "the batched branch was not isolated in one top-level if"
    ns: Dict[str, Any] = dict(
        torch=torch, F=F, math=math, _arm2=_arm2, Dict=Dict, List=List, Tuple=Tuple, Any=Any, Sequence=Sequence, Tensor=torch.Tensor,
        topological_task_order=topological_task_order, make_partial_state_score_fn=make_partial_state_score_fn,
        build_partial_state_context_from_graph=build_partial_state_context_from_graph, _TRAIN_SEED=1,
        _ANCHOR_STATS={s: {"ce": 0.0, "floor": 0.0, "top1": 0.0, "groups": 0.0} for s in ("train", "val")},
        NEAR_CFG=types.SimpleNamespace(dag_alpha_key=ALPHA, tied_max_plans=0, anchor_ce_weight=0.5, score_task_rows_only=False,
                                       batched_tf=batched),
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "train_near_rtt.py:loss_tied_teacher_forced_ce", "exec"), ns)
    return ns["loss_tied_teacher_forced_ce"]


def _sequential_logps(graph, model, ctx, plans, order, with_full):
    """Per (plan, task) log-prob of the label under the sequential walk: prefix states and full-context states."""
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn

    score = make_partial_state_score_fn(model, graph, ctx)
    place = graph.task_logit_to_placement
    pre = np.zeros((len(plans), int(graph.n_tasks))); full = np.zeros_like(pre)
    for k, plan in enumerate(plans):
        committed: Dict[int, Any] = {}
        for t in order:
            pre[k, t] = float(F.log_softmax(score(t, committed), dim=-1)[int(plan[t])])
            committed[t] = tuple(int(v) for v in place[t][int(plan[t])])
        if with_full:
            allp = {t: tuple(int(v) for v in place[t][int(plan[t])]) for t in order}
            for t in order:
                pre_full = {j: p for j, p in allp.items() if j != t}
                full[k, t] = float(F.log_softmax(score(t, pre_full), dim=-1)[int(plan[t])])
    return pre, full


@pytest.mark.parametrize("which", ["typical", "group101_256", "multi_plan"])
def test_per_step_logps_equal_the_sequential_walk(graphs, which):
    from src.policy.gnn.batched_tf import BatchedTeacherForced

    graph = _pick(graphs)[which]
    model = _model(graph).eval()
    ctx, order = _ctx(graph), _order(graph)
    plans = graph.tied_optimal_logit_plans[ALPHA]
    with torch.no_grad():
        b = BatchedTeacherForced(graph, ctx, plans, order, with_full=True)
        task_emb, platform_emb = model._encode(graph)
        lp = b.state_logps(model, graph, task_emb, platform_emb).numpy()
        seq_pre, seq_full = _sequential_logps(graph, model, ctx, plans, order, True)
    got_pre, got_full = lp[b.state_of["prefix"]], lp[b.state_of["full"]]
    assert np.abs(got_pre - seq_pre).max() <= 1e-5, np.abs(got_pre - seq_pre).max()
    assert np.abs(got_full - seq_full).max() <= 1e-5, np.abs(got_full - seq_full).max()


def test_caps_are_empty_on_the_registered_rung_and_the_capped_columns_agree(graphs):
    """L2: the block is per prefix row. The registered rung is uncapped; a capped rung (if the cache has one) must agree too."""
    from src.policy.gnn.batched_tf import BatchedTeacherForced

    graph = _pick(graphs)["typical"]
    caps = graph.partial_state_ctx["node_caps_by_alpha"]
    assert not any(math.isfinite(float(v)) for v in caps[ALPHA].values()), "caps are not empty at the registered rung"
    capped = [a for a in caps if a != ALPHA and any(math.isfinite(float(v)) for v in caps[a].values())]
    if not capped:
        pytest.skip("no capped rung in this cache")
    model = _model(graph).eval()
    ctx, order = _ctx(graph, capped[0]), _order(graph)
    plan = graph.tied_optimal_logit_plans[ALPHA][:1]
    with torch.no_grad():
        b = BatchedTeacherForced(graph, ctx, plan, order, with_full=False)
        lp = b.state_logps(model, graph, *model._encode(graph)).numpy()
        seq, _ = _sequential_logps(graph, model, ctx, plan, order, False)
    assert np.abs(lp[b.state_of["prefix"]] - seq).max() <= 1e-5


def test_feature_blocks_equal_the_sequential_blocks(graphs):
    """The 27 columns themselves, row for row, for every prefix state of one plan (the quantity the log-probs hide)."""
    from src.policy.gnn.batched_tf import BatchedTeacherForced
    from src.policy.tabular.reduced_features import partial_state_columns

    graph = _pick(graphs)["multi_plan"]
    ctx, order = _ctx(graph), _order(graph)
    plan = graph.tied_optimal_logit_plans[ALPHA][0]
    b = BatchedTeacherForced(graph, ctx, [plan], order, with_full=False)
    place = graph.task_logit_to_placement
    committed: Dict[int, Any] = {}
    worst = 0.0
    for t in order:
        s = int(b.state_of["prefix"][0, t])
        ref = partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed)
        got = b.ps[b.offs[s]:b.offs[s] + b.counts[s]].numpy()
        worst = max(worst, float(np.abs(ref - got).max()))
        committed[t] = tuple(int(v) for v in place[t][int(plan[t])])
    assert worst <= 1e-5, worst


def test_the_encode_is_prefix_free(graphs):
    """L5: the partial-state block never reaches the node embeddings."""
    graph = _pick(graphs)["typical"]
    model = _model(graph).eval()
    with torch.no_grad():
        graph.partial_state_edge_attr = None
        a = model._encode(graph)
        graph.partial_state_edge_attr = torch.randn(int(graph.edge_index.size(1)), model.partial_state_edge_dim)
        b = model._encode(graph)
        graph.partial_state_edge_attr = None
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("which", ["typical", "multi_plan", "group101_256"])
def test_causality_later_placements_do_not_reach_earlier_steps(graphs, which):
    """(b): perturb the placements of every task after position s of the commit order. Steps <= s keep bit-equal log-probs, and the
    gradient of the steps <= s with respect to the feature rows of the later steps is exactly zero (rows are scored independently:
    no pooling, attention or normalisation across steps)."""
    from src.policy.gnn.batched_tf import BatchedTeacherForced

    graph = _pick(graphs)[which]
    model = _model(graph).eval()
    ctx, order = _ctx(graph), _order(graph)
    n = int(graph.n_tasks)
    plan = [int(v) for v in graph.tied_optimal_logit_plans[ALPHA][0]]
    s = n // 2
    rng = np.random.RandomState(3)
    pert = list(plan)
    changed = 0
    for t in order[s + 1:]:
        c = len(graph.task_logit_to_placement[t])
        if c > 1:
            pert[t] = int((plan[t] + 1 + rng.randint(c - 1)) % c)
            changed += int(pert[t] != plan[t])
    assert changed > 0, "the perturbation changed nothing"
    early = order[:s + 1]
    with torch.no_grad():
        task_emb, platform_emb = model._encode(graph)
        lps = []
        for pl in (plan, pert):
            b = BatchedTeacherForced(graph, ctx, [pl], order, with_full=False)
            lps.append(b.state_logps(model, graph, task_emb, platform_emb)[torch.as_tensor(b.state_of["prefix"][0, early])])
    assert torch.equal(lps[0], lps[1]), float((lps[0] - lps[1]).abs().max())

    b = BatchedTeacherForced(graph, ctx, [pert], order, with_full=False)
    ps = b.ps.clone().requires_grad_(True)
    with torch.no_grad():
        task_emb, platform_emb = model._encode(graph)
    lp = b.state_logps(model, graph, task_emb, platform_emb, ps=ps)
    sid = b.state_of["prefix"][0]
    lp[torch.as_tensor(sid[early])].sum().backward()
    later_rows = np.concatenate([np.arange(b.offs[sid[t]], b.offs[sid[t]] + b.counts[sid[t]]) for t in order[s + 1:]])
    early_rows = np.concatenate([np.arange(b.offs[sid[t]], b.offs[sid[t]] + b.counts[sid[t]]) for t in early])
    assert float(ps.grad[later_rows].abs().max()) == 0.0
    assert float(ps.grad[early_rows].abs().max()) > 0.0


def test_batched_loss_equals_the_sequential_loss_in_eval_and_differs_by_dropout_in_train(graphs):
    graph = _pick(graphs)["multi_plan"]
    model = _model(graph)
    dev = torch.device("cpu")
    seq, bat = _loss_fn(False), _loss_fn(True)
    model.eval()
    with torch.no_grad():
        a, _ = seq(model, graph, dev, full_context_weight=1.0)
        b, _ = bat(model, graph, dev, full_context_weight=1.0)
    assert abs(float(a) - float(b)) <= 1e-5 * max(1.0, abs(float(a))), (float(a), float(b))
    model.train()
    torch.manual_seed(5)
    a, _ = seq(model, graph, dev, full_context_weight=1.0)
    torch.manual_seed(5)
    b, _ = bat(model, graph, dev, full_context_weight=1.0)
    # Different dropout masks (one per scored row instead of one per step over the whole edge tensor): same distribution, so
    # the two train-mode losses are estimates of one quantity and land close, but not equal.
    assert math.isfinite(float(b)) and abs(float(a) - float(b)) <= 0.5 * max(1.0, abs(float(a))), (float(a), float(b))
    b.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_flag_off_leaves_the_sequential_loss_unchanged(graphs):
    """Arm 1 byte-unchanged: the function with the flag off is bit-equal, in train mode and the same seed, to the function with the
    batched branch removed from its source."""
    graph = _pick(graphs)["multi_plan"]
    dev = torch.device("cpu")
    off, bare = _loss_fn(False), _loss_fn(False, strip_branch=True)
    outs = []
    for fn in (off, bare):
        model = _model(graph).train()
        torch.manual_seed(11)
        loss, _ = fn(model, graph, dev, full_context_weight=1.0)
        loss.backward()
        outs.append((loss.detach().clone(), [p.grad.clone() for p in model.parameters() if p.grad is not None]))
    assert torch.equal(outs[0][0], outs[1][0])
    assert all(torch.equal(x, y) for x, y in zip(outs[0][1], outs[1][1]))


def test_the_flag_is_wired_only_into_the_training_loss_and_the_sidecar():
    src = (REPO_ROOT / "src/notebooks/train_near_rtt.py").read_text()
    loss_body = src[src.index("def loss_tied_teacher_forced_ce"):src.index("class NearRttRankingLoss")]
    assert "NEAR_CFG.batched_tf" in loss_body
    assert src.count("NEAR_CFG.batched_tf") == 2  # the loss branch and the sidecar key
    for path in ("src/policy/gnn/prefix_serving.py", "src/policy/gnn/scheduler.py", "src/policy/gnn/seq_decode.py",
                 "src/policy/gnn/partial_state_edges.py"):
        assert "batched_tf" not in (REPO_ROOT / path).read_text(), path
    assert '"batched_teacher_forced"' in src
    assert '"batched_teacher_forced" in payload' in (REPO_ROOT / "src/executesimulation.py").read_text()


def _three_steps(graph, device, deterministic=True):
    from src.policy.gnn.batched_tf import batched_tf_loss

    if deterministic:
        torch.use_deterministic_algorithms(True)
    model = _model(graph, seed=3).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctx, order = _ctx(graph), _order(graph)
    data = graph.to(device) if hasattr(graph, "to") else graph
    torch.manual_seed(9)
    for _ in range(3):
        loss = batched_tf_loss(model, data, ctx, graph.tied_optimal_logit_plans[ALPHA], order, 1.0, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def test_batched_training_is_bit_identical_on_cpu(graphs):
    graph = _pick(graphs)["multi_plan"]
    torch.set_num_threads(1)
    a = _three_steps(graph, torch.device("cpu"))
    b = _three_steps(graph, torch.device("cpu"))
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_batched_training_is_bit_identical_on_cuda(graphs):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    graph = _pick(graphs)["multi_plan"]
    a = _three_steps(graph, torch.device("cuda"))
    b = _three_steps(graph, torch.device("cuda"))
    assert all(torch.equal(a[k], b[k]) for k in a)
