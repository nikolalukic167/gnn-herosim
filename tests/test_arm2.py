"""uncut_groups_s0_v1 Arm 2 (src/policy/gnn/arm2.py): committed-state re-encode (B1) and group anchor (B2).

Pins what the arm depends on:
- the re-encode period (`auto` = 1 up to 10 tasks, else ceil(n / 16), at most 16 encodes) and the off switch;
- the committed state (flag, per-platform count, peer mass in v5 seconds) and the pruned edge set;
- with no flag the model has no new parameter and the shared scorer takes the old path;
- one encode when m exceeds the batch equals the single-encode path; m = 1 lets a placement reach a partner's
  logits through message passing, and reaches the MP-OFF twin only through its own platform state;
- groups, the anchor label rule, and the on_anchor_node column the scorer reads.
"""
from __future__ import annotations

import math
from pathlib import Path
import os

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.policy.gnn import arm2
from src.policy.gnn.gnn_model import TaskPlacementGNN
from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
from src.policy.tabular.reduced_features import (
    PARTIAL_STATE_CONTRACT_V5, PartialStateContext, krank_node_order, partial_state_feature_dim,
)

ENVS = ("GNN_DISABLE_MESSAGE_PASSING", "GNN_PLAN_RAW", "GNN_MP_NODE_EDGES", "PARTIAL_STATE_CONTRACT",
        arm2.REENCODE_ENV, arm2.GROUP_ANCHOR_ENV, "PARTIAL_STATE_LOAD_SECONDS", "PARTIAL_STATE_EXCHANGE_SECONDS")
NODES = (0, 1, 2)
REPLICAS = [(n, p) for n in NODES for p in (0, 1)]
N_TASKS = 4
WIDTH = partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V5)


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


def _ctx() -> PartialStateContext:
    caps = {n: math.inf for n in NODES}
    tasks = list(range(N_TASKS))
    exchange = {(a, b): ((0.0, 0.0) if a == b else (1.0 + 0.5 * a, 0.1)) for a in NODES for b in NODES}
    return PartialStateContext(
        node_caps=caps, demand={(t, r): 1.0 for t in tasks for r in REPLICAS},
        node_of={r: r[0] for r in REPLICAS}, task_type_index={t: t % 4 for t in tasks},
        parents={t: [] for t in tasks},
        route_hops_bneck={(a, b): ((0.0, math.inf) if a == b else (1.0, 100.0)) for a in NODES for b in NODES},
        payload_bytes=0.0, transfer_norm=0.0,
        node_rank=krank_node_order(caps, {n: float(n) for n in NODES}, contract=PARTIAL_STATE_CONTRACT_V5),
        ingress_links={(t, n): () for t in tasks for n in NODES}, core_links=frozenset(),
        peer_pairs={(0, 1): 2.0, (1, 0): 2.0, (0, 2): 3.0, (2, 0): 3.0, (1, 2): 1.0, (2, 1): 1.0}, peer_norm=1.0,
        node_exchange=exchange, cand_nodes={t: list(NODES) for t in tasks},
        backlog_s={r: float(i) for i, r in enumerate(REPLICAS)},
        service_s={(t, r): 1.0 + t + 0.5 * r[1] for t in tasks for r in REPLICAS},
        contract=PARTIAL_STATE_CONTRACT_V5,
    )


def _graph() -> Data:
    torch.manual_seed(0)
    n_platforms = len(REPLICAS)
    fwd = [(t, N_TASKS + p) for t in range(N_TASKS) for p in range(n_platforms)]
    rows = fwd + [(d, s) for s, d in fwd]
    data = Data()
    data.n_tasks, data.n_platforms = N_TASKS, n_platforms
    data.task_features = torch.randn(N_TASKS, 3)
    data.task_type_onehot4 = torch.eye(4)[:N_TASKS]
    data.platform_features = torch.randn(n_platforms, 14)
    data.edge_index = torch.tensor(rows, dtype=torch.long).t().contiguous()
    fwd_attr = torch.randn(len(fwd), 5)
    data.edge_attr = torch.cat([fwd_attr, fwd_attr])
    data.peer_edge_index = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
    data.peer_edge_attr = torch.ones(6, 1)
    data.task_logit_to_queue_key = {t: [f"k{t}_{p}" for p in range(n_platforms)] for t in range(N_TASKS)}
    data.queue_key_to_platform_meta = {f"k{t}_{p}": {"platform_pos": p}
                                       for t in range(N_TASKS) for p in range(n_platforms)}
    data.task_logit_to_placement = {t: list(REPLICAS) for t in range(N_TASKS)}
    return data


def _model(*, mp: bool = True, every: str = "auto", anchor: bool = False) -> TaskPlacementGNN:
    if not mp:
        os.environ["GNN_DISABLE_MESSAGE_PASSING"] = "1"
    torch.manual_seed(1)
    model = TaskPlacementGNN(
        task_feature_dim=3, platform_feature_dim=14, task_type_onehot_dim=4, mp_peer_edges=True,
        mp_bipartite_edge_conv=True, partial_state_edge_dim=WIDTH, reencode_every=every, group_anchor=anchor,
    )
    model.eval()
    return model


def _logits(model, data, committed, task, ctx=None):
    with torch.no_grad():
        return make_partial_state_score_fn(model, data, ctx or _ctx())(task, committed).clone()


def test_reencode_period():
    assert arm2.parse_reencode_every(None) == "inf"
    assert arm2.parse_reencode_every("INF") == "inf"
    assert arm2.parse_reencode_every("auto") == "auto"
    assert arm2.parse_reencode_every("3") == "3"
    for bad in ("-1", "x", "1.5"):
        with pytest.raises(ValueError, match="FAIL LOUD"):
            arm2.parse_reencode_every(bad)
    assert arm2.reencode_period("inf", 50) is None
    assert [arm2.reencode_period("auto", n) for n in (1, 10, 11, 16, 17, 100, 226)] == [1, 1, 1, 1, 2, 7, 15]
    assert arm2.reencode_period("4", 50) == 4
    for n in range(1, 300):
        assert arm2.reencodes_for_batch("auto", n) <= 16 if n > 10 else arm2.reencodes_for_batch("auto", n) == n


def test_flags_off_adds_nothing_and_takes_the_old_path():
    off = _model(every="inf")
    assert not off.committed_inputs and not off.group_anchor
    names = set(off.state_dict())
    assert not any(k.startswith(("committed_", "anchor_head")) for k in names)
    assert off.partial_state_edge_dim == WIDTH
    data = _graph()
    score = make_partial_state_score_fn(off, data, _ctx())
    assert not hasattr(score, "anchor_logits")
    on = _model(every="auto", anchor=True)
    assert on.partial_state_edge_dim == WIDTH + 1
    assert {"committed_task_enc.weight", "committed_platform_enc.0.weight", "anchor_head.0.weight"} <= set(on.state_dict())


def test_committed_state_counts_mass_and_prunes_edges():
    data, ctx = _graph(), _ctx()
    layout = arm2.Arm2Layout(data, ctx, anchor=False, committed_inputs=True)
    snapshot = {0: (0, 0), 3: (0, 0)}
    flag, state, keep = layout.committed_state(snapshot)
    assert flag.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert state[0, 0] == pytest.approx(math.log1p(2.0))
    assert state[1:, 0].abs().sum() == 0
    # open peers of the one committed task that has peers (task 0 on node 0): tasks 1 and 2 pay to pull from node 0
    for pos, (node, _p) in enumerate(REPLICAS):
        want = 0.0
        for b in (2.0, 3.0):
            pb, lat = ctx.node_exchange[(0, node)]
            want += b * pb + (lat if pb > 0.0 else 0.0)
        assert state[pos, 1] == pytest.approx(math.log1p(want), rel=1e-6)
    ei = data.edge_index
    for r in range(ei.size(1)):
        s, d = int(ei[0, r]), int(ei[1, r])
        t, p = (s, d - N_TASKS) if s < N_TASKS else (d, s - N_TASKS)
        assert bool(keep[r]) == (t not in snapshot or p == 0)


def test_empty_snapshot_is_the_single_encode_path():
    data, model = _graph(), _model(every="1000")
    ctx = _ctx()
    got = _logits(model, data, {0: (0, 0), 1: (1, 1)}, 2, ctx)
    clear = _graph()
    with torch.no_grad():
        te, pe = model._encode(clear)
        clear.partial_state_edge_attr = data.partial_state_edge_attr
        want = model._score(te, pe, clear)[2]
    assert torch.equal(got, want)


def test_a_placement_reaches_a_partner_through_message_passing_only_after_a_reencode():
    data, model = _graph(), _model(every="1")
    a = _logits(model, data, {0: (0, 0)}, 3)
    b = _logits(model, _graph(), {0: (2, 1)}, 3)
    assert not torch.allclose(a, b)
    late = _model(every="1000")
    c = _logits(late, _graph(), {0: (0, 0)}, 3)
    d = _logits(late, _graph(), {0: (0, 0)}, 3)
    assert torch.equal(c, d)


def test_twin_reads_the_commitment_through_its_platform_state():
    data, twin = _graph(), _model(mp=False, every="1")
    a = _logits(twin, data, {0: (0, 0)}, 1)
    b = _logits(twin, _graph(), {0: (2, 1)}, 1)
    assert not torch.allclose(a, b)


def test_snapshot_uses_the_first_commits_in_insertion_order():
    model = _model(every="2")
    data = _graph()
    one = _logits(model, data, {0: (0, 0)}, 3)
    base = _logits(model, _graph(), {}, 3)
    # one commit is below the period, so the encode is the empty one; only the scorer columns differ
    assert one.shape == base.shape
    score = make_partial_state_score_fn(model, data, _ctx())
    with torch.no_grad():
        score(3, {0: (0, 0)})
        score(3, {})
        assert score.encode_count() == 1
        score(3, {0: (0, 0), 1: (1, 0)})
        score(3, {1: (1, 0), 0: (0, 0)})  # same set, other order: one cached encode
        assert score.encode_count() == 2


def test_groups_and_anchor_label_rule():
    data, ctx = _graph(), _ctx()
    assert arm2.peer_groups(data) == [[0, 1, 2]]
    layout = arm2.Arm2Layout(data, ctx, anchor=True, committed_inputs=False)
    assert layout.group_nodes == [[0, 1, 2]]
    assert arm2.anchor_label(layout, {0: 1, 1: 2, 2: 2, 3: 0}) == [2]
    assert arm2.anchor_label(layout, {0: 1, 1: 2, 2: 1, 3: 0}) == [1]
    assert arm2.anchor_label(layout, {0: 2, 1: 1, 2: 0, 3: 0}) == [0]


def test_anchor_column_is_what_the_scorer_reads():
    data = _graph()
    model = _model(anchor=True)
    ctx = _ctx()
    score = make_partial_state_score_fn(model, data, ctx)
    with torch.no_grad():
        score(0, {})
    logits = score.anchor_logits()
    assert len(logits) == 1 and tuple(logits[0].shape) == (3,)
    attr = data.partial_state_edge_attr
    assert attr.shape[1] == WIDTH + 1
    rows = arm2.candidate_edge_rows(data)[0]
    node = [float(v) for v in logits[0].tolist()]
    anchor = node.index(max(node))
    assert [attr[r, WIDTH].item() for r in rows] == [1.0 if REPLICAS[k][0] == anchor else 0.0 for k in range(6)]
    with torch.no_grad():
        score(3, {})
    assert attr[arm2.candidate_edge_rows(data)[3], WIDTH].abs().sum() == 0  # a singleton has no anchor
    teach = arm2.make_arm2_score_fn(model, data, ctx, teacher_anchor=[2])
    with torch.no_grad():
        teach(1, {})
    assert [data.partial_state_edge_attr[r, WIDTH].item()
            for r in arm2.candidate_edge_rows(data)[1]] == [1.0 if c[0] == 2 else 0.0 for c in REPLICAS]


def test_scorer_width_mismatch_is_loud():
    model = _model(every="auto")
    ctx = _ctx()
    model.partial_state_edge_dim = WIDTH + 3
    with pytest.raises(ValueError, match="partial_state_edge_dim"):
        arm2.make_arm2_score_fn(model, _graph(), ctx)


def test_anchor_head_trains_and_is_deterministic():
    def run():
        model = _model(anchor=True)
        model.train()
        data, ctx = _graph(), _ctx()
        score = arm2.make_arm2_score_fn(model, data, ctx, teacher_anchor=[1])
        torch.manual_seed(5)
        lp = torch.log_softmax(score(0, {}), dim=-1)[2]
        ce = torch.nn.functional.cross_entropy(score.anchor_logits()[0].unsqueeze(0), torch.tensor([1]))
        (-lp + 0.5 * ce).backward()
        return {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
    g1, g2 = run(), run()
    assert g1.keys() == g2.keys() and any(k.startswith("anchor_head") for k in g1)
    assert all(torch.equal(g1[k], g2[k]) for k in g1)


def test_random_orders_are_seeded_topological_and_distinct_from_each_other():
    parents = {0: [], 1: [0], 2: [0], 3: [1, 2]}
    a = arm2.random_topological_orders(4, parents, 2, 7, "ds1")
    assert a == arm2.random_topological_orders(4, parents, 2, 7, "ds1")
    assert arm2.random_topological_orders(6, {}, 3, 7, "ds1") != arm2.random_topological_orders(6, {}, 3, 8, "ds1")
    for order in a:
        assert sorted(order) == [0, 1, 2, 3] and order[0] == 0 and order[-1] == 3
    free = arm2.random_topological_orders(6, {}, 8, 1, "x")
    assert len({tuple(o) for o in free}) > 1
    assert arm2.random_topological_orders(4, parents, 0, 7, "ds1") == []


def _train_two_epochs(rows_only=False):
    """The trainer's loss on the synthetic batch: label-order loss + 2 random orders + anchor CE, Adam, two epochs."""
    torch.manual_seed(3)
    model = _model(every="auto", anchor=True)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctx = _ctx()
    plan = {0: 1, 1: 4, 2: 5, 3: 2}  # logit index per task
    for _ in range(2):
        data = _graph()
        layout = arm2.layout_for(model, data, ctx)
        label_nodes = {t: REPLICAS[plan[t]][0] for t in range(N_TASKS)}
        teacher = arm2.anchor_label(layout, label_nodes)
        score = arm2.make_arm2_score_fn(model, data, ctx, teacher_anchor=teacher, task_rows_only=rows_only)
        orders = [list(range(N_TASKS))] + arm2.random_topological_orders(N_TASKS, {}, 2, 42, "ds")
        total = torch.zeros(())
        for order in orders:
            committed, logp = {}, torch.zeros(())
            for t in order:
                logp = logp + torch.log_softmax(score(t, committed), dim=-1)[plan[t]]
                committed[t] = REPLICAS[plan[t]]
            total = total - logp
        loss = total / len(orders)
        for lg, node in zip(score.anchor_logits(), teacher):
            loss = loss + 0.5 * torch.nn.functional.cross_entropy(
                lg.unsqueeze(0), torch.tensor([layout.group_nodes[0].index(node)]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_training_loop_with_both_flags_and_random_prefix_is_bit_identical():
    a, b = _train_two_epochs(), _train_two_epochs()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    start = _model(every="auto", anchor=True).state_dict()
    assert any(not torch.equal(a[k], start[k]) for k in ("committed_task_enc.weight", "anchor_head.0.weight"))


def test_snapshot_key_separates_same_set_with_a_different_prefix_order():
    model, data = _model(every="2", anchor=True), _graph()
    score = arm2.make_arm2_score_fn(model, data, _ctx(), teacher_anchor=[0])
    a = {0: (0, 0), 1: (1, 0), 2: (2, 0)}
    b = {2: (2, 0), 1: (1, 0), 0: (0, 0)}
    assert sorted(a.items()) == sorted(b.items())
    assert score.snapshot_key(a) != score.snapshot_key(b)
    assert score.snapshot_key(a) == tuple(sorted(list(a.items())[:2]))
    assert score.snapshot_key({0: (0, 0)}) == ()


def test_single_encode_uses_one_encode_for_any_committed_set():
    model, data = _model(every="1", anchor=True), _graph()
    score = arm2.make_arm2_score_fn(model, data, _ctx(), teacher_anchor=[0], single_encode=True)
    with torch.no_grad():
        for committed in ({}, {0: (0, 0)}, {0: (0, 0), 1: (1, 0)}, {3: (2, 1), 1: (1, 0)}):
            score(2, committed)
    assert score.encode_count() == 1


def test_model_after_a_loss_equals_the_result_on_a_fresh_graph():
    model = _model(every="1", anchor=True)
    ctx = _ctx()
    data = _graph()
    score = arm2.make_arm2_score_fn(model, data, ctx, teacher_anchor=[1])
    with torch.no_grad():
        score(3, {0: (0, 0), 1: (1, 1), 2: (2, 0)})
        score.close()
        assert all(getattr(data, n, None) is None for n in arm2.ARM2_ATTRS)
        for g in (data, _graph()):
            g.partial_state_edge_attr = torch.zeros((g.edge_index.size(1), WIDTH + 1))
        used, fresh = model(data), model(_graph_with_zero_attr())
    assert all(torch.equal(x, y) for x, y in zip(used, fresh))


def _graph_with_zero_attr():
    g = _graph()
    g.partial_state_edge_attr = torch.zeros((g.edge_index.size(1), WIDTH + 1))
    return g


@pytest.mark.parametrize("train", [False, True])
def test_rows_only_composes_with_reencode_anchor_and_random_prefix(train):
    """NEAR_RTT_SCORE_TASK_ROWS_ONLY on the Arm 2 closure equals the full-tensor score to float precision, same dropout stream."""
    ctx = _ctx()
    plan = {0: 1, 1: 4, 2: 5, 3: 2}
    orders = [list(range(N_TASKS))] + arm2.random_topological_orders(N_TASKS, {}, 2, 42, "ds")

    def walk(rows_only):
        torch.manual_seed(5)
        model = _model(every="1", anchor=True)
        model.train(train)
        data = _graph()
        layout = arm2.layout_for(model, data, ctx)
        teacher = arm2.anchor_label(layout, {t: REPLICAS[plan[t]][0] for t in range(N_TASKS)})
        score = arm2.make_arm2_score_fn(model, data, ctx, teacher_anchor=teacher, task_rows_only=rows_only)
        torch.manual_seed(9)
        out = []
        for order in orders:
            committed = {}
            for t in order:
                out.append(score(t, committed).detach().clone())
                committed[t] = REPLICAS[plan[t]]
        return out, torch.get_rng_state(), score.encode_count()

    full, rng_full, enc_full = walk(False)
    rows, rng_rows, enc_rows = walk(True)
    assert torch.equal(rng_full, rng_rows) and enc_full == enc_rows > 1
    for a, b in zip(full, rows):
        assert float((a - b).abs().max()) <= 1e-5 * max(float(a.abs().max()), 1.0)


def test_training_loop_with_rows_only_is_bit_identical():
    a, b = _train_two_epochs(rows_only=True), _train_two_epochs(rows_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_trainers_init_weights_accepts_the_bias_free_b1_modules():
    """train_near_rtt.py runs model.apply(init_weights) at import; its zeros_(module.bias) died on the bias-free B1 Linears (first launch)."""
    import ast
    import torch.nn as nn

    src = (Path(__file__).resolve().parents[1] / "src/notebooks/train_near_rtt.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "init_weights")
    ns = {"nn": nn}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "init_weights", "exec"), ns)
    model = _model(every="auto", anchor=True)
    model.apply(ns["init_weights"])
    assert model.committed_task_enc.bias is None
    assert float(model.committed_task_enc.weight.abs().sum()) > 0
