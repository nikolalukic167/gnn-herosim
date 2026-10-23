"""partial_state_v8 in the batched teacher-forced loss: S1-S3 (columns 30-32) are constant per row, so they live in the template and must equal the single definition
on every plan, mask and state; the per-step log-probs of the gnn (with the conv's static attribute), the twin and the MLP equal the sequential walk."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import test_batched_tf_link as T  # noqa: E402
from test_batched_tf_link import N_TASKS, REPLICAS, scenario  # noqa: E402

from src.policy.gnn import batched_tf as B  # noqa: E402
from src.policy.gnn import link_static as LS  # noqa: E402
from src.policy.tabular import reduced_features as RF  # noqa: E402

V8 = RF.PARTIAL_STATE_CONTRACT_V8


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", V8)
    monkeypatch.setenv("PARTIAL_STATE_LOAD_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_EXCHANGE_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_PEER_MASS", "1")


def v8_scenario(seed, monkeypatch):
    rnd, ctx, g, place = scenario(seed, monkeypatch)
    ctx.contract = V8
    return rnd, ctx, g, place


def test_columns_30_to_32_equal_the_definition_on_random_plans_masks_and_prefixes(monkeypatch):
    for seed in range(5):
        rnd, ctx, g, place = v8_scenario(seed, monkeypatch)
        S = B._Static(g, ctx)
        assert S.template.shape[1] == 33
        nonzero = 0
        for _ in range(6):
            plan = [rnd.randrange(len(place[t])) for t in range(N_TASKS)]
            for M in (np.tril(np.ones((N_TASKS, N_TASKS)), -1), 1.0 - np.eye(N_TASKS), np.zeros((N_TASKS, N_TASKS))):
                out = B._blocks(S, ctx, g, plan, M)
                assert out.shape == (S.R, 33)
                for t in range(N_TASKS):
                    committed = {j: tuple(place[j][int(plan[j])]) for j in range(N_TASKS) if M[t, j] > 0}
                    want = RF.partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed).astype(np.float32)
                    got = out[S.start[t]:S.start[t] + S.C[t]]
                    assert np.array_equal(got[:, 30:33], want[:, 30:33])
                    nonzero += int((got[:, 30:33] != 0).any(1).sum())
                    assert np.abs(got[:, :27] - want[:, :27]).max() <= 1e-5
        assert nonzero > 0  # not vacuous: the static block is live in the loss input


def test_the_static_block_is_the_same_in_every_state_of_a_batch(monkeypatch):
    rnd, ctx, g, place = v8_scenario(9, monkeypatch)
    plans = [[rnd.randrange(len(place[t])) for t in range(N_TASKS)] for _ in range(3)]
    b = B.BatchedTeacherForced(g, ctx, plans, list(range(N_TASKS)), with_full=True)
    assert b.ps.shape[1] == 33
    for name in ("prefix", "full"):
        for t in range(N_TASKS):
            seen = {b.ps[b.offs[int(b.state_of[name][k, t])]:b.offs[int(b.state_of[name][k, t])] + b.counts[int(b.state_of[name][k, t])], 30:33].numpy().tobytes()
                    for k in range(len(plans))}
            assert len(seen) == 1, (name, t)


def _graph_with_static(g, ctx):
    g = T._model_graph(g)
    g.link_static_attr = LS.link_static_attr_from_ctx(g, ctx)
    return g


@pytest.mark.parametrize("arm", ["gnn", "twin", "mlp"])
def test_batched_log_probs_equal_the_sequential_walk_with_the_static_block(arm, monkeypatch):
    from src.policy.gnn.arm_models import TaskPlacementMLPSame
    from src.policy.gnn.gnn_model import TaskPlacementGNN
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn

    if arm == "twin":
        monkeypatch.setenv("GNN_DISABLE_MESSAGE_PASSING", "1")
    else:
        monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)
    rnd, ctx, g, place = v8_scenario(21, monkeypatch)
    g = _graph_with_static(g, ctx)
    dims = dict(task_feature_dim=5, platform_feature_dim=7, hidden_dim=16, edge_dim=5, task_type_onehot_dim=4, partial_state_edge_dim=33)
    torch.manual_seed(0)
    if arm == "mlp":
        model = TaskPlacementMLPSame(**dims)
    elif arm == "gnn":
        model = TaskPlacementGNN(**dims, embedding_dim=16, num_layers=2, mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True, mp_link_static_attr=True)
    else:
        model = TaskPlacementGNN(**dims, embedding_dim=16, num_layers=2, mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True)
    model = model.eval()
    plans = [[rnd.randrange(len(place[t])) for t in range(N_TASKS)] for _ in range(3)]
    b = B.BatchedTeacherForced(g, ctx, plans, list(range(N_TASKS)), with_full=True)
    with torch.no_grad():
        te, pe = model._encode(g)
        lp = b.state_log_softmax(model, g, te, pe)
        got_pre, got_full = b.step_logps(lp, "prefix").numpy(), b.step_logps(lp, "full").numpy()
        score = make_partial_state_score_fn(model, g, ctx, cache_encode=False)
        for k, plan in enumerate(plans):
            for t in range(N_TASKS):
                for name, got, committed in (("prefix", got_pre, {j: tuple(place[j][plan[j]]) for j in range(t)}),
                                             ("full", got_full, {j: tuple(place[j][plan[j]]) for j in range(N_TASKS) if j != t})):
                    seq = torch.log_softmax(score(t, committed), dim=-1)[plan[t]].item()
                    assert abs(got[k, t] - seq) <= 1e-5, (arm, name, k, t, got[k, t], seq)


def test_the_conv_attribute_changes_the_gnn_encoding_and_leaves_the_twin_alone(monkeypatch):
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)
    rnd, ctx, g, place = v8_scenario(4, monkeypatch)
    g = _graph_with_static(g, ctx)
    kw = dict(task_feature_dim=5, platform_feature_dim=7, hidden_dim=16, edge_dim=5, task_type_onehot_dim=4, partial_state_edge_dim=33, embedding_dim=16, num_layers=2,
              mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True, mp_link_static_attr=True)
    torch.manual_seed(0)
    m = TaskPlacementGNN(**kw).eval()
    with torch.no_grad():
        a = m._encode(g)
        g2 = _graph_with_static(g.clone(), ctx)
        g2.link_static_attr = g2.link_static_attr + 1.0
        b2 = m._encode(g2)
    assert not torch.equal(a[0], b2[0])  # the conv reads the attribute
    bad = g.clone()
    bad.link_static_attr = bad.link_static_attr[:-1]
    with pytest.raises(Exception):
        m._encode(bad)
