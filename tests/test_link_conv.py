"""live_link_features_v1 variant (b): the re-encode-per-placement GNN (src/policy/gnn/link_conv.py)."""
import math
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import test_batched_tf_link as TB  # noqa: E402
from src.placement import link_features as LF  # noqa: E402
from src.policy.gnn import batched_tf as B  # noqa: E402
from src.policy.gnn import link_conv as LC  # noqa: E402
from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402
from src.policy.tabular import reduced_features as RF  # noqa: E402

N = TB.N_TASKS


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", TB.V7)
    monkeypatch.setenv("PARTIAL_STATE_LOAD_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_EXCHANGE_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_PEER_MASS", "1")
    monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)


def model_for(g):
    g = TB._model_graph(g)
    torch.manual_seed(0)
    m = TaskPlacementGNN(task_feature_dim=5, platform_feature_dim=7, embedding_dim=16, hidden_dim=16, num_layers=2, edge_dim=5, task_type_onehot_dim=4,
                         partial_state_edge_dim=30, mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True, mp_link_edge_conv=True, mp_peer_edges=False).eval()
    return g, m


def random_states(rnd, place, count):
    out = []
    for _ in range(count):
        plan = [rnd.randrange(len(place[t])) for t in range(N)]
        cs = [j for j in range(N) if rnd.random() < rnd.choice([0.2, 0.5, 0.9])]
        out.append((plan, cs))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_rows_equal_the_definition_for_every_task_with_its_own_placement_left_out(seed, monkeypatch):
    rnd, ctx, g, place = TB.scenario(seed, monkeypatch)
    LS = LC.static_of(g, ctx)
    states = random_states(rnd, place, 10)
    S = LS.S
    cm = np.zeros((len(states), N))
    nd = np.zeros((len(states), N), dtype=np.int64)
    for s, (plan, cs) in enumerate(states):
        for j in cs:
            cm[s, j] = 1.0
        nd[s] = S.row_node[S.start + np.asarray(plan)]
    rows = LC.link_rows(LS, torch.from_numpy(cm), torch.from_numpy(nd)).numpy().astype(np.float32)
    worst = 0.0
    for s, (plan, cs) in enumerate(states):
        for t in range(N):
            committed = {j: tuple(place[j][plan[j]]) for j in sorted(cs) if j != t}  # t's own placement is not in its prefix
            want = RF.partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed).astype(np.float32)[:, 27:30]
            got = rows[s, S.start[t]:S.start[t] + S.C[t]]
            assert np.array_equal(got, want), (s, t, got, want)
            worst = max(worst, float(np.abs(got).max()))
    assert worst > 0.0


def test_rows_to_edges_aligns_forward_and_reverse_rows_with_the_candidates(monkeypatch):
    rnd, ctx, g, place = TB.scenario(3, monkeypatch)
    g, m = model_for(g)
    LS = LC.static_of(g, ctx)
    plan = [rnd.randrange(len(place[t])) for t in range(N)]
    committed = {j: tuple(place[j][plan[j]]) for j in range(0, N, 2)}
    block = LC.link_edge_block(g, ctx, committed)
    assert block.shape == (g.edge_index.size(1), 3) and block.dtype == torch.float32
    cm, nd = LS.placement_arrays(committed)
    rows = LC.link_rows(LS, torch.from_numpy(cm)[None], torch.from_numpy(nd)[None])[0].to(torch.float32)
    ei = g.edge_index
    n_t = int(g.n_tasks)
    seen = 0
    for e in range(ei.size(1)):
        a, b = int(ei[0, e]), int(ei[1, e])
        t, plat = (a, b - n_t) if a < n_t else (b, a - n_t)  # forward rows are task -> platform, reverse rows platform -> task
        k = [int(p) for p in range(len(REPLICAS_POS(g, t)))][REPLICAS_POS(g, t).index(plat)]
        assert torch.equal(block[e], rows[LS.start[t] + k])
        seen += 1
    assert seen == ei.size(1)


def REPLICAS_POS(g, t):
    return [int(g.queue_key_to_platform_meta[k]["platform_pos"]) for k in g.task_logit_to_queue_key[t]]


def test_the_closure_encodes_once_per_distinct_committed_set_and_matches_a_manual_walk(monkeypatch):
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn, refresh_partial_state_edge_attr

    rnd, ctx, g, place = TB.scenario(4, monkeypatch)
    g, m = model_for(g)
    calls = []
    orig = m._encode
    m._encode = lambda d: (calls.append(1), orig(d))[1]
    score = make_partial_state_score_fn(m, g, ctx)
    plan = [rnd.randrange(len(place[t])) for t in range(N)]
    with torch.no_grad():
        for t in range(N):
            committed = {j: tuple(place[j][plan[j]]) for j in range(t)}
            got = score(t, committed)
            setattr(g, LC.ATTR, LC.link_edge_block(g, ctx, committed))
            te, pe = orig(g)
            refresh_partial_state_edge_attr(g, ctx, t, committed)
            want = m._score(te, pe, g)[t]
            assert torch.equal(got, want)
        n_first = len(calls)
        for t in range(N):  # the same committed sets again: no new encode
            score(t, {j: tuple(place[j][plan[j]]) for j in range(t)})
    assert n_first == N and len(calls) == N


def test_a_committed_task_is_scored_with_its_own_placement_excluded_from_its_conv_rows(monkeypatch):
    rnd, ctx, g, place = TB.scenario(6, monkeypatch)
    g, m = model_for(g)
    LS = LC.static_of(g, ctx)
    plan = [rnd.randrange(len(place[t])) for t in range(N)]
    full = {j: tuple(place[j][plan[j]]) for j in range(N)}
    for t in range(N):
        without = {j: p for j, p in full.items() if j != t}
        a = LC.link_edge_block(g, ctx, full)
        b = LC.link_edge_block(g, ctx, without)
        rows_t = [e for e in range(g.edge_index.size(1)) if (int(g.edge_index[0, e]) == t) or (int(g.edge_index[1, e]) == t)]
        assert torch.equal(a[rows_t], b[rows_t]), t


def test_the_big_graph_encode_equals_the_per_copy_encodes(monkeypatch):
    rnd, ctx, g, place = TB.scenario(8, monkeypatch)
    g, m = model_for(g)
    LS = LC.static_of(g, ctx)
    plans = [[rnd.randrange(len(place[t])) for t in range(N)] for _ in range(3)]
    blocks = []
    for plan in plans:
        committed = {j: tuple(place[j][plan[j]]) for j in range(0, N, 2)}
        blocks.append(LC.link_edge_block(g, ctx, committed))
    attr = torch.stack(blocks)
    big = LC.replicate_graph(g, 3, attr)
    with torch.no_grad():
        te, pe = m._encode(big)
        for s, blk in enumerate(blocks):
            setattr(g, LC.ATTR, blk)
            t1, p1 = m._encode(g)
            assert torch.allclose(te[s * N:(s + 1) * N], t1, atol=1e-6, rtol=1e-6)
            assert torch.allclose(pe[s * len(TB.REPLICAS):(s + 1) * len(TB.REPLICAS)], p1, atol=1e-6, rtol=1e-6)
    assert not torch.equal(blocks[0], blocks[1]) or not torch.equal(blocks[1], blocks[2])


def test_the_batched_loss_equals_the_sequential_closure(monkeypatch):
    import torch.nn.functional as F
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn

    rnd, ctx, g, place = TB.scenario(9, monkeypatch)
    g, m = model_for(g)
    plans = [[rnd.randrange(len(place[t])) for t in range(N)] for _ in range(3)]
    order = list(range(N))
    w = 0.7
    with torch.no_grad():
        got = LC.link_conv_loss(m, g, ctx, plans, order, w, torch.device("cpu"))
        score = make_partial_state_score_fn(m, g, ctx, cache_encode=False)
        pre, full = [], []
        for plan in plans:
            lp_pre = torch.zeros(())
            lp_full = torch.zeros(())
            for t in order:
                c_pre = {j: tuple(place[j][plan[j]]) for j in order[:t]}
                lp_pre = lp_pre + F.log_softmax(score(t, c_pre), -1)[plan[t]]
                c_full = {j: tuple(place[j][plan[j]]) for j in order if j != t}
                lp_full = lp_full + F.log_softmax(score(t, c_full), -1)[plan[t]]
            pre.append(lp_pre)
            full.append(lp_full)
        want = -torch.logsumexp(torch.stack(pre), 0) + w * -torch.logsumexp(torch.stack(full), 0)
    assert abs(float(got) - float(want)) <= 1e-4 * max(1.0, abs(float(want))), (float(got), float(want))


def test_the_loss_has_gradients_in_train_mode_and_the_conv_reads_the_link_block(monkeypatch):
    rnd, ctx, g, place = TB.scenario(10, monkeypatch)
    g, m = model_for(g)
    m.train()
    plans = [[rnd.randrange(len(place[t])) for t in range(N)]]
    loss = LC.link_conv_loss(m, g, ctx, plans, list(range(N)), 0.5, torch.device("cpu"))
    loss.backward()
    w = m.bip_convs[0].message_mlp[0].weight.grad
    assert w is not None and float(w[:, -3:].abs().sum()) > 0.0, "the new conv columns get no gradient"


def test_a_context_rebuilt_from_json_style_tables_gives_bitwise_identical_conv_blocks(monkeypatch):
    rnd, ctx, g, place = TB.scenario(12, monkeypatch)
    # the cache pickles hops as lists / the live fabric hands tuples: the conv block must not care (tests/test_link_features.py covers live vs stored flows)
    live = LC.static_of(g, ctx)
    ctx2 = RF.PartialStateContext(
        node_caps=ctx.node_caps, demand=ctx.demand, node_of=ctx.node_of, task_type_index=ctx.task_type_index, parents=ctx.parents, route_hops_bneck=ctx.route_hops_bneck,
        payload_bytes=0.0, transfer_norm=0.0, node_rank=ctx.node_rank, ingress_links=ctx.ingress_links, core_links=ctx.core_links, peer_pairs=ctx.peer_pairs, peer_norm=1.0,
        node_exchange=ctx.node_exchange, cand_nodes=ctx.cand_nodes, contract=TB.V7, backlog_s=ctx.backlog_s, service_s=ctx.service_s,
        link_counts=dict(ctx.link_counts), link_routes={k: [list(h) for h in v] for k, v in ctx.link_routes.items()})
    import copy

    g2 = copy.copy(g)
    g2._link_conv_static = None
    g2.partial_state_ctx = None
    cache = LC.LinkStatic(g2, ctx2)
    plan = [rnd.randrange(len(place[t])) for t in range(N)]
    for _ in range(6):
        committed = {j: tuple(place[j][plan[j]]) for j in range(N) if rnd.random() < 0.6}
        cm, nd = live.placement_arrays(committed)
        a = LC.link_rows(live, torch.from_numpy(cm)[None], torch.from_numpy(nd)[None])
        b = LC.link_rows(cache, torch.from_numpy(cm)[None], torch.from_numpy(nd)[None])
        assert torch.equal(a, b)
