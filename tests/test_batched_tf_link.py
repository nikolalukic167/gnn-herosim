"""partial_state_v7 in the batched teacher-forced loss: the vectorised link block (batched_tf._link_block) against the single definition
(reduced_features.partial_state_columns -> link_features.edge_columns + pending_counts), on random plans, random commit masks, prefix states and the refine ("every other task committed") states.

Columns 27-29 are compared bitwise (float32), the other 27 within 1e-5 as test_batched_tf does. Self-contained: a real SimPy fair-share fabric with random flows, five servers behind two core routers, nine tasks with random candidate sets,
symmetric and one-directional peer pairs, finite node caps."""
import math
import os
import random
import sys

import numpy as np
import pytest
import simpy
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.placement import link_features as LF  # noqa: E402
from src.placement.network_fabric import NetworkFabric, directed_hops_from_topology, link_key  # noqa: E402
from src.policy.gnn import batched_tf as B  # noqa: E402
from src.policy.tabular import reduced_features as RF  # noqa: E402

V7 = RF.PARTIAL_STATE_CONTRACT_V7
MIB = 1024 ** 2
SERVERS = ["s0", "s1", "s2", "s3", "s4"]
NAMES = dict(enumerate(SERVERS))
REPLICAS = [(n, 10 * n + k) for n in range(5) for k in range(2)]
N_TASKS = 9


def topology():
    links, routes = {}, {}
    access = {"s0": (4.0, 75.0), "s1": (10.0, 10.0), "s2": (20.0, 20.0), "s3": (4.0, 75.0), "s4": (10.0, 20.0)}
    cores = {"s0": "core_0", "s1": "core_0", "s2": "core_0", "s3": "core_1", "s4": "core_1"}
    for s, (o, i) in access.items():
        links[link_key(s, cores[s])] = {"bandwidth_mbps": min(o, i), "latency": 0.01, "access_node": s, "bandwidth_out_mbps": o, "bandwidth_in_mbps": i}
    links[link_key("core_0", "core_1")] = {"bandwidth_mbps": 50.0, "latency": 0.002}
    for a in SERVERS:
        routes[a] = {}
        for b in SERVERS:
            if a != b:
                routes[a][b] = [a, cores[a], b] if cores[a] == cores[b] else [a, cores[a], cores[b], b]
    return {"links": links, "routes": routes}


def exchange_tables(lt):
    out = {}
    for a, na in NAMES.items():
        for b, nb in NAMES.items():
            if a == b:
                out[(a, b)] = (0.0, 0.0)
                continue
            hops = directed_hops_from_topology(lt, na, nb)
            path = lt["routes"][na][nb]
            lat = sum(float(lt["links"][link_key(x, y)]["latency"]) for x, y in zip(path, path[1:]))
            out[(a, b)] = (1.0 / (min(bw for _k, bw in hops) * MIB), lat)
    return out


def scenario(seed, monkeypatch, n_tasks=None, density=0.4, n_flows=(0, 14), n_cands=(3, 6)):
    N = N_TASKS if n_tasks is None else int(n_tasks)
    rnd = random.Random(seed)
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE", "1")
    env = simpy.Environment()
    fabric = NetworkFabric(env, topology())
    for _ in range(rnd.randint(*n_flows)):
        a, b = rnd.sample(SERVERS, 2)
        fabric.fairshare.start(fabric.directed_hops(a, b), rnd.uniform(5, 80) * MIB)
    if fabric.fairshare.flows:
        env.run(until=rnd.uniform(0.01, 0.8))
    flows = LF.flows_live(fabric.fairshare, env.now)
    counts = LF.counts_of(flows)
    routes = LF.routes_between(fabric.link_topology, NAMES)
    nx = exchange_tables(fabric.link_topology)
    cands = {t: sorted(rnd.sample(range(len(REPLICAS)), rnd.randint(*n_cands))) for t in range(N)}
    place = {t: [REPLICAS[p] for p in cands[t]] for t in range(N)}
    pairs = {}
    for i in range(N):
        for j in range(i + 1, N):
            if rnd.random() < density:
                b = rnd.uniform(1, 30) * MIB
                pairs[(i, j)] = b
                pairs[(j, i)] = b if rnd.random() < 0.8 else rnd.uniform(1, 30) * MIB
    for _ in range(3):  # one-directional pairs
        i, j = rnd.sample(range(N), 2)
        pairs.pop((j, i), None)
        pairs[(i, j)] = rnd.uniform(1, 30) * MIB
    nodes = list(range(5))
    caps = {n: rnd.choice([math.inf, 3.0, 5.0]) for n in nodes}
    ctx = RF.PartialStateContext(
        node_caps=caps, demand={(t, r): rnd.choice([1.0, 2.0]) for t in range(N) for r in place[t]}, node_of={r: r[0] for r in REPLICAS},
        task_type_index={t: t % 4 for t in range(N)}, parents={t: [] for t in range(N)},
        route_hops_bneck={(a, b): ((0.0, math.inf) if a == b else (2.0, 10.0)) for a in nodes for b in nodes}, payload_bytes=0.0, transfer_norm=0.0,
        node_rank=RF.krank_node_order(caps, {n: float(n) for n in nodes}, contract=V7), ingress_links={(t, n): () for t in range(N) for n in nodes},
        core_links=frozenset(), peer_pairs=pairs, peer_norm=1.0, node_exchange=nx, cand_nodes={t: sorted({r[0] for r in place[t]}) for t in range(N)}, contract=V7,
        backlog_s={r: rnd.uniform(0, 3) for r in REPLICAS}, service_s={(t, r): rnd.uniform(0.5, 4) for t in range(N) for r in place[t]},
        link_counts=counts, link_routes=routes)
    src, dst = [], []
    for t in range(N):
        for p in cands[t]:
            src.append(t)
            dst.append(N + p)
    ei, _ea = to_undirected(torch.tensor([src, dst], dtype=torch.long), torch.zeros(len(src), 5), num_nodes=N + len(REPLICAS))
    g = Data(edge_index=ei, n_tasks=N, n_platforms=len(REPLICAS))
    g.task_logit_to_placement = place
    g.task_logit_to_queue_key = {t: [f"n{r[0]}:{r[1]}" for r in place[t]] for t in range(N)}
    g.queue_key_to_platform_meta = {f"n{r[0]}:{r[1]}": {"platform_pos": p} for p, r in enumerate(REPLICAS)}
    return rnd, ctx, g, place


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", V7)
    monkeypatch.setenv("PARTIAL_STATE_LOAD_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_EXCHANGE_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_PEER_MASS", "1")


SEEN = {"nonzero_rows": 0, "rows": 0}


def check_states(S, ctx, g, place, plan, M):
    out = B._blocks(S, ctx, g, plan, M)
    SEEN["rows"] += out.shape[0]
    SEEN["nonzero_rows"] += int((out[:, 27:30] != 0).any(1).sum())
    assert out.shape == (S.R, RF.partial_state_feature_dim(V7)) and out.dtype == np.float32
    for t in range(N_TASKS):
        committed = {j: tuple(place[j][int(plan[j])]) for j in range(N_TASKS) if M[t, j] > 0}  # ascending task index = the commit order of a corpus without parents
        want = RF.partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed).astype(np.float32)
        got = out[S.start[t]:S.start[t] + S.C[t]]
        assert np.array_equal(got[:, 27:30], want[:, 27:30]), (t, got[:, 27:30], want[:, 27:30])
        assert np.abs(got[:, :27] - want[:, :27]).max() <= 1e-5


@pytest.mark.parametrize("seed", range(6))
def test_link_columns_equal_the_definition_on_random_plans_and_masks(seed, monkeypatch):
    rnd, ctx, g, place = scenario(seed, monkeypatch)
    S = B._Static(g, ctx)
    assert S.link is not None and S.template.shape[1] == 30
    for _ in range(8):
        plan = [rnd.randrange(len(place[t])) for t in range(N_TASKS)]
        M = np.zeros((N_TASKS, N_TASKS))
        for t in range(N_TASKS):
            for j in range(N_TASKS):
                if j != t and rnd.random() < rnd.choice([0.2, 0.5, 0.9]):
                    M[t, j] = 1.0
        check_states(S, ctx, g, place, plan, M)


@pytest.mark.parametrize("seed", range(4))
def test_prefix_and_refine_states_equal_the_definition(seed, monkeypatch):
    rnd, ctx, g, place = scenario(100 + seed, monkeypatch)
    S = B._Static(g, ctx)
    for _ in range(6):
        plan = [rnd.randrange(len(place[t])) for t in range(N_TASKS)]
        pre = np.tril(np.ones((N_TASKS, N_TASKS)), -1)  # strict prefix in ascending order
        check_states(S, ctx, g, place, plan, pre)
        check_states(S, ctx, g, place, plan, 1.0 - np.eye(N_TASKS))  # a refine pass / the full-context term: every other task committed
        check_states(S, ctx, g, place, plan, np.zeros((N_TASKS, N_TASKS)))  # nothing committed: the link columns are 0


def test_the_batched_states_carry_the_link_columns_end_to_end(monkeypatch):
    rnd, ctx, g, place = scenario(7, monkeypatch)
    plans = [[rnd.randrange(len(place[t])) for t in range(N_TASKS)] for _ in range(2)]
    b = B.BatchedTeacherForced(g, ctx, plans, list(range(N_TASKS)), with_full=True)
    assert b.ps.shape[1] == 30
    nonzero = (b.ps[:, 27:30] != 0).any(1)
    assert bool(nonzero.any()), "E1-E3 are zero in every batched state: the link block did not reach the loss input"
    for name, mask in (("prefix", np.tril(np.ones((N_TASKS, N_TASKS)), -1)), ("full", 1.0 - np.eye(N_TASKS))):
        for k, plan in enumerate(plans):
            for t in range(N_TASKS):
                s = int(b.state_of[name][k, t])
                committed = {j: tuple(place[j][plan[j]]) for j in range(N_TASKS) if mask[t, j] > 0}
                want = RF.partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed).astype(np.float32)
                got = b.ps[b.offs[s]:b.offs[s] + b.counts[s]].numpy()
                assert np.array_equal(got[:, 27:30], want[:, 27:30])


def test_a_link_contract_without_its_tables_is_refused(monkeypatch):
    rnd, ctx, g, place = scenario(3, monkeypatch)

    class M:
        partial_state_edge_dim = 30
        arm_kind = None
        edge_scorer = type("Sc", (), {"fc1": 1, "fc2": 1, "dropout": 1})()

    B._unsupported(M(), ctx)  # supported now
    ctx.link_counts = None
    with pytest.raises(ValueError, match="link_counts and link_routes"):
        B._unsupported(M(), ctx)


def test_the_comparisons_are_not_vacuous(monkeypatch):
    SEEN.update(nonzero_rows=0, rows=0)
    rnd, ctx, g, place = scenario(11, monkeypatch)
    S = B._Static(g, ctx)
    for _ in range(10):
        plan = [rnd.randrange(len(place[t])) for t in range(N_TASKS)]
        check_states(S, ctx, g, place, plan, 1.0 - np.eye(N_TASKS))
    assert SEEN["nonzero_rows"] > 0.4 * SEEN["rows"], SEEN


def test_the_prefix_flows_alone_move_the_columns_on_an_empty_fabric(monkeypatch):
    rnd, ctx, g, place = scenario(5, monkeypatch)
    ctx.link_counts = {}  # no flow in flight: only the batch's own committed pairs can load a link
    S = B._Static(g, ctx)
    moved = 0
    for _ in range(10):
        plan = [rnd.randrange(len(place[t])) for t in range(N_TASKS)]
        check_states(S, ctx, g, place, plan, 1.0 - np.eye(N_TASKS))
        full = B._blocks(S, ctx, g, plan, 1.0 - np.eye(N_TASKS))
        none = B._blocks(S, ctx, g, plan, np.zeros((N_TASKS, N_TASKS)))
        assert (none[:, 27:30] == 0).all()
        moved += int((full[:, 27:30] != 0).any(1).sum())
    assert moved > 0
    two_hop = [e for e in LF.pending_counts(ctx.link_routes, {0: [(1, 1.0)], 1: [(0, 1.0)]}, {0: 0, 1: 3}).values()]
    assert two_hop and max(two_hop) == 1


def _model_graph(g):
    n_p = len(REPLICAS)
    gen = torch.Generator().manual_seed(2)
    g.task_features = torch.rand(N_TASKS, 5, generator=gen)
    g.platform_features = torch.rand(n_p, 7, generator=gen)
    g.edge_attr = torch.rand(g.edge_index.size(1), 5, generator=gen)
    g.task_type_onehot4 = torch.eye(4)[torch.arange(N_TASKS) % 4]
    g.dag_parents = {t: [] for t in range(N_TASKS)}
    return g


@pytest.mark.parametrize("arm", ["gnn", "mlp"])
def test_batched_log_probs_equal_the_sequential_walk_with_the_link_columns(arm, monkeypatch):
    from src.policy.gnn.arm_models import TaskPlacementMLPSame
    from src.policy.gnn.gnn_model import TaskPlacementGNN
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn

    monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)
    rnd, ctx, g, place = scenario(21, monkeypatch)
    g = _model_graph(g)
    dims = dict(task_feature_dim=5, platform_feature_dim=7, hidden_dim=16, edge_dim=5, task_type_onehot_dim=4, partial_state_edge_dim=30)
    torch.manual_seed(0)
    model = (TaskPlacementMLPSame(**dims) if arm == "mlp" else TaskPlacementGNN(**dims, embedding_dim=16, num_layers=2)).eval()
    plans = [[rnd.randrange(len(place[t])) for t in range(N_TASKS)] for _ in range(3)]
    order = list(range(N_TASKS))
    b = B.BatchedTeacherForced(g, ctx, plans, order, with_full=True)
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
                    assert abs(got[k, t] - seq) <= 1e-5, (name, k, t, got[k, t], seq)


def test_a_commit_order_other_than_ascending_is_refused_for_the_link_block(monkeypatch):
    rnd, ctx, g, place = scenario(8, monkeypatch)
    plans = [[rnd.randrange(len(place[t])) for t in range(N_TASKS)]]
    B.BatchedTeacherForced(g, ctx, plans, list(range(N_TASKS)), with_full=False)
    with pytest.raises(ValueError, match="ascending task index"):
        B.BatchedTeacherForced(g, ctx, plans, list(reversed(range(N_TASKS))), with_full=False)
