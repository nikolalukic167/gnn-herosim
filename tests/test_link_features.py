"""partial_state_v7: the link-sharing columns (placement/link_features.py) and the train/serve parity gate."""
import math
import os
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
from src.policy.tabular import reduced_features as RF  # noqa: E402

V7 = RF.PARTIAL_STATE_CONTRACT_V7
MIB = 1024 ** 2
SERVERS = ["s0", "s1", "s2"]


def topology():
    """Three servers on directional access links to one core router (MB/s: s0 out 4 / in 75, s1 and s2 symmetric 10 / 20)."""
    access = {"s0": (4.0, 75.0), "s1": (10.0, 10.0), "s2": (20.0, 20.0)}
    links = {}
    for s, (out_bw, in_bw) in access.items():
        links[link_key(s, "core_0")] = {"bandwidth_mbps": min(out_bw, in_bw), "latency": 0.01, "access_node": s,
                                       "bandwidth_out_mbps": out_bw, "bandwidth_in_mbps": in_bw}
    routes = {a: {b: [a, "core_0", b] for b in SERVERS if b != a} for a in SERVERS}
    return {"links": links, "routes": routes}


@pytest.fixture
def fabric_env(monkeypatch):
    monkeypatch.setenv("HEROSIM_EXCHANGE_FAIRSHARE", "1")
    env = simpy.Environment()
    return env, NetworkFabric(env, topology())


def start_flows(env, fabric, specs, until):
    for src, dst, nbytes in specs:
        fabric.fairshare.start(fabric.directed_hops(src, dst), nbytes)
    if until > env.now:
        env.run(until=until)


def stored_block(share, specs):
    """A `fidelity["flows"]` block (flow_snapshot schema v1) the way the capture writes it: FairShare advanced to now first."""
    share._advance()
    exchanges, active = [], []
    for i, f in enumerate(share.flows):
        src, dst = specs[i][0], specs[i][1]
        exchanges.append({"owner_tid": 100 + i, "dst_node": dst, "stage": "input_io", "planned": False, "read_remaining": 0.0, "transfers": [{
            "partner_id": i, "src_node": src, "payload": f.nbytes, "latency": 0.02, "hops": [list(h) for h in f.hops], "b": 0.0, "rest": f.nbytes,
            "scale": 1.0, "status": "active", "startup_remaining": 0.0, "rem": f.rem, "rate": f.rate, "latency_remaining": 0.02}]})
        active.append([100 + i, 0])
    return {"version": 1, "model": "processor_sharing_v1", "partners": "serial", "tcp": False, "fairshare_last": 0.0, "active": active, "exchanges": exchanges}


SPECS = [("s0", "s1", 40 * MIB), ("s0", "s2", 30 * MIB), ("s2", "s1", 20 * MIB)]


def test_v7_is_valid_30_wide_and_in_every_family_it_extends():
    assert RF.validate_partial_state_contract("partial_state_v7") == V7
    assert RF.partial_state_feature_dim(V7) == RF.partial_state_feature_dim(RF.PARTIAL_STATE_CONTRACT_V6) + 3 == 30
    for group in (RF.PEER_BLOCK_CONTRACTS, RF.LOAD_SECONDS_CONTRACTS, RF.BOTH_DIRECTION_CONTRACTS, RF.FOUR_TYPE_CONTRACTS, RF.INFLIGHT_CONTRACTS):
        assert V7 in group and RF.PARTIAL_STATE_CONTRACT_V6 in group
    assert RF.LINK_CONTRACTS == frozenset({V7})
    assert RF.partial_state_feature_dim(RF.PARTIAL_STATE_CONTRACT_V6) == 27
    assert not LF.link_enabled(RF.PARTIAL_STATE_CONTRACT_V6) and LF.link_enabled(V7)
    with pytest.raises(RF.PartialStateContractMismatchError):
        RF.require_matching_partial_state_contract("partial_state_v6", V7, model_label="m")


def test_directed_hops_is_one_definition_for_the_fabric_and_the_stored_readers(fabric_env):
    _env, fabric = fabric_env
    for a in SERVERS:
        for b in SERVERS:
            if a != b:
                assert fabric.directed_hops(a, b) == directed_hops_from_topology(topology(), a, b)
    assert fabric.directed_hops("s0", "s1") == [("s0>core_0", 4.0), ("core_0>s1", 10.0)]


def test_node_raw_counts_the_access_links_and_drains_the_busier_one(fabric_env):
    env, fabric = fabric_env
    start_flows(env, fabric, SPECS, until=0.0)
    flows = LF.flows_live(fabric.fairshare, env.now)
    # s0 sends two flows (40 + 30 MiB over a 4 MB/s out link), receives none; s1 receives two (40 and 20 MiB over 10 MB/s in), sends none
    assert LF.node_raw(flows, "s0") == (2, 0, 70 * MIB / (4.0 * MIB))
    assert LF.node_raw(flows, "s1") == (0, 2, 60 * MIB / (10.0 * MIB))
    assert LF.node_raw(flows, "s2") == (1, 1, max(20 * MIB / (20.0 * MIB), 30 * MIB / (20.0 * MIB)))
    p = LF.platform_columns(LF.node_raw(flows, "s0"))
    assert p.tolist() == [math.log1p(2), 0.0, math.log1p(17.5)]


def test_a_flow_that_holds_no_link_is_not_counted(fabric_env):
    env, fabric = fabric_env
    assert LF.node_raw(LF.flows_live(fabric.fairshare, env.now), "s0") == (0, 0, 0.0)
    assert LF.counts_of(()) == {}


def test_live_front_end_never_advances_the_fabric_and_equals_the_stored_one(fabric_env):
    env, fabric = fabric_env
    share = fabric.fairshare
    start_flows(env, fabric, SPECS, until=0.7)
    before = ([f.rem for f in share.flows], share._last)
    live = LF.flows_live(share, env.now + 0.123)  # a later instant than the last event: rem is brought forward without touching the object
    assert ([f.rem for f in share.flows], share._last) == before
    live_now = LF.flows_live(share, env.now)
    assert live_now == tuple((f.rem - f.rate * (env.now - share._last), tuple(f.hops)) for f in share.flows)
    block = stored_block(share, SPECS)  # advances to env.now exactly as the capture does
    stored = LF.flows_stored(block)
    assert stored == live_now  # bitwise: the same subtraction rem - rate * dt
    assert live != live_now
    for node in SERVERS:
        assert LF.node_raw(stored, node) == LF.node_raw(live_now, node)
    assert LF.counts_of(stored) == LF.counts_of(live_now)


def test_stored_front_end_fails_loud_on_a_bad_block(fabric_env):
    env, fabric = fabric_env
    start_flows(env, fabric, SPECS, until=0.2)
    block = stored_block(fabric.fairshare, SPECS)
    block["version"] = 2
    with pytest.raises(ValueError, match="version"):
        LF.flows_stored(block)


def node_exchange(link_topology, names):
    out = {}
    for a, na in names.items():
        for b, nb in names.items():
            if a == b:
                out[(a, b)] = (0.0, 0.0)
                continue
            hops = directed_hops_from_topology(link_topology, na, nb)
            lat = sum(float(link_topology["links"][link_key(x, y)]["latency"]) for x, y in zip(link_topology["routes"][na][nb], link_topology["routes"][na][nb][1:]))
            out[(a, b)] = (1.0 / (min(bw for _k, bw in hops) * MIB), lat)
    return out


NAMES = {0: "s0", 1: "s1", 2: "s2"}


def test_edge_columns_on_an_empty_fabric_are_the_solo_price(fabric_env):
    _env, fabric = fabric_env
    routes = LF.routes_between(fabric.link_topology, NAMES)
    nx = node_exchange(fabric.link_topology, NAMES)
    e1, e2, e3 = LF.edge_columns({}, routes, nx, [(0, 8 * MIB)], 1)
    pb, lat = nx[(0, 1)]
    assert e1 == math.log1p(8 * MIB * pb + lat) and e2 == 0.0 and e3 == 0.0
    assert LF.edge_columns({}, routes, nx, [(1, 8 * MIB)], 1) == (0.0, 0.0, 0.0)  # a co-located partner moves nothing
    assert LF.edge_columns({}, routes, nx, [], 1) == (0.0, 0.0, 0.0)


def test_edge_columns_price_the_shared_rate_and_report_the_busiest_hop(fabric_env):
    env, fabric = fabric_env
    start_flows(env, fabric, SPECS, until=0.0)
    counts = LF.counts_of(LF.flows_live(fabric.fairshare, env.now))
    routes = LF.routes_between(fabric.link_topology, NAMES)
    nx = node_exchange(fabric.link_topology, NAMES)
    b = 8 * MIB
    e1, e2, e3 = LF.edge_columns(counts, routes, nx, [(0, b)], 1)  # s0>core_0 carries 2 flows, core_0>s1 carries 2
    rate = min(4.0 * MIB / 3, 10.0 * MIB / 3)
    shared = b / rate + nx[(0, 1)][1]
    solo = b * nx[(0, 1)][0] + nx[(0, 1)][1]
    assert e1 == math.log1p(shared) and e2 == math.log(shared / solo) and e3 == math.log1p(2.0)
    assert e2 > 0.0
    # the batch's own other transfers are not counted: a second committed partner only adds its own price
    two = LF.edge_columns(counts, routes, nx, [(0, b), (2, b)], 1)
    assert two[0] > e1


def base_ctx_kwargs(n_tasks, replicas, link_counts, link_routes, nx):
    nodes = sorted({r[0] for r in replicas})
    caps = {n: math.inf for n in nodes}
    return dict(
        node_caps=caps, demand={(t, r): 1.0 for t in range(n_tasks) for r in replicas}, node_of={r: r[0] for r in replicas},
        task_type_index={t: t % 4 for t in range(n_tasks)}, parents={t: [] for t in range(n_tasks)},
        route_hops_bneck={(a, b): ((0.0, math.inf) if a == b else (2.0, 10.0)) for a in nodes for b in nodes}, payload_bytes=0.0, transfer_norm=0.0,
        node_rank=RF.krank_node_order(caps, {n: float(n) for n in nodes}, contract=V7), ingress_links={(t, n): () for t in range(n_tasks) for n in nodes},
        core_links=frozenset(), peer_pairs={(0, 1): 8.0 * MIB, (1, 0): 8.0 * MIB, (1, 2): 4.0 * MIB, (2, 1): 4.0 * MIB, (0, 2): 6.0 * MIB, (2, 0): 6.0 * MIB},
        peer_norm=1.0, node_exchange=nx, cand_nodes={t: list(nodes) for t in range(n_tasks)}, contract=V7,
        backlog_s={r: 0.5 * r[1] for r in replicas}, service_s={(t, r): 1.0 + t + 0.5 * r[1] for t in range(n_tasks) for r in replicas},
        link_counts=link_counts, link_routes=link_routes)


REPLICAS = [(n, 10 * n + k) for n in range(3) for k in range(2)]


def test_partial_state_columns_carry_e1_to_e3_after_the_push_columns(fabric_env):
    env, fabric = fabric_env
    start_flows(env, fabric, SPECS, until=0.0)
    counts = LF.counts_of(LF.flows_live(fabric.fairshare, env.now))
    routes = LF.routes_between(fabric.link_topology, NAMES)
    nx = node_exchange(fabric.link_topology, NAMES)
    ctx = RF.PartialStateContext(**base_ctx_kwargs(3, REPLICAS, counts, routes, nx))
    committed = {0: (0, 0), 1: (2, 20)}
    block = RF.partial_state_columns(ctx, 2, REPLICAS, committed)
    assert block.shape == (len(REPLICAS), 30)
    # the prefix: task 0 (node 0) and task 1 (node 2) are peers, so two flows 2 -> 0 and 0 -> 2 are added to the live counts
    pending = LF.pending_counts(routes, {0: [(1, 8.0 * MIB), (2, 6.0 * MIB)], 1: [(0, 8.0 * MIB), (2, 4.0 * MIB)]}, {0: 0, 1: 2})
    assert pending == {key: 1 for key, _bw in routes[(2, 0)] + routes[(0, 2)]}
    eff = LF.with_pending(counts, pending)
    for i, cand in enumerate(REPLICAS):
        want = LF.edge_columns(eff, routes, nx, [(0, 6.0 * MIB), (2, 4.0 * MIB)], cand[0])
        assert block[i, 27:30].tolist() == np.float32(want).tolist()
    v6 = RF.PartialStateContext(**{**base_ctx_kwargs(3, REPLICAS, None, None, nx), "contract": RF.PARTIAL_STATE_CONTRACT_V6, "link_counts": None, "link_routes": None,
                                   "node_rank": RF.krank_node_order({n: math.inf for n in range(3)}, {n: float(n) for n in range(3)}, contract=RF.PARTIAL_STATE_CONTRACT_V6)})
    old = RF.partial_state_columns(v6, 2, REPLICAS, committed)
    assert old.shape == (len(REPLICAS), 27) and np.array_equal(old, block[:, :27])


def test_the_edge_columns_refresh_with_the_committed_prefix(fabric_env):
    _env, fabric = fabric_env
    routes = LF.routes_between(fabric.link_topology, NAMES)
    nx = node_exchange(fabric.link_topology, NAMES)
    ctx = RF.PartialStateContext(**base_ctx_kwargs(3, REPLICAS, {}, routes, nx))  # an empty fabric: only the batch's own flows can load a link
    alone = RF.partial_state_columns(ctx, 2, REPLICAS, {0: (0, 0)})  # one placed batch-mate: no flow of its own yet (its peers are unplaced)
    both = RF.partial_state_columns(ctx, 2, REPLICAS, {0: (0, 0), 1: (2, 20)})  # 0 and 1 are placed peers: flows 2 -> 0 and 0 -> 2 now load the links
    assert np.all(alone[:, 29] == 0.0)
    cand_on_2 = [i for i, c in enumerate(REPLICAS) if c[0] == 2]
    assert all(both[i, 29] > 0.0 for i in range(len(REPLICAS)))  # task 2's partner 0 sits on node 0; every route from node 0 crosses a loaded link
    assert both[cand_on_2[0], 28] > 0.0  # the exchange 0 -> 2 is slower than solo
    # a refine pass scores a task with every OTHER task committed, and the task itself never counts against itself
    refine = RF.partial_state_columns(ctx, 1, REPLICAS, {0: (0, 0), 2: (2, 21)})
    assert refine.shape == (len(REPLICAS), 30)


def test_a_v7_context_without_link_tables_fails_loud(fabric_env):
    _env, fabric = fabric_env
    nx = node_exchange(fabric.link_topology, NAMES)
    with pytest.raises(ValueError, match="link_counts and link_routes"):
        RF.PartialStateContext(**{**base_ctx_kwargs(3, REPLICAS, {}, {}, nx), "link_counts": None, "link_routes": None})


# ---- the gate: one state, the trainer path and the live serving path, a full decode, bitwise ----------------------------------------------------

PLAT_DIM_BASE, TASK_DIM = 14, 7


def synthetic_graph(link_platform_extra, psc):
    from src.placement.four_type_features import PLATFORM_EXTRA_DIM  # noqa: F401

    n_tasks, n_platforms = 3, len(REPLICAS)
    g = torch.Generator().manual_seed(3)
    src, dst, attrs = [], [], []
    for t in range(n_tasks):
        for p in range(n_platforms):
            src.append(t)
            dst.append(n_tasks + p)
            attrs.append(torch.rand(5, generator=g).tolist())
    ei, ea = to_undirected(torch.tensor([src, dst], dtype=torch.long), torch.tensor(attrs, dtype=torch.float32), num_nodes=n_tasks + n_platforms)
    plat = torch.rand(n_platforms, PLAT_DIM_BASE, generator=g)
    plat = torch.cat([plat, torch.from_numpy(link_platform_extra).to(torch.float32)], dim=1)
    data = Data(edge_index=ei, n_tasks=n_tasks, n_platforms=n_platforms, task_features=torch.rand(n_tasks, TASK_DIM, generator=g), platform_features=plat)
    data.edge_attr = ea
    data.task_logit_to_placement = {t: [tuple(r) for r in REPLICAS] for t in range(n_tasks)}
    data.task_logit_to_queue_key = {t: [f"n{r[0]}:{r[1]}" for r in REPLICAS] for t in range(n_tasks)}
    data.queue_key_to_platform_meta = {f"n{r[0]}:{r[1]}": {"platform_pos": p} for p, r in enumerate(REPLICAS)}
    data.partial_state_ctx = psc
    data.dag_parents = {t: [] for t in range(n_tasks)}
    return data


def psc_of(counts, routes, nx):
    kw = base_ctx_kwargs(3, REPLICAS, counts, routes, nx)
    return {"node_caps": kw["node_caps"], "demand": kw["demand"], "task_type_index": kw["task_type_index"], "parents": kw["parents"],
            "route_hops_bneck": kw["route_hops_bneck"], "payload_bytes": 0.0, "transfer_norm": 0.0, "node_rank": kw["node_rank"],
            "ingress_links": kw["ingress_links"], "core_links": [], "peer_pairs": kw["peer_pairs"], "node_exchange": nx, "peer_norm": 1.0,
            "cand_nodes": kw["cand_nodes"], "backlog_s": kw["backlog_s"], "service_s": kw["service_s"], "link_counts": counts, "link_routes": routes}


def run_path(plat, psc, mp_off, monkeypatch):
    """One path (a platform block and a context) through a model and a full decode plus two refine passes; returns what each step saw."""
    from src.policy.gnn.gnn_model import TaskPlacementGNN
    from src.policy.gnn.partial_state_edges import candidate_edge_rows, make_partial_state_score_fn
    from src.policy.gnn.prefix_serving import _self_refine
    from src.policy.gnn.seq_decode import decode_masked_topo_placement

    if mp_off:
        monkeypatch.setenv("GNN_DISABLE_MESSAGE_PASSING", "1")
    else:
        monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)
    graph = synthetic_graph(plat, psc)
    ctx = RF.build_partial_state_context_from_graph(graph)
    torch.manual_seed(0)
    model = TaskPlacementGNN(task_feature_dim=TASK_DIM, platform_feature_dim=PLAT_DIM_BASE + LF.LINK_PLATFORM_DIM, embedding_dim=8, hidden_dim=16,
                             num_layers=2, edge_dim=5, partial_state_edge_dim=RF.partial_state_feature_dim(V7)).eval()
    score = make_partial_state_score_fn(model, graph, ctx, cache_encode=False)
    rows = candidate_edge_rows(graph)
    trace = []

    def traced(t, committed):
        logits = score(t, committed)
        block = graph.partial_state_edge_attr[rows[t]].clone()
        trace.append((t, dict(committed), block, logits.detach().clone()))
        return logits

    tl = {t: [tuple(r) for r in REPLICAS] for t in range(3)}
    with torch.no_grad():
        task_emb, platform_emb = model._encode(graph)
        combo = decode_masked_topo_placement([[0.0] * len(REPLICAS)] * 3, tl, 3, dag_parents={t: [] for t in range(3)}, node_caps={},
                                             demands={t: [1.0] * len(REPLICAS) for t in range(3)}, score_fn=traced)
        n_decode = len(trace)
        refined, _moved = _self_refine(traced, tl, combo, 2)
    return {"combo": tuple(combo), "refined": refined, "trace": trace, "n_decode": n_decode, "task_emb": task_emb.clone(), "platform_emb": platform_emb.clone(),
            "platform_features": graph.platform_features.clone()}


def test_trainer_and_live_serving_give_bitwise_identical_columns_conv_inputs_and_logits_across_a_decode(fabric_env, monkeypatch):
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", V7)
    env, fabric = fabric_env
    share = fabric.fairshare
    start_flows(env, fabric, SPECS, until=0.7)
    nx = node_exchange(fabric.link_topology, NAMES)
    node_names = [NAMES[r[0]] for r in REPLICAS]

    # live serving path: the flows of the running fabric, the fabric's own topology
    live_flows = LF.flows_live(share, env.now)
    live_psc = psc_of(LF.counts_of(live_flows), LF.routes_between(fabric.link_topology, NAMES), nx)
    live_plat = np.asarray([LF.platform_columns(LF.node_raw(live_flows, n)) for n in node_names])
    # trainer path: the stored block of the same state, the dataset's (routes, links) as the cache reads them
    stored_flows = LF.flows_stored(stored_block(share, SPECS))
    cache_psc = psc_of(LF.counts_of(stored_flows), LF.routes_between({"routes": topology()["routes"], "links": topology()["links"]}, NAMES), nx)
    cache_plat = np.asarray([LF.platform_columns(LF.node_raw(stored_flows, n)) for n in node_names])

    assert np.array_equal(live_plat, cache_plat)
    assert live_psc["link_counts"] == cache_psc["link_counts"] and live_psc["link_routes"] == cache_psc["link_routes"]

    runs = {(path, arm): run_path(plat, psc, arm == "twin", monkeypatch)
            for path, plat, psc in (("live", live_plat, live_psc), ("cache", cache_plat, cache_psc)) for arm in ("gnn", "twin")}
    ref = runs[("cache", "gnn")]
    assert ref["n_decode"] == 3 and len(ref["trace"]) > 3  # the decode plus at least one refine pass
    for key, r in runs.items():
        assert r["combo"] == ref["combo"] or key[1] == "twin"  # the twin is a different model, so its plan may differ; the columns may not
        assert torch.equal(r["platform_features"], ref["platform_features"])
    # same path, two paths: live vs cache, per arm, bitwise: columns, conv inputs (the node embeddings) and logits at every step
    for arm in ("gnn", "twin"):
        a, b = runs[("live", arm)], runs[("cache", arm)]
        assert a["combo"] == b["combo"] and a["refined"] == b["refined"] and len(a["trace"]) == len(b["trace"])
        assert torch.equal(a["task_emb"], b["task_emb"]) and torch.equal(a["platform_emb"], b["platform_emb"])
        for (ta, ca, ea, la), (tb, cb, eb, lb) in zip(a["trace"], b["trace"]):
            assert ta == tb and ca == cb and torch.equal(ea, eb) and torch.equal(la, lb) and ea.shape[1] == 30
    # the columns are a function of (task, committed prefix) alone: every step either model took, on either path, reads exactly what the single
    # definition gives, so the gnn and the twin carry byte-identical v7 columns at every decode step and refine pass they share
    ctx_by_path = {"live": RF.build_partial_state_context_from_graph(synthetic_graph(live_plat, live_psc)),
                   "cache": RF.build_partial_state_context_from_graph(synthetic_graph(cache_plat, cache_psc))}
    seen = {}
    for (path, arm), r in runs.items():
        for t, committed, block, _logits in r["trace"]:
            want = torch.as_tensor(RF.partial_state_columns(ctx_by_path[path], t, [tuple(c) for c in REPLICAS], committed)).to(block.dtype)
            assert torch.equal(block, want)
            seen.setdefault((path, t, tuple(sorted(committed.items()))), {})[arm] = block
    shared = [k for k, v in seen.items() if set(v) == {"gnn", "twin"}]
    assert shared  # the first decode step is the same prefix for both
    for k in shared:
        assert seen[k]["gnn"].numpy().tobytes() == seen[k]["twin"].numpy().tobytes()
    # the new columns are live inputs of the decode, not constants: a quiet fabric gives other edge columns for the same prefix
    quiet = psc_of({}, cache_psc["link_routes"], nx)
    ctx_q = RF.build_partial_state_context_from_graph(synthetic_graph(cache_plat * 0.0, quiet))
    ctx_l = RF.build_partial_state_context_from_graph(synthetic_graph(live_plat, live_psc))
    committed = ref["trace"][2][1]
    assert not np.array_equal(RF.partial_state_columns(ctx_q, 2, REPLICAS, committed)[:, 27:], RF.partial_state_columns(ctx_l, 2, REPLICAS, committed)[:, 27:])


def test_widen_graph_appends_and_removes_the_held_columns():
    g = Data(platform_features=torch.zeros(2, 5), link_platform_extra=torch.ones(2, 3))
    LF.widen_graph(g)
    assert g.platform_features.shape == (2, 8) and not hasattr(g, "link_platform_extra")
    with pytest.raises(ValueError, match="no link columns"):
        LF.widen_graph(Data(platform_features=torch.zeros(2, 5)))
