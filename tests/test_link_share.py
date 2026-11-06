"""partial_state_v10: the link-sharing block SH1-SH4 (link_features.share_edge_columns), in the sequential columns and in the batched teacher-forced loss."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import test_batched_tf_link as T  # noqa: E402,F401
from test_batched_tf_link import N_TASKS, scenario  # noqa: E402
from test_link_features import NAMES, SPECS, MIB, fabric_env, node_exchange, start_flows  # noqa: E402,F401

from src.placement import link_features as LF  # noqa: E402
from src.policy.gnn import batched_tf as B  # noqa: E402
from src.policy.tabular import reduced_features as RF  # noqa: E402

V9, V10 = RF.PARTIAL_STATE_CONTRACT_V9, RF.PARTIAL_STATE_CONTRACT_V10
SH0 = 37  # first SH column: after the 37 v9 columns


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", V10)
    monkeypatch.setenv("PARTIAL_STATE_LOAD_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_EXCHANGE_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_PEER_MASS", "1")


def tables(fabric_env):
    env, fabric = fabric_env
    start_flows(env, fabric, SPECS, until=0.0)
    counts = LF.counts_of(LF.flows_live(fabric.fairshare, env.now))
    return counts, LF.routes_between(fabric.link_topology, NAMES), node_exchange(fabric.link_topology, NAMES)


def test_v10_is_41_wide_and_extends_v9_in_every_family():
    assert RF.partial_state_feature_dim(V10) == RF.partial_state_feature_dim(V9) + LF.LINK_SHARE_DIM == 41
    for fam in (RF.VALID_PARTIAL_STATE_CONTRACTS, RF.PEER_BLOCK_CONTRACTS, RF.LOAD_SECONDS_CONTRACTS, RF.BOTH_DIRECTION_CONTRACTS, RF.FOUR_TYPE_CONTRACTS,
                RF.INFLIGHT_CONTRACTS, RF.LINK_CONTRACTS, RF.STATIC_LINK_CONTRACTS, RF.OUT_LINK_CONTRACTS):
        assert V10 in fam and V9 in fam
    assert RF.SHARE_LINK_CONTRACTS == frozenset({V10})


def test_default_contract_is_untouched():
    assert RF.DEFAULT_PARTIAL_STATE_CONTRACT != V10 and RF.partial_state_feature_dim(V9) == 37


def test_share_columns_against_hand_computed_values(fabric_env):
    counts, routes, nx = tables(fabric_env)
    b = 8 * MIB
    assert LF.share_edge_columns(counts, routes, nx, [], 1) == (0.0, 0.0, 0.0, 0.0)
    assert LF.share_edge_columns(counts, routes, nx, [(1, b)], 1) == (0.0, 0.0, 0.0, 0.0)  # co-located: nothing moves
    # one remote partner: a single new flow, so SH1 = SH2 = the E1 price, SH3 = 0, SH4 = log(1) = 0
    e1 = LF.edge_columns(counts, routes, nx, [(0, b)], 1)
    one = LF.share_edge_columns(counts, routes, nx, [(0, b)], 1)
    assert one[0] == e1[0] and one[1] == one[0] and one[2] == 0.0 and one[3] == 0.0
    # two pulls that cross the same hop share it: each gets bw*MiB/(c + 2), so the shared price exceeds the E1 prices
    two = LF.share_edge_columns(counts, routes, nx, [(0, b), (2, 2 * b)], 1)
    hops0, hops2 = routes[(0, 1)], routes[(2, 1)]
    own = {}
    for hops in (hops0, hops2):
        for k, _bw in hops:
            own[k] = own.get(k, 0) + 1
    assert max(own.values()) >= 2, "fixture: the two routes must share a hop"
    want = 0.0
    longest = 0.0
    for nj, bb in ((0, b), (2, 2 * b)):
        pb, lat = nx[(nj, 1)]
        hops = routes[(nj, 1)]
        act = [counts.get(k, 0) for k, _ in hops]
        m = [own[k] for k, _ in hops]
        t = bb / min(bw * MIB / (n + mm) for (_k, bw), n, mm in zip(hops, act, m)) + (lat if pb > 0.0 else 0.0)
        want += t
        longest = max(longest, t)
    assert two[0] == math.log1p(want) and two[1] == math.log1p(longest) and two[3] == math.log(float(max(own.values())))
    ind = LF.edge_columns(counts, routes, nx, [(0, b), (2, 2 * b)], 1)
    assert two[0] >= ind[0] and two[2] >= 0.0
    # an empty fabric with one pull prices solo
    e = LF.share_edge_columns({}, routes, nx, [(0, b)], 1)
    assert e == (math.log1p(b * nx[(0, 1)][0] + nx[(0, 1)][1]),) * 2 + (0.0, 0.0)
    # outside partners are extra pulls: same value as listing them with the committed ones
    assert LF.share_edge_columns(counts, routes, nx, [(0, b)], 1, [(2, 2 * b)]) == two


def test_v9_columns_are_untouched_and_sh_is_prefix_dependent(monkeypatch):
    for seed in range(3):
        rnd, ctx, g, place = scenario(seed, monkeypatch)
        ctx.contract = V10
        ctx9 = scenario(seed, monkeypatch)[1]
        ctx9.contract = V9
        for t in range(N_TASKS):
            committed = {j: tuple(place[j][0]) for j in range(t)}
            w10 = RF.partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed)
            w9 = RF.partial_state_columns(ctx9, t, [tuple(c) for c in place[t]], committed)
            assert w10.shape[1] == 41 and np.array_equal(w10[:, :SH0], w9)
            if not committed:
                assert (w10[:, SH0:] == 0).all()


def test_batched_share_block_equals_the_definition_on_random_plans_masks_and_prefixes(monkeypatch):
    for seed in range(5):
        rnd, ctx, g, place = scenario(seed, monkeypatch)
        ctx.contract = V10
        S = B._Static(g, ctx)
        assert S.template.shape[1] == 41
        nonzero = 0
        for _ in range(6):
            plan = [rnd.randrange(len(place[t])) for t in range(N_TASKS)]
            for M in (np.tril(np.ones((N_TASKS, N_TASKS)), -1), 1.0 - np.eye(N_TASKS), np.zeros((N_TASKS, N_TASKS))):
                out = B._blocks(S, ctx, g, plan, M)
                assert out.shape == (S.R, 41)
                for t in range(N_TASKS):
                    committed = {j: tuple(place[j][int(plan[j])]) for j in range(N_TASKS) if M[t, j] > 0}
                    want = RF.partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed).astype(np.float32)
                    got = out[S.start[t]:S.start[t] + S.C[t]]
                    assert np.array_equal(got[:, SH0:], want[:, SH0:])
                    assert np.abs(got[:, :SH0] - want[:, :SH0]).max() <= 1e-5
                    nonzero += int((got[:, SH0:] != 0).any(1).sum())
        assert nonzero > 0


def test_outside_partners_enter_the_columns_and_the_batched_twin_refuses_them(monkeypatch):
    rnd, ctx, g, place = scenario(0, monkeypatch)
    ctx.contract = V10
    t = N_TASKS - 1
    committed = {j: tuple(place[j][0]) for j in range(t)}
    base = RF.partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed)
    node_ids = sorted({ctx.node_of[tuple(c)] for c in place[t]})
    ctx.outside_partners = {t: [(node_ids[0], 6 * MIB)]}
    withx = RF.partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed)
    assert np.array_equal(base[:, :SH0], withx[:, :SH0]) and not np.array_equal(base[:, SH0:], withx[:, SH0:])
    with pytest.raises(ValueError, match="outside partners"):
        B._Static(g, ctx)


@pytest.mark.parametrize("trained,serving", [(V9, V10), (V10, V9)])
def test_a_checkpoint_of_the_other_contract_is_refused(trained, serving):
    with pytest.raises(RF.PartialStateContractMismatchError):
        RF.require_matching_partial_state_contract(trained, serving, model_label="ckpt")
    RF.require_matching_partial_state_contract(trained, trained, model_label="ckpt")


def test_a_sidecarless_checkpoint_is_refused_for_v10():
    with pytest.raises(RF.PartialStateContractMismatchError):
        RF.require_matching_partial_state_contract(None, V10, model_label="ckpt")


def test_a_v10_sidecar_reaches_serving(tmp_path):
    import json
    from src.executesimulation import checkpoint_mp_config

    m = tmp_path / "ckpt.pt"
    m.write_bytes(b"")
    m.with_suffix(".contract.json").write_text(json.dumps({"partial_state_contract": V10, "mp_residual": False}))
    assert checkpoint_mp_config(m)["partial_state_contract"] == V10


def _run_path(plat, psc, arm, monkeypatch):
    """test_link_features.run_path at the v10 width: one path through a model and a full decode plus two refine passes; returns what each step saw."""
    import torch
    import test_link_features as TL
    from src.policy.gnn.arm_models import TaskPlacementMLPSame
    from src.policy.gnn.gnn_model import TaskPlacementGNN
    from src.policy.gnn.partial_state_edges import candidate_edge_rows, make_partial_state_score_fn
    from src.policy.gnn.prefix_serving import _self_refine
    from src.policy.gnn.seq_decode import decode_masked_topo_placement

    if arm == "twin":
        monkeypatch.setenv("GNN_DISABLE_MESSAGE_PASSING", "1")
    else:
        monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)
    graph = TL.synthetic_graph(plat, psc)
    ctx = RF.build_partial_state_context_from_graph(graph)
    torch.manual_seed(0)
    dims = dict(task_feature_dim=TL.TASK_DIM, platform_feature_dim=TL.PLAT_DIM_BASE + LF.LINK_PLATFORM_DIM, hidden_dim=16, edge_dim=5,
                partial_state_edge_dim=RF.partial_state_feature_dim(V10))
    model = (TaskPlacementMLPSame(**dims) if arm == "mlp" else TaskPlacementGNN(**dims, embedding_dim=8, num_layers=2)).eval()
    score = make_partial_state_score_fn(model, graph, ctx, cache_encode=False)
    rows = candidate_edge_rows(graph)
    trace = []

    def traced(t, committed):
        logits = score(t, committed)
        trace.append((t, dict(committed), graph.partial_state_edge_attr[rows[t]].clone(), logits.detach().clone()))
        return logits

    tl = {t: [tuple(r) for r in TL.REPLICAS] for t in range(3)}
    with torch.no_grad():
        combo = decode_masked_topo_placement([[0.0] * len(TL.REPLICAS)] * 3, tl, 3, dag_parents={t: [] for t in range(3)}, node_caps={},
                                             demands={t: [1.0] * len(TL.REPLICAS) for t in range(3)}, score_fn=traced)
        refined, _moved = _self_refine(traced, tl, combo, 2)
    return {"combo": tuple(combo), "refined": refined, "trace": trace}


def test_cache_and_live_paths_give_bitwise_identical_v10_columns_and_logits_on_one_captured_state(fabric_env, monkeypatch):
    """One captured fabric state seen by the live scheduler (flows_live, the fabric's topology) and by the cache builder (the stored block of the same state, the dataset's routes):
    the same inputs go through the one definition, so the SH columns and every logit of a decode and two refine passes are equal to the last bit."""
    import torch
    import test_link_features as TL

    env, fabric = fabric_env
    TL.start_flows(env, fabric, TL.SPECS, until=0.7)
    nx = TL.node_exchange(fabric.link_topology, TL.NAMES)
    node_names = [TL.NAMES[r[0]] for r in TL.REPLICAS]
    live_flows = LF.flows_live(fabric.fairshare, env.now)
    live_psc = TL.psc_of(LF.counts_of(live_flows), LF.routes_between(fabric.link_topology, TL.NAMES), nx)
    live_plat = np.asarray([LF.platform_columns(LF.node_raw(live_flows, n)) for n in node_names])
    stored = LF.flows_stored(TL.stored_block(fabric.fairshare, TL.SPECS))
    cache_psc = TL.psc_of(LF.counts_of(stored), LF.routes_between({"routes": TL.topology()["routes"], "links": TL.topology()["links"]}, TL.NAMES), nx)
    cache_plat = np.asarray([LF.platform_columns(LF.node_raw(stored, n)) for n in node_names])
    assert np.array_equal(live_plat, cache_plat) and live_psc["link_counts"] == cache_psc["link_counts"]
    for arm in ("gnn", "twin", "mlp"):
        a = _run_path(live_plat, live_psc, arm, monkeypatch)
        b = _run_path(cache_plat, cache_psc, arm, monkeypatch)
        assert a["combo"] == b["combo"] and a["refined"] == b["refined"] and len(a["trace"]) == len(b["trace"]) > 3
        live_sh = 0
        for (ta, ca, ea, la), (tb, cb, eb, lb) in zip(a["trace"], b["trace"]):
            assert ta == tb and ca == cb and ea.shape[1] == 41 and torch.equal(ea, eb) and torch.equal(la, lb)
            live_sh += int((ea[:, SH0:] != 0).any(1).sum())
        assert live_sh > 0  # the new columns are live inputs of the decode, not constants
