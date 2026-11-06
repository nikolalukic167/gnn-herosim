"""S5 best-of-K (HEROSIM_BOK_*): the sampled decoder against the served greedy decode, on a small captured fabric state (the fixtures of
tests/test_link_features.py: 3 tasks, a handful of candidates, live fair-share flows, partial_state_v8 so the static link columns exist)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import test_link_features as TL  # noqa: E402
from test_link_features import NAMES, REPLICAS, SPECS, fabric_env, node_exchange, start_flows  # noqa: E402,F401

from src.placement import link_features as LF  # noqa: E402
from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402
from src.policy.gnn.prefix_serving import PrefixServingOptions, decode_prefix_conditioned  # noqa: E402
from src.policy.tabular import reduced_features as RF  # noqa: E402

CONTRACT = RF.PARTIAL_STATE_CONTRACT_V8
BOK_ENV = ("HEROSIM_BOK_K", "HEROSIM_BOK_TAU", "HEROSIM_BOK_NOISE", "HEROSIM_BOK_BATCHED", "HEROSIM_PS_STATIC_MEMO", "HEROSIM_PS_FAST_SPLIT",
           "HEROSIM_BOK_CHARGE_RATIO", "GNN_PREFIX_SELF_REFINE")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in BOK_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", CONTRACT)
    monkeypatch.setenv("PARTIAL_STATE_LOAD_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_EXCHANGE_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_PEER_MASS", "1")
    monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)


@pytest.fixture
def state(fabric_env):
    env, fabric = fabric_env
    start_flows(env, fabric, SPECS, until=0.7)
    nx = node_exchange(fabric.link_topology, NAMES)
    flows = LF.flows_live(fabric.fairshare, env.now)
    psc = TL.psc_of(LF.counts_of(flows), LF.routes_between(fabric.link_topology, NAMES), nx)
    psc["node_caps_by_alpha"] = {"inf": {}}
    plat = LF.np.asarray([LF.platform_columns(LF.node_raw(flows, NAMES[r[0]])) for r in REPLICAS])
    return plat, psc


def _model():
    torch.manual_seed(0)
    return TaskPlacementGNN(task_feature_dim=TL.TASK_DIM, platform_feature_dim=TL.PLAT_DIM_BASE + LF.LINK_PLATFORM_DIM, hidden_dim=16, edge_dim=5,
                            partial_state_edge_dim=RF.partial_state_feature_dim(CONTRACT), embedding_dim=8, num_layers=2).eval()


def _options():
    return PrefixServingOptions(alpha_key="inf", allow_replica_reuse=True, relax_on_stuck=False, task_type_vocab=("a", "b", "c", "d"), peer_mass=True,
                                mp_peer_edges=False, partial_state_contract=CONTRACT)


def decode(state, monkeypatch, **env):
    """One decode on a fresh graph; returns (served plan, the K sampled plans or None)."""
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    plat, psc = state
    graph = TL.synthetic_graph(plat, psc)
    graph._bok_seed = [7, 0, 3]
    with torch.no_grad():
        combo = decode_prefix_conditioned(_model(), graph, _options())
    for k in env:
        monkeypatch.delenv(k, raising=False)
    return tuple(combo), getattr(graph, "_bok_combos", None)


def test_the_fixture_decodes_greedily(state, monkeypatch):
    plan, combos = decode(state, monkeypatch)
    assert len(plan) == 3 and combos is None


@pytest.mark.parametrize("passes", [0, 2])
def test_k1_without_noise_is_the_greedy_plan_bit_for_bit(state, monkeypatch, passes):
    served, _ = decode(state, monkeypatch, GNN_PREFIX_SELF_REFINE=passes)
    plan, combos = decode(state, monkeypatch, GNN_PREFIX_SELF_REFINE=passes, HEROSIM_BOK_K=1, HEROSIM_BOK_TAU=1.0, HEROSIM_BOK_NOISE=0)
    assert plan == served and combos == [served]
    plan_b, combos_b = decode(state, monkeypatch, GNN_PREFIX_SELF_REFINE=passes, HEROSIM_BOK_K=1, HEROSIM_BOK_TAU=1.0, HEROSIM_BOK_NOISE=0, HEROSIM_BOK_BATCHED=1)
    assert plan_b == served and combos_b == [served]


@pytest.mark.parametrize("passes", [0, 2])
@pytest.mark.parametrize("K", [4, 8])
def test_batched_k_equals_sequential_k_for_fixed_seeds(state, monkeypatch, K, passes):
    kw = dict(GNN_PREFIX_SELF_REFINE=passes, HEROSIM_BOK_K=K, HEROSIM_BOK_TAU=0.5, HEROSIM_BOK_NOISE=1)
    seq_plan, seq = decode(state, monkeypatch, **kw)
    bat_plan, bat = decode(state, monkeypatch, HEROSIM_BOK_BATCHED=1, **kw)
    assert seq is not None and bat is not None and len(seq) == len(bat) == K
    assert [tuple(map(tuple, c)) for c in bat] == [tuple(map(tuple, c)) for c in seq] and bat_plan == seq_plan
    assert len({tuple(map(tuple, c)) for c in seq}) > 1, "the noise must sample different plans (else the equality is vacuous)"


def test_the_static_memo_leaves_greedy_plans_unchanged(state, monkeypatch):
    calls = {"n": 0}
    real = LF.static_edge_columns

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(LF, "static_edge_columns", counting)
    for passes in (0, 2):
        calls["n"] = 0
        plain, _ = decode(state, monkeypatch, GNN_PREFIX_SELF_REFINE=passes)
        n_plain = calls["n"]
        calls["n"] = 0
        memo, _ = decode(state, monkeypatch, GNN_PREFIX_SELF_REFINE=passes, HEROSIM_PS_STATIC_MEMO=1)
        assert memo == plain
        assert 0 < calls["n"] < n_plain, "the memo must actually skip recomputations (else the equality is vacuous)"


def test_two_runs_at_one_seed_give_the_same_plan_and_other_seeds_may_not(state, monkeypatch):
    kw = dict(HEROSIM_BOK_K=4, HEROSIM_BOK_TAU=0.5, HEROSIM_BOK_NOISE=1)
    a, ca = decode(state, monkeypatch, **kw)
    b, cb = decode(state, monkeypatch, **kw)
    assert a == b and ca == cb
    plat, psc = state
    graph = TL.synthetic_graph(plat, psc)
    graph._bok_seed = [8, 0, 3]
    for k, v in kw.items():
        monkeypatch.setenv(k, str(v))
    with torch.no_grad():
        decode_prefix_conditioned(_model(), graph, _options())
    assert graph._bok_combos != ca  # a different seed draws different Gumbel noise
