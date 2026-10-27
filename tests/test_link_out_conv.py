"""Lever 2 (S7, 2026-10-27): R1-R3/U1 of the committed prefix in the bipartite conv's edge attributes (mp_link_out_conv, src/policy/gnn/link_conv.out_rows)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import test_batched_tf_link as TB  # noqa: E402
from src.policy.gnn import link_conv as LC  # noqa: E402
from src.policy.gnn import link_static as LSt  # noqa: E402
from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402
from src.policy.tabular import reduced_features as RF  # noqa: E402

N = TB.N_TASKS
V8, V9 = RF.PARTIAL_STATE_CONTRACT_V8, RF.PARTIAL_STATE_CONTRACT_V9
OUT0 = 33


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", V9)
    monkeypatch.setenv("PARTIAL_STATE_LOAD_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_EXCHANGE_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_PEER_MASS", "1")
    monkeypatch.delenv("GNN_DISABLE_MESSAGE_PASSING", raising=False)
    monkeypatch.delenv("GNN_MP_LINK_OUT_CONV", raising=False)


def v9_scenario(seed, monkeypatch):
    rnd, ctx, g, place = TB.scenario(seed, monkeypatch)
    ctx.contract = V9
    return rnd, ctx, g, place


def model_for(g, ctx, out_conv=True, prenorm=True):
    g = TB._model_graph(g)
    g.link_static_attr = LSt.link_static_attr_from_ctx(g, ctx)
    torch.manual_seed(0)
    m = TaskPlacementGNN(task_feature_dim=5, platform_feature_dim=7, embedding_dim=16, hidden_dim=16, num_layers=2, edge_dim=5, task_type_onehot_dim=4,
                         partial_state_edge_dim=37, mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True, mp_link_static_attr=True, mp_link_out_conv=out_conv,
                         mp_peer_edges=False, mp_conv_prenorm_residual=prenorm, mp_conv_prenorm_final_norm=prenorm).eval()
    return g, m


@pytest.mark.parametrize("seed", range(5))
def test_out_rows_equal_the_scorer_definition_with_the_tasks_own_placement_left_out(seed, monkeypatch):
    rnd, ctx, g, place = v9_scenario(seed, monkeypatch)
    LS = LC.static_of(g, ctx)
    S = LS.S
    states = []
    for _ in range(10):
        plan = [rnd.randrange(len(place[t])) for t in range(N)]
        states.append((plan, [j for j in range(N) if rnd.random() < rnd.choice([0.2, 0.5, 0.9])]))
    cm = np.zeros((len(states), N))
    nd = np.zeros((len(states), N), dtype=np.int64)
    for s, (plan, cs) in enumerate(states):
        cm[s, cs] = 1.0
        nd[s] = S.row_node[S.start + np.asarray(plan)]
    rows = LC.out_rows(LS, torch.from_numpy(cm), torch.from_numpy(nd)).numpy().astype(np.float32)
    live = 0
    for s, (plan, cs) in enumerate(states):
        for t in range(N):
            committed = {j: tuple(place[j][plan[j]]) for j in sorted(cs) if j != t}
            want = RF.partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed)[:, OUT0:OUT0 + 4]
            got = rows[s, S.start[t]:S.start[t] + S.C[t]]
            assert np.array_equal(got, want.astype(np.float32)), (s, t, got, want)
            live += int((got[:, 3] != 0).sum())
    assert live > 0  # U1 is live, so the comparison is not vacuous


def test_out_rows_refuse_a_contract_without_the_outbound_block(monkeypatch):
    rnd, ctx, g, place = TB.scenario(1, monkeypatch)
    ctx.contract = V8
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", V8)
    LS = LC.static_of(g, ctx)
    with pytest.raises(ValueError, match="outbound conv block"):
        LC.out_rows(LS, torch.zeros(1, N, dtype=torch.float64), torch.zeros(1, N, dtype=torch.long))


def test_the_flag_is_off_by_default_and_leaves_the_v9_architecture_unchanged(monkeypatch):
    rnd, ctx, g, place = v9_scenario(2, monkeypatch)
    g, plain = model_for(g, ctx, out_conv=False, prenorm=False)
    torch.manual_seed(0)
    default = TaskPlacementGNN(task_feature_dim=5, platform_feature_dim=7, embedding_dim=16, hidden_dim=16, num_layers=2, edge_dim=5, task_type_onehot_dim=4,
                               partial_state_edge_dim=37, mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True, mp_link_static_attr=True, mp_peer_edges=False).eval()
    assert default.mp_link_out_conv is False and default.conv_edge_extras == [("link_static_attr", 3)]
    sd_a, sd_b = plain.state_dict(), default.state_dict()
    assert list(sd_a) == list(sd_b) and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    assert not LC.is_per_prefix(default)
    with torch.no_grad():
        ta, pa = plain._encode(g)
        tb, pb = default._encode(g)
    assert torch.equal(ta, tb) and torch.equal(pa, pb)
    monkeypatch.setenv("GNN_MP_LINK_OUT_CONV", "1")
    torch.manual_seed(0)
    from_env = TaskPlacementGNN(task_feature_dim=5, platform_feature_dim=7, embedding_dim=16, hidden_dim=16, num_layers=2, edge_dim=5, task_type_onehot_dim=4,
                                partial_state_edge_dim=37, mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True, mp_link_static_attr=True, mp_peer_edges=False)
    assert from_env.mp_link_out_conv is True


def test_the_flag_is_weight_visible_and_a_checkpoint_does_not_load_across_it(monkeypatch):
    rnd, ctx, g, place = v9_scenario(3, monkeypatch)
    g, on = model_for(g, ctx, out_conv=True)
    _g, off = model_for(g, ctx, out_conv=False)
    assert on.conv_edge_extras == [("link_static_attr", 3), ("link_out_conv_attr", 4)]
    assert on.bip_convs[0].message_mlp[0].in_features == off.bip_convs[0].message_mlp[0].in_features + 4
    with pytest.raises(RuntimeError):
        off.load_state_dict(on.state_dict(), strict=True)


def test_the_flag_is_on_the_serving_whitelists():
    import inspect

    from src.policy.gnn import arm_models, prefix_serving
    import src.executesimulation as ES

    assert "mp_link_out_conv" in arm_models.GRAPH_FLAGS
    assert 'sidecar.get("mp_link_out_conv"' in inspect.getsource(prefix_serving)
    src = inspect.getsource(ES)
    assert '"mp_link_out_conv",' in src and 'mp_cfg.get("mp_link_out_conv"' in src


def test_the_batched_loss_equals_the_sequential_closure_with_static_and_outbound_blocks(monkeypatch):
    import torch.nn.functional as F
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn

    rnd, ctx, g, place = v9_scenario(9, monkeypatch)
    g, m = model_for(g, ctx)
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


def test_the_outbound_columns_get_gradients_and_move_the_encoding(monkeypatch):
    rnd, ctx, g, place = v9_scenario(10, monkeypatch)
    g, m = model_for(g, ctx)
    plan = [rnd.randrange(len(place[t])) for t in range(N)]
    with torch.no_grad():
        setattr(g, LC.OUT_ATTR, LC.link_edge_block(g, ctx, {}, rows_fn=LC.out_rows))
        empty = m._encode(g)[1]
        setattr(g, LC.OUT_ATTR, LC.link_edge_block(g, ctx, {j: tuple(place[j][plan[j]]) for j in range(N)}, rows_fn=LC.out_rows))
        full = m._encode(g)[1]
    assert not torch.equal(empty, full)
    m.train()
    loss = LC.link_conv_loss(m, g, ctx, [plan], list(range(N)), 0.5, torch.device("cpu"))
    loss.backward()
    grad = m.bip_convs[0].message_mlp[0].weight.grad
    assert grad is not None and float(grad[:, -4:].abs().sum()) > 0.0


def test_the_sidecheck_tells_the_lever_arm_its_control_and_the_v9_gnn_apart(monkeypatch, tmp_path):
    import hashlib

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
    import joint_burst_v2_sidecheck as SC

    monkeypatch.setenv("R1A_SLATE", "declared_pruning_v1_uncapped")
    split = tmp_path / "split.json"
    split.write_text("{}")
    base = {"arm_kind": "gnn", "disable_message_passing": False, "partial_state_contract": V9, "load_seconds": True, "exchange_seconds": True, "full_context_ce_weight": 1.0,
            "dag_alpha_key": "inf", "mp_residual": False, "plan_raw": False, "plan_raw_sum": False, "plan_raw_local": False, "mp_bipartite_edge_conv": True,
            "mp_bipartite_edge_attr_zero": True, "mp_peer_edges": True, "partial_state_feature_dim": 37, "mp_link_static_attr": True,
            "candidate_slate": "declared_pruning_v1_uncapped", "physics_env": {"transfer_model": "pipelined"}, "split_artifact": {"sha256": hashlib.sha256(split.read_bytes()).hexdigest()}}
    pre = {"mp_conv_prenorm_residual": True, "mp_conv_prenorm_final_norm": True}
    sidecars = {"ra_gnn_eng_v9": base, "ra_gnn_eng_v9prenorm": {**base, **pre}, "ra_gnn_eng_v9outconv": {**base, **pre, "mp_link_out_conv": True}}
    for arm in sidecars:
        for kind, sc in sidecars.items():
            assert SC.check_ra(dict(sc), arm, str(split), "inf") == (0 if kind == arm else 1), (arm, kind)
