"""Combined-lever gate wiring (G8): scripts_cosim/fresh_topo_burst_v1_gate.py and joint_burst_v2_sidecheck.py."""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
import fresh_topo_burst_v1_gate as G  # noqa: E402
import joint_burst_v2_sidecheck as SC  # noqa: E402


def test_comb_kinds_are_nosplit_gate_arms_outside_the_default_grid():
    assert set(G.COMB_KINDS) == {"ra_gnn_eng_combf", "ra_gnn_eng_combl", "ra_gnn_eng_combfl", "ra_mlp_same_combfl"}
    for base in G.COMB_KINDS:
        k = base + "_nosplit"
        assert k in G.R1A_ARMS and k in G.R1A_NOSPLIT and base in G.RA_KINDS and base not in G.RA_MP_OFF
        assert next((k[:-len(s)] for s in G.SUFFIXES if k.endswith(s)), k) == base
        assert G.split_dc(f"{k}_k100") == (k, 1.0)
    assert G.COMB_STEMS["ra_gnn_eng_combfl"] == "uncut-comb-fl-gnn-eng" and G.COMB_STEMS["ra_mlp_same_combfl"] == "uncut-comb-fl-mlp-same"


def test_comb_contract_per_kind():
    assert [G.comb_contract(k) for k in G.COMB_KINDS] == ["partial_state_v6", "partial_state_v5", "partial_state_v6", "partial_state_v6"]
    assert G.comb_contract("ra_gnn_eng_arm1") == "partial_state_v5" and G.comb_contract("ra_gnn_eng") == "partial_state_v5"


@pytest.mark.parametrize("kind,sc,ok", [
    ("ra_gnn_eng_combf", {"batched_teacher_forced": True, "partial_state_contract": "partial_state_v6"}, True),
    ("ra_gnn_eng_combf", {"batched_teacher_forced": True, "partial_state_contract": "partial_state_v5"}, False),
    ("ra_gnn_eng_combl", {"batched_teacher_forced": True, "partial_state_contract": "partial_state_v5"}, True),
    ("ra_gnn_eng_combl", {"batched_teacher_forced": True, "partial_state_contract": "partial_state_v6"}, False),
    ("ra_gnn_eng_combfl", {"batched_teacher_forced": None, "partial_state_contract": "partial_state_v6"}, False),
    ("ra_gnn_eng_combfl", {"batched_teacher_forced": True, "partial_state_contract": "partial_state_v6", "regret_lambda": 2.0}, False),
    ("ra_mlp_same_combfl", {"batched_teacher_forced": True, "partial_state_contract": "partial_state_v6"}, True),
])
def test_comb_trainer_check(tmp_path, kind, sc, ok):
    p = tmp_path / "c.contract.json"
    p.write_text(json.dumps(sc))
    if ok:
        G.comb_trainer_check(kind, str(p))
    else:
        with pytest.raises(SystemExit, match="FAIL LOUD"):
            G.comb_trainer_check(kind, str(p))


def test_comb_trainer_check_ignores_other_kinds(tmp_path):
    p = tmp_path / "c.contract.json"
    p.write_text("{}")
    G.comb_trainer_check("ra_gnn_eng_arm1", str(p))


def _sidecar(tmp_path, **over):
    split = tmp_path / "split.json"
    split.write_text("{}")
    sc = {"arm_kind": "gnn", "disable_message_passing": False, "partial_state_contract": "partial_state_v6",
          "load_seconds": True, "exchange_seconds": True, "full_context_ce_weight": 1.0, "dag_alpha_key": "inf",
          "mp_residual": False, "plan_raw": False, "plan_raw_sum": False, "plan_raw_local": False,
          "mp_bipartite_edge_conv": True, "mp_bipartite_edge_attr_zero": True, "mp_peer_edges": True,
          "partial_state_feature_dim": 27, "candidate_slate": "declared_pruning_v1_uncapped", "physics_env": {"x": 1},
          "split_artifact": {"sha256": hashlib.sha256(split.read_bytes()).hexdigest()}}
    sc.update(over)
    return sc, str(split)


def test_sidecheck_v6_arms_want_v6_and_v5_arms_refuse_it(tmp_path, monkeypatch):
    monkeypatch.setenv("R1A_SLATE", "declared_pruning_v1_uncapped")
    sc, split = _sidecar(tmp_path)
    assert SC.check_ra(sc, "ra_gnn_eng_v6", split, "inf") == 0
    assert SC.check_ra(sc, "ra_gnn_eng", split, "inf") == 1
    sc5, _ = _sidecar(tmp_path, partial_state_contract="partial_state_v5")
    assert SC.check_ra(sc5, "ra_gnn_eng_v6", split, "inf") == 1 and SC.check_ra(sc5, "ra_gnn_eng", split, "inf") == 0
    mlp, _ = _sidecar(tmp_path, arm_kind="mlp_same", mp_bipartite_edge_conv=False, mp_bipartite_edge_attr_zero=False, mp_peer_edges=False)
    assert SC.check_ra(mlp, "ra_mlp_same_v6", split, "inf") == 0


def test_comb_arm_charges_its_own_frozen_w(tmp_path, monkeypatch):
    for k in G.COMB_KINDS:
        for rung in ("moderate", "heavy"):
            env, val = G.dc_cost_ms(f"{k}_nosplit", 1.0, rung)
            assert env == G.DC_BUCKET_ENV and all(v > 0 for v in json.loads(val).values())
    f = json.load(open(G.DC_W_FILE))
    for k in G.COMB_KINDS:
        f["w_ms_per_plan"].pop(f"{k}_nosplit", None)
        f.get("w_ms_per_plan_by_bucket", {}).pop(f"{k}_nosplit", None)
    p = tmp_path / "w.json"
    p.write_text(json.dumps(f))
    monkeypatch.setattr(G, "DC_W_FILE", str(p))
    for k in G.COMB_KINDS:
        with pytest.raises(SystemExit, match="no calibrated w|null"):
            G.dc_cost_ms(f"{k}_nosplit", 1.0, "heavy")


def test_wide_mlp_is_a_nosplit_gate_arm_outside_the_other_kind_sets():
    (base,) = G.COMB_WIDE
    k = base + "_nosplit"
    assert base == "ra_mlp_wide_combfl" and G.COMB_STEMS[base] == "uncut-comb-fl-mlp-wide"
    assert k in G.R1A_ARMS and k in G.R1A_NOSPLIT and base in G.RA_KINDS and base not in G.RA_MP_OFF and base not in G.COMB_KINDS
    assert G.split_dc(f"{k}_k100") == (k, 1.0) and G.comb_contract(base) == "partial_state_v6" and G.COMB_CHECK[base] == "ra_mlp_same_v6"
