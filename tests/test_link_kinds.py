"""live_link_features_v1: the four v7 gate kinds, their stems, sidechecks, MP-OFF switch and the untrained-checkpoint shape rule."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))

KINDS = ("ra_gnn_eng_combfl_link", "ra_twin_eng_combfl_link", "ra_mlp_same_combfl_link", "ra_mlp_wide_combfl_link")


def test_kinds_stems_contract_and_mp_off():
    import fresh_topo_burst_v1_gate as G

    assert G.COMB_LINK == KINDS
    assert [G.COMB_STEMS[k] for k in KINDS] == ["uncut-link-gnn-eng", "uncut-link-twin", "uncut-link-mlp-same", "uncut-link-mlp-wide"]
    assert [G.COMB_CHECK[k] for k in KINDS] == ["ra_gnn_eng_v7", "ra_twin_eng_v7", "ra_mlp_same_v7", "ra_mlp_same_v7"]
    for k in KINDS:
        assert k in G.RA_KINDS and G.comb_contract(k) == "partial_state_v7" and k + "_nosplit" in G.R1A_ARMS
    assert "ra_twin_eng_combfl_link" in G.RA_MP_OFF and not {"ra_gnn_eng_combfl_link", "ra_mlp_same_combfl_link"} & set(G.RA_MP_OFF)
    assert G.comb_contract("ra_gnn_eng_combfl") == "partial_state_v6" and G.comb_contract("ra_gnn_eng_combfl_dgu") == "partial_state_v6"


def test_the_graph_free_link_kinds_serve_as_mlp_same():
    src = open(REPO_ROOT / "scripts_cosim" / "fresh_topo_burst_v1_gate.py").read()
    assert 'if base_kind in ("ra_mlp_same_combfl_link", "ra_mlp_wide_combfl_link"):\n                env["GNN_ARM_KIND"] = "mlp_same"' in src


def test_every_link_kind_has_no_w_alias_to_a_v6_key():
    import json

    import fresh_topo_burst_v1_gate as G

    f = json.load(open(G.DC_W_FILE))
    for k in KINDS:
        for top in ("w_ms_per_plan", "w_ms_per_plan_by_bucket"):
            assert k + "_nosplit" not in f.get(top, {}) or f[top][k + "_nosplit"] not in [v for kk, v in f[top].items() if kk != k + "_nosplit" and "link" not in kk]


def _sidecar(arm_kind, mp_off, conv):
    return {"arm_kind": arm_kind, "disable_message_passing": mp_off, "partial_state_contract": "partial_state_v7", "load_seconds": True, "exchange_seconds": True,
            "full_context_ce_weight": 1.0, "dag_alpha_key": "inf", "mp_residual": False, "plan_raw": False, "plan_raw_sum": False, "plan_raw_local": False,
            "mp_bipartite_edge_conv": conv, "mp_bipartite_edge_attr_zero": conv, "mp_peer_edges": arm_kind == "gnn", "partial_state_feature_dim": 30,
            "candidate_slate": "declared_pruning_v1_uncapped", "physics_env": {"transfer_model": "pipelined"}}


@pytest.mark.parametrize("arm,kind,mp_off,conv", [("ra_gnn_eng_v7", "gnn", False, True), ("ra_twin_eng_v7", "gnn", True, False), ("ra_mlp_same_v7", "mlp_same", False, False)])
def test_sidecheck_wants_a_v7_30_wide_sidecar(arm, kind, mp_off, conv, monkeypatch, tmp_path):
    import hashlib

    import joint_burst_v2_sidecheck as S

    monkeypatch.setenv("R1A_SLATE", "declared_pruning_v1_uncapped")
    split = tmp_path / "split.json"
    split.write_text("{}")
    good = {**_sidecar(kind, mp_off, conv), "split_artifact": {"sha256": hashlib.sha256(split.read_bytes()).hexdigest()}}
    assert S.check_ra(dict(good), arm, str(split), "inf") == 0
    for bad in ({"partial_state_contract": "partial_state_v6"}, {"partial_state_feature_dim": 27}, {"split_artifact": None}):
        assert S.check_ra({**good, **bad}, arm, str(split), "inf") == 1


def test_untrained_checkpoint_shape_rule_names_the_grown_inputs():
    import torch

    import v7_untrained_arm_checkpoints as C

    old = {"platform_encoder.net.0.weight": torch.zeros(8, 18), "edge_scorer.fc1.weight": torch.zeros(8, 40), "other": torch.zeros(3)}
    new = {"platform_encoder.net.0.weight": torch.zeros(8, 21), "edge_scorer.fc1.weight": torch.zeros(8, 43), "other": torch.zeros(3)}
    C.check_shapes("ra_gnn_eng_combfl_link", new, old)
    with pytest.raises(SystemExit, match="grown tensors"):
        C.check_shapes("ra_gnn_eng_combfl_link", {**new, "other": torch.zeros(4)}, old)
    with pytest.raises(SystemExit, match="grown tensors"):
        C.check_shapes("ra_gnn_eng_combfl_link", {**new, "edge_scorer.fc1.weight": torch.zeros(8, 42)}, old)
    mlp_old, mlp_new = {"net.0.weight": torch.zeros(5, 59)}, {"net.0.weight": torch.zeros(5, 65)}
    C.check_shapes("ra_mlp_wide_combfl_link", mlp_new, mlp_old)
