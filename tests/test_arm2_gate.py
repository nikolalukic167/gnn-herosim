"""Arm 2 gate wiring (scripts_cosim/fresh_topo_burst_v1_gate.py, joint_burst_v2_sidecheck.py)."""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
import fresh_topo_burst_v1_gate as G  # noqa: E402
import joint_burst_v2_sidecheck as SC  # noqa: E402

GNN, TWIN = "ra_gnn_eng_arm2", "ra_twin_eng_arm2"


def test_arm2_kinds_are_gate_arms_but_not_default_grid():
    for base in (GNN, TWIN):
        k = base + "_nosplit"
        assert k in G.R1A_ARMS and base in G.RA_KINDS and next((k[:-len(s)] for s in G.SUFFIXES if k.endswith(s)), k) == base
        assert k in G.R1A_CDX and k not in G.R1A_NOSPLIT
    assert TWIN in G.RA_MP_OFF and GNN not in G.RA_MP_OFF
    src = open(G.__file__).read()
    assert 'stem = os.environ.get("R1A_ARM2_STEM", "uncut-arm2") + "-" + base_kind[3:-len("_arm2")]' in src
    assert G.split_dc(f"{GNN}_nosplit_k100") == (f"{GNN}_nosplit", 1.0)


def _sidecar(tmp_path, **over):
    split = tmp_path / "split.json"
    split.write_text("{}")
    sc = {"arm_kind": "gnn", "disable_message_passing": False, "partial_state_contract": "partial_state_v5",
          "load_seconds": True, "exchange_seconds": True, "full_context_ce_weight": 1.0, "dag_alpha_key": "inf",
          "mp_residual": False, "plan_raw": False, "plan_raw_sum": False, "plan_raw_local": False,
          "mp_bipartite_edge_conv": True, "mp_bipartite_edge_attr_zero": True, "mp_peer_edges": True,
          "partial_state_feature_dim": 27, "candidate_slate": "declared_pruning_v1_uncapped", "physics_env": {"x": 1},
          "split_artifact": {"sha256": hashlib.sha256(split.read_bytes()).hexdigest()},
          "reencode_every": "auto", "group_anchor": True, "random_prefix_k": 2, "anchor_ce_weight": 0.5}
    sc.update(over)
    path = tmp_path / "c.contract.json"
    path.write_text(json.dumps(sc))
    return str(path), str(split)


def test_sidecheck_wants_the_arm2_flags_and_refuses_them_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setenv("R1A_SLATE", "declared_pruning_v1_uncapped")
    c, s = _sidecar(tmp_path)
    assert SC.check_ra(json.load(open(c)), GNN, s, "inf") == 0
    assert SC.check_ra(json.load(open(c)), "ra_gnn_eng", s, "inf") == 1  # an Arm 2 checkpoint served as Arm 1
    c2, s2 = _sidecar(tmp_path, reencode_every="inf", group_anchor=False)
    assert SC.check_ra(json.load(open(c2)), GNN, s2, "inf") == 1  # an Arm 1 checkpoint served as Arm 2
    assert SC.check_ra(json.load(open(c2)), "ra_gnn_eng", s2, "inf") == 0
    c3, s3 = _sidecar(tmp_path, disable_message_passing=True, mp_bipartite_edge_conv=False, mp_bipartite_edge_attr_zero=False)
    assert SC.check_ra(json.load(open(c3)), TWIN, s3, "inf") == 0


def test_arm2_arm_without_its_own_w_fails_loud_and_never_borrows_nosplit():
    for kind in (f"{GNN}_nosplit", f"{TWIN}_nosplit"):
        with pytest.raises(SystemExit, match="no calibrated w|null"):
            G.dc_cost_ms(kind, 1.0, "heavy")


def test_arm2k0_kinds_are_their_own_gate_arms_with_k0_sidecars(tmp_path):
    for base in ("ra_gnn_eng_arm2k0", "ra_twin_eng_arm2k0"):
        k = base + "_nosplit"
        assert k in G.R1A_ARMS and base in G.RA_KINDS and base in G.ARM2_ALL and base not in G.ARM2_KINDS
        assert next((k[:-len(s)] for s in G.SUFFIXES if k.endswith(s)), k) == base
        assert G.split_dc(f"{k}_k100") == (k, 1.0)
        assert SC.ARM2_RANDOM_PREFIX_K[base] == 0
    assert "ra_twin_eng_arm2k0" in G.RA_MP_OFF and "ra_gnn_eng_arm2k0" not in G.RA_MP_OFF
    assert 'stem = os.environ.get("R1A_ARM2K0_STEM", "uncut-arm2k0")' in open(G.__file__).read()
