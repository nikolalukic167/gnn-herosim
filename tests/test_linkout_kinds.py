"""outbound_link_features_v1: the four v9 gate kinds, their stems, contract, sidecheck arms, MP-OFF switch and the absence of a w alias."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))

KINDS = ("ra_gnn_eng_combfl_linkout", "ra_twin_eng_combfl_linkout", "ra_mlp_same_combfl_linkout", "ra_mlp_wide_combfl_linkout")


def test_kinds_stems_contract_and_mp_off():
    import fresh_topo_burst_v1_gate as G

    assert G.COMB_OUT == KINDS and G.COMB_OUT_E40 == tuple(k + "_e40" for k in KINDS)
    assert [G.COMB_STEMS[k] for k in KINDS] == ["uncut-linkout-gnn-eng", "uncut-linkout-twin", "uncut-linkout-mlp-same", "uncut-linkout-mlp-wide"]
    assert [G.COMB_STEMS[k + "_e40"] for k in KINDS] == [G.COMB_STEMS[k] + "-e40" for k in KINDS]
    assert [G.COMB_CHECK[k] for k in KINDS] == ["ra_gnn_eng_v9", "ra_twin_eng_v9", "ra_mlp_same_v9", "ra_mlp_same_v9"]
    for k in KINDS + G.COMB_OUT_E40:
        assert k in G.RA_KINDS and G.comb_contract(k) == "partial_state_v9" and k + "_nosplit" in G.R1A_ARMS
    mp_off = {"ra_twin_eng_combfl_linkout", "ra_twin_eng_combfl_linkout_e40"}
    assert mp_off <= set(G.RA_MP_OFF) and not set(k for k in KINDS + G.COMB_OUT_E40 if "twin" not in k) & set(G.RA_MP_OFF)
    assert G.comb_contract("ra_gnn_eng_combfl_linkstatic") == "partial_state_v8"


def test_the_graph_free_v9_kinds_serve_as_mlp_same():
    src = open(REPO_ROOT / "scripts_cosim" / "fresh_topo_burst_v1_gate.py").read()
    assert '"ra_mlp_same_combfl_linkout", "ra_mlp_wide_combfl_linkout", "ra_mlp_same_combfl_linkout_e40", "ra_mlp_wide_combfl_linkout_e40"):\n                env["GNN_ARM_KIND"] = "mlp_same"' in src


def test_no_w_entry_is_aliased_for_a_v9_kind():
    import fresh_topo_burst_v1_gate as G

    f = json.load(open(G.DC_W_FILE))
    for top in ("w_ms_per_plan", "w_ms_per_plan_by_bucket"):
        for k, v in f.get(top, {}).items():
            if "linkout" in k:
                others = [vv for kk, vv in f[top].items() if "linkout" not in kk]
                assert v not in others, f"{top}[{k}] equals another kind's entry: an alias"


def test_the_gate_refuses_any_v9_kind_in_the_w_alias_table(tmp_path, monkeypatch):
    import pytest

    import fresh_topo_burst_v1_gate as G

    f = json.load(open(G.DC_W_FILE))
    al = f.get("w_ms_per_plan_alias", {})
    assert {k: al[k] for k in G.OUT_W_KINDS & set(al)} == G.E40_W_FROM_DEV  # the only aliased v9 entries
    for kind in KINDS + G.COMB_OUT_E40:
        for sfx in ("", "_nosplit"):
            bad = tmp_path / "w.json"
            bad.write_text(json.dumps({**f, "w_ms_per_plan_alias": {**f["w_ms_per_plan_alias"], kind + sfx: "ra_gnn_eng_combfl_linkstatic_e40_nosplit"}}))
            monkeypatch.setattr(G, "DC_W_FILE", str(bad))
            with pytest.raises(SystemExit, match="never an alias"):
                G.dc_cost_ms("ra_gnn_eng_combfl_linkstatic_e40_nosplit", 1.0, "moderate")


def test_ckpt_suffix_applies_only_to_v9_kinds_and_the_gate_fails_loud_on_a_missing_sidecar(monkeypatch):
    import fresh_topo_burst_v1_gate as G

    monkeypatch.delenv("R1A_CKPT_SUFFIX", raising=False)
    assert all(G.ckpt_suffix(k) == "" for k in KINDS + G.COMB_OUT_E40)
    monkeypatch.setenv("R1A_CKPT_SUFFIX", "-final")
    assert all(G.ckpt_suffix(k) == "-final" for k in KINDS + G.COMB_OUT_E40)
    assert G.ckpt_suffix("ra_gnn_eng_combfl_linkstatic_e40") == ""
    src = open(REPO_ROOT / "scripts_cosim" / "fresh_topo_burst_v1_gate.py").read()
    assert 'f"{stem}-seed{seed}{sfx}.pt"' in src and "R1A_CKPT_SUFFIX={sfx!r} but the sidecar {side} is missing" in src
    rd = open(REPO_ROOT / "scripts_cosim" / "llf_confirm_read.py").read()
    assert "V9_CKPT_SUFFIX if arm_set_name == \"v9\" else None" in rd


def test_only_the_three_e40_kinds_alias_and_each_to_its_own_dev_kind(tmp_path, monkeypatch):
    import pytest

    import fresh_topo_burst_v1_gate as G

    f = json.load(open(G.DC_W_FILE))
    assert set(G.E40_W_FROM_DEV) == {f"{k}_e40_nosplit" for k in ("ra_gnn_eng_combfl_linkout", "ra_twin_eng_combfl_linkout", "ra_mlp_wide_combfl_linkout")}
    for e, d in G.E40_W_FROM_DEV.items():
        assert d == e.replace("_e40", "") and d in f["w_ms_per_plan"]
        assert G.dc_cost_ms(e, 1.0, "heavy") == G.dc_cost_ms(d, 1.0, "heavy")
    for e, d in G.E40_W_FROM_DEV.items():  # a re-pointed alias, or one for any other v9 kind, is refused
        for key, val in ((e, "ra_gnn_eng_combfl_linkstatic_e40_nosplit"), ("ra_mlp_same_combfl_linkout_e40_nosplit", d), ("ra_gnn_eng_combfl_linkout_nosplit", d)):
            bad = tmp_path / "w.json"
            bad.write_text(json.dumps({**f, "w_ms_per_plan_alias": {**f["w_ms_per_plan_alias"], key: val}}))
            monkeypatch.setattr(G, "DC_W_FILE", str(bad))
            with pytest.raises(SystemExit, match="never an alias"):
                G.dc_cost_ms("ra_gnn_eng_combfl_linkstatic_e40_nosplit", 1.0, "moderate")


def test_mlp_e40_entry_is_the_dev_moderate_w_at_both_rungs():
    import fresh_topo_burst_v1_gate as G

    f = json.load(open(G.DC_W_FILE))
    e, d = "ra_mlp_same_combfl_linkout_e40_nosplit", "ra_mlp_same_combfl_linkout_nosplit"
    assert e not in f["w_ms_per_plan_alias"]
    assert f["w_ms_per_plan"][e] == {"moderate": f["w_ms_per_plan"][d]["moderate"], "heavy": f["w_ms_per_plan"][d]["moderate"]}
    assert f["w_ms_per_plan_by_bucket"][e]["heavy"] == f["w_ms_per_plan_by_bucket"][d]["moderate"] == f["w_ms_per_plan_by_bucket"][e]["moderate"]
    assert f["w_ms_per_plan"][d]["heavy"] > 2 * f["w_ms_per_plan"][d]["moderate"]  # the runaway dev cell's w is not used
    assert "_source_" + e in f
