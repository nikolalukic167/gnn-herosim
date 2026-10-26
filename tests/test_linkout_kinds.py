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
    assert not (G.OUT_W_KINDS & set(f.get("w_ms_per_plan_alias", {})))
    for kind in KINDS + G.COMB_OUT_E40:
        for sfx in ("", "_nosplit"):
            bad = tmp_path / "w.json"
            bad.write_text(json.dumps({**f, "w_ms_per_plan_alias": {**f["w_ms_per_plan_alias"], kind + sfx: "ra_gnn_eng_combfl_linkstatic_e40_nosplit"}}))
            monkeypatch.setattr(G, "DC_W_FILE", str(bad))
            with pytest.raises(SystemExit, match="never an alias"):
                G.dc_cost_ms("ra_gnn_eng_combfl_linkstatic_e40_nosplit", 1.0, "moderate")
