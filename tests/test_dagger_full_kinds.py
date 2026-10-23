"""Own-state round, full version (6d6e1e54, A1 c810a270): the two new gate kinds, their stems, sidechecks, MP-OFF switch, _kr serving and frozen w."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))

GNN, TWIN = "ra_gnn_eng_combfl_dgu", "ra_twin_eng_combfl_dgu"



def test_kinds_stems_checks_and_mp_off():
    import fresh_topo_burst_v1_gate as G
    assert G.COMB_STEMS[GNN] == "uncut-dagger-full-gnn" and G.COMB_STEMS[TWIN] == "uncut-dagger-full-twin"
    assert G.COMB_CHECK[GNN] == "ra_gnn_eng_v6" and G.COMB_CHECK[TWIN] == "ra_twin_eng_v6"
    for k in (GNN, TWIN):
        assert k in G.RA_KINDS and k in G.COMB_DGU and G.comb_contract(k) == "partial_state_v6"
        assert k + "_nosplit" in G.R1A_ARMS
    assert TWIN in G.RA_MP_OFF and GNN not in G.RA_MP_OFF


def test_both_serve_k100_and_kr_like_the_fl_kinds():
    import fresh_topo_burst_v1_gate as G
    for k in (GNN, TWIN):
        assert G.split_dc(k + "_nosplit_k100") == (k + "_nosplit", 1.0)
        assert G.split_dc(k + "_nosplit_kr")[0] == k + "_nosplit"


OLD = {GNN: "ra_gnn_eng_combfl_nosplit", TWIN: "ra_twin_eng_combfl_nosplit"}


def test_w_entries_are_the_copied_values_of_the_old_arms():
    import fresh_topo_burst_v1_gate as G
    f = json.load(open(G.DC_W_FILE))
    for k, old in OLD.items():
        for top in ("w_ms_per_plan", "w_ms_per_plan_by_bucket"):
            assert f[top][k + "_nosplit"] == f[top][old], top
        assert f["_source_" + k + "_nosplit"]
        for rung in ("moderate", "heavy"):
            assert G.dc_cost_ms(k + "_nosplit", 1.0, rung) == G.dc_cost_ms(old, 1.0, rung)
    with pytest.raises(SystemExit, match="no calibrated w"):
        G.dc_cost_ms(GNN + "_nosplit_x", 1.0, "moderate")
