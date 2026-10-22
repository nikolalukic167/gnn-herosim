"""physMP quick look: FL gnn with the physics columns live inside message passing (NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO=0), 20 epochs, seed 1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))

KIND = "ra_gnn_eng_physmp_combfl"


def test_the_config_is_fl_gnn_with_exactly_the_two_changes():
    c = yaml.safe_load((REPO_ROOT / "experiments/uncut_comb_fl_gnn_physmp.yaml").read_text())
    assert c["env"]["NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO"] == "0"
    assert c["env"]["NEAR_RTT_MP_BIPARTITE_EDGE_CONV"] == "1" and c["env"]["NEAR_RTT_MP_PEER_EDGES"] == "1"
    assert c["env"]["PARTIAL_STATE_CONTRACT"] == "partial_state_v6" and c["env"]["NEAR_RTT_SPLIT_ARTIFACT"] == "experiments/uncut_arm1_split.json"
    assert c["env"]["NEAR_RTT_LABEL_OVERRIDE_JSON"] == "/share/nikola.lukic/uncut_arm1/label_screen/sim_labels_arm1.json"
    assert c["cache_dir"] == "/share/nikola.lukic/uncut_groups/inflight/cache_v6_a/cache" and c["unset_env"] == ["TRAIN_INIT_CHECKPOINT", "NEAR_RTT_LABEL_OVERRIDE_JSON"]
    a = c["args"]
    assert (a["epochs"], a["min-epochs"], a["patience"]) == (20, 20, 20) and a["learning-rate"] == 0.001 and a["ce-loss-weight"] == 1 and a["regret-loss-weight"] == 0
    assert c["wandb"]["project"] == "gnn-uncut-arm1" and c["wandb"]["run_name"] == "uncut-comb-fl-gnn-physmp"


def test_the_train_sbatch_checks_the_v6_physmp_sidecar():
    text = (REPO_ROOT / "scripts_cosim/datalab/physmp_train.sbatch").read_text()
    assert "experiments/uncut_comb_fl_gnn_physmp.yaml" in text and '"ra_gnn_eng_physmp_v6"' in text and "613c8cbb1cf494b60a215d3fccb4d30e6cfe17943304d7078a7b8f58f6f34808" in text
    import joint_burst_v2_sidecheck as S
    assert S.COMB_V6_ARMS["ra_gnn_eng_physmp_v6"] == S.RA_ARMS["ra_gnn_eng_physmp"] and S.RA_ARMS["ra_gnn_eng_physmp"][4] is True


def test_the_gate_serves_the_kind_like_fl_gnn_with_its_own_stem_and_the_v6_check():
    import fresh_topo_burst_v1_gate as G
    assert G.COMB_STEMS[KIND] == "uncut-comb-fl-gnn-physmp" and KIND in G.RA_KINDS and KIND in G.COMB_PHYSMP
    assert G.COMB_CHECK[KIND] == "ra_gnn_eng_physmp_v6" and G.comb_contract(KIND) == "partial_state_v6"
    k = KIND + "_nosplit"
    assert k in G.R1A_ARMS and G.split_dc(k + "_k100") == (k, 1.0)
    assert "ra_gnn_eng_combfl_nosplit" == json.load(open(G.DC_W_FILE))["w_ms_per_plan_alias"][k]
    for rung in ("moderate", "heavy"):
        assert G.dc_cost_ms(k, 1.0, rung) == G.dc_cost_ms("ra_gnn_eng_combfl_nosplit", 1.0, rung)
    with pytest.raises(SystemExit, match="no calibrated w"):
        G.dc_cost_ms(KIND + "_nosplit_x", 1.0, "moderate")
