"""Fair-share label arms: the filed FL gnn / twin / MLP recipes, 20 epochs, seed 1, trained on the fair-share label override."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts_cosim"))

ARMS = {"gnn": "ra_gnn_eng", "twin": "ra_twin_eng", "mlp": "ra_mlp_same"}


@pytest.mark.parametrize("arm", ARMS)
def test_the_config_is_the_filed_fl_recipe_with_the_three_changes(arm):
    c = yaml.safe_load((REPO_ROOT / f"experiments/uncut_comb_fl_{arm}_fslabel.yaml").read_text())
    e, a = c["env"], c["args"]
    assert e["NEAR_RTT_LABEL_OVERRIDE_JSON"].endswith("/label_override_fs.json")
    assert e["NEAR_RTT_LABEL_OVERRIDE_JSON"] != "/share/nikola.lukic/uncut_arm1/label_screen/sim_labels_arm1.json"
    assert (a["epochs"], a["min-epochs"], a["patience"]) == (20, 20, 20)
    assert c["wandb"]["run_name"] == f"uncut-comb-fl-{arm}-fslabel" and c["wandb"]["project"] == "gnn-uncut-arm1"
    # unchanged FL recipe
    assert c["cache_dir"] == "/share/nikola.lukic/uncut_groups/inflight/cache_v6_a/cache" and c["unset_env"] == ["TRAIN_INIT_CHECKPOINT", "NEAR_RTT_LABEL_OVERRIDE_JSON"]
    assert e["PARTIAL_STATE_CONTRACT"] == "partial_state_v6" and e["NEAR_RTT_SPLIT_ARTIFACT"] == "experiments/uncut_arm1_split.json" and e["NEAR_RTT_TRAIN_OBJECTIVE"] == "ce_only"
    assert a["learning-rate"] == 0.001 and a["ce-loss-weight"] == 1 and a["regret-loss-weight"] == 0
    if arm == "gnn":
        assert e["NEAR_RTT_MP_BIPARTITE_EDGE_CONV"] == "1" and e["NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO"] == "1" and "GNN_DISABLE_MESSAGE_PASSING" not in e
    if arm == "twin":
        assert e["GNN_DISABLE_MESSAGE_PASSING"] == "1" and "NEAR_RTT_MP_BIPARTITE_EDGE_CONV" not in e
    if arm == "mlp":
        assert e["NEAR_RTT_ARM"] == "mlp_same"


def test_the_three_configs_name_one_override_file():
    paths = {yaml.safe_load((REPO_ROOT / f"experiments/uncut_comb_fl_{a}_fslabel.yaml").read_text())["env"]["NEAR_RTT_LABEL_OVERRIDE_JSON"] for a in ARMS}
    assert len(paths) == 1


def test_the_train_sbatch_asserts_the_override_sha_and_the_v6_sidecheck_per_arm():
    t = (REPO_ROOT / "scripts_cosim/datalab/fslabel_train.sbatch").read_text()
    assert "LABEL_SHA256" in t and "sha256sum \"$LABELF\"" in t and "613c8cbb1cf494b60a215d3fccb4d30e6cfe17943304d7078a7b8f58f6f34808" in t
    for arm, kind in (("gnn", "ra_gnn_eng_v6"), ("twin", "ra_twin_eng_v6"), ("mlp", "ra_mlp_same_v6")):
        assert f"{arm}) KIND={kind}" in t
    assert "uncut_comb_fl_${ARM}_fslabel.yaml" in t


@pytest.mark.parametrize("arm,base", ARMS.items())
def test_the_gate_serves_the_kind_like_its_filed_fl_arm(arm, base):
    import fresh_topo_burst_v1_gate as G
    kind = f"{base}_fslabel_combfl"
    assert G.COMB_STEMS[kind] == f"uncut-comb-fl-{arm}-fslabel" and kind in G.RA_KINDS and kind in G.COMB_FSLABEL
    assert G.COMB_CHECK[kind] == f"{base}_v6" and G.comb_contract(kind) == "partial_state_v6"
    k = kind + "_nosplit"
    assert k in G.R1A_ARMS and G.split_dc(k + "_k100") == (k, 1.0)
    filed = f"{base}_combfl_nosplit"
    assert json.load(open(G.DC_W_FILE))["w_ms_per_plan_alias"][k] == filed
    for rung in ("moderate", "heavy"):
        assert G.dc_cost_ms(k, 1.0, rung) == G.dc_cost_ms(filed, 1.0, rung)
    assert (kind in G.RA_MP_OFF) == (arm == "twin")


def test_the_mlp_kind_gets_the_mlp_arm_kind_and_the_twin_the_mp_off_switch():
    src = (REPO_ROOT / "scripts_cosim/fresh_topo_burst_v1_gate.py").read_text()
    assert '"ra_mlp_same_fslabel_combfl"):' in src and '.removesuffix("_combfl").removesuffix("_fslabel").removesuffix("_flreg")' in src
    assert "mlp_same_fslabel_combfl"[:].removesuffix("_combfl").removesuffix("_fslabel") == "mlp_same"
