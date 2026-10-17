import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))


def test_the_bar_sensitivity_kind_is_cd_ib0_plus_the_inflight_flag_and_outside_the_default_grid():
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    k = "cd_ib0_inflight"
    assert G.RULE_POLICY[k] == G.RULE_POLICY["cd_ib0"] == "peer_greedy_network_cd"
    assert G.ib0_alpha(k) == G.ib0_alpha("cd_ib0") == {"HEROSIM_PG_BATCH_ALPHA": "0"}
    assert G.check_alias(k) == "cd_inflight" and G.check_alias("cd_ib0") == "cd"
    assert k in G.R1A_DIAG and k in G.R1A_ARMS and k not in G.R1A_CLASSICAL
    assert G.split_dc(k + "_k100") == (k, 1.0)
