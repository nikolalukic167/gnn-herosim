"""slate_bar_v1 live arms: cd_exactS restricted to the declared slate (primary) and to CD's own top-5 (descriptive), both decoded whole."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import fresh_topo_burst_v1_gate as G  # noqa: E402


def test_arms_are_registered_single_seed_and_outside_the_default_grid():
    for k in ("cd_exactS_slate", "cd_exactS_top5cd"):
        assert k in G.RULE_POLICY and k in G.R1A_DIAG and k in G.SLATE_EXACT
    assert G.RULE_POLICY["cd_exactS_slate"] == G.RULE_POLICY["cd_exactS"] == "peer_greedy_network_cd"


def test_slate_cost_env_is_scrubbed_and_recorded():
    src = Path(G.__file__).read_text()
    assert '"HEROSIM_PG_SLATE_COST", "HEROSIM_PG_SHARED_LINK"' in src  # popped before the per-kind env is set
    assert "HEROSIM_PG_SLATE_COST" in (Path(G.__file__).resolve().parents[1] / "src/executesimulation.py").read_text()  # recorded in run_provenance.env
    assert 'env["HEROSIM_PG_SLATE_COST"] = "cd"' in src and src.count("HEROSIM_PG_SLATE_COST") >= 5
