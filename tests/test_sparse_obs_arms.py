"""sparse_obs_s0_v1: the degraded-observation arms are registered, single-seed, outside the default grid, and search exactly as cd_exactS_ib0."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import fresh_topo_burst_v1_gate as G  # noqa: E402


def test_arms_registered_like_ib0():
    assert set(G.OBS_ARMS) <= set(G.R1A_DIAG) and set(G.OBS_ARMS) <= set(G.RULE_POLICY)
    for k in G.OBS_ARMS:
        assert G.RULE_POLICY[k] == G.RULE_POLICY["cd_exactS_ib0"] and G.EXACT_ALPHA[k] == G.EXACT_ALPHA["cd_exactS_ib0"]
        assert set(G.OBS_ARMS[k]) <= set(G.OBS_ENV)
        assert G.OBS_ARMS[k].get("HEROSIM_OBS_ESTIMATOR", "naive") in ("naive", "ec")


def test_bundle_degrades_queue_exec_partners_and_leaves_warmth_and_inflight():
    b = G.OBS_ARMS["cd_obs_bundle_ec"]
    assert {"HEROSIM_OBS_QUEUE", "HEROSIM_OBS_EXEC_SIGMA", "HEROSIM_OBS_PARTNERS"} <= set(b)
    assert not any("WARM" in k or "INFLIGHT" in k for k in b)  # the reference does not read in-flight; warmth is stage 2


def test_obs_env_is_scrubbed_served_and_recorded():
    src = Path(G.__file__).read_text()
    assert "for k in OBS_ENV:\n        env.pop(k, None)" in src and "env.update(OBS_ARMS[kind]" in src
    rec = (Path(G.__file__).resolve().parents[1] / "src/executesimulation.py").read_text()
    assert all(k in rec for k in G.OBS_ENV)


def test_stage2_warmth_arms():
    a, n, e = (G.OBS_ARMS[k] for k in ("cd_obs_a0plus", "cd_obs_w1_naive", "cd_obs_w1_ec"))
    assert a == {"HEROSIM_PG_COLD_TAIL": "1"}
    assert n["HEROSIM_PG_COLD_TAIL"] == e["HEROSIM_PG_COLD_TAIL"] == "1" and n["HEROSIM_OBS_WARM"] == e["HEROSIM_OBS_WARM"] == "seconds"
    assert (n["HEROSIM_OBS_ESTIMATOR"], e["HEROSIM_OBS_ESTIMATOR"]) == ("naive", "ec")
