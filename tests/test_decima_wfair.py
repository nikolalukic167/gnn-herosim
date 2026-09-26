"""decima_rule_v1: Decima's tuned weighted-fair baseline mapped onto the burst seat."""
import pytest

from src.policy.decima_wfair.scheduler import (
    DECIMA_ALPHA_ENV, DECIMA_DEFAULT_ALPHA, DecimaWeightedFairBatchScheduler, decima_alpha,
    earliest_platforms, fair_share_count, job_weight, peer_groups,
)


def test_peer_groups_are_connected_components_named_by_smallest_id():
    table = {1: {2: 5.0}, 2: {1: 5.0, 3: 1.0}, 3: {2: 1.0}, 7: {9: 2.0}, 9: {7: 2.0}}
    assert peer_groups(table) == {1: 1, 2: 1, 3: 1, 7: 7, 9: 7}
    assert peer_groups({}) == {}


def test_weight_family_spans_fair_naive_and_tuned():
    assert job_weight(4.0, 0.0) == 1.0          # fair
    assert job_weight(4.0, 1.0) == 4.0          # naive weighted fair
    assert job_weight(4.0, -1.0) == 0.25        # Decima's tuned alpha: short jobs get more
    with pytest.raises(ValueError):
        job_weight(0.0, -1.0)


def test_fair_share_count_rounds_and_clamps():
    assert fair_share_count(8, 1.0, 1.0) == 8   # alone in the system: every platform
    assert fair_share_count(8, 1.0, 4.0) == 2
    assert fair_share_count(8, 1.0, 100.0) == 1  # never below one platform
    assert fair_share_count(3, 1.0, 2.0) == 2    # 1.5 rounds half up
    with pytest.raises(ValueError):
        fair_share_count(0, 1.0, 1.0)
    with pytest.raises(ValueError):
        fair_share_count(4, 2.0, 1.0)


def test_earliest_platforms_orders_by_finish_then_ids():
    ranked = [(3.0, 1, 10), (1.0, 2, 20), (1.0, 0, 5), (9.0, 3, 30)]
    assert earliest_platforms(ranked, 2) == [(0, 5), (2, 20)]
    assert earliest_platforms(ranked, 9) == [(0, 5), (2, 20), (1, 10), (3, 30)]


def test_alpha_env(monkeypatch):
    monkeypatch.delenv(DECIMA_ALPHA_ENV, raising=False)
    assert decima_alpha() == DECIMA_DEFAULT_ALPHA == -1.0
    monkeypatch.setenv(DECIMA_ALPHA_ENV, "0.5")
    assert decima_alpha() == 0.5
    for bad in ("abc", "nan", "inf"):
        monkeypatch.setenv(DECIMA_ALPHA_ENV, bad)
        with pytest.raises(ValueError):
            decima_alpha()


def test_rule_is_locality_blind_and_registered():
    assert DecimaWeightedFairBatchScheduler.exchange_on is False
    import src.placement.simulation as sim
    src = open(sim.__file__).read()
    assert '"decima_wfair_network_decima_wfair_network": (GNNOrchestrator, GNNAutoscaler, DecimaWeightedFairBatchScheduler)' in src


def test_gate_driver_tunes_off_the_study():
    import scripts_cosim.fresh_topo_burst_v1_gate as G
    study = {9119, 9414, 9420, 9423, 9434, 9435, 9444, 9446, 9456, 9461, 9466, 9469}
    assert not study & set(G.DECIMA_TUNE_TOPOS)
    tune = G.tasks_for("decimatune", None)
    assert {t["topo"] for t in tune} == set(G.DECIMA_TUNE_TOPOS)
    assert {t["kind"] for t in tune} == set(G.DECIMA_TUNE_ALPHAS) | {"cd"}
    gate = G.tasks_for("decima", {"topologies": sorted(study)})
    assert len(gate) == 12 * 4 and {t["kind"] for t in gate} == {"decima"}


def test_tune_read_picks_the_fastest_alpha_vs_cd(tmp_path):
    import json
    import scripts_cosim.fresh_topo_burst_v1_gate as G
    from scripts_cosim.decima_tune_read import main

    def put(topo, w, kind, elapsed):
        name = f"cc40s{topo}__{w}__{kind}_s0"
        (tmp_path / f"{name}.summary.json").write_text(json.dumps(
            {"arm": name, "topology": topo, "window": w, "checkpoint_seed": 0, "averageElapsedTime": elapsed}))

    for t in G.DECIMA_TUNE_TOPOS:
        for w in G.WINDOWS:
            put(t, w, "cd", 10.0)
            for kind, alpha in G.DECIMA_TUNE_ALPHAS.items():
                put(t, w, kind, 20.0 - alpha)  # larger alpha faster in this synthetic
    out = tmp_path / "tune.json"
    assert main(["--dir", str(tmp_path), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["chosen"]["alpha"] == 1.0
    assert res["decima_a1"]["median_pct_vs_cd"] == 90.0
