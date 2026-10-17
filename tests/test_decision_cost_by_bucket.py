"""uncut_groups_s0_v1: the decision-cost clock charges w(n) by served batch-size bucket for arms with a frozen
w_ms_per_plan_by_bucket entry, and the one calibrated rate for every other arm."""
import json
import os
import sys

import pytest

from src.placement import decision_cost as dc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts_cosim"))
import fresh_topo_burst_v1_gate as G  # noqa: E402

RATES = {"le10": 0.1, "11_50": 0.4, "gt50": 0.9}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(dc.ENV, raising=False)
    monkeypatch.delenv(dc.BUCKET_ENV, raising=False)


def test_unset_is_the_scalar_path(monkeypatch):
    assert dc.bucket_rates() is None and not dc.enabled()
    monkeypatch.setenv(dc.ENV, "0.25")
    assert dc.enabled() and all(dc.rate_ms(n) == 0.25 for n in (1, 10, 11, 50, 51, 3000))


def test_bucket_rate_follows_tasks_decided(monkeypatch):
    monkeypatch.setenv(dc.BUCKET_ENV, json.dumps(RATES))
    assert dc.enabled()
    assert [dc.rate_ms(n) for n in (1, 10, 11, 50, 51, 3000)] == [0.1, 0.1, 0.4, 0.4, 0.9, 0.9]


def test_all_zero_buckets_are_off(monkeypatch):
    monkeypatch.setenv(dc.BUCKET_ENV, json.dumps({"le10": 0, "11_50": 0, "gt50": 0}))
    assert not dc.enabled()


@pytest.mark.parametrize("bad", ['{"le10": 0.1}', '{"le10": 0.1, "11_50": 0.2, "gt50": -1}', "[1, 2, 3]", "abc",
                                 '{"le10": 0.1, "11_50": 0.2, "gt50": 0.3, "gt100": 1}', '{"le10": true, "11_50": 0.2, "gt50": 0.3}'])
def test_malformed_bucket_rates_fail_loud(monkeypatch, bad):
    monkeypatch.setenv(dc.BUCKET_ENV, bad)
    with pytest.raises(ValueError, match="FAIL LOUD"):
        dc.bucket_rates()


def test_both_rules_set_fails_loud(monkeypatch):
    monkeypatch.setenv(dc.ENV, "0.25")
    monkeypatch.setenv(dc.BUCKET_ENV, json.dumps(RATES))
    with pytest.raises(ValueError, match="FAIL LOUD"):
        dc.enabled()


def test_bucket_tags_are_the_timing_buckets():
    from src.placement.decision_timing import SIZE_BUCKETS
    assert [dc.bucket_of(n) for n in (10, 11, 50, 51)] == ["le10", "11_50", "11_50", "gt50"]
    assert {t for _h, t in SIZE_BUCKETS} == set(RATES)


def test_gate_charges_what_the_file_says():
    f = json.load(open(G.DC_W_FILE))
    byb = f["w_ms_per_plan_by_bucket"]
    for rung in ("moderate", "heavy"):
        var, val = G.dc_cost_ms("cd_exactS_ib0", 1.0, rung)  # bucketed arm
        assert var == dc.BUCKET_ENV and json.loads(val) == byb["cd_exactS_ib0"][rung]
        var, val = G.dc_cost_ms("cd_exactS_ib0", 0.25, rung)
        assert json.loads(val) == {b: 0.25 * v for b, v in byb["cd_exactS_ib0"][rung].items()}
        var, val = G.dc_cost_ms("cd_ib0", 1.0, rung)  # scalar arm, no bucket entry
        assert "cd_ib0" not in byb and var == dc.ENV and float(val) == f["w_ms_per_plan"]["cd_ib0"][rung]
    assert G.dc_cost_ms("cd_ib0", None, "heavy") is None


def test_frozen_buckets_cover_both_rungs_and_le10_is_the_scalar():
    f = json.load(open(G.DC_W_FILE))
    for arm, rungs in f["w_ms_per_plan_by_bucket"].items():
        assert set(rungs) == {"moderate", "heavy"}
        for rung, b in rungs.items():
            assert set(b) == set(RATES) and all(float(v) > 0 for v in b.values())
            # hand rules' buckets were imputed from the scalar (33629f1f); a learned arm's buckets are calibrated per bucket and
            # its scalar is a separate pooled fit (Arm 1/1b/2/K0/3 at their w pins, and the FAST kinds' copy of Arm 3's)
            if not arm.startswith("ra_"):
                assert b["le10"] == f["w_ms_per_plan"][arm][rung]
    for arm in ("cd", "cd_ib0"):
        assert arm not in f["w_ms_per_plan_by_bucket"]


def test_null_bucket_entry_fails_loud(monkeypatch, tmp_path):
    f = json.load(open(G.DC_W_FILE))
    f["w_ms_per_plan_by_bucket"]["cd_exactS"]["heavy"] = None
    p = tmp_path / "w.json"
    p.write_text(json.dumps(f))
    monkeypatch.setattr(G, "DC_W_FILE", str(p))
    with pytest.raises(SystemExit, match="FAIL LOUD"):
        G.dc_cost_ms("cd_exactS", 1.0, "heavy")


def test_the_scheduler_charges_the_rate_of_the_decided_batch():
    import inspect
    from src.policy.gnn import scheduler

    src = inspect.getsource(scheduler)
    assert "decision_cost.rate_ms(len(decodable)) * 1e-3 *" in src and "decision_cost.cost_ms()" not in src
