import math

from scripts_cosim.load_recalibration_v1_bisect import allowed, cell_metrics, guards, search
from scripts_cosim.load_recalibration_v1_identity import compare


def test_search_stops_in_band_and_reports_steps():
    share = lambda m: (0.05 * m, True)  # share 0.25 at m=5
    r = search("moderate", share, 0.05, 64.0, 8)
    assert r["status"] == "BRACKETED" and r["answer"]["kind"] == "IN-BAND"
    assert 0.2 <= r["answer"]["share"] <= 0.3 and len(r["steps"]) <= 8


def test_search_guard_failure_counts_as_above_target_and_heavy_falls_back_to_highest_stable():
    # share rises with m but CD fails a guard above m=10
    def ev(m):
        return (min(0.35, 0.05 * m), m <= 10.0)
    r = search("heavy", ev, 0.05, 64.0, 8)
    assert r["answer"]["kind"] == "HIGHEST-STABLE" and r["answer"]["m"] <= 10.0 and r["answer"]["allowed"]
    assert len(r["steps"]) <= 8


def test_unbracketed_when_ends_do_not_bracket():
    assert search("heavy", lambda m: (0.01, True), 0.05, 64.0, 8)["status"] == "UNBRACKETED-HIGH"
    assert search("light", lambda m: (0.9, True), 0.05, 64.0, 8)["status"] == "UNBRACKETED-LOW"


def _summary(**kw):
    s = {"num_tasks": 50000, "requestFailures": 0, "queue_share": 0.1, "averageElapsedTime": 5.0, "endTime": 1000.0,
         "averageExecutionTime": 1.0, "latency_percentiles": {"p95": 10.0, "p99": 20.0},
         "arrival_end": {"end_over_last_arrival": 1.01}, "queue_drift": {"quarter_mean_queue": [1.0, 1.0, 1.0, 1.5]},
         "replica_count_series": {"time_mean": 400.0}, "placement_wait": {"mean": 0.5, "p95": 1.0, "max": 3.0}}
    s.update(kw)
    return s


def test_guards_per_rung():
    c = cell_metrics(_summary())
    assert math.isclose(c["busy_fraction"], 50000 / (400 * 1000)) and c["backlog_ratio"] == 1.5
    g = guards([c], 1)
    assert allowed(g, "light") and allowed(g, "moderate") and allowed(g, "heavy")
    bad = guards([cell_metrics(_summary(latency_percentiles={"p95": 301.0, "p99": 1.0}))], 1)
    assert not allowed(bad, "moderate")
    assert not allowed(guards([c], 2), "moderate")  # a cell did not finish
    drift = guards([cell_metrics(_summary(queue_drift={"quarter_mean_queue": [1, 1, 1, 3]}))], 1)
    assert allowed(drift, "moderate") and not allowed(drift, "heavy")
    busy = guards([cell_metrics(_summary(averageExecutionTime=3.0))], 1)
    assert not allowed(busy, "light") and allowed(busy, "heavy")


def test_identity_compare(tmp_path):
    import json
    b, a = tmp_path / "b", tmp_path / "a"
    b.mkdir(); a.mkdir()
    old = {"x": 1, "nested": {"y": [1, 2]}, "wallclock_s": 5, "code": "c1"}
    (b / "c.summary.json").write_text(json.dumps(old))
    (a / "c.summary.json").write_text(json.dumps({**old, "wallclock_s": 9, "code": "c2", "placement_wait": {}, "arrival_end": {}}))
    assert compare(b, a) == []
    (a / "c.summary.json").write_text(json.dumps({**old, "nested": {"y": [1, 3]}, "placement_wait": {}, "arrival_end": {}}))
    assert compare(b, a) == ["c.summary.json: nested differs"]
