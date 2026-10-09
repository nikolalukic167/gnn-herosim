import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
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
         "arrival_end": {"end_over_last_arrival": 1.01}, "backlog_profile": {"last_over_mid": 1.5, "in_system_ratio": 1.0},
         "replica_count_series": {"time_mean": 400.0}, "placement_wait": {"mean": 0.5, "p95": 1.0, "max": 3.0},
         "lock_wait": {"mean": 0.0, "p95": 0.0, "max": 0.0, "effective_queue_share": 0.1}}
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
    drift = guards([cell_metrics(_summary(backlog_profile={"last_over_mid": 3.0, "in_system_ratio": 3.0}))], 1)
    assert allowed(drift, "moderate") and not allowed(drift, "heavy")
    crowd = guards([cell_metrics(_summary(backlog_profile={"last_over_mid": 1.0, "in_system_ratio": 2.5, "in_system_at": {"half": 40, "three_quarter": 100}}))], 1)
    assert allowed(crowd, "moderate") and not allowed(crowd, "heavy")
    few = guards([cell_metrics(_summary(backlog_profile={"last_over_mid": 1.0, "in_system_ratio": 2.5, "in_system_at": {"half": 2, "three_quarter": 5}}))], 1)
    assert allowed(few, "heavy")  # in-system(1/2) < 20: reported, not applied
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


def test_backlog_profile_counts_unplaced_wait_and_tasks_in_system():
    from scripts_cosim.fresh_topo_burst_v1_gate import backlog_profile

    # 8 tasks arriving at t=0..7; the last two wait 10 s before placement, none queues; each runs 1 s after placement
    tr = [{"taskId": i, "dispatchedTime": float(i), "scheduledTime": i + (10.0 if i >= 6 else 0.0), "queueTime": 0.0,
           "doneTime": i + (10.0 if i >= 6 else 0.0) + 1.0} for i in range(8)]
    bp = backlog_profile(tr)
    assert bp["quarter_mean_backlog"] == [0.0, 0.0, 0.0, 10.0] and bp["last_over_mid"] == math.inf  # middle quarters are 0
    assert bp["in_system_at"] == {"half": 1, "three_quarter": 1}  # t=3.5: task 3 in system; t=5.25: task 5
    assert backlog_profile([]) is None


def test_linear_midpoint_and_seeded_end_places_bracket_despite_guards():
    import scripts_cosim.load_recalibration_v1_bisect as lb
    from scripts_cosim.load_recalibration_v1_bisect import next_multiplier

    assert next_multiplier(0.2666, 5.9402) == 1.2584  # log midpoint (default)
    lb.MIDPOINT = "linear"
    assert next_multiplier(0.2666, 11.6139) == 5.9402
    # light: the low end (seeded, 7/8 cells so its guards fail) is below the band, the high end above it
    def ev(m):
        return {0.2666: (0.025, False), 11.6139: (0.104, True)}.get(m, (0.0085 * m, True))
    r = search("light", ev, 0.2666, 11.6139, 8, seeded=(0.2666, 11.6139))
    assert r["status"] == "BRACKETED" and r["steps"][0]["position"] == "low"
    assert r["answer"]["kind"] == "IN-BAND" and r["answer"]["m"] > 5
    # without the seed, a failing end counts as above target
    assert search("light", ev, 0.2666, 11.6139, 8)["status"] == "UNBRACKETED-LOW"
    lb.MIDPOINT = "log"


def test_choose_final_falls_back_to_highest_passing_step():
    from scripts_cosim.load_recalibration_v1_bisect import choose_final

    class Ev:
        def __init__(self, bad):
            self.bad, self.calls = bad, []

        def finalize(self, m):
            self.calls.append(m)
            g = guards([cell_metrics(_summary(latency_percentiles={"p95": 999.0 if m in self.bad else 10.0}))], 1)
            return {"guards": {"cd": g, "reactive": g}, "cd_median_share": 0.3, "reactive_median_share": 0.1}

    steps = [{"m": m, "share": 0.1, "allowed": True} for m in (11.6, 20.0, 30.0)]
    res = {"rung": "moderate", "answer": {"m": 30.0, "kind": "IN-BAND"}, "steps": steps}
    ev = Ev(bad={30.0})
    out = choose_final(res, ev)
    assert out["chosen"]["m"] == 20.0 and out["chosen"]["kind"] == "FALLBACK-HIGHEST-PASSING" and ev.calls == [30.0, 20.0]
    assert choose_final(res, Ev(bad={11.6, 20.0, 30.0}))["chosen"] is None
    assert choose_final({"rung": "light", "answer": None, "steps": []}, Ev(set()))["chosen"] is None


def test_lock_wait_profile_counts_the_wait_after_started():
    from scripts_cosim.fresh_topo_burst_v1_gate import lock_wait_profile

    # two tasks, elapsed 10 s each; queue 1 s each; the second waits 8 s for the compute lock
    tr = [{"taskId": 0, "dispatchedTime": 0.0, "doneTime": 10.0, "queueTime": 1.0, "ioEndTime": 5.0, "computeStartTime": 5.0},
          {"taskId": 1, "dispatchedTime": 0.0, "doneTime": 10.0, "queueTime": 1.0, "ioEndTime": 1.0, "computeStartTime": 9.0},
          {"taskId": -1, "dispatchedTime": 0.0, "doneTime": 99.0, "queueTime": 9.0, "ioEndTime": 0.0, "computeStartTime": 50.0}]
    lw = lock_wait_profile(tr)
    assert lw["n"] == 2 and lw["mean"] == 4.0 and lw["max"] == 8.0 and lw["n_unstamped"] == 0
    assert lw["queue_share_same_tasks"] == 0.1 and lw["effective_queue_share"] == 0.5
    tr[0]["computeStartTime"] = None  # never served: no lock wait, reported
    assert lock_wait_profile(tr)["n_unstamped"] == 1
    assert lock_wait_profile([]) is None


def test_backlog_v2_counts_lock_wait_and_guard_prefers_it():
    from scripts_cosim.fresh_topo_burst_v1_gate import backlog_profile

    # 8 tasks; the last quarter waits 10 s for the compute lock, nothing else
    tr = [{"taskId": i, "dispatchedTime": float(i), "scheduledTime": float(i), "queueTime": 0.5, "doneTime": i + 20.0,
           "ioEndTime": i + 1.0, "computeStartTime": i + 1.0 + (10.0 if i >= 6 else 0.0)} for i in range(8)]
    assert backlog_profile(tr)["quarter_mean_backlog"] == [0.5, 0.5, 0.5, 0.5]
    v2 = backlog_profile(tr, with_lock_wait=True)
    assert v2["quarter_mean_backlog"] == [0.5, 0.5, 0.5, 10.5] and v2["last_over_mid"] == 21.0
    g = guards([cell_metrics(_summary(backlog_profile={"last_over_mid": 1.0, "in_system_ratio": 1.0},
                                      backlog_profile_v2=v2))], 1)
    assert not allowed(g, "heavy") and allowed(g, "moderate")
