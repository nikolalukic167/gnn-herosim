"""workload_fix_v1 run tooling: input builder, class-split exchange telemetry, the stage reader."""
import json
import os
import sys
from pathlib import Path

import pytest
import simpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_cosim"))
import workload_fix_v1_build as B  # noqa: E402
import workload_fix_v1_read as R  # noqa: E402

from src.placement.network_fabric import NetworkFabric  # noqa: E402


def _window(i, n=40):
    events = [{"timestamp": 10.0 * k + (0.001 * (k % 4)), "peer_group": k // 4, "node_name": f"client_node{(k * 3) % 7}",
               "application": {"name": "nofs-dnn1"}} for k in range(n)]
    pairs = [[a, a + 1, 1e8 + a] for a in range(0, n - 1, 2)]
    return {"rps": 1, "duration": 1, "events": events, "peer_exchange": pairs,
            "grounded_workload_v1": {"seed": 7400 + i}}


@pytest.fixture()
def x1(tmp_path):
    d = tmp_path / "x1"
    d.mkdir()
    for i, name in enumerate(B.WINDOWS):
        (d / name).write_text(json.dumps(_window(i)))
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "cc40s9601.json").write_text(json.dumps({"scheduler": {"batch_timeout": 16.0}}))
    return d, cfg


def _build(tmp_path, x1_dir, cfg, sampler, factor=0.5, tag="a"):
    out = tmp_path / f"out_{sampler}"
    seeds = B.apply_sampler(x1_dir, out / "_x1_windows", sampler)
    assert set(seeds.values()) == {7400, 7401, 7402, 7403}
    return B.build_rung(out / "_x1_windows", cfg, out, tag, factor, [9601])


def test_builder_changes_payloads_and_rate_only(tmp_path, x1):
    d, cfg = x1
    legacy = _build(tmp_path, d, cfg, "legacy")
    new = _build(tmp_path, d, cfg, "wf1_v1")
    for name in B.WINDOWS:
        a, b = json.loads((legacy / "wl" / name).read_text()), json.loads((new / "wl" / name).read_text())
        assert a["events"] == b["events"]
        assert [p[:2] for p in a["peer_exchange"]] == [p[:2] for p in b["peer_exchange"]]
        assert [p[2] for p in a["peer_exchange"]] != [p[2] for p in b["peer_exchange"]]
        assert b["workload_fix_v1"]["sampler"] == "wf1_v1" and "workload_fix_v1" not in a
        ts = [e["timestamp"] for e in a["events"]]
        assert ts[1] == pytest.approx(10.0 * 0.5 + 0.001 * 0.5)  # factor scales every timestamp
        assert len({e["node_name"] for e in a["events"] if e["peer_group"] == 0}) == 1  # single origin
    cfg_new = json.loads((new / "cfg" / "cc40s9601.json").read_text())
    assert cfg_new["scheduler"]["batch_timeout"] == 8.0 and cfg_new["cd_gap_v1_rate_scale"]["factor"] == 0.5


def test_legacy_build_is_deterministic_and_verifiable(tmp_path, x1):
    d, cfg = x1
    first = _build(tmp_path, d, cfg, "legacy")
    second_out = tmp_path / "again"
    B.apply_sampler(d, second_out / "_x1_windows", "legacy")
    second = B.build_rung(second_out / "_x1_windows", cfg, second_out, "a", 0.5, [9601])
    assert all(r["identical"] for r in B.verify_against(first, second / "wl").values())


def test_a_built_rung_is_frozen(tmp_path, x1):
    d, cfg = x1
    _build(tmp_path, d, cfg, "legacy")
    with pytest.raises(SystemExit, match="frozen"):
        B.build_rung(tmp_path / "out_legacy" / "_x1_windows", cfg, tmp_path / "out_legacy", "a", 0.5, [9601])


def test_fabric_records_exchange_by_class_pair():
    lt = {"links": {"core0|n0": {"latency": 0.01, "bandwidth_mbps": 100.0}, "core0|n1": {"latency": 0.01, "bandwidth_mbps": 100.0}},
          "routes": {"n0": {"n1": ["n0", "core0", "n1"]}},
          "access_classes": {"n0": {"class": "wired"}, "n1": {"class": "cellular"}}}
    fabric = NetworkFabric(simpy.Environment(), lt)
    fabric.record_exchange("n0", "n1", 2.0)
    fabric.record_exchange("n1", "n0", 3.0)
    assert fabric.exchange_by_class == {"cellular+wired": {"seconds": 5.0, "transfers": 2}}
    plain = NetworkFabric(simpy.Environment(), {k: v for k, v in lt.items() if k != "access_classes"})
    assert plain.access_classes == {} and plain.exchange_by_class == {}


def _row(topo, window, arm, lat, q=0.1, ex=1.0, classes=None):
    r = {"arm": f"cc40s{topo}__{window}__{arm}_s0", "num_tasks": 100, "averageElapsedTime": lat, "queue_share": q,
         "totalPeerExchangeTime": ex * 100, "cold_start_pct": 5.0}
    if classes is not None:
        r["peerExchangeByAccessClass"] = classes
    return r


def _store(arms, topos, windows, classes=None):
    ok = {}
    for t in topos:
        for w in windows:
            for arm, lat in arms.items():
                ok[(t, w, arm)] = _row(t, w, arm, lat, classes=classes)
    return ok


def test_reader_pairs_on_topology_and_holm_within_family():
    topos = list(range(1, 11))
    ok = _store({"cd": 1.0, "selfpredict": 1.1, "locality": 1.2, "batched": 1.3, "reactive": 2.0}, topos,
                [f"{w}lo" for w in R.WINDOWS])
    res = R.compute(ok, {}, topos, ["lo"])
    assert res["lo"]["_cd_first"] is True
    assert res["lo"]["selfpredict"]["vs_cd"] == pytest.approx(10.0)
    assert res["lo"]["selfpredict"]["label"] == "CD-FASTER" and res["lo"]["selfpredict"]["faster"] == 0
    assert "p_holm" not in res["lo"]["reactive"]  # context only, outside the family
    assert res["_meta"]["family_size"] == 3


def test_reader_failure_drops_only_that_arms_cell_and_sensitivity_excludes_topology():
    topos = [1, 2, 3, 4, 5, 6]
    ok = _store({"cd": 1.0, "selfpredict": 1.1, "locality": 1.2, "batched": 1.3, "reactive": 2.0}, topos,
                [f"{w}lo" for w in R.WINDOWS])
    for w in R.WINDOWS:
        del ok[(6, f"{w}lo", "reactive")]
    bad = {(6, f"{w}lo", "reactive"): {} for w in R.WINDOWS}
    res = R.compute(ok, bad, topos, ["lo"])
    assert res["lo"]["reactive"]["n_failed"] == 4 and res["lo"]["reactive"]["n_topologies"] == 5
    assert res["lo"]["cd"]["n_topologies"] == 6
    sens = R.compute(ok, bad, topos, ["lo"], excluded=[6])
    assert sens["lo"]["cd"]["n_topologies"] == 5


def test_reader_exchange_split_by_access_class():
    classes = {"wired+wired": {"seconds": 5.0, "transfers": 10}, "cellular+wired": {"seconds": 45.0, "transfers": 4}}
    ok = _store({"cd": 1.0}, [1, 2], ["g0lo"], classes=classes)
    split = R.compute(ok, {}, [1, 2], ["lo"])["lo"]["cd"]["exchange_by_access_class"]
    assert split["available"] and split["latency_seconds"] == pytest.approx(2 * 100 * 1.0)
    assert split["by_pair"]["wired+wired"]["share_of_latency"] == pytest.approx(10.0 / 200.0)
    assert split["by_pair"]["cellular+wired"]["share_of_latency"] == pytest.approx(90.0 / 200.0)
    plain = R.compute(_store({"cd": 1.0}, [1], ["g0lo"]), {}, [1], ["lo"])["lo"]["cd"]["exchange_by_access_class"]
    assert plain["available"] is False


def test_driver_refuses_a_non_r1_environment(monkeypatch):
    monkeypatch.setenv("WF1_RUNGS", "a")
    import importlib
    import fresh_topo_burst_v1_gate as G
    importlib.reload(G)
    for k in ("HEROSIM_TRANSFER_MODEL", "HEROSIM_REPLICA_RELEASE", "HEROSIM_SCALEOUT", "GATE_FIXED_POLICY_TIME_SCALE"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit, match="R1"):
        G.tasks_for("wf1cal", None)
    monkeypatch.setenv("HEROSIM_TRANSFER_MODEL", "pipelined")
    monkeypatch.setenv("HEROSIM_REPLICA_RELEASE", "1")
    monkeypatch.setenv("HEROSIM_SCALEOUT", "kpa")
    monkeypatch.setenv("GATE_FIXED_POLICY_TIME_SCALE", "1.0")
    monkeypatch.setenv("WF1_TOPOS", "9601,9602")
    cal = G.tasks_for("wf1cal", None)
    assert {t["kind"] for t in cal} == {"cd"} and len(cal) == 4 and {t["window"] for t in cal} == {"g0a", "g1a"}
    full = G.tasks_for("wf1", {"topologies": [1, 2]})
    assert len(full) == 5 * 2 * 4  # five rules x topologies x four windows at one rung
    monkeypatch.delenv("WF1_RUNGS")
    importlib.reload(G)


def test_bisection_finds_the_multiplier_and_stops_at_eight_steps():
    import math

    import workload_fix_v1_bisect as X

    calls = []

    def share(m):
        calls.append(m)
        return 0.1 + 0.076 * math.log(m)

    r = X.search(0.3, share, 0.5, 24.0, 8)
    assert r["status"] == "BRACKETED" and len(r["steps"]) == 8
    assert abs(r["closest"]["share"] - 0.3) < 0.02
    assert r["steps"][0]["m"] == 0.5 and r["steps"][1]["m"] == 24.0 and r["steps"][2]["m"] == X.next_multiplier(0.5, 24.0)


def test_bisection_reports_an_unbracketed_target_and_treats_failed_cells_as_overload():
    import math

    import workload_fix_v1_bisect as X

    flat = X.search(0.3, lambda m: 0.05, 0.5, 24.0, 8)
    assert flat["status"] == "UNBRACKETED-HIGH" and len(flat["steps"]) == 2
    low = X.search(0.1, lambda m: 0.2, 0.5, 24.0, 8)
    assert low["status"] == "UNBRACKETED-LOW"
    hang = X.search(0.3, lambda m: 0.1 if m < 6 else math.inf, 0.5, 24.0, 8)
    assert hang["status"] == "BRACKETED" and hang["bracket"][1] <= 6.5
    assert X.tag_for(1.4142) == "m1p4142"


def test_fixed_batch_window_changes_only_batch_timeout(tmp_path, x1):
    d, cfg = x1
    default = _build(tmp_path, d, cfg, "legacy", factor=0.25, tag="a")
    out = tmp_path / "out_fixed"
    B.apply_sampler(d, out / "_x1_windows", "legacy")
    fixed = B.build_rung(out / "_x1_windows", cfg, out, "a", 0.25, [9601], batch_timeout_fixed=16.0)
    for name in B.WINDOWS:  # timestamps still scale, workload bytes untouched
        assert (default / "wl" / name).read_bytes() == (fixed / "wl" / name).read_bytes()
    a = json.loads((default / "cfg" / "cc40s9601.json").read_text())
    b = json.loads((fixed / "cfg" / "cc40s9601.json").read_text())
    assert a["scheduler"]["batch_timeout"] == 4.0 and b["scheduler"]["batch_timeout"] == 16.0
    assert b["workload_fix_v1_batch_timeout"] == {"fixed_s": 16.0, "ladder_value_s": 4.0}
    for k in set(a) | set(b):
        if k not in ("scheduler", "workload_fix_v1_batch_timeout"):
            assert a.get(k) == b.get(k)
    assert {k: v for k, v in b["scheduler"].items() if k != "batch_timeout"} == {k: v for k, v in a["scheduler"].items() if k != "batch_timeout"}
    assert "workload_fix_v1_batch_timeout" not in a


def test_default_build_has_no_fixed_window_record(tmp_path, x1):
    d, cfg = x1
    rung = _build(tmp_path, d, cfg, "legacy")
    assert "workload_fix_v1_batch_timeout" not in json.loads((rung / "cfg" / "cc40s9601.json").read_text())
    with pytest.raises(SystemExit, match="> 0"):
        B.fix_batch_timeout(rung / "cfg", 0)


def test_driver_arm_filter(monkeypatch):
    import importlib

    import fresh_topo_burst_v1_gate as G
    monkeypatch.setenv("WF1_RUNGS", "a")
    for k, v in (("HEROSIM_TRANSFER_MODEL", "pipelined"), ("HEROSIM_REPLICA_RELEASE", "1"), ("HEROSIM_SCALEOUT", "kpa"),
                 ("GATE_FIXED_POLICY_TIME_SCALE", "1.0"), ("WF1_ARMS", "cd,batched,locality")):
        monkeypatch.setenv(k, v)
    importlib.reload(G)
    tasks = G.tasks_for("wf1", {"topologies": [1, 2]})
    assert {t["kind"] for t in tasks} == {"cd", "batched", "locality"} and len(tasks) == 3 * 2 * 4
    monkeypatch.setenv("WF1_ARMS", "cd,bogus")
    with pytest.raises(SystemExit, match="WF1_ARMS"):
        G.tasks_for("wf1", {"topologies": [1]})
    monkeypatch.delenv("WF1_RUNGS")
    monkeypatch.delenv("WF1_ARMS")
    importlib.reload(G)


def _full(topo, window, arm, lat, wait, queue, ex, rv, cold):
    r = _row(topo, window, arm, lat, ex=ex)
    r.update(averageWaitTime=wait, averageQueueTime=queue, totalPeerRendezvousWait=rv * 100, averageColdStartTime=cold)
    return r


def test_decomposition_sums_to_latency_with_an_explicit_remainder():
    ok = {(t, f"{w}lo", "cd"): _full(t, f"{w}lo", "cd", 3.0, 1.0, 0.5, 0.25, 0.25, 0.5) for t in (1, 2) for w in R.WINDOWS}
    d = R.decompose(ok, [1, 2], ["lo"])["lo"]
    assert d["batching_wait"] == 1.0 and d["queue"] == 0.5 and d["exchange"] == 0.25 and d["cold_start"] == 0.5
    assert d["other"] == pytest.approx(3.0 - 2.5)
    assert sum(d[k] for k in R.COMPONENTS) + d["other"] == pytest.approx(d["latency"])
    old = {(1, f"{w}lo", "cd"): _row(1, f"{w}lo", "cd", 3.0) for w in R.WINDOWS}
    assert R.decompose(old, [1], ["lo"])["available"] is False  # missing field is reported, not zero


def _emit(n_lines, width=100):
    return [sys.executable, "-c", f"import sys\nfor i in range({n_lines}):\n    print('line%07d ' % i + 'x' * {width - 12})"]


def test_log_cap_keeps_head_and_tail_and_counts_the_dropped_bytes(tmp_path):
    import capped_log as C

    log = tmp_path / "run.log"
    total = 20000 * 101  # 100 characters and a newline per line
    rc = C.run_logged(_emit(20000), os.environ, str(tmp_path), str(log), cap=500_000)
    text = log.read_bytes()
    assert rc == 0 and len(text) < 500_000 + 200
    assert text.startswith(b"line0000000") and b"line0019999" in text[-200:]
    note = text[text.index(b"\n[log capped"):].split(b"\n")[1].decode()
    dropped = int(note.split("bytes: ")[1].split(" bytes dropped")[0])
    kept = len(text) - len(note) - 2
    assert kept + dropped == total  # nothing unaccounted for


def test_log_under_the_cap_is_complete_and_uncapped_mode_is_the_old_behaviour(tmp_path):
    import capped_log as C

    small, plain = tmp_path / "small.log", tmp_path / "plain.log"
    C.run_logged(_emit(100), os.environ, str(tmp_path), str(small), cap=1_000_000)
    C.run_logged(_emit(100), os.environ, str(tmp_path), str(plain), cap=None)
    assert small.read_bytes() == plain.read_bytes() and b"log capped" not in small.read_bytes()
    assert C.run_logged([sys.executable, "-c", "import sys; print('e', file=sys.stderr); sys.exit(3)"],
                        os.environ, str(tmp_path), str(small), cap=1000) == 3
    assert b"e" in small.read_bytes()


def test_cap_setting_comes_from_the_environment(monkeypatch):
    import capped_log as C

    monkeypatch.delenv(C.CAP_ENV, raising=False)
    assert C.cap_bytes() == 50 * 1024 * 1024
    monkeypatch.setenv(C.CAP_ENV, "0")
    assert C.cap_bytes() is None
    monkeypatch.setenv(C.CAP_ENV, "1.5")
    assert C.cap_bytes() == int(1.5 * 1024 * 1024)


def test_tune_counts_a_hung_run_as_infinite_and_picks_the_geometric_mean_minimum():
    inf = float("inf")
    ok, bad = {}, {}
    lat = {(1, "lo"): 3.0, (2, "lo"): 2.0, (4, "lo"): 2.5, (1, "hi"): 0.5, (2, "hi"): 0.6, (4, "hi"): 0.4}
    for (w, rung), v in lat.items():
        for t in (1, 2):
            for g in R.WINDOWS:
                if (w, rung, t, g) == (4, "hi", 2, "g3"):
                    bad[(t, f"{g}{rung}b{w}", "cd")] = {}
                    continue
                ok[(t, f"{g}{rung}b{w}", "cd")] = _row(t, f"{g}{rung}b{w}", "cd", v)
    res = R.tune(ok, bad, [1, 2], ["lo", "hi"], [1, 2, 4])
    assert res["table"][1]["geomean"] == pytest.approx((3.0 * 0.5) ** 0.5)
    assert res["table"][2]["geomean"] == pytest.approx((2.0 * 0.6) ** 0.5)
    assert res["table"][4]["hi"]["n_hung"] == 1 and res["table"][4]["hi"]["median_latency"] == pytest.approx(0.4)
    assert res["table"][4]["geomean"] == pytest.approx((2.5 * 0.4) ** 0.5)
    assert res["best_window_s"] == 4 and res["tie"] is None  # one hung cell of eight does not move the median
    # when half the cells of a window hang its median is infinite and it cannot win
    for g in R.WINDOWS[:2]:
        for t in (1, 2):
            ok.pop((t, f"{g}hib4", "cd"))
    res = R.tune(ok, bad, [1, 2], ["lo", "hi"], [1, 2, 4])
    assert res["table"][4]["geomean"] == inf and res["best_window_s"] == 2


def test_reader_reports_request_failures_next_to_latency():
    topos = [1, 2, 3]
    ok = _store({"cd": 1.0, "selfpredict": 1.1}, topos, [f"{w}lo" for w in R.WINDOWS])
    assert R.compute(ok, {}, topos, ["lo"])["lo"]["cd"]["request_failures"] == {"available": False}  # pre-R1.1 summaries
    for w in R.WINDOWS:
        for t in topos:
            ok[(t, f"{w}lo", "cd")]["requestFailures"] = 0
            ok[(t, f"{w}lo", "selfpredict")]["requestFailures"] = 0
    ok[(2, "g1lo", "selfpredict")]["requestFailures"] = 5
    row = R.compute(ok, {}, topos, ["lo"])["lo"]
    assert row["cd"]["request_failures"] == {"available": True, "tasks": 0, "runs_with_failure": 0, "n_runs": 12}
    assert row["selfpredict"]["request_failures"] == {"available": True, "tasks": 5, "runs_with_failure": 1, "n_runs": 12}
