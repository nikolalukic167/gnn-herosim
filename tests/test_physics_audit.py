"""physics_audit_v1: the invariant checkers must be able to fail.

Every checker is run on a synthetic trace that satisfies its invariant (PASS) and on copies with one seeded defect
(FAIL, or NOT-TESTED where the data cannot support a verdict). A checker that passes everything is the failure this
file exists to catch. No simulation is run; ~1 s.

Integration checks that need a real cell (default path replays byte for byte, trace on == trace off) run only when
HEROSIM_AUDIT_TEST_CFG and HEROSIM_AUDIT_TEST_WORKLOAD point at one; see test_default_path_and_trace_identity.
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts_cosim" / "physics_audit"))

import check_invariants as ci  # noqa: E402
from replay_identity import compare  # noqa: E402

MB = 1024.0 * 1024.0
N_TASKS = 300
NODE = "n0"
Q = f"{NODE}:1"


# ---------------------------------------------------------------------------------------------------------------
# a synthetic trace that satisfies every trace-level invariant
# ---------------------------------------------------------------------------------------------------------------
def good_rows():
    route = [["a|b", 100.0], ["b|c", 50.0]]  # bottleneck 50 MB/s
    nbytes, route_latency = 25.0 * MB, 0.02
    ingress = route_latency + nbytes / (50.0 * MB)
    rows = [{"k": "header", "t": 0.0,
             "env": {"HEROSIM_TRANSFER_MODEL": "pipelined", "HEROSIM_REPLICA_RELEASE": "1", "HEROSIM_SCALEOUT": "kpa",
                     "HEROSIM_POLICY_TIME_SCALE": None},
             "policy": "SynthScheduler", "autoscaler": "A", "keep_alive": 30.0, "reconcile_interval": 1.0,
             "queue_length": 0.7, "kpa": {"stable_window_s": 60.0, "panic_window_s": 6.0}, "warmth_physics": "node_disk_v2",
             "fabric": True, "ingress_pipe": False,
             "nodes": [{"node": NODE, "id": 0, "memory": 32.0, "available_memory": 32.0, "platforms": 4}]}]
    svc = []
    for i in range(N_TASKS):
        arr = float(i)
        pop = arr + 0.1
        first = i == 0
        svc.append({
            "k": "svc", "t": arr + 1.9, "task": i, "type": "f", "q": Q, "ptype": "xavierCpu", "release": True,
            "pop": pop, "arrived": pop + ingress, "cold_end": pop + ingress + (0.5 if first else 0.0),
            "io_start": arr + 0.2, "storage_got": arr + 0.2, "rendezvous_end": arr + 0.2, "io_end": arr + 1.7,
            "started": arr + 0.2, "compute_start": arr + 1.7, "exec_end": arr + 1.85, "done": arr + 1.9,
            "scheduled": arr, "dispatched": arr, "cold": 0.5 if first else 0.0, "cold_started": first, "warm": not first,
            "prev_type": None if first else "f", "inflight_at_pop": 0 if first else 1, "rendezvous": 0.0,
            "exchange": 0.0, "exec": 0.15, "incarnation": 1})
        rows.append({"k": "arrive", "t": arr, "task": i, "type": "f", "src": "c0"})
        rows.append({"k": "enqueue", "t": arr, "task": i, "q": Q})
        rows.append({"k": "xfer", "t": pop + ingress, "kind": "ingress", "task": i, "src": "c0", "dst": NODE,
                     "bytes": nbytes, "route": route, "model": "pipelined", "charged": nbytes / (50.0 * MB),
                     "latency": route_latency, "wait": 0.0, "sf": False, "route_latency": route_latency})
        rows.append({"k": "decision", "t": arr, "before": arr, "after": arr, "wall": 0.001, "n": 1, "policy": "S"})
    # the exchange of task 5 with two partners, summing to what the task reports
    peer_bytes = [10.0 * MB, 20.0 * MB]
    for b in peer_bytes:
        rows.append({"k": "xfer", "t": 5.5, "kind": "peer", "task": 5, "src": "n1", "dst": NODE, "bytes": b, "route": route,
                     "model": "pipelined", "charged": b / (50.0 * MB), "latency": 0.01, "wait": None, "sf": False,
                     "route_latency": None})
    svc[5]["exchange"] = sum(b / (50.0 * MB) + 0.01 for b in peer_bytes)
    rows += svc
    # replicas: 60 creations, 36 load-caused; load_live follows the in-flight wave
    for i in range(60):
        rows.append({"k": "rep_up", "t": float(i), "fn": "f", "q": f"{NODE}:{100 + i}", "cause": "load" if i % 5 < 3 else "reachability",
                     "mem": 0.07, "node_mem": 32.0, "node_avail": 31.0, "inc": 1})
    for t in range(0, N_TASKS, 1):
        wave = 5 + 4 * math.sin(t / 7.0)
        in_system = sum(1 for j in range(N_TASKS) if j <= t < j + 1.9)  # ties counted inclusively (upper bound)
        rows.append({"k": "kpa", "t": float(t), "fn": "f", "obs": float(round(wave)), "cur": 4, "ready": 4, "desired": 4,
                     "panic": False, "stable": wave, "load_live": int(round(wave)), "occ": {Q: in_system}})
    rows.append({"k": "end", "t": float(N_TASKS + 2), "created": N_TASKS, "dispatched": N_TASKS, "done": N_TASKS, "failed": 0,
                 "failed_ids": [], "not_done_ids": [], "undispatched_ids": []})
    return rows


def write(tmp_path, rows, name="t.jsonl"):
    p = tmp_path / name
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def mutated(rows, fn):
    out = copy.deepcopy(rows)
    fn(out)
    return out


def status(check, tmp_path, rows, **kw):
    return check(ci.Trace(write(tmp_path, rows)), **kw)["status"]


@pytest.fixture
def rows():
    return good_rows()


# ---------------------------------------------------------------------------------------------------------------
# every invariant: the good trace passes, a seeded defect fails
# ---------------------------------------------------------------------------------------------------------------
def test_good_trace_passes_everything(tmp_path, rows):
    tr = ci.Trace(write(tmp_path, rows))
    for check in (ci.i1_littles_law, ci.i2_transfer_time, ci.i3_no_store_and_forward, ci.i4_released_replicas,
                  ci.i6_memory, ci.i7_conservation, ci.i9_cold_start_accounting, ci.i10_decision_time):
        r = check(tr)
        assert r["status"] == "PASS", (r["id"], r["status"], r["detail"], r["numbers"])
    assert ci.i5_scaleout_causality(tr)["status"] == "PASS"


def test_i1_counter_that_leaks_fails(tmp_path, rows):
    def leak(rs):
        k = next(r for r in rs if r["k"] == "kpa" and r["t"] == 100.0)
        k["occ"][Q] += 3
    assert status(ci.i1_littles_law, tmp_path, mutated(rows, leak)) == "FAIL"


def test_i1_counter_that_drops_tasks_fails(tmp_path, rows):
    def drop(rs):
        for r in rs:
            if r["k"] == "kpa" and r["t"] >= 50.0:
                r["occ"][Q] = 0
    assert status(ci.i1_littles_law, tmp_path, mutated(rows, drop)) == "FAIL"


def test_i1_too_few_tasks_is_not_tested(tmp_path, rows):
    assert status(ci.i1_littles_law, tmp_path, rows, min_tasks=10_000) == "NOT-TESTED"


def test_i2_transfer_slower_than_analytic_fails(tmp_path, rows):
    def slow(rs):
        for r in rs:
            if r["k"] == "svc" and r["task"] % 2 == 0:
                r["arrived"] += 0.05
    assert status(ci.i2_transfer_time, tmp_path, mutated(rows, slow)) == "FAIL"


def test_i2_exchange_sum_that_disagrees_fails(tmp_path, rows):
    def off(rs):
        next(r for r in rs if r["k"] == "svc" and r["task"] == 5)["exchange"] *= 1.5
    assert status(ci.i2_transfer_time, tmp_path, mutated(rows, off)) == "FAIL"


def test_i3_store_and_forward_fails(tmp_path, rows):
    def sf(rs):
        x = next(r for r in rs if r["k"] == "xfer")
        x["sf"] = True
    def charged_per_hop(rs):
        x = next(r for r in rs if r["k"] == "xfer" and r["kind"] == "ingress")
        x["charged"] *= len(x["route"])
    def model(rs):
        rs[0]["env"]["HEROSIM_TRANSFER_MODEL"] = "store_forward"
    for fn in (sf, charged_per_hop, model):
        assert status(ci.i3_no_store_and_forward, tmp_path, mutated(rows, fn)) == "FAIL", fn.__name__


def test_i4_compute_overlap_fails(tmp_path, rows):
    def overlap(rs):
        a = next(r for r in rs if r["k"] == "svc" and r["task"] == 10)
        a["done"] += 1.0  # past the next task's compute_start
    assert status(ci.i4_released_replicas, tmp_path, mutated(rows, overlap)) == "FAIL"


def test_i4_no_io_overlap_fails(tmp_path, rows):
    def serial(rs):
        for r in rs:
            if r["k"] == "svc":
                r["io_end"] = r["io_start"] + 0.5
    assert status(ci.i4_released_replicas, tmp_path, mutated(rows, serial)) == "FAIL"


def test_i4_cold_start_while_function_is_warm_fails(tmp_path, rows):
    def cold(rs):
        r = next(r for r in rs if r["k"] == "svc" and r["task"] == 40)
        r["cold_started"], r["warm"], r["cold"], r["prev_type"] = True, False, 0.5, None
    assert status(ci.i4_released_replicas, tmp_path, mutated(rows, cold)) == "FAIL"


def test_i4_without_release_is_not_tested(tmp_path, rows):
    def held(rs):
        rs[0]["env"]["HEROSIM_REPLICA_RELEASE"] = "0"
    assert status(ci.i4_released_replicas, tmp_path, mutated(rows, held)) == "NOT-TESTED"


def test_i5_uncorrelated_replicas_fail(tmp_path, rows):
    def scramble(rs):
        ks = [r for r in rs if r["k"] == "kpa"]
        for i, r in enumerate(ks):
            r["load_live"] = (i * 7919) % 11  # deterministic noise
    assert status(ci.i5_scaleout_causality, tmp_path, mutated(rows, scramble)) == "FAIL"


def test_i5_reachability_dominated_is_fail_with_cause(tmp_path, rows):
    def reach(rs):
        for r in rs:
            if r["k"] == "rep_up":
                r["cause"] = "reachability"
    assert status(ci.i5_scaleout_causality, tmp_path, mutated(rows, reach)) == "FAIL-WITH-CAUSE"


def test_i6_over_committed_node_fails(tmp_path, rows):
    def over(rs):
        for r in rs:
            if r["k"] == "rep_up" and r["fn"] == "f":
                r["mem"] = 1.0  # 60 replicas x 1 GB on a 32 GB node
    assert status(ci.i6_memory, tmp_path, mutated(rows, over)) == "FAIL"


def test_i6_refusals_must_match_counter(tmp_path, rows):
    tr = ci.Trace(write(tmp_path, rows))
    fake = {"stats": {"scaleOut": {"memory_cap_refusals": 7}}}
    assert ci.i6_memory(tr, fake)["status"] == "FAIL"


def test_i7_silent_drop_fails(tmp_path, rows):
    def drop(rs):
        rs.remove(next(r for r in rs if r["k"] == "svc" and r["task"] == 77))
    assert status(ci.i7_conservation, tmp_path, mutated(rows, drop)) == "FAIL"


def test_i7_logged_failure_is_not_a_drop(tmp_path, rows):
    def failed(rs):
        rs.remove(next(r for r in rs if r["k"] == "svc" and r["task"] == 77))
        e = next(r for r in rs if r["k"] == "end")
        e["done"], e["failed"], e["failed_ids"] = N_TASKS - 1, 1, [77]
    assert status(ci.i7_conservation, tmp_path, mutated(rows, failed)) == "PASS"


def test_i9_warm_start_charged_fails(tmp_path, rows):
    def charged(rs):
        next(r for r in rs if r["k"] == "svc" and r["task"] == 12)["cold"] = 0.3
    assert status(ci.i9_cold_start_accounting, tmp_path, mutated(rows, charged)) == "FAIL"


def test_i9_extra_cold_start_fails(tmp_path, rows):
    def extra(rs):
        r = next(r for r in rs if r["k"] == "svc" and r["task"] == 12)
        r["cold_started"], r["warm"], r["cold"] = True, False, 0.5
    assert status(ci.i9_cold_start_accounting, tmp_path, mutated(rows, extra)) == "FAIL"


def test_i10_decision_that_advances_the_clock_fails(tmp_path, rows):
    def advance(rs):
        next(r for r in rs if r["k"] == "decision")["after"] += 0.01
    assert status(ci.i10_decision_time, tmp_path, mutated(rows, advance)) == "FAIL"


def test_i10_uninstrumented_path_is_not_tested(tmp_path, rows):
    assert status(ci.i10_decision_time, tmp_path, [r for r in rows if r["k"] != "decision"]) == "NOT-TESTED"


def test_i12_rungs_with_different_time_scale_fail(tmp_path, rows):
    a = ci.Trace(write(tmp_path, rows, "a.jsonl"))
    scaled = mutated(rows, lambda rs: rs[0]["env"].update(HEROSIM_POLICY_TIME_SCALE="0.5"))
    b = ci.Trace(write(tmp_path, scaled, "b.jsonl"))
    r = ci.i12_rung_config({"x20": a, "x50": b})
    assert r["status"] == "FAIL" and "policy_time_scale" in r["numbers"]["differing"]
    assert ci.i12_rung_config({"x20": a, "x30": ci.Trace(write(tmp_path, rows, "c.jsonl"))})["status"] == "PASS"
    assert ci.i12_rung_config({"x20": a})["status"] == "NOT-TESTED"


def test_i8_determinism_needs_twelve_cells_and_identical_results(tmp_path):
    def result(path, rtt, wall):
        json.dump({"stats": {"total_rtt": rtt, "averageGNNDecisionTime": wall}, "run_provenance": {"code": {"dirty": False}}},
                  open(path, "w"))
        return str(path)
    pairs = [(result(tmp_path / f"a{i}.json", 100.0 + i, 0.001), result(tmp_path / f"b{i}.json", 100.0 + i, 0.002))
             for i in range(12)]
    assert ci.i8_determinism(pairs)["status"] == "PASS"  # wall-clock fields are masked
    assert ci.i8_determinism(pairs[:11])["status"] == "NOT-TESTED"
    bad = list(pairs)
    bad[3] = (bad[3][0], result(tmp_path / "bad.json", 999.0, 0.002))
    assert ci.i8_determinism(bad)["status"] == "FAIL"


# ---------------------------------------------------------------------------------------------------------------
# helpers the checkers rest on
# ---------------------------------------------------------------------------------------------------------------
def test_spearman_known_values():
    assert ci.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert ci.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert ci.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert ci.spearman([1, 2, 2, 3], [1, 2, 3, 3]) == pytest.approx(5 / 6, abs=1e-9)  # ranks [1,2.5,2.5,4] vs [1,2,3.5,3.5]; checked against scipy


def test_replay_identity_masks_only_wall_clock(tmp_path):
    a = {"stats": {"total_rtt": 5.0, "averageGNNDecisionTime": 0.1, "taskResults": [{"gnn_decision_time": 0.1, "doneTime": 3.0}]},
         "run_provenance": {"code": {"dirty": False}}}
    b = copy.deepcopy(a)
    b["stats"]["averageGNNDecisionTime"] = 0.9
    b["stats"]["taskResults"][0]["gnn_decision_time"] = 0.9
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    json.dump(a, open(pa, "w")); json.dump(b, open(pb, "w"))
    assert compare(str(pa), str(pb)) == []
    assert compare(str(pa), str(pb), strict=True) != []
    b["stats"]["taskResults"][0]["doneTime"] = 3.0001
    json.dump(b, open(pb, "w"))
    assert compare(str(pa), str(pb)) != []


def test_env_step_finding_reproduces_standalone():
    proc = subprocess.run([sys.executable, str(REPO / "scripts_cosim/physics_audit/env_step_repro.py")],
                          capture_output=True, text=True)
    assert proc.returncode == 0 and "DIFFERENT" in proc.stdout, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------------------------------------------
# the fidelity block's pure parts
# ---------------------------------------------------------------------------------------------------------------
def fidelity_block():
    ev = lambda gid, fn: {"gid": gid, "fn": fn, "src": "client_node0", "app": {"name": f"nofs-{fn}", "dag": {fn: []}}, "qos": {"name": "medium"}}
    queued = [dict(ev(4, "dnn1"), q="node1:5", node_id=2, platform_id=5), dict(ev(9, "dnn2"), q="node1:6", node_id=2, platform_id=6)]
    return {"version": 1, "queued": queued, "batch": [ev(11, "dnn1"), ev(12, "dnn1")], "pairs": [[4, 11, 1.0], [11, 12, 2.0]],
            "peers": {}, "platforms": {"node1:5": {"initialized": True}, "node1:7": {"initialized": False},
                                       "node2:3": {"initialized": False}}}


def test_replay_workload_orders_queued_before_batch():
    from src.placement import snapshot_fidelity as sf

    block = fidelity_block()
    wl, forced, ids = sf.replay_workload(block)
    assert ids == {4: 0, 9: 1, 11: 2, 12: 3}
    assert forced == {0: (2, 5), 1: (2, 6)}
    assert [e["timestamp"] for e in wl["events"]] == [0.0] * 4
    assert wl["peer_exchange"] == [[0, 2, 1.0], [2, 3, 2.0]]


def test_free_platforms_stay_uninitialised_in_the_replay():
    from src.placement import snapshot_fidelity as sf

    assert sf.uninitialized_keys(fidelity_block()) == {("node1", 7), ("node2", 3)}
    assert sf.uninitialized_keys(None) == set()


# ---------------------------------------------------------------------------------------------------------------
# integration: needs a real cell (HEROSIM_AUDIT_TEST_CFG / HEROSIM_AUDIT_TEST_WORKLOAD)
# ---------------------------------------------------------------------------------------------------------------
CFG = os.environ.get("HEROSIM_AUDIT_TEST_CFG")
WL = os.environ.get("HEROSIM_AUDIT_TEST_WORKLOAD")


@pytest.mark.skipif(not (CFG and WL), reason="set HEROSIM_AUDIT_TEST_CFG and HEROSIM_AUDIT_TEST_WORKLOAD to a cell")
def test_default_path_and_trace_identity(tmp_path):
    """The audit trace, the snapshot capture and the fidelity block must not change a run: trace off, trace on, and
    trace + snapshots + fidelity all give results identical up to wall-clock fields."""
    base_env = dict(os.environ, PYTHONPATH=str(REPO), HEROSIM_TRANSFER_MODEL="pipelined", HEROSIM_REPLICA_RELEASE="1",
                    HEROSIM_SCALEOUT="kpa", HEROSIM_PEER_EXCHANGE="1", HEROSIM_SERVER_ONLY_REPLICAS="1",
                    HEROSIM_WARMTH_PHYSICS="node_disk_v2", PYTHONHASHSEED="0", SIM_FORCE_FULL_STATS="1",
                    GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1", HEROSIM_MAX_EVENTS="1500")
    scale = os.environ.get("HEROSIM_AUDIT_TEST_TIME_SCALE")
    if scale:
        base_env["HEROSIM_POLICY_TIME_SCALE"] = scale

    def run(name, **extra):
        out = tmp_path / f"{name}.json"
        env = dict(base_env, **extra)
        proc = subprocess.run([sys.executable, str(REPO / "src/executesimulation.py"), "--config", CFG, "--workload", WL,
                               "--policy", "peer_greedy_network_cd", "--output", str(out)], env=env, cwd=str(REPO),
                              capture_output=True, text=True)
        assert proc.returncode == 0 and out.exists(), proc.stderr[-500:]
        return str(out)

    off = run("off")
    on = run("on", HEROSIM_AUDIT_TRACE=str(tmp_path / "t.jsonl"))
    full = run("full", HEROSIM_AUDIT_TRACE=str(tmp_path / "t2.jsonl"), HEROSIM_SNAPSHOT_FIDELITY="1",
               LIVE_AUDIT_SNAPSHOT_PATH=str(tmp_path / "s.jsonl"), LIVE_AUDIT_MIN_BATCH_SIZE="1", LIVE_AUDIT_MIN_CANDIDATES="1")
    assert compare(off, on) == []
    assert compare(off, full) == []
    assert (tmp_path / "s.jsonl").stat().st_size > 0


# ---------------------------------------------------------------------------------------------------------------
# gate summaries: latency percentiles and replica-count series (physics_audit_v1 pass 2)
# ---------------------------------------------------------------------------------------------------------------
def _gate():
    import importlib

    return importlib.import_module("scripts_cosim.fresh_topo_burst_v1_gate")


def test_latency_percentiles_nearest_rank_and_filters_warmup_rows():
    g = _gate()
    rows = [{"taskId": i, "dispatchedTime": 0.0, "doneTime": float(i + 1)} for i in range(100)]
    rows.append({"taskId": -1, "dispatchedTime": 0.0, "doneTime": 1e6})  # a non-workload row must not count
    got = g.latency_percentiles(rows)
    assert got == {"n": 100, "p50": 50.0, "p95": 95.0, "p99": 99.0, "max": 100.0}
    assert g.latency_percentiles([]) is None and g.latency_percentiles(None) is None


def test_replica_count_series_matches_the_event_record():
    g = _gate()
    ev = []
    for t in range(0, 11):
        ev.append({"name": "a", "timestamp": t, "count": 0 if t < 2 else (2 if t < 6 else 1)})
        ev.append({"name": "b", "timestamp": t, "count": 1 if t >= 4 else 0})
    got = g.replica_count_series(ev, 10.0, points=10)
    assert got["total"] == [0, 0, 2, 2, 3, 3, 2, 2, 2, 2, 2]
    assert got["peak"] == 3
    # a: 0 on [0,2), 2 on [2,6), 1 on [6,10) = 12 replica-seconds; b: 1 on [4,10) = 6
    assert abs(got["time_mean"] - (2 * 4 + 1 * 4 + 6) / 10.0) < 1e-9
    assert g.replica_count_series(None, 10.0) is None and g.replica_count_series(ev, 0) is None
