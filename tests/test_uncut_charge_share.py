import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_charge_share as C  # noqa: E402

HOST = {"hostname": "a", "cpu_model": "AMD EPYC 9335 32-Core Processor", "smt_control": "off", "nproc": 8}
W = {"w_ms_per_plan": {"cd_ib0": {"moderate": 0.01, "heavy": 0.01}, "ra_gnn_eng_arm1_nosplit": {"moderate": 5.0, "heavy": 5.0}},
     "w_ms_per_plan_by_bucket": {"ra_gnn_eng_arm1_nosplit": {"heavy": {"le10": 6.0, "11_50": 2.0, "gt50": 1.0}}}}


def write(d, arm, rung, ms, evals=(1000, 1000, 1000), mean=2.0, tasks=100, host=HOST, topo=19405):
    d.mkdir(exist_ok=True)
    by = {b: {"total_s": m * 1e-3 * n, "plan_evals": n} for b, m, n in zip(C.BUCKETS, ms, evals)}
    s = {"averageElapsedTime": mean, "num_tasks": tasks, "schedulerCounters": {"sched_plan_evals": sum(evals)},
         "decisionTiming": {"total_s": sum(v["total_s"] for v in by.values()), "by_bucket": by}}
    if host is not None:
        s["host"] = host
    (d / f"cc40s{topo}__g0{rung}__{arm}_s0.summary.json").write_text(json.dumps(s))


@pytest.fixture
def wf(tmp_path):
    f = tmp_path / "w.json"
    f.write_text(json.dumps(W))
    return str(f)


def bar(tmp_path):
    for r in ("moderate", "heavy"):
        write(tmp_path / "bar", "cd_ib0_k100", r, (0.01, 0.01, 0.01), host=None)
    return str(tmp_path / "bar")


def test_q_bound_math_and_keep(tmp_path, wf):
    b = bar(tmp_path)
    for r in ("moderate", "heavy"):
        write(tmp_path / "new", "cd_ib0", r, (0.012, 0.01, 0.005))
    out = C.main(["q", "--new", str(tmp_path / "new"), "--arm", "cd_ib0", "--bar", b, "--w-file", wf])
    h = out["rungs"]["heavy"]
    share = 0.01 * 1e-3 * 1000 / (2.0 * 100)
    assert abs(h["by_bucket"]["le10"]["share"] - share) < 1e-15 and abs(h["by_bucket"]["le10"]["q"] - 1.2) < 1e-9
    assert abs(h["bound"] - share * (0.2 + 0.0 + 0.5)) < 1e-15 and out["verdict"] == "KEEP"


def test_q_refreeze_when_bound_trips(tmp_path, wf):
    b = bar(tmp_path)
    for r in ("moderate", "heavy"):
        write(tmp_path / "new", "cd_ib0", r, (0.01, 0.01, 0.01) if r == "moderate" else (10.0, 0.01, 0.01))
    out = C.main(["q", "--new", str(tmp_path / "new"), "--arm", "cd_ib0", "--bar", b, "--w-file", wf])
    assert out["rungs"]["heavy"]["bound"] >= C.BOUND and out["verdict"] == "REFREEZE"


def test_q_needs_host_and_one_venue(tmp_path, wf):
    b = bar(tmp_path)
    write(tmp_path / "new", "cd_ib0", "heavy", (0.01,) * 3, host=None)
    with pytest.raises(SystemExit, match="no host block"):
        C.main(["q", "--new", str(tmp_path / "new"), "--arm", "cd_ib0", "--bar", b, "--w-file", wf])
    write(tmp_path / "new", "cd_ib0", "heavy", (0.01,) * 3)
    write(tmp_path / "new2", "cd_ib0", "moderate", (0.01,) * 3, host={**HOST, "smt_control": "on"})
    with pytest.raises(SystemExit, match="mixes"):
        C.main(["q", "--new", f"{tmp_path / 'new'},{tmp_path / 'new2'}", "--arm", "cd_ib0", "--bar", b, "--w-file", wf])


def test_learned_share_uses_bucket_rates(tmp_path, wf):
    write(tmp_path / "cal", "ra_gnn_eng_arm1_nosplit", "heavy", (6.0, 2.0, 1.0), topo=19461)
    write(tmp_path / "cal", "ra_gnn_eng_arm1_nosplit", "moderate", (5.0, 5.0, 5.0), topo=19461)
    out = C.main(["share", "--cells", str(tmp_path / "cal"), "--arm", "ra_gnn_eng_arm1_nosplit", "--w-file", wf])
    assert abs(out["rungs"]["heavy"]["by_bucket"]["le10"] - 6.0 * 1e-3 * 1000 / 200.0) < 1e-12
    assert abs(out["rungs"]["moderate"]["share_total"] - 3 * 5.0 * 1e-3 * 1000 / 200.0) < 1e-12


@pytest.mark.parametrize("end_ms, verdict", [(0.0101, "OK"), (0.0103, "STOP")])
def test_drift_stop(tmp_path, wf, end_ms, verdict):
    for r in ("moderate", "heavy"):
        write(tmp_path / "s", "cd_ib0", r, (0.01,) * 3, topo=19461)
        write(tmp_path / "e", "cd_ib0", r, (end_ms,) * 3, topo=19461)
    out = C.main(["drift", "--start", str(tmp_path / "s"), "--end", str(tmp_path / "e"), "--arm", "cd_ib0", "--w-file", wf])
    assert out["verdict"] == verdict
