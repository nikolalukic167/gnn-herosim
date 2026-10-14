import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_cotenancy_r as C  # noqa: E402


HOST = {"hostname": "os-cpu-slurm-9", "cpu_model": "AMD EPYC 9335 32-Core Processor", "smt_control": "off", "threads_per_core": "1",
        "nproc": 64, "cpuset": "0-63"}


def write(d, rung, by, mean=2.0, host=HOST, threads=1):
    d.mkdir(exist_ok=True)
    ev = sum(v[1] for v in by.values())
    s = {"averageElapsedTime": mean, "schedulerCounters": {"sched_plan_evals": ev}, "python_env": {"torch_num_threads": threads},
         "decisionTiming": {"total_s": sum(v[0] for v in by.values()), "by_bucket": {b: {"total_s": t, "plan_evals": n} for b, (t, n) in by.items()}}}
    if host is not None:
        s["host"] = host
    (d / f"cc40s19461__g0{rung}__cd_ib0_s0.summary.json").write_text(json.dumps(s))


EX = {"le10": (1.0, 1000), "11_50": (1.0, 1000), "gt50": (1.0, 1000)}


def test_pooled_when_no_bucket_stands_out(tmp_path):
    write(tmp_path / "e", "heavy", EX)
    write(tmp_path / "s", "heavy", {b: (1.4 * t, n) for b, (t, n) in EX.items()})
    out = C.main(["--excl", str(tmp_path / "e"), "--shared", str(tmp_path / "s"), "--arm", "cd_ib0"])
    assert out["rungs"]["heavy"]["rule"] == "pooled" and abs(out["r"] - 1.4) < 1e-9


def test_max_bucket_when_an_eligible_bucket_exceeds_1_25x_pooled(tmp_path):
    write(tmp_path / "e", "heavy", EX)
    write(tmp_path / "s", "heavy", {"le10": (1.0, 1000), "11_50": (1.0, 1000), "gt50": (2.5, 1000)})
    out = C.main(["--excl", str(tmp_path / "e"), "--shared", str(tmp_path / "s"), "--arm", "cd_ib0"])
    assert out["rungs"]["heavy"]["rule"] == "max eligible bucket r" and abs(out["r"] - 2.5) < 1e-9


def test_small_bucket_is_not_eligible(tmp_path):
    ex = {"le10": (1.0, 1000), "11_50": (1.0, 1000), "gt50": (0.1, 100)}
    write(tmp_path / "e", "heavy", ex)
    write(tmp_path / "s", "heavy", {"le10": (1.0, 1000), "11_50": (1.0, 1000), "gt50": (0.5, 100)})
    out = C.main(["--excl", str(tmp_path / "e"), "--shared", str(tmp_path / "s"), "--arm", "cd_ib0"])
    assert out["rungs"]["heavy"]["rule"] == "pooled" and not out["rungs"]["heavy"]["by_bucket"]["gt50"]["eligible"]


def test_not_sim_identical_fails(tmp_path):
    write(tmp_path / "e", "heavy", EX)
    write(tmp_path / "s", "heavy", EX, mean=2.1)
    with pytest.raises(SystemExit, match="sim-identical"):
        C.main(["--excl", str(tmp_path / "e"), "--shared", str(tmp_path / "s"), "--arm", "cd_ib0"])


def run2(tmp_path, *extra):
    return C.main(["--excl", str(tmp_path / "e"), "--shared", str(tmp_path / "s"), "--arm", "cd_ib0", *extra])


def test_venue_is_recorded_and_one_state(tmp_path):
    write(tmp_path / "e", "heavy", EX)
    write(tmp_path / "s", "heavy", EX, host={**HOST, "nproc": 8, "cpuset": "56-63"})
    out = run2(tmp_path)
    assert out["venue"]["smt_control"] == "off" and out["venue"]["shared"][0]["nproc"] == 8
    write(tmp_path / "s", "heavy", EX, host={**HOST, "smt_control": "on"})
    with pytest.raises(SystemExit, match="mix CPU model / SMT"):
        run2(tmp_path)
    write(tmp_path / "s", "heavy", EX, host={**HOST, "cpu_model": "AMD EPYC 9475F 48-Core Processor"})
    with pytest.raises(SystemExit, match="mix CPU model / SMT"):
        run2(tmp_path)


def test_host_block_required_and_threads_within_nproc(tmp_path):
    write(tmp_path / "e", "heavy", EX)
    write(tmp_path / "s", "heavy", EX, host=None)
    with pytest.raises(SystemExit, match="no host block"):
        run2(tmp_path)
    assert run2(tmp_path, "--legacy-no-host")["venue"].startswith("LEGACY")
    write(tmp_path / "s", "heavy", EX, host={**HOST, "nproc": 8}, threads=9)
    with pytest.raises(SystemExit, match="void"):
        run2(tmp_path)


def test_node_balanced_reps_pool_and_must_agree(tmp_path):
    for d in ("e", "e2"):
        write(tmp_path / d, "heavy", EX)
    write(tmp_path / "s", "heavy", {b: (1.2 * t, n) for b, (t, n) in EX.items()})
    write(tmp_path / "s2", "heavy", {b: (1.6 * t, n) for b, (t, n) in EX.items()}, host={**HOST, "hostname": "os-cpu-slurm-10"})
    out = C.main(["--excl", f"{tmp_path / 'e'},{tmp_path / 'e2'}", "--shared", f"{tmp_path / 's'},{tmp_path / 's2'}", "--arm", "cd_ib0"])
    assert abs(out["r"] - 1.4) < 1e-9 and out["venue"]["hosts"] == ["os-cpu-slurm-10", "os-cpu-slurm-9"]
    write(tmp_path / "e2", "heavy", EX, mean=2.1)
    with pytest.raises(SystemExit, match="sim-identical"):
        C.main(["--excl", f"{tmp_path / 'e'},{tmp_path / 'e2'}", "--shared", str(tmp_path / "s"), "--arm", "cd_ib0"])
