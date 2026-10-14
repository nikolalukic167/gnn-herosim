import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_cotenancy_r as C  # noqa: E402


def write(d, rung, by, mean=2.0):
    d.mkdir(exist_ok=True)
    ev = sum(v[1] for v in by.values())
    s = {"averageElapsedTime": mean, "schedulerCounters": {"sched_plan_evals": ev},
         "decisionTiming": {"total_s": sum(v[0] for v in by.values()), "by_bucket": {b: {"total_s": t, "plan_evals": n} for b, (t, n) in by.items()}}}
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
