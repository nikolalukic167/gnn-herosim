import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_interim_share as S  # noqa: E402

W = {"w_ms_per_plan": {"x": {"heavy": 2.0}}, "w_ms_per_plan_by_bucket": {"x": {"heavy": {"le10": 3.0, "11_50": 1.0, "gt50": 1.5}}}}


def write(d, sbs, lat=2.0, evals=None, name="cc40s19537__g0heavy__x_k100_s1"):
    d.mkdir(exist_ok=True)
    sbs = np.asarray(sbs, dtype=float)
    n = len(sbs)
    ev = evals or {"le10": int((sbs <= 10).sum()), "11_50": int(((sbs > 10) & (sbs <= 50)).sum()), "gt50": int((sbs > 50).sum())}
    s = {"averageElapsedTime": lat, "num_tasks": n,
         "decisionTiming": {"tasks": n, "by_bucket": {b: {"plan_evals": v, "total_s": 0.0} for b, v in ev.items()}}}
    (d / f"{name}.summary.json").write_text(json.dumps(s))
    arr = np.column_stack([np.arange(n), np.zeros(n), np.full(n, lat), sbs, sbs])
    np.save(d / f"{name}.tasks_bsz.npy", arr)


@pytest.fixture
def wf(tmp_path):
    f = tmp_path / "w.json"
    f.write_text(json.dumps(W))
    return str(f)


def test_pooled_and_felt_shares(tmp_path, wf):
    write(tmp_path / "c", [1] * 6 + [200] * 4)
    out = S.main(["--cells", str(tmp_path / "c"), "--arm", "x_k100", "--w-file", wf])
    h = out["rungs"]["heavy"]
    lat = 2.0 * 10
    assert abs(h["share_by_bucket"]["le10"] - 3.0e-3 * 6 / lat) < 1e-12
    assert abs(h["share_by_bucket"]["101_256"] - 1.5e-3 * 4 / lat) < 1e-12
    assert abs(h["felt_by_bucket"]["101_256"] - 1.5e-3 * 200 / 2.0) < 1e-12
    assert abs(h["felt_by_bucket"]["le10"] - 3.0e-3 * 1 / 2.0) < 1e-12
    assert abs(h["felt_total"] - (6 * 3.0e-3 + 4 * 1.5e-3 * 200) / lat) < 1e-12


def test_gt50_split_is_proportional_to_the_sidecar(tmp_path, wf):
    write(tmp_path / "c", [1] * 4 + [60] * 3 + [200] * 3, evals={"le10": 6, "11_50": 0, "gt50": 4})
    out = S.main(["--cells", str(tmp_path / "c"), "--arm", "x_k100", "--w-file", wf])
    h = out["rungs"]["heavy"]
    assert abs(h["evals"]["51_100"] - 2.0) < 1e-12 and abs(h["gt50_sidecar_gap_max"] - 0.5) < 1e-12


def test_not_one_plan_per_task_fails(tmp_path, wf):
    write(tmp_path / "c", [1] * 4, evals={"le10": 9, "11_50": 0, "gt50": 0})
    with pytest.raises(SystemExit, match="one-plan-per-task"):
        S.main(["--cells", str(tmp_path / "c"), "--arm", "x_k100", "--w-file", wf])


def test_arm2_mode_with_ratio_one_equals_default(tmp_path, wf):
    write(tmp_path / "c", [1] * 6 + [30] * 3 + [200] * 4)
    a = S.main(["--cells", str(tmp_path / "c"), "--arm", "x_k100", "--w-file", wf])
    b = S.main(["--cells", str(tmp_path / "c"), "--arm", "x_k100", "--w-file", wf, "--evals-per-task-from-buckets"])
    for k in ("share_total", "felt_total", "felt_by_bucket", "share_by_bucket"):
        assert a["rungs"]["heavy"][k] == b["rungs"]["heavy"][k]
    assert b["rungs"]["heavy"]["evals_per_task_ratio"]["le10"] == [1.0, 1.0] and "felt_label" in b and "felt_label" not in a


def test_arm2_mode_scales_felt_by_bucket_ratio_and_keeps_pooled(tmp_path, wf):
    write(tmp_path / "c", [1] * 6 + [200] * 4, evals={"le10": 42, "11_50": 0, "gt50": 32})
    with pytest.raises(SystemExit, match="one-plan-per-task"):
        S.main(["--cells", str(tmp_path / "c"), "--arm", "x_k100", "--w-file", wf])
    h = S.main(["--cells", str(tmp_path / "c"), "--arm", "x_k100", "--w-file", wf, "--evals-per-task-from-buckets"])["rungs"]["heavy"]
    assert abs(h["felt_by_bucket"]["le10"] - 3.0e-3 * 1 * 7 / 2.0) < 1e-12
    assert abs(h["felt_by_bucket"]["101_256"] - 1.5e-3 * 200 * 8 / 2.0) < 1e-12
    assert abs(h["share_by_bucket"]["le10"] - 3.0e-3 * 42 / 20.0) < 1e-12
