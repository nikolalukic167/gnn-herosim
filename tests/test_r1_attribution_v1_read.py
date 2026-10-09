import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import r1_attribution_v1_read as R  # noqa: E402


def test_exact_wilcoxon_matches_enumeration():
    # all 5 differences positive: only the all-positive and all-negative patterns reach the extreme, p = 2 / 32
    assert R.exact_wilcoxon([1, 2, 3, 4, 5]) == pytest.approx(2 / 32)
    assert R.exact_wilcoxon([0, 0]) is None
    # symmetric: 1 +, 2 -, 3 + -> W+ = 4, mean 3, |dev| 1 -> every pattern with |W+ - 3| >= 1 : sums {0,1,2,4,5,6} = 6 of 8
    assert R.exact_wilcoxon([1, -2, 3]) == pytest.approx(6 / 8)


def test_holm_step_down_and_missing_counts_as_one():
    adj = R.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj["a"] == pytest.approx(0.04) and adj["c"] == pytest.approx(0.09) and adj["b"] == pytest.approx(0.09)
    assert adj["d"] is None


def _write(d: Path, kind, seed, topo, win, rung, lat, **kw):
    s = {"averageElapsedTime": lat, "num_tasks": 100, "effective_queue_share": 0.1, "latency_percentiles": {"p95": 5.0},
         "arrival_end": {"end_over_last_arrival": 1.0}, "requestFailures": 0, "wallclock_s": 10, "code": "x", **kw}
    (d / f"cc40s{topo}__{win}{rung}__{kind}_s{seed}.summary.json").write_text(json.dumps(s))


def test_reader_pairs_on_the_same_cell_and_seed(tmp_path):
    for t in range(9500, 9512):
        for w in ("g0", "g1"):
            for r in R.RUNGS:
                _write(tmp_path, "cd", 0, t, w, r, 2.0)
                for sd in (1, 2):
                    _write(tmp_path, "ra_gnn_eng", sd, t, w, r, 2.0 * 0.9)  # 10 % faster than CD everywhere
                    _write(tmp_path, "ra_twin_eng", sd, t, w, r, 2.0 * 0.95)
    (tmp_path / "cc40s9500__g0heavy__ra_gnn_eng_s1.summary.json").unlink()
    (tmp_path / "cc40s9500__g0heavy__ra_gnn_eng_s1.failed.json").write_text(json.dumps({"why": "watchdog: stall"}))
    rep = R.read(str(tmp_path), "ra_gnn_eng")
    prim = rep["main"]["primary"]["tests"]["heavy"]
    assert prim["median_pct"] == pytest.approx(-10.0) and prim["n_topologies"] == 12 and prim["wins"] == 12
    assert prim["p"] == pytest.approx(2 / 2 ** 12)
    s1 = rep["main"]["S1"]["tests"]["ra_gnn_eng vs ra_twin_eng|light"]
    assert s1["median_pct"] == pytest.approx(100 * (0.9 - 0.95) / 0.95)
    assert rep["main"]["S2"]["tests"]["cd_random vs cd|light"]["note"] == "not run (no implementation)"
    assert rep["arms"]["ra_gnn_eng|heavy"]["failed"] == 1 and rep["arms"]["ra_gnn_eng|heavy"]["failed_why"] == {"watchdog": 1}
    assert rep["sensitivity"]["excluded_topologies"] == [9500]
    assert rep["sensitivity"]["families"]["primary"]["tests"]["heavy"]["n_topologies"] == 11
