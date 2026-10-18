import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import uncut_ext_inflight_read as R  # noqa: E402

ARM = "ra_gnn_eng_arm1b_nosplit"


def cell(whole, a360):
    return {"averageElapsedTime": whole, "steady_state": {"after_360s": {"mean": a360}}, "_path": "x"}


def cells(topos, arm_vals, seeds):
    arm, bar = {}, {}
    for t in topos:
        for w in ("g0", "g1", "g2", "g3"):
            for r in ("moderate", "heavy"):
                bar[(R.BAR_KIND, 0, t, w, r)] = cell(10.0, 8.0)
                for sd in seeds:
                    arm[(ARM, sd, t, w, r)] = cell(arm_vals[0], arm_vals[1])
    return arm, bar


def test_read_matches_the_reader_statistic_and_counts_wins():
    topos = list(range(19537, 19549))
    arm, bar = cells(topos, (9.0, 8.4), (11, 12))
    res = R.read({"_k100": arm}, bar, ARM, [11, 12], topos)["_k100"]
    by = {(e["rung"], e["stat"]): e for e in res}
    assert by[("moderate", "whole")]["median_pct"] == pytest.approx(-10.0) and by[("moderate", "whole")]["ahead"] == 12
    assert by[("heavy", "after360")]["median_pct"] == pytest.approx(5.0) and by[("heavy", "after360")]["ahead"] == 0
    assert by[("moderate", "whole")]["p_unadjusted"] < 0.001 and by[("moderate", "whole")]["n_cells"] == 48


def test_a_missing_seed_removes_the_cell_for_that_topology():
    topos = [19537, 19538]
    arm, bar = cells(topos, (9.0, 8.4), (11, 12))
    del arm[(ARM, 12, 19537, "g0", "moderate")]
    e = {(x["rung"], x["stat"]): x for x in R.read({"_k100": arm}, bar, ARM, [11, 12], topos)["_k100"]}[("moderate", "whole")]
    assert e["n_cells"] == 7 and e["n_topologies"] == 2
