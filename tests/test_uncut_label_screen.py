import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import uncut_label_screen as L  # noqa: E402


def frame(counts, topo=16201):
    f, i = {}, 0
    for n, how_many in counts.items():
        for _ in range(how_many):
            f[f"gnn_datasets_wf1_train/ds_{i:05d}"] = {"n_tasks": n, "state_id": f"C1/cc40s{topo}__g1heavy__cd_ib0_s0#{i}", "plan_space_size": 100 * n, "label_source": "exact"}
            i += 1
    return f


def test_draw_takes_125_per_bucket_or_all_and_is_deterministic():
    f = frame({5: 300, 30: 300, 70: 300, 150: 40})
    a, b = L.draw(f), L.draw(f)
    assert a == b and a["state_list_sha256"] == b["state_list_sha256"]
    assert {k: v["drawn"] for k, v in a["counts"].items()} == {"le10": 125, "11_50": 125, "51_100": 125, "101_256": 40}
    assert a["n"] == 415 and len({s["gid"] for s in a["states"]}) == 415


def test_draw_changes_with_seed():
    f = frame({5: 300, 30: 300, 70: 300, 150: 300})
    assert L.draw(f)["state_list_sha256"] != L.draw(f, seed=1)["state_list_sha256"]


def test_forbidden_topology_fails_loud():
    for t in (19537, 19548, 19554, 19589):
        with pytest.raises(SystemExit, match="forbidden"):
            L.draw(frame({5: 10}, topo=t))
    L.draw(frame({5: 10}, topo=19549))      # not in either set


def test_accel_test_id_clash_fails_loud():
    f = frame({5: 10})
    with pytest.raises(SystemExit, match="accel"):
        L.draw(f, accel_test_ids=[next(iter(f))])


def test_bucket_edges_and_out_of_range():
    assert [L.bucket_of(n) for n in (1, 10, 11, 50, 51, 100, 101, 256)] == ["le10", "le10", "11_50", "11_50", "51_100", "51_100", "101_256", "101_256"]
    with pytest.raises(ValueError):
        L.bucket_of(257)


def test_small_flag_follows_plan_space_600():
    f = frame({5: 3})
    k = sorted(f)
    f[k[0]]["plan_space_size"] = 600
    f[k[1]]["plan_space_size"] = 601
    f[k[2]]["plan_space_size"] = None
    out = {s["gid"]: s["small"] for s in L.draw(f)["states"]}
    assert out == {k[0]: True, k[1]: False, k[2]: False}
