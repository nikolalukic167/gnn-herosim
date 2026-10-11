import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import gnn_overcap_batches_read as B  # noqa: E402


def tasks(lat, plats, cold=None):
    return {i: {"disp": float(i), "lat": lat[i], "cold": (cold or {}).get(i, 0), "plat": plats[i]} for i in range(len(lat))}


def test_buckets_add_to_the_gap_and_split_by_cap_and_size():
    lat_a = [3.0] * 8 + [1.0] * 4
    lat_b = [2.0] * 8 + [1.0] * 4
    a, b = tasks(lat_a, ["p"] * 12), tasks(lat_b, ["p", "q"] * 6)
    batches = [{"tasks": list(range(8)), "n": 8, "over": True}, {"tasks": [8, 9, 10, 11], "n": 4, "over": False}]
    gb, base, gap = B.gap_by_bucket(a, b, batches)
    assert gap == pytest.approx(8 / 12) and base == pytest.approx(20 / 12)
    assert gb[("opening", "over", "5to8")] == pytest.approx(8 / 12) and gb[("opening", "within", "le4")] == pytest.approx(0.0)
    assert gb[("all", "all", "all")] == pytest.approx(gap)
    assert sum(v for k, v in gb.items() if k[0] == "all" and k[1] != "all" and k[2] != "all") == pytest.approx(gap)


def test_stack_and_cold_are_counted_per_batch():
    t = tasks([1.0] * 4, ["p", "p", "q", "q"], cold={0: 1, 2: 1})
    r = B.stack_cold(t, [{"tasks": [0, 1, 2, 3], "n": 4, "over": True}])
    assert r[("over", "le4")] == [1, 1, 2, 4, 4] and r[("all", "all")] == [1, 1, 2, 4, 4]


def test_a_log_that_misses_or_repeats_tasks_fails_loudly():
    with pytest.raises(SystemExit, match="two batches"):
        B.check_partition([{"tasks": [0, 1]}, {"tasks": [1, 2]}], 3)
    with pytest.raises(SystemExit, match="covers 2"):
        B.check_partition([{"tasks": [0, 1]}], 3)


def test_load_batches_reads_the_scheduler_log(tmp_path):
    p = tmp_path / "o.jsonl"
    p.write_text(json.dumps({"tasks": [0, 1], "n": 2, "over": True, "s_gnn": 1.0, "s_expansion": 2.0}) + "\n")
    assert B.load_batches(p) == [{"tasks": [0, 1], "n": 2, "over": True}]
