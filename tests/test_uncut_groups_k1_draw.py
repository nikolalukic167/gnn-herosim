"""uncut_groups_s0_v1 K1(b): the sealed draw is seeded, stratified by size bucket, and names first-member triggers."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_groups_k1_draw as D  # noqa: E402


def _wl(sizes):
    ev = []
    for g, n in enumerate(sizes):
        ev += [{"peer_group": g}] * n
    return {"events": ev}


def test_draw_is_seeded_stratified_and_names_first_members():
    sizes = [3] * 50 + [20] * 30 + [60] * 10 + [150] * 4
    wls = {(w, r): _wl(sizes) for w in D.WINDOWS for r in D.RUNGS}
    s1, p1 = D.draw(wls, [1, 2, 3], seed=7, per_cell=25)
    s2, p2 = D.draw(wls, [1, 2, 3], seed=7, per_cell=25)
    assert s1 == s2 and p1 == p2 and len(s1) == 200
    firsts = {sum(sizes[:g]) for g in range(len(sizes))}
    assert all(s["trigger_task_id"] in firsts and s["group_size"] > 10 for s in s1)
    cell = [s for s in s1 if s["window"] == "g0" and s["rung"] == "heavy"]
    w = {"11_50": 20 * 30, "51_100": 60 * 10, "gt100": 150 * 4}
    assert {b: sum(1 for s in cell if s["bucket"] == b) for b in w} == D.allocate(25, w)
    assert sum(len(v) for v in p1.values()) == 200
    assert len({(s['topology'], s['window'], s['rung'], s['trigger_task_id']) for s in s1}) == 200
    assert D.draw(wls, [1, 2, 3], seed=8)[0] != s1


def test_allocation_is_largest_remainder():
    assert D.allocate(25, {"a": 1, "b": 1, "c": 1}) in ({"a": 9, "b": 8, "c": 8},)
    assert sum(D.allocate(25, {"a": 600, "b": 600, "c": 600}).values()) == 25
