import pytest

from scripts_cosim.cd_gap_v1_build_b import surge_after


def wl():
    ev = []
    # group 0 before the switch, group 1 straddling by 2 ms, groups 2-3 after
    for g, ts in ((0, [1.0, 1.001]), (1, [9.999, 10.001]), (2, [12.0, 12.002]), (3, [20.0])):
        ev += [{"timestamp": t, "peer_group": g, "application": {"name": "a"}} for t in ts]
    ev.sort(key=lambda e: e["timestamp"])
    return {"events": ev, "duration": 30.0, "peer_exchange": [[0, 1, 5.0]]}


def test_k1_is_identity_on_events():
    w = wl()
    o = surge_after(w, 10.0, 1.0)
    assert [e["timestamp"] for e in o["events"]] == [e["timestamp"] for e in w["events"]]
    assert o["duration"] == 30.0


def test_groups_anchored_after_are_compressed_and_before_untouched():
    w = wl()
    o = surge_after(w, 10.0, 2.0)
    got = {(e["peer_group"], i): e["timestamp"] for i, e in enumerate(o["events"])}
    by = {}
    for e in o["events"]:
        by.setdefault(e["peer_group"], []).append(e["timestamp"])
    assert by[0] == [1.0, 1.001] and by[1] == [9.999, 10.001]
    assert by[2] == pytest.approx([11.0, 11.001]) and by[3] == [15.0]
    assert o["duration"] == 20.0
    r = o["warm_surge_s0_v1"]
    assert (r["n_pre"], r["n_post"], r["straddling_groups"]) == (4, 3, 1)
    assert o["peer_exchange"] == w["peer_exchange"]


def test_order_break_fails_loud():
    w = wl()
    w["events"].append({"timestamp": 10.0005, "peer_group": 1, "application": {"name": "a"}})
    w["events"].sort(key=lambda e: e["timestamp"])
    w["events"].insert(0, w["events"].pop())
    with pytest.raises(SystemExit):
        surge_after(w, 10.0, 3.0)
