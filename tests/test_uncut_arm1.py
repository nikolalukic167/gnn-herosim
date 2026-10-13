from scripts_cosim import uncut_arm1 as A


def _wl(sizes, step=10.0):
    ev, t = [], 0.0
    for g, s in enumerate(sizes):
        ev += [{"timestamp": t + 0.001 * k, "peer_group": g} for k in range(s)]
        t += step
    return {"events": ev, "peer_exchange": [[0, 1, 5.0]], "rps": len(ev)}


def test_groups_of_gives_trigger_size_and_arrival():
    assert A.groups_of(_wl([3, 12])) == [(0, 3, 0.0), (3, 12, 10.0)]


def test_plan_pool_strata_pool_and_over256():
    wls = {("A", "g0", "heavy"): _wl([20] * 100 + [300])}                    # 10 s spacing: groups 0..35 start before 360 s
    r = A.plan_pool(wls, [1, 2], seed=1, mult=5)
    c = {(x["bucket"], x["early"]): x for x in r["census"]}
    assert r["groups_over_256"] == {"heavy": 1}
    assert c[("11_50", True)]["available"] == 36 * 2 and c[("11_50", False)]["available"] == 64 * 2
    assert c[("11_50", True)]["pool"] == c[("11_50", True)]["available"] and not c[("11_50", True)]["ok"]
    assert len({(p["window"], p["rung"], p["topology"], p["trigger"]) for p in r["pool"]}) == len(r["pool"])
    assert all(isinstance(v, list) for d in r["plans"].values() for v in d.values())


def test_truncate_workload_keeps_arrival_prefix_and_valid_pairs():
    wl = _wl([2, 2, 2])
    out = A.truncate_workload(wl, 10.5)
    assert len(out["events"]) == 4 and out["peer_exchange"] == [[0, 1, 5.0]] and out["rps"] == 4


def test_open_sets_contribute_only_their_opening_groups():
    wl = _wl([20] * 100)           # 10 s spacing: groups 0..35 start before 360 s
    r = A.plan_pool({("A", "g0", "heavy"): wl, ("C1", "g0", "heavy"): wl}, [1], seed=1, mult=100, open_sets=["C1"])
    late = [c for c in r["census"] if c["bucket"] == "11_50" and not c["early"]][0]
    early = [c for c in r["census"] if c["bucket"] == "11_50" and c["early"]][0]
    assert late["available"] == 64 and early["available"] == 36 * 2
