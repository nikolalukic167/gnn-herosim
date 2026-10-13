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


def test_served_cross_pairs_and_draw_strata():
    snap = {"fidelity": {"pairs": [[1, 2, 1.0], [2, 3, 1.0], [3, 9, 1.0]]}}
    chosen = {"chosen": [[1, 5, 0], [2, 5, 1], [3, 6, 0]], "deferred": [4]}
    assert A.served_cross_pairs(snap, chosen) == (1, 1)
    rows = []
    for b, n_pool in (("le10", 6000), ("11_50", 5000), ("51_100", 2500), ("101_256", 400)):
        for early in (True, False):
            for i in range(2000 if (b == "101_256" and not early) else n_pool):
                rows.append({"bucket": b, "early": early, "cross": i % 7, "size": 20 if b == "11_50" else 70 if b == "51_100" else 150 if b == "101_256" else 4, "key": f"{b}{early}{i}", "n_deferred": 0})
    res = A.draw_states(rows, 1)
    assert res["counts"]["le10"] == 4800 and res["counts"]["101_256"] == 600
    fb = [x for x in res["census"] if x["bucket"] == "101_256" and x["early"]]
    assert all(x["fallback"] for x in fb) and sum(x["draw"] for x in fb) == 400 // 3
    c = {(x["bucket"], x["early"], x["side"]): x for x in res["census"]}
    assert c[("11_50", True, "above")]["draw"] == round(1380 * 2 / 3) and c[("11_50", True, "below")]["draw"] == 1380 - round(1380 * 2 / 3)
    assert all(r["cross"] > 3 for r in res["drawn"] if r["bucket"] == "11_50" and r["early"] and r["side"] == "above")
    assert res["opening_share"] > 0.25 or res["problems"]


def test_dataset_plan_maps_gid_order_to_dataset_task_indexing():
    from scripts_cosim import uncut_arm1_build as B
    plan = [[1, 10], [2, 20], [3, 30]]                       # snapshot gid order: gids 5, 6, 7
    assert B.dataset_plan(plan, [5, 6, 7], [7, 5, 6]) == {"0": [3, 30], "1": [1, 10], "2": [2, 20]}
