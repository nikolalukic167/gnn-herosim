import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import cost_to_go_handv as hv  # noqa: E402


def ev(ts, typ):
    return {"timestamp": ts, "application": {"dag": {typ: []}}}


def feat(load_after, type_platforms, replicas=(), cold=None, scale_in=100.0):
    return {"load_after": load_after, "type_platforms": type_platforms, "replicas": list(replicas),
            "cold_cost_s": cold or {}, "scale_in_after_s": scale_in}


def test_rates_use_only_events_up_to_t0():
    evs = [ev(10, "a"), ev(40, "a"), ev(50, "b"), ev(61, "a"), ev(9999, "b")]
    r = hv.type_rates(evs, t0=60.0, lookback_s=60.0)
    assert r == {"a": 2 / 60.0, "b": 1 / 60.0}
    future = copy.deepcopy(evs)
    future[-1]["timestamp"] = 70.0
    future.append(ev(61.5, "a"))
    assert hv.type_rates(future, 60.0, 60.0) == r


def test_rates_shorter_than_lookback_and_empty():
    assert hv.type_rates([ev(1, "a"), ev(2, "a")], t0=4.0, lookback_s=60.0) == {"a": 0.5}
    assert hv.type_rates([ev(1, "a")], t0=0.0, lookback_s=60.0) == {}


def test_load_term_weights_backlog_by_expected_arrivals_share():
    f = feat({"p1": 10.0, "p2": 2.0}, {"a": ["p1", "p2"], "b": ["p2"]})
    out = hv.hand_v(f, {"a": 0.1, "b": 0.2}, H=10.0)
    # a: 1 expected arrival split over p1,p2 (0.5 each); b: 2 expected on p2
    assert out["load"] == pytest.approx(0.5 * 10.0 + (0.5 + 2.0) * 2.0)
    assert out["cold"] == 0.0 and out["v"] == out["load"]


def test_load_term_zero_rate_is_zero_and_plan_with_more_backlog_costs_more():
    f1 = feat({"p1": 5.0}, {"a": ["p1"]})
    f2 = feat({"p1": 9.0}, {"a": ["p1"]})
    assert hv.hand_v(f1, {}, 10.0)["v"] == 0.0
    assert hv.hand_v(f2, {"a": 0.1}, 10.0)["v"] > hv.hand_v(f1, {"a": 0.1}, 10.0)["v"]


def test_cold_term_counts_only_unused_replicas_reaching_scale_in():
    reps = [{"key": "p1", "type": "a", "idle_s": 90.0, "used": False},
            {"key": "p2", "type": "a", "idle_s": 90.0, "used": True},
            {"key": "p3", "type": "a", "idle_s": 10.0, "used": False}]
    f = feat({}, {"a": ["p1", "p2", "p3"]}, reps, cold={"a": 6.0}, scale_in=100.0)
    out = hv.hand_v(f, {"a": 0.3}, H=10.0)
    assert out["cold"] == pytest.approx(0.3 * 10.0 / 3 * 6.0)   # only p1: idle 90 + 10 >= 100, unused
    assert hv.hand_v(f, {"a": 0.3}, H=5.0)["cold"] == 0.0         # 95 < 100: nobody scales in within 5 s


def test_choose_ties_go_to_policy_and_lambda_zero_is_argmin_s():
    c = {"policy": (0.0, 0.0), "s0": (-1.0, 5.0), "s1": (0.0, -3.0)}
    assert hv.choose(c, 0.0) == "s0"
    assert hv.choose({"policy": (0.0, 0.0), "s0": (0.0, 0.0)}, 1.0) == "policy"
    assert hv.choose(c, 1.0) == "s1"      # -1+5=4 vs 0-3=-3
    assert hv.choose(c, 0.1) == "s0"      # -0.5 vs -0.3


def mk_state(dq, cands, cell="cc40s16251_heavy_g0"):
    return {"cell": cell, "cands": cands, "dq": dq}


def test_fit_lambda_recovers_a_v_that_corrects_s():
    # S prefers s0 but Q_H prefers s1; V sees it. lambda large enough flips the pick.
    st = mk_state({"policy": 0.0, "s0": 2.0, "s1": -3.0}, {"policy": (0, 0), "s0": (-1.0, 4.0), "s1": (0.0, -2.0)})
    lam, small = hv.fit_lambda([st] * 10)
    assert lam > 1 / 6          # s1 beats s0 once -2*lam < -1 + 4*lam
    assert hv.choose(st["cands"], lam) == "s1"
    m = hv.score([st], lam)
    assert m["hit"] == 1.0 and m["hit_lambda0"] == 0.0 and m["captured_share"] == pytest.approx(1.0)
    # the near-tie breaker never changes a pick it was not allowed to on the fit set
    assert small <= lam


def test_fit_lambda_prefers_smallest_on_ties():
    st = mk_state({"policy": 0.0, "s0": -1.0}, {"policy": (0, 0), "s0": (-1.0, 0.0)})
    assert hv.fit_lambda([st] * 5)[0] == 0.0


def test_spearman_of_a_perfect_v():
    st = mk_state({"policy": 0.0, "s0": 1.0, "s1": 2.0, "s2": 3.0},
                  {"policy": (0, 0), "s0": (0, 10.0), "s1": (0, 20.0), "s2": (0, 30.0)})
    sp = hv.spearmans([st])
    assert sp["pooled"] == pytest.approx(1.0) and sp["per_state_mean"] == pytest.approx(1.0)


def test_split_must_be_disjoint_and_nonempty():
    hv.check_split([1], [2], [3])
    with pytest.raises(ValueError, match="empty --fit"):
        hv.check_split([], [2], [3])
    with pytest.raises(ValueError, match="empty --eval"):
        hv.check_split([1], [], [])
    with pytest.raises(ValueError, match="eval"):
        hv.check_split([1, 2], [2], [])
    with pytest.raises(ValueError, match="gate"):
        hv.check_split([1, 5], [2], [5, 6])


def test_topo_of_cell():
    assert hv.topo_of_cell("shard_cc40s16251_heavy_g0.jsonl") == 16251
    with pytest.raises(ValueError):
        hv.topo_of_cell("nonsense")


def row(ds, H, slot, q, plan, **kw):
    return dict(ds=ds, H=H, eps=0.0, continuation="cd_exacts", tag=f"s0|{slot}", q=q, plan=plan, **kw)


def test_build_states_pairs_against_policy_and_drops_gnn_and_duplicates():
    ds = "/x/ds_1"
    P0, P1, P2, P3 = [[1, 1]], [[2, 2]], [[3, 3]], [[4, 4]]
    rows = [row(ds, 5.0, "policy", 10.0, P0), row(ds, 5.0, "s0", 10.0, P0), row(ds, 5.0, "s1", 9.0, P1),
            row(ds, 5.0, "s2", 12.0, P2), row(ds, 5.0, "gnn", 8.0, P3)]
    tops = {ds: {"s": 1.0, "top_s": [{"s": 1.0}, {"s": 1.5}, {"s": 2.5}]}}
    base = {"load_after": {"p": 0.0}, "type_platforms": {"a": ["p"]}, "replicas": [], "cold_cost_s": {}, "scale_in_after_s": 1.0}
    feats = {}
    for slot, load in (("policy", 0.0), ("s1", 4.0), ("s2", 8.0), ("gnn", 1.0)):
        f = copy.deepcopy(base)
        f["load_after"] = {"p": load}
        feats[(ds, slot)] = f
    out = hv.build_states(rows, tops, feats, lambda d: {"a": 0.1}, lambda d: "cc40s16251_heavy_g0")
    st = out[(ds, 5.0)]
    assert set(st["cands"]) == {"policy", "s1", "s2"}          # s0 equals the policy plan; gnn has no S
    assert st["dq"] == {"policy": 0.0, "s1": -1.0, "s2": 2.0}
    assert st["cands"]["s1"][0] == pytest.approx(0.5)           # dS paired against the policy S
    assert st["cands"]["s2"][1] == pytest.approx(0.1 * 5.0 * 8.0)  # dV: rate * H * load_after


def test_build_states_refuses_perturbed_or_other_continuation_rows():
    ds = "/x/ds_1"
    bad = row(ds, 5.0, "policy", 1.0, [[1, 1]])
    bad["eps"] = 0.01
    with pytest.raises(ValueError, match="eps=0"):
        hv.build_states([bad], {}, {}, lambda d: {}, lambda d: "")


def test_post_commit_features_with_fakes(monkeypatch):
    import importlib
    import types
    live = types.ModuleType("src.placement.live_audit")
    live.platform_queue_drain_seconds = lambda platform, orch, memo, exec_scale=1.0: platform.drain * exec_scale
    live.inflight_remaining_seconds = lambda platform: platform.inflight
    for name, mod in (("src", types.ModuleType("src")), ("src.placement", types.ModuleType("src.placement")),
                      ("src.placement.live_audit", live)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "cost_to_go_handv_features", raising=False)
    fe = importlib.import_module("cost_to_go_handv_features")

    class N:
        def __init__(self, i): self.id = i

    class P:
        def __init__(self, i, drain, inflight, idle_since, last_allocated=0.0):
            self.id, self.drain, self.inflight = i, drain, inflight
            self.idle_since, self.last_allocated = idle_since, last_allocated

    n1, n2 = N(1), N(2)
    p1, p2, p3 = P(10, 3.0, 1.0, 50.0), P(20, 0.0, 0.0, float("inf"), last_allocated=80.0), P(30, 5.0, None, 10.0)

    class Sched:
        pg_inflight = True
        nodes = types.SimpleNamespace(items={})
        def _pg_orchestrator(self): return object()
        def _pg_xf(self, platform): return 1.0
        def _pg_plan_cost(self, tasks, plan, orch, memo, nodes): return 9.9, [2.0, 0.5], [1.0, 1.0]

    state = types.SimpleNamespace(replicas={"a": {(n1, p1), (n2, p2)}, "b": {(n1, p3)}})
    out = fe.post_commit_features(Sched(), ["t0", "t1"], [(n1, p1), (n1, p1)], state, now=100.0,
                                  cold_cost_s={"a": 4.0, "b": 6.0}, scale_in_after_s=120.0, ds="d", slot="policy", t0=100.0, workload="w.json")
    assert out["load_after"]["1:10"] == pytest.approx(3.0 + 1.0 + 2.5)     # drain + in-flight + the plan's own service (both tasks)
    assert out["load_after"]["2:20"] == 0.0
    assert out["load_after"]["1:30"] == 5.0                              # in-flight None counts as 0
    idle = {r["key"]: (r["idle_s"], r["used"]) for r in out["replicas"]}
    assert idle["1:10"] == (50.0, True) and idle["2:20"] == (20.0, False) and idle["1:30"] == (90.0, False)
    assert out["type_platforms"] == {"a": ["1:10", "2:20"], "b": ["1:30"]}
    with pytest.raises(KeyError, match="cold_cost_s"):
        fe.post_commit_features(Sched(), ["t0", "t1"], [(n1, p1), (n1, p1)], state, 100.0, {"a": 4.0}, 120.0, ds="d", slot="s", t0=1.0, workload="w")


def test_read_end_to_end_on_synthetic_files(tmp_path, capsys):
    import argparse
    rows, tops, feats = [], [], []
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps({"events": [ev(float(t), "a") for t in range(1, 100)]}))
    for topo, n in ((16301, 6), (16251, 6)):
        for i in range(n):
            ds = tmp_path / f"ds_{topo}_{i}"
            ds.mkdir()
            (ds / "generation_provenance.json").write_text(json.dumps({"argv": ["--snapshots", f"/x/shard_cc40s{topo}_heavy_g0.jsonl"]}))
            d = str(ds)
            tops.append({"ds": d, "argmin_s": {"s": 1.0, "top_s": [{"s": 1.0}, {"s": 1.2}]}})
            for H in hv.HS:
                rows.append(row(d, H, "policy", 10.0, [[1, 1]]))
                rows.append(row(d, H, "s1", 8.0, [[2, 2]]))     # S says policy, Q_H says s1; V should say s1 too
            for slot, load in (("policy", 5.0), ("s1", 1.0)):
                feats.append({"ds": d, "slot": slot, "t0": 90.0, "workload": str(wl), "load_after": {"p": load},
                              "type_platforms": {"a": ["p"]}, "replicas": [], "cold_cost_s": {}, "scale_in_after_s": 1e9})
    paths = {}
    for name, lst in (("s0", rows), ("tops", tops), ("features", feats)):
        paths[name] = tmp_path / f"{name}.jsonl"
        paths[name].write_text("".join(json.dumps(r) + "\n" for r in lst))
    a = argparse.Namespace(s0=str(paths["s0"]), tops=str(paths["tops"]), features=str(paths["features"]),
                           fit_topos="16301", eval_topos="16251", gate_topos="16251", lookback=60.0)
    hv.read(a)
    out = capsys.readouterr().out
    assert "H = 5 s: fit states 6, eval states 6" in out
    assert "picks the Q_H-best plan on 100.0 %" in out        # held-out, at the fitted lambda
    a.fit_topos = "16251"
    with pytest.raises(ValueError, match="overlaps"):
        hv.read(a)
