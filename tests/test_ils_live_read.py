"""scripts_cosim/ils_live_read.py on a synthetic tree (3 topologies): the freeze file, A2's budget rule, the grid, the asserts and the statistics."""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import ils_live_cells as C  # noqa: E402
import ils_live_read as R  # noqa: E402

TOPOS = [1, 2, 3]
PIN = "a" * 40
MODELS = {}
FACT = {"bar": 1.0, "gnn": 0.95, "exact": 1.0, "ils200": 1.10, "ils50": 1.02, "ils10": 0.99}


def rho(heavy, moderate=None):
    """A projection's rho block: {rung: {budget: {rho_median}}}; heavy / moderate are {budget: rho}."""
    moderate = moderate or {b: v * 0.8 for b, v in heavy.items()}
    return {"heavy": {b: {"rho_median": v} for b, v in heavy.items()}, "moderate": {b: {"rho_median": v} for b, v in moderate.items()}}


def w_file(path):
    byb = {"cd_exactS_ib0_live": {r: {"le10": 0.05, "11_50": 0.2, "gt50": 0.4} for r in R.RUNGS}}
    d = {"w_ms_per_plan": {k: {r: 0.0116 for r in R.RUNGS} for k in R.ILS_KIND.values()}, "w_ms_per_plan_by_bucket": byb, "exact_live_G": 8,
         "w_S_ms_per_plan_by_bucket": {k: {r: {"le10": 0.07, "11_50": 0.23, "gt50": 0.47} for r in R.RUNGS} for k in R.ILS_KIND.values()}}
    json.dump(d, open(path, "w"))


def summary(arm, sd, t, w, r, wf):
    e = (10.0 + t + int(w[1:]) + (5 if r == "heavy" else 0)) * FACT[arm]
    s = {"num_tasks": 100, "code": {"commit": ("7cb483c5" + "0" * 32) if arm in ("bar", "gnn") else PIN, "dirty": False}, "servers": 24, "topology": t,
         "scaleOut": {"target": 0.7}, "env": {}, "averageElapsedTime": e, "steady_state": {"after_360s": {"mean": e * 0.9}},
         "endTime": 1000.0, "arrival_end": {"last_arrival_s": 990.0},
         "schedulerCounters": {"sched_queue_wait_seconds": 0.0, "sched_busy_seconds": 2.0}}
    if arm in R.ARM_KIND:
        k = R.ARM_KIND[arm]
        want = R.expected_charge(wf, k, r)
        env = {"HEROSIM_PG_BATCH_ALPHA": "0", "HEROSIM_PG_INFLIGHT_FULL": "1"}
        ils = arm.startswith("ils")
        env.update({"HEROSIM_PG_ILS_PER": arm[3:], "HEROSIM_PG_ILS_SEED": "0"} if ils else {"HEROSIM_PG_CD_EXACT": "1", "HEROSIM_PG_EXACT_G": "8"})
        for var, v in want.items():
            env[var] = json.dumps(v) if isinstance(v, dict) else repr(v)
        s["env"] = env
        s["schedulerCounters"].update(sched_felt_task_seconds=3.0, pg_ils_batches=10 if ils else 0, pg_ils_improved=4 if ils else 0,
                                      pg_ils_evals=500 if ils else 0, sched_plan_evals_s=500 if ils else 0,
                                      pg_exact_batches=7 if arm == "exact" else 0, pg_exact_g_skipped=3 if arm == "exact" else 0)
    if arm == "gnn":
        s["env"]["GNN_MODEL_PATH"] = MODELS[sd]
    return s


def make_tree(tmp_path, monkeypatch, arms, rho_block):
    U = str(tmp_path)
    monkeypatch.setattr(R, "TOPOS", TOPOS)
    monkeypatch.setattr(C, "TOPOS", TOPOS)
    wf = f"{U}/w.json"
    w_file(wf)
    os.makedirs(f"{U}/comb")
    md5s = {}
    for sd in (1, 2):
        MODELS[sd] = f"{U}/comb/{R.CKPT_FILE.format(sd)}"
        open(MODELS[sd], "wb").write(f"m{sd}".encode())
        md5s[f"ra_gnn_eng_combfl_nosplit|s{sd}"] = hashlib.md5(f"m{sd}".encode()).hexdigest()
    json.dump(md5s, open(f"{U}/comb/md5_comball.json", "w"))
    os.makedirs(f"{U}/ils_live")
    json.dump({"rho": rho_block}, open(f"{U}/ils_live/projection.json", "w"))
    fz = {"arms": {}, "projection": {"file": "ils_live/projection.json", "md5": R.md5_file(f"{U}/ils_live/projection.json")}}
    for arm in arms:
        lst = "exact" if arm == "exact" else "ils"
        lines = C.lines(lst, None if arm == "exact" else arm[3:])
        lp = f"ils_live/list_{arm}.txt"
        open(f"{U}/{lp}", "w").write("\n".join(lines) + "\n")
        fz["arms"][arm] = {"pin": PIN, "list": lp, "list_md5": R.md5_file(f"{U}/{lp}"), "w_md5": R.w_entries_md5(wf, arm)}
    fzp = f"{U}/freeze.json"
    json.dump(fz, open(fzp, "w"))
    fz2, rho2 = R.load_freeze(U, fzp, wf)
    rt = R.roots(U, fz2)
    g = R.grid(rt, R.required_arms(fz2, rho2))
    for key, stem in g.items():
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        json.dump(summary(key[0], key[1], key[2], key[3], key[4], wf), open(stem + ".summary.json", "w"))
    return U, wf, fzp


def test_lists_have_96_cells_and_the_exact_list_two_identity_lines():
    assert len(C.lines("ils", "50")) == 96
    ls = C.lines("exact")
    assert len(ls) == 96 + 2 and ls[-1].startswith("ils_live/ident_exact_gnn")


@pytest.mark.parametrize("heavy,expect_bud", [({"200": 3.0, "50": 1.4, "10": 0.4}, {"heavy": "10", "moderate": "10"}),
                                              ({"200": 3.0, "50": 0.9, "10": 0.3}, {"heavy": "50", "moderate": "50"}),
                                              ({"200": 0.9, "50": 0.5, "10": 0.2}, {"heavy": "200", "moderate": "200"}),
                                              ({"200": 3.0, "50": 2.0, "10": 1.2}, {"heavy": None, "moderate": "10"})])   # moderate rho = 0.8 x heavy
def test_a2_budget_rule(heavy, expect_bud):
    bud, primary = R.ils_budget_per_load(rho(heavy))
    assert bud == expect_bud and primary == expect_bud["heavy"]


def test_a2_budget_rule_none_at_heavy_reads_moderate_at_the_largest_feasible():
    bud, primary = R.ils_budget_per_load(rho({"200": 3.0, "50": 2.0, "10": 1.2}, {"200": 1.5, "50": 0.8, "10": 0.4}))
    assert bud == {"heavy": None, "moderate": "50"} and primary is None
    bud, primary = R.ils_budget_per_load(rho({"200": 3.0, "50": 0.9, "10": 0.3}, {"200": 1.5, "50": 1.1, "10": 0.4}))
    assert bud == {"heavy": "50", "moderate": None} and primary == "50"      # feasible at heavy, infeasible at moderate: that test enters as infeasible


def test_freeze_file_is_checked(tmp_path, monkeypatch):
    U, wf, fzp = make_tree(tmp_path, monkeypatch, ["exact", "ils50"], rho({"200": 3.0, "50": 0.9, "10": 0.3}))
    R.load_freeze(U, fzp, wf)
    fz = json.load(open(fzp))
    open(f"{U}/{fz['arms']['exact']['list']}", "a").write("ils_live/cells_exact\textra\tx\n")
    with pytest.raises(SystemExit, match="cell list md5"):
        R.load_freeze(U, fzp, wf)
    fz = json.load(open(fzp))
    fz["arms"]["exact"]["list_md5"] = R.md5_file(f"{U}/{fz['arms']['exact']['list']}")
    fz["arms"]["ils50"]["w_md5"] = "0" * 32
    json.dump(fz, open(fzp, "w"))
    with pytest.raises(SystemExit, match="w entries"):
        R.load_freeze(U, fzp, wf)


def test_read_and_statistics_at_the_primary_budget(tmp_path, monkeypatch):
    U, wf, fzp = make_tree(tmp_path, monkeypatch, ["exact", "ils50", "ils10"], rho({"200": 3.0, "50": 0.9, "10": 0.3}))
    fz, rh = R.load_freeze(U, fzp, wf)
    cells, rows, g, req = R.load_all(U, fz, rh, wf)
    n_cells, n_ident = R.check_lists(U, fz, g)
    assert n_cells == 3 * 24 and n_ident == 2          # 3 synthetic topologies x 4 windows x 2 loads per arm
    res = R.build(cells, rows, req, rh)
    assert res["primary"] == "50" and res["budget"] == {"heavy": "50", "moderate": "50"}
    ib = {(e["rung"], e["stat"]): e for e in res["fam"]["F-ils-bar"]}
    assert ib[("moderate", "whole")]["median_pct"] == pytest.approx(2.0) and ib[("moderate", "whole")]["budget"] == "50"
    ig = res["fam"]["F-ils-gnn"][0]
    assert ig["median_pct"] == pytest.approx(100 * (0.95 - 1.02) / 1.02)
    assert all(abs(e["median_pct"]) < 1e-9 for e in res["fam"]["F-exact-bar"])
    assert all(e["p_holm"] >= e["p"] for e in res["fam"]["F-ils-bar"])
    # the ladder: both feasible budgets at both loads, vs bar and vs gnn
    assert {(k[0], k[1], k[2]) for k in res["ladder"]} == {(a, r, o) for a in ("ils50", "ils10") for r in R.RUNGS for o in ("bar", "gnn")}
    pa = R.per_arm(rows, cells, req)
    assert pa[("ils50", "moderate")]["busy_per_task"] == pytest.approx(0.02) and pa[("bar", "heavy")]["share_felt"] is None
    txt = R.report(fz, rh, res, pa, len(g), (n_cells, n_ident), req)
    assert "budget ladder" in txt and "batches on the bar's plan" in txt and "plan changed in" in txt


def test_an_infeasible_budget_enters_with_p_one_and_is_not_read(tmp_path, monkeypatch):
    # no budget feasible at heavy, moderate feasible only at 10: the heavy tests are infeasible entries; no ILS cell at heavy is required
    rb = rho({"200": 3.0, "50": 2.0, "10": 1.2}, {"200": 1.5, "50": 1.1, "10": 0.4})
    U, wf, fzp = make_tree(tmp_path, monkeypatch, ["ils10"], rb)
    fz, rh = R.load_freeze(U, fzp, wf)
    cells, rows, g, req = R.load_all(U, fz, rh, wf)
    assert ("ils10", "heavy") not in req and ("ils10", "moderate") in req
    res = R.build(cells, rows, req, rh)
    es = res["fam"]["F-ils-bar"]
    heavy = [e for e in es if e["rung"] == "heavy"]
    assert all(e.get("infeasible") and e["p"] == 1.0 for e in heavy)
    assert all(e["p_holm"] >= 1.0 - 1e-12 for e in heavy)
    assert all(e["median_pct"] is not None for e in es if e["rung"] == "moderate")
    txt = R.report(fz, rh, res, R.per_arm(rows, cells, req), len(g), R.check_lists(U, fz, g), req)
    assert "INFEASIBLE LIVE" in txt


def test_asserts_fail_loud(tmp_path, monkeypatch):
    U, wf, fzp = make_tree(tmp_path, monkeypatch, ["exact", "ils50"], rho({"200": 3.0, "50": 0.9, "10": 0.3}))
    fz, rh = R.load_freeze(U, fzp, wf)
    rt = R.roots(U, fz)
    ck = {"seen": {}, "md5": json.load(open(f"{U}/comb/md5_comball.json"))}
    key = ("ils50", 0, 1, "g0", "moderate")
    stem = R.stem_of(rt, "ils50", "moderate", 1, "g0", 0)
    s = json.load(open(stem + ".summary.json"))
    for mutate, why in ((lambda x: x["env"].__setitem__("HEROSIM_PG_ILS_PER", "200"), "HEROSIM_PG_ILS_PER"),
                        (lambda x: x["env"].__setitem__("HEROSIM_DECISION_COST_MS", "9.9"), "HEROSIM_DECISION_COST_MS"),
                        (lambda x: x["env"].__setitem__("HEROSIM_EXCHANGE_TCP", "1"), "env"),
                        (lambda x: x["code"].__setitem__("dirty", True), "pin"),
                        (lambda x: x["schedulerCounters"].__setitem__("pg_ils_batches", 0), "instrument"),
                        (lambda x: x["schedulerCounters"].__setitem__("pg_exact_batches", 1), "exact search")):
        t = json.loads(json.dumps(s))
        mutate(t)
        json.dump(t, open(stem + ".summary.json", "w"))
        with pytest.raises(SystemExit, match=why):
            R.one(rt, key, stem, ck, wf)
    json.dump(s, open(stem + ".summary.json", "w"))
    open(stem + ".failed.json", "w").write("{}")
    with pytest.raises(SystemExit, match="failed cell"):
        R.one(rt, key, stem, ck, wf)


def test_a_cell_outside_the_grid_fails(tmp_path, monkeypatch):
    U, wf, fzp = make_tree(tmp_path, monkeypatch, ["exact"], rho({"200": 3.0, "50": 0.9, "10": 0.3}))
    fz, rh = R.load_freeze(U, fzp, wf)
    d = R.roots(U, fz)[("exact", "heavy")][0]
    open(f"{d}/cc40s99__g0heavy__cd_exactS_ib0_live_k100_s0.summary.json", "w").write("{}")
    with pytest.raises(SystemExit, match="outside the grid"):
        R.load_all(U, fz, rh, wf)
