"""scripts_cosim/ils_live_read.py on a synthetic tree (3 topologies): the grid, the asserts, the frozen-ness refusal and the statistics."""
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
FACT = {"bar": 1.0, "gnn": 0.95, "ils": 1.02, "exact": 1.0}


def w_file(path):
    byb = {"cd_exactS_ib0_live": {r: {"le10": 0.05, "11_50": 0.2, "gt50": 0.4} for r in R.RUNGS}}
    d = {"w_ms_per_plan": {"cd_ib0_ils": {r: 0.0116 for r in R.RUNGS}}, "w_ms_per_plan_by_bucket": byb,
         "w_S_ms_per_plan_by_bucket": {"cd_ib0_ils": {r: {"le10": 0.07, "11_50": 0.23, "gt50": 0.47} for r in R.RUNGS}}}
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
        env.update({"HEROSIM_PG_ILS_PER": "200", "HEROSIM_PG_ILS_SEED": "0"} if arm == "ils" else {"HEROSIM_PG_CD_EXACT": "1", "HEROSIM_PG_EXACT_G": "8"})
        for var, v in want.items():
            env[var] = json.dumps(v) if isinstance(v, dict) else repr(v)
        s["env"] = env
        s["schedulerCounters"].update(sched_felt_task_seconds=3.0, pg_ils_batches=10 if arm == "ils" else 0, pg_ils_improved=4 if arm == "ils" else 0,
                                      pg_ils_evals=500 if arm == "ils" else 0, sched_plan_evals_s=500 if arm == "ils" else 0,
                                      pg_exact_batches=7 if arm == "exact" else 0, pg_exact_g_skipped=3 if arm == "exact" else 0)
    if arm == "gnn":
        s["env"]["GNN_MODEL_PATH"] = MODELS[sd]
    return s


@pytest.fixture
def tree(tmp_path, monkeypatch):
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
    for key, stem in R.grid(U).items():
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        json.dump(summary(key[0], key[1], key[2], key[3], key[4], wf), open(stem + ".summary.json", "w"))
    cl = f"{U}/cells.txt"
    open(cl, "w").write("\n".join(C.lines()) + "\n")
    monkeypatch.setattr(R, "ARM_PIN", PIN)
    monkeypatch.setattr(R, "W_MD5", hashlib.md5(open(wf, "rb").read()).hexdigest())
    monkeypatch.setattr(R, "CELLS_MD5", hashlib.md5(open(cl, "rb").read()).hexdigest())
    ck = {"seen": {}, "md5": json.load(open(f"{U}/comb/md5_comball.json"))}
    return U, wf, cl, ck


def test_the_list_has_192_cells_and_one_identity_line():
    ls = C.lines()
    assert len(ls) == 2 * 12 * 4 * 2 + 1 and ls[-1].startswith("ils_live/ident_bar")


def test_unfrozen_reader_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "CELLS_MD5", None)
    with pytest.raises(SystemExit, match="not frozen"):
        R.check_list(str(tmp_path), str(tmp_path / "x"))


def test_list_matches_the_grid_and_a_changed_list_fails(tree, monkeypatch):
    U, wf, cl, ck = tree
    assert R.check_list(U, cl) == (2 * 3 * 4 * 2, 1)
    open(cl, "a").write("ils_live/cells_ils\textra\tx\n")
    with pytest.raises(SystemExit, match="md5"):
        R.check_list(U, cl)


def test_read_and_statistics(tree):
    U, wf, cl, ck = tree
    cells, rows = {}, {}
    for key, stem in R.grid(U).items():
        cells[key], rows[key] = R.one(U, key, stem, ck, wf)
    res, nc = R.build(cells, rows)
    ib = {(e["rung"], e["stat"]): e for e in res["F-ils-bar"]}
    assert ib[("moderate", "whole")]["median_pct"] == pytest.approx(2.0) and ib[("moderate", "whole")]["faster"] == 0
    assert all(e["p_holm"] >= e["p"] for e in res["F-ils-bar"])
    ig = res["F-ils-gnn"][0]
    assert ig["median_pct"] == pytest.approx(100 * (0.95 - 1.02) / 1.02)
    assert all(abs(e["median_pct"]) < 1e-9 for e in res["F-exact-bar"])
    pa = R.per_arm(rows, cells)
    assert pa[("ils", "moderate")]["busy_per_task"] == pytest.approx(0.02) and pa[("ils", "moderate")]["share_felt"] is not None
    assert pa[("bar", "heavy")]["share_felt"] is None      # the filed bar cells predate the felt counter
    txt = R.report(res, nc, pa, (96, 1), 10, cells)
    assert "F-ils-bar" in txt and "F-exact-bar" in txt and "batches on the bar's plan" in txt and "plan changed in" in txt


def test_asserts_fail_loud(tree):
    U, wf, cl, ck = tree
    key = ("ils", 0, 1, "g0", "moderate")
    stem = R.stem_of(U, *[key[0], key[4], key[2], key[3], key[1]])
    s = json.load(open(stem + ".summary.json"))
    for mutate, why in ((lambda x: x["env"].__setitem__("HEROSIM_PG_ILS_PER", "50"), "HEROSIM_PG_ILS_PER"),
                        (lambda x: x["env"].__setitem__("HEROSIM_DECISION_COST_MS", "9.9"), "HEROSIM_DECISION_COST_MS"),
                        (lambda x: x["env"].__setitem__("HEROSIM_EXCHANGE_TCP", "1"), "env"),
                        (lambda x: x["code"].__setitem__("dirty", True), "pin"),
                        (lambda x: x["schedulerCounters"].__setitem__("pg_ils_batches", 0), "instrument"),
                        (lambda x: x["schedulerCounters"].__setitem__("pg_exact_batches", 1), "exact search")):
        t = json.loads(json.dumps(s))
        mutate(t)
        json.dump(t, open(stem + ".summary.json", "w"))
        with pytest.raises(SystemExit, match=why):
            R.one(U, key, stem, ck, wf)
    json.dump(s, open(stem + ".summary.json", "w"))
    open(stem + ".failed.json", "w").write("{}")
    with pytest.raises(SystemExit, match="failed cell"):
        R.one(U, key, stem, ck, wf)


def test_a_cell_outside_the_grid_fails(tree):
    U, wf, cl, ck = tree
    d = R.roots(U)[("exact", "heavy")][0]
    open(f"{d}/cc40s99__g0heavy__cd_exactS_ib0_live_k100_s0.summary.json", "w").write("{}")
    with pytest.raises(SystemExit, match="outside the grid"):
        R.load_all(U, wf)
