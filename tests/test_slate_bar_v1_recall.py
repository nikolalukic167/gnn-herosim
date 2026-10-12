"""slate_bar_v1 recall audit: the slate rules and the reader's arithmetic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))

import slate_bar_v1_recall as R  # noqa: E402


def _ranked(k):
    # couple i on node "n{i // 2}": two platforms per node, standalone order = index order
    return [(i, 100 + i, f"n{i // 2}") for i in range(k)]


def test_top5_top6_are_standalone_prefixes():
    sl = R.slates_for(_ranked(8), {(i, 100 + i): 0.0 for i in range(8)}, [])
    assert sl["top5"] == {(i, 100 + i) for i in range(5)}
    assert sl["top6"] == {(i, 100 + i) for i in range(6)}
    assert sl["top5cd"] == sl["top5"]


def test_union_takes_top3_then_cheapest_exchange_from_the_rest():
    ranked = _ranked(8)
    exch = {(i, 100 + i): 10.0 for i in range(8)}
    exch[(7, 107)] = 0.0
    exch[(6, 106)] = 1.0
    sl = R.slates_for(ranked, exch, [])
    assert sl["union"] == {(0, 100), (1, 101), (2, 102), (7, 107), (6, 106)}


def test_union_exchange_tie_goes_to_standalone_order():
    sl = R.slates_for(_ranked(8), {(i, 100 + i): 0.0 for i in range(8)}, [])
    assert sl["union"] == sl["top5"]


def test_partner_adds_anchor_node_couples_then_fills():
    ranked = _ranked(10)
    sl = R.slates_for(ranked, {(i, 100 + i): 0.0 for i in range(10)}, ["n4", "n0", "n3"])
    # top3 = 0,1,2; n4 -> couple 8; n0 already kept (0,1) -> nothing; n3 -> couple 6
    assert sl["partner"] == {(0, 100), (1, 101), (2, 102), (8, 108), (6, 106)}


def test_partner_fills_in_standalone_order_without_anchors():
    sl = R.slates_for(_ranked(10), {(i, 100 + i): 0.0 for i in range(10)}, ["nX"])
    assert sl["partner"] == sl["top5"]


def test_small_candidate_sets_are_whole():
    sl = R.slates_for(_ranked(3), {(i, 100 + i): 0.0 for i in range(3)}, ["n1"])
    for rule in R.RULES:
        assert sl[rule] == {(i, 100 + i) for i in range(3)}


def _row(src, rung, n, cont, tasks, fb=0, rep=False, ident="x", plan=None):
    return {"src": src, "rung": rung, "n": n, "cont": cont, "repeat": rep, "id": ident, "plan": plan or [[0, 0]],
            "exact_fallbacks": fb, "decisions": 1, "hook": {"tasks": tasks, "decl_mismatch": 0}}


def _task(nv, ins, sizes=None):
    return {"n_valid": nv, "rank": 1, "pick": [1, 2], "in": ins, "size": sizes or {r: min(5, nv) if r != "top6" else min(6, nv) for r in R.RULES}}


def test_summarise_recall_and_ratio_and_population_weights():
    allin = {r: True for r in R.RULES}
    miss5 = {"top5": False, "top5cd": False, "union": True, "top6": True, "partner": False}
    rows = [
        _row("accel", "heavy", 2, "cd", [_task(8, allin), _task(8, miss5)]),
        _row("accel", "heavy", 2, "cd", [_task(3, allin), _task(3, allin)]),
        _row("accel", "heavy", 1, "cd", [_task(8, allin)]),
    ]
    cells, pooled = R.summarise(rows, {"accel|heavy|2": 300, "accel|heavy|1": 100})
    c = cells["accel|heavy|2|cd"]
    assert c["task_top5"] == 0.75 and c["batch_top5"] == 0.5 and c["batch_union"] == 1.0
    assert c["task_gt5_top5"] == 0.5
    # top6 on the 8/8 batch is 36/25, on the 3/3 batch 1
    assert abs(c["ratio_top6_gmean"] - (36 / 25) ** 0.5) < 1e-12
    p = pooled["accel|heavy|cd"]
    assert abs(p["batch_top5"] - (300 * 0.5 + 100 * 1.0) / 400) < 1e-12


def test_checks_count_repeat_pairs_and_errors():
    t = [_task(3, {r: True for r in R.RULES})]
    rows = [_row("g8", "heavy", 8, "cd", t, ident="a", plan=[[1, 2]]), _row("g8", "heavy", 8, "cd", t, rep=True, ident="a", plan=[[1, 2]]),
            {"src": "g8", "rung": "heavy", "n": 8, "cont": "cd", "id": "b", "error": "boom"}]
    c = R.checks(rows)
    assert c["repeat_pairs"] == 1 and c["repeat_identical"] == 1 and c["errors"] == 1
    assert c["pick_not_final"] == 0


def test_recommend_rule():
    def pooled(top5, union, partner, top6):
        return {f"g8|{r}|cd_exacts": {"batch_top5": top5, "batch_top5cd": top5, "batch_union": union, "batch_partner": partner,
                                      "batch_top6": top6}
                for r in ("moderate", "heavy")}
    assert R.recommend(pooled(0.6, 0.61, 0.6, 0.62))["pick"] == "top5"
    assert R.recommend(pooled(0.6, 0.65, 0.63, 0.66))["pick"] == "union"
    assert R.recommend(pooled(0.6, 0.65, 0.63, 0.71))["pick"] == "top6"
