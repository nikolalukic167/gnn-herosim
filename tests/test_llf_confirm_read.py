"""live_link_features_v1 confirmation reader (scripts_cosim/llf_confirm_read.py) on a synthetic grid: the statistic, the runaway readings, the label order, the rulings gate."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
import llf_confirm_read as R  # noqa: E402

TOPOS = R.IDS
RUL = {"primary_metric": "whole", "runaway_metric": "whole", "arm_median_pool": "all_seeds_pooled", "runaway_unit": "any_seed_flags_cell", "no_window_left": "drop_topology",
       "weaker_order": ["MP-ATTRIBUTED", "MP-DIRECTION", "TIE", "TWIN-FASTER"], "not_slower_than_mlp_pct": 0.0, "seed_clause_reading": "same_reading_as_label", "mp_direction_min_faster": 8}


def base(t, w, rung):
    return (4.0 + 0.13 * (t % 7) + 0.2 * (w == "g1")) * (1.5 if rung == "heavy" else 1.0)


def grid(tmp, f, pin="0ec7cdcb1234", pin_of=None):
    """f(arm, seed, t, w, rung) -> the task mean; writes one summary per expected cell into two subdirectories."""
    n = 0
    for arm, kind in R.ARMS.items():
        for s in ((0,) if arm == "bar" else R.SEEDS):
            for t in TOPOS:
                for w in R.WINDOWS:
                    for r in R.RUNGS:
                        v = f(arm, s, t, w, r)
                        d = tmp / ("main" if n % 2 else "r1")
                        d.mkdir(exist_ok=True)
                        (d / f"cc40s{t}__{w}{r}__{kind}_s{s}.summary.json").write_text(json.dumps(
                            {"averageElapsedTime": v, "steady_state": {"after_360s": {"mean": v * 0.9}}, "code": {"commit": pin_of(arm) if pin_of else pin, "dirty": False}, "servers": 24}))
                        n += 1
    return n


def make(factor_twin=1.0, factor_mlp=1.0, jitter=0.0):
    ratio = {"gnn": 1.0, "twin": factor_twin, "mlp": factor_mlp, "wide": 1.1, "fs_gnn": 1.05, "fs_twin": 1.05, "fs_mlp": 1.05, "fs_wide": 1.1, "bar": 1.2}

    def f(arm, s, t, w, r):
        return base(t, w, r) * ratio[arm] * (1 + jitter * ((t * 7 + s) % 5 - 2) / 100.0) * (1 + 0.003 * s if arm != "bar" else 1.0)
    return f


def read(tmp, f, rul=None, runaway=None):
    ff = f if runaway is None else (lambda a, s, t, w, r: f(a, s, t, w, r) * (50.0 if (a, s, t, w, r) in runaway else 1.0))
    assert grid(tmp, ff) == 12 * 2 * 2 * (9 * 3 - 2)  # 12 topologies x 2 windows x 2 rungs x (8 learned arms x 3 seeds + bar)
    return R.run(str(tmp), rul or RUL, TOPOS, "0ec7cdcb", None, "x")


def test_the_sealed_id_list_hashes_to_the_filed_sha():
    import hashlib
    assert hashlib.sha256(",".join(map(str, R.IDS)).encode()).hexdigest() == R.IDS_SHA256


def test_mp_attributed_when_gnn_is_clearly_faster_than_twin_and_not_slower_than_mlp(tmp_path):
    # gnn 5 % faster than the twin; the MLP slower than the gnn (factor 1.0 = same speed counts as not slower at threshold 0 only if equal, so 1.02)
    res = read(tmp_path, make(factor_twin=1.05, factor_mlp=1.02, jitter=0.5))
    assert res["label"] == "MP-ATTRIBUTED" and res["label_by_reading"] == {"all": "MP-ATTRIBUTED", "no_runaway": "MP-ATTRIBUTED"}
    assert all(res["evidence"]["all"]["rows"][r]["median_pct"] <= -2 for r in R.RUNGS)


def test_twin_faster_beats_every_other_label(tmp_path):
    res = read(tmp_path, make(factor_twin=0.95, factor_mlp=1.02, jitter=0.5))
    assert res["label"] == "TWIN-FASTER"


def test_mp_direction_when_gnn_is_slightly_faster_everywhere(tmp_path):
    res = read(tmp_path, make(factor_twin=1.01, factor_mlp=1.02, jitter=0.2))
    assert res["label"] == "MP-DIRECTION"
    assert all(res["evidence"]["all"]["rows"][r]["median_pct"] < 0 for r in R.RUNGS)


def test_tie_when_the_arms_are_the_same(tmp_path):
    res = read(tmp_path, make(factor_twin=1.0, factor_mlp=1.02, jitter=0.0))
    assert res["label"] == "TIE"  # all differences are exactly -0 after the seed means are equal: no Wilcoxon p, no negative median


def test_seed_clause_blocks_attribution(tmp_path):
    # gnn 5 % faster than the twin on seeds 1 and 2 only; seed 3 equal: the pooled median is below -2 % but a seed alone is not <= -1 %
    f0 = make(factor_twin=1.0, factor_mlp=1.02, jitter=0.5)

    def f(arm, s, t, w, r):
        v = f0(arm, s, t, w, r)
        return v * 1.05 if (arm == "twin" and s in (1, 2)) else v
    res = read(tmp_path, f)
    assert res["label"] != "MP-ATTRIBUTED"
    assert res["evidence"]["all"]["per_seed_median_pct"][3]["moderate"] > -1


def test_a_gnn_slower_than_mlp_blocks_attribution_at_threshold_zero(tmp_path):
    res = read(tmp_path, make(factor_twin=1.05, factor_mlp=0.9, jitter=0.5))
    assert res["label"] == "MP-DIRECTION"


def test_a_runaway_pair_is_dropped_and_changes_the_no_runaway_reading_only(tmp_path):
    f = make(factor_twin=1.05, factor_mlp=1.02, jitter=0.5)
    # the twin runs away on topology 30285 g0 heavy in every seed: the pair is dropped, the topology keeps its g1 window
    run = {("twin", s, 30285, "g0", "heavy") for s in R.SEEDS}
    res = read(tmp_path, f, runaway=run)
    flagged = {(r["arm"], r["topology"], r["window"], r["rung"]) for r in res["runaways"] if r["stat"] == "whole"}
    assert flagged == {("twin", 30285, "g0", "heavy")}
    ev = res["evidence"]
    assert ev["no_runaway"]["rows"]["heavy"]["n"] == 12 and ev["no_runaway"]["rows"]["moderate"] == {k: v for k, v in ev["all"]["rows"]["moderate"].items()}
    # with the runaway kept the gnn looks far faster than the twin on that topology; dropped, it does not
    assert ev["all"]["rows"]["heavy"]["median_pct"] < ev["no_runaway"]["rows"]["heavy"]["median_pct"] + 1e-9


def test_a_topology_with_no_window_left_is_dropped(tmp_path):
    f = make(factor_twin=1.05, factor_mlp=1.02, jitter=0.5)
    run = {("twin", s, 30285, w, "heavy") for s in R.SEEDS for w in R.WINDOWS}
    res = read(tmp_path, f, runaway=run)
    assert res["evidence"]["no_runaway"]["rows"]["heavy"]["n"] == 11 and res["evidence"]["no_runaway"]["rows"]["moderate"]["n"] == 12


def test_the_final_label_is_the_weaker_reading(tmp_path):
    order = RUL["weaker_order"]
    assert max(["MP-ATTRIBUTED", "TIE"], key=order.index) == "TIE" and max(["MP-DIRECTION", "TWIN-FASTER"], key=order.index) == "TWIN-FASTER"


def test_paired_percent_and_the_seed_mean_cell(tmp_path):
    f = make()
    grid(tmp_path, f)
    V, _, _ = R.load(str(tmp_path), TOPOS, "0ec7cdcb")
    g = R.Grid(V, TOPOS, RUL)
    x, flagged = g.cell("gnn", 30285, "g0", "moderate", "whole", "all")
    assert not flagged and x == pytest.approx(sum(f("gnn", s, 30285, "g0", "moderate") for s in R.SEEDS) / 3)
    # gnn is 1/1.2 of the bar in every cell up to the seed factor (1.003 s): the paired % is the same everywhere, so the median is that value
    c = g.contrast("gnn", "bar", "moderate", "whole", "all")
    want = 100 * ((1.0 + 0.003 * 2) / 1.2 - 1)  # the seed mean of 1 + 0.003 s over s = 1, 2, 3
    assert c["n"] == 12 and c["median_pct"] == pytest.approx(want, abs=1e-6) and c["faster"] == 12
    assert all(v == pytest.approx(want, abs=1e-6) for v in c["per_topology"].values())


def test_the_after_360_statistic_is_read_from_steady_state(tmp_path):
    grid(tmp_path, make())
    V, _, _ = R.load(str(tmp_path), TOPOS, "0ec7cdcb")
    k = (R.ARMS["gnn"], 1, 30285, "g0", "moderate")
    assert V[k]["a360"] == pytest.approx(V[k]["whole"] * 0.9)


def test_load_refuses_a_missing_duplicated_unexpected_or_wrong_pin_cell(tmp_path):
    grid(tmp_path, make())
    victim = next((tmp_path / "main").glob("*.summary.json"))
    text = victim.read_text()
    victim.unlink()
    with pytest.raises(SystemExit, match="missing"):
        R.load(str(tmp_path), TOPOS, "0ec7cdcb")
    (tmp_path / "r1" / victim.name).write_text(text)  # the same cell in the other subdirectory is a duplicate once the original is back
    (tmp_path / "main" / victim.name).write_text(text)
    with pytest.raises(SystemExit, match="twice"):
        R.load(str(tmp_path), TOPOS, "0ec7cdcb")
    (tmp_path / "r1" / victim.name).unlink()
    with pytest.raises(SystemExit, match="not ffffffff"):
        R.load(str(tmp_path), TOPOS, "ffffffff")
    (tmp_path / "main" / "cc40s39999__g0moderate__cd_ib0_inflightfull_k100_s0.summary.json").write_text(text)
    with pytest.raises(SystemExit, match="unexpected"):
        R.load(str(tmp_path), TOPOS, "0ec7cdcb")


def test_rulings_gate_refuses_a_missing_or_invalid_point():
    for k in R.RULING_POINTS:
        r = dict(RUL)
        del r[k]
        with pytest.raises(SystemExit, match="rulings lacks"):
            R.check_rulings(r)
    with pytest.raises(SystemExit, match="not one of"):
        R.check_rulings({**RUL, "primary_metric": "median"})
    with pytest.raises(SystemExit, match="permutation"):
        R.check_rulings({**RUL, "weaker_order": ["TIE"]})


def test_output_is_written_once(tmp_path):
    (tmp_path / "cells").mkdir()
    grid(tmp_path / "cells", make(1.05, 1.02, 0.5))
    out = tmp_path / "out" / "read.json"
    res = R.run(str(tmp_path / "cells"), RUL, TOPOS, "0ec7cdcb", str(out), "x")
    assert json.load(open(out))["label"] == res["label"]
    with pytest.raises(SystemExit, match="read once"):
        R.run(str(tmp_path / "cells"), RUL, TOPOS, "0ec7cdcb", str(out), "x")


STATIC_PIN, REF_PIN = "02668ecf0000", "0ec7cdcb1111"
two = lambda arm: STATIC_PIN if arm in ("gnn", "twin", "mlp", "wide") else REF_PIN


def test_group_pins_accept_each_group_at_its_own_commit_and_record_them(tmp_path):
    grid(tmp_path, make(), pin_of=two)
    V, pins, by_group = R.load(str(tmp_path), TOPOS, "static=02668ecf,refs=0ec7cdcb,bar=0ec7cdcb")
    assert len(V) == 12 * 2 * 2 * 25 and pins == sorted([REF_PIN, STATIC_PIN]) and by_group == {"static": [STATIC_PIN], "refs": [REF_PIN], "bar": [REF_PIN]}
    with pytest.raises(SystemExit, match="not 0ec7cdcb"):  # a single pin is the old behaviour: the static cells are at the other commit
        R.load(str(tmp_path), TOPOS, "0ec7cdcb")


def test_group_pins_refuse_a_static_cell_at_the_refs_commit_an_unused_pin_or_a_bad_group_set(tmp_path):
    grid(tmp_path, make(), pin_of=two)
    with pytest.raises(SystemExit, match="not 02668ecf"):
        R.load(str(tmp_path), TOPOS, "static=02668ecf,refs=02668ecf,bar=02668ecf")
    with pytest.raises(SystemExit, match="exactly static, refs and bar"):
        R.load(str(tmp_path), TOPOS, "static=02668ecf,refs=0ec7cdcb")
    with pytest.raises(SystemExit, match="not 02668ecf"):
        R.load(str(tmp_path), TOPOS, "static=02668ecf,refs=0ec7cdcb,bar=02668ecf")


def test_group_pins_refuse_a_listed_pin_that_no_cell_ran_at(tmp_path):
    grid(tmp_path, make(), pin="0ec7cdcb1234")
    with pytest.raises(SystemExit, match="not 02668ecf"):
        R.load(str(tmp_path), TOPOS, "static=02668ecf,refs=0ec7cdcb,bar=0ec7cdcb")
    # all cells at one commit but two listed prefixes of that same commit: both are used, the read goes through
    V, pins, _ = R.load(str(tmp_path), TOPOS, "static=0ec7,refs=0ec7cdcb,bar=0ec7cdcb1234")
    assert pins == ["0ec7cdcb1234"]


def test_run_records_pins_by_group(tmp_path):
    (tmp_path / "cells").mkdir()
    grid(tmp_path / "cells", make(1.05, 1.02, 0.5), pin_of=two)
    res = R.run(str(tmp_path / "cells"), RUL, TOPOS, "static=02668ecf,refs=0ec7cdcb,bar=0ec7cdcb", None, "x")
    assert res["pins_by_group"]["static"] == [STATIC_PIN] and res["pins_by_group"]["refs"] == [REF_PIN] and res["label"] == "MP-ATTRIBUTED"


CD_TOPOS = tuple(range(40001, 40013))


def cd_env(arm, seed, rung):
    if arm in R.CD_CKPT:
        return {"GNN_MODEL_PATH": f"/x/models/{R.CD_CKPT[arm]}{seed}.pt", "HEROSIM_DECISION_COST_MS_BY_BUCKET": json.dumps(R.CD_CHARGE[(arm, rung)])}
    return {"HEROSIM_DECISION_COST_MS": str(R.CD_BAR_CHARGE[rung])} if arm == "bar" else {}


def cd_grid(tmp, f, pin_of=lambda arm: "abcd1234", env_of=cd_env, layout=None):
    n = 0
    for arm, kind in R.CD_ARMS.items():
        for s in ((0,) if arm in R.S0_ARMS else R.CD_SEEDS):
            for t in CD_TOPOS:
                for w in R.WINDOWS:
                    for r in R.RUNGS:
                        d = tmp / ("a" if n % 2 else "b") if layout is None else layout(tmp, arm, s)
                        d.mkdir(parents=True, exist_ok=True)
                        v = f(arm, s, t, w, r)
                        (d / f"cc40s{t}__{w}{r}__{kind}_s{s}.summary.json").write_text(json.dumps(
                            {"averageElapsedTime": v, "steady_state": {"after_360s": {"mean": v * 0.9}}, "code": {"commit": pin_of(arm), "dirty": False}, "servers": 24,
                             "env": env_of(arm, s, r)}))
                        n += 1
    return n


def cd_f(arm, s, t, w, r):
    ratio = {"gnn": 1.0, "wide": 1.05, "mlp": 1.05, "bar": 1.2, "reactive": 1.5, "random": 2.0}[arm]
    return base(t, w, r) * ratio * (1 + 0.01 * ((t * 7 + s) % 5 - 2))


def test_cd_mode_primary_is_four_tests_with_holm_over_four(tmp_path):
    assert cd_grid(tmp_path, cd_f) == 12 * 2 * 2 * (3 * 6 + 3)
    res = R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x")
    keys = {f"gnn_vs_{b}|{r}" for b in ("bar", "wide") for r in R.RUNGS}
    assert set(res["primary"]["all"]) == keys == set(res["labels"])
    row = res["primary"]["no_runaway"]["gnn_vs_bar|moderate"]
    assert row["median_pct"] < -10 and row["faster"] == 12
    assert row["holm_p"] == pytest.approx(min(1.0, 4 * row["p"]), rel=1e-9)
    assert res["labels"]["gnn_vs_bar|moderate"] == "CONFIRMED"
    assert any(k.startswith("all|whole|gnn_vs_reactive|") for k in res["secondary"]) and any(k.startswith("no_runaway|a360|bar_vs_random|") for k in res["secondary"])
    assert set(res["per_seed_median_pct"]["gnn_vs_wide|heavy|all"]) == set(R.CD_SEEDS) and res["seeds"] == [1, 2, 3, 4, 5, 6]
    assert set(res["per_seed_block"]["gnn_vs_bar|moderate|no_runaway"]) == {"s123", "s456"}


def test_cd_labels():
    ok = {"median_pct": -5.0, "holm_p": 0.01}
    assert R.cd_label(ok, {"median_pct": -1.0}) == "CONFIRMED"
    assert R.cd_label(ok, {"median_pct": 0.5}) == "DIRECTION"          # the all-cells guard
    assert R.cd_label({"median_pct": -5.0, "holm_p": 0.2}, {"median_pct": -1.0}) == "DIRECTION"
    assert R.cd_label({"median_pct": 0.0, "holm_p": 0.001}, {"median_pct": -1.0}) == "NOT-FASTER"
    assert R.cd_label({"median_pct": None, "holm_p": None}, {"median_pct": None}) == "NOT-FASTER"


def test_cd_mode_all_cells_guard_blocks_confirmed(tmp_path):
    # gnn runs away on 2 cells of topology 40001: the pair drops without runaways, but the all-cells heavy median vs bar turns positive via many runaways
    bad = {(t, w) for t in CD_TOPOS[:8] for w in R.WINDOWS}
    cd_grid(tmp_path, lambda a, s, t, w, r: cd_f(a, s, t, w, r) * (50.0 if a == "gnn" and r == "heavy" and (t, w) in bad else 1.0))
    res = R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x")
    assert res["primary"]["all"]["gnn_vs_bar|heavy"]["median_pct"] > 0
    assert res["labels"]["gnn_vs_bar|heavy"] != "CONFIRMED"


def test_cd_mode_pin_groups_and_missing_cell_fail_loud(tmp_path):
    cd_grid(tmp_path, cd_f, pin_of=lambda arm: "bbbb0000" if arm in R.S0_ARMS else "aaaa1111")
    with pytest.raises(SystemExit, match="exactly static and bar"):
        R.load(str(tmp_path), CD_TOPOS, "static=aaaa1111,refs=x,bar=bbbb0000", R.CD_ARMS, seeds=R.CD_SEEDS)
    V, pins, by = R.load(str(tmp_path), CD_TOPOS, "static=aaaa1111,bar=bbbb0000", R.CD_ARMS, seeds=R.CD_SEEDS)
    assert by == {"static": ["aaaa1111"], "bar": ["bbbb0000"]}
    next(tmp_path.glob("*/cc40s40001__g0moderate__random_s0.summary.json")).unlink()
    with pytest.raises(SystemExit, match="missing"):
        R.load(str(tmp_path), CD_TOPOS, "static=aaaa1111,bar=bbbb0000", R.CD_ARMS, seeds=R.CD_SEEDS)


@pytest.mark.parametrize("bad", ["ckpt", "seed", "charge", "bar", "bar_rung"])
def test_cd_mode_checkpoint_and_charge_checks_fail_loud(tmp_path, bad):
    def env_of(arm, s, r):
        e = cd_env(arm, s, r)
        if bad == "ckpt" and arm == "wide" and s == 2:
            e["GNN_MODEL_PATH"] = "/x/models/uncut-linkstatic-gnn-eng-e40-seed2.pt"
        if bad == "seed" and arm == "gnn" and s == 3:
            e["GNN_MODEL_PATH"] = "/x/models/" + R.CD_CKPT["gnn"] + "1.pt"
        if bad == "charge" and arm == "mlp" and r == "heavy":
            e["HEROSIM_DECISION_COST_MS_BY_BUCKET"] = json.dumps(R.CD_CHARGE[("mlp", "moderate")])
        if bad == "bar" and arm == "bar":
            e["HEROSIM_DECISION_COST_MS"] = "0.5"
        if bad == "bar_rung" and arm == "bar" and r == "moderate":
            e["HEROSIM_DECISION_COST_MS"] = str(R.CD_BAR_CHARGE["heavy"])
        return e
    cd_grid(tmp_path, cd_f, env_of=env_of)
    with pytest.raises(SystemExit, match="FAIL LOUD"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x")


def test_cd_mode_runaway_pairs_drop_without_runaways(tmp_path):
    cd_grid(tmp_path, lambda a, s, t, w, r: cd_f(a, s, t, w, r) * (50.0 if (a, t, w, r) == ("gnn", 40001, "g0", "heavy") else 1.0))
    res = R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x")
    assert any(x["arm"] == "gnn" and x["topology"] == 40001 for x in res["runaways"])
    assert res["primary"]["all"]["gnn_vs_bar|heavy"]["n"] == 12


def cd_layout(tmp, arm, s):
    """The fresh-24 layout: s0 arms but reactive and seeds 1-3 in d_cd/<sub>, seeds 4-6 in d_cd_s456/<sub>, reactive flat in cells_react."""
    if arm == "reactive":
        return tmp / "cells_react"
    return tmp / ("d_cd_s456" if s in (4, 5, 6) else "d_cd") / f"sub_{arm}_{s}"


def test_cd_mode_reads_two_roots_and_the_flat_reactive_dir(tmp_path):
    cd_grid(tmp_path, cd_f, layout=cd_layout)
    (tmp_path / "cells_react" / "cc40s40001__g2heavy__reactive_s0.summary.json").write_text("{}")   # g2/g3 admission cells are ignored
    (tmp_path / "cells_react" / "cc40s40001__g3heavy__reactive_s0.failed.json").write_text("{}")
    roots = [str(tmp_path / "d_cd"), str(tmp_path / "d_cd_s456")]
    flat = (str(tmp_path / "cells_react"), R.CD_ARMS["reactive"])
    res = R.run_cd(roots, RUL, CD_TOPOS, "abcd1234", None, "x", flat=flat)
    assert res["cells"] == 12 * 2 * 2 * (3 * 6 + 3)
    with pytest.raises(SystemExit, match="missing"):
        R.run_cd(roots[:1], RUL, CD_TOPOS, "abcd1234", None, "x", flat=flat)
    with pytest.raises(SystemExit, match="missing"):
        R.run_cd(roots, RUL, CD_TOPOS, "abcd1234", None, "x")


def test_cd_mode_seed_blocks_read_each_block_alone(tmp_path):
    # seeds 4-6 of the gnn are 20 % slower: pooled and s123 favour the gnn vs wide, s456 does not
    cd_grid(tmp_path, lambda a, s, t, w, r: cd_f(a, s, t, w, r) * (1.2 if a == "gnn" and s >= 4 else 1.0))
    res = R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x")
    blk = res["per_seed_block"]["gnn_vs_wide|moderate|all"]
    assert blk["s123"]["median_pct"] < 0 < blk["s456"]["median_pct"]
    assert res["primary"]["all"]["gnn_vs_wide|moderate"]["median_pct"] == pytest.approx(R.Grid(R.load(str(tmp_path), CD_TOPOS, "abcd1234", R.CD_ARMS, seeds=R.CD_SEEDS)[0], CD_TOPOS, RUL, R.CD_ARMS, seeds=R.CD_SEEDS).contrast("gnn", "wide", "moderate", "whole", "all")["median_pct"])


def test_cd_mode_allow_missing_is_random_only_and_named(tmp_path):
    cd_grid(tmp_path, cd_f)
    gone = ("random", 0, 40001, "g0", "moderate")
    next(tmp_path.glob("*/cc40s40001__g0moderate__random_s0.summary.json")).unlink()
    with pytest.raises(SystemExit, match="1 of .* cells missing"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x")
    res = R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", allow_missing=[gone])
    assert res["allowed_missing"] == ["cc40s40001__g0moderate__random_s0"]
    assert res["secondary"]["all|whole|gnn_vs_random|moderate"]["n"] == 12              # 40001 keeps g1
    assert res["secondary"]["all|whole|gnn_vs_random|moderate"]["per_topology"][40001] is not None
    with pytest.raises(SystemExit, match="only cells of the random arm"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", allow_missing=[gone, ("cd_ib0_inflightfull_k100", 0, 40001, "g0", "moderate")])
    with pytest.raises(SystemExit, match="which exist"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", allow_missing=[gone, ("random", 0, 40002, "g0", "moderate")])
    next(tmp_path.glob("*/cc40s40002__g1heavy__reactive_s0.summary.json")).unlink()
    with pytest.raises(SystemExit, match="missing"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", allow_missing=[gone])


XRATIO = {"gnn": 1.0, "twin": 1.04, "wide": 1.05, "mlp": 1.05, "bar": 1.2, "reactive": 1.5, "random": 2.0, "heft": 1.3}


def xf(arm, s, t, w, r):
    return base(t, w, "moderate" if r not in R.RUNGS else r) * XRATIO[arm] * (1 + 0.01 * ((t * 7 + s) % 5 - 2))


def xenv(arm, seed, crung):
    if arm in R.CD_CKPT:
        return {"GNN_MODEL_PATH": f"/x/models/{R.CD_CKPT[arm]}{seed}.pt", "HEROSIM_DECISION_COST_MS_BY_BUCKET": json.dumps(R.CD_CHARGE[(arm, crung)])}
    if arm in ("bar", "heft"):
        return {"HEROSIM_DECISION_COST_MS": str((R.CD_BAR_CHARGE if arm == "bar" else R.CD_HEFT_CHARGE)[crung])}
    return {}


def xgrid(tmp, arms, seeds, rungs, f=xf, servers=24, clients=None, charge_rung=None, env_of=xenv, fields=None, skip=()):
    """Cells of the extended cd mode; fields(arm, s, t, w, r) -> extra summary keys."""
    n = 0
    for arm in arms:
        kind = R.CD_ARM_REGISTRY[arm]
        for s in ((0,) if arm in R.S0_ARMS else seeds):
            for t in CD_TOPOS:
                for w in R.WINDOWS:
                    for r in rungs:
                        if (arm, s, t, w, r) in skip:
                            continue
                        d = tmp / f"sub{n % 3}"
                        d.mkdir(parents=True, exist_ok=True)
                        v = f(arm, s, t, w, r)
                        body = {"averageElapsedTime": v, "steady_state": {"after_360s": {"mean": v * 0.9}}, "code": {"commit": "abcd1234", "dirty": False},
                                "servers": servers, "env": env_of(arm, s, (charge_rung or {}).get(r, r))}
                        if clients is not None:
                            body["clients"] = clients
                        body.update(fields(arm, s, t, w, r) if fields else {})
                        (d / f"cc40s{t}__{w}{r}__{kind}_s{s}.summary.json").write_text(json.dumps(body))
                        n += 1
    return n


def stable_fields(bad=()):
    def fields(arm, s, t, w, r):
        if arm != "reactive":
            return {}
        drain = 0.5 if (t, w) in bad else 0.05
        return {"endTime": 1000.0 * (1 + drain), "arrival_end": {"last_arrival_s": 1000.0}, "queue_share": 0.001,
                "peerExchangeFairshare": {"flow_seconds": 200.0, "extra_seconds": 100.0}}
    return fields


def test_reactive_stability_rule():
    ok = {"endTime": 1100.0, "last_arrival_s": 1000.0, "flow_seconds": 300.0, "extra_seconds": 200.0, "queue_share": 0.8}
    assert R.reactive_stable(ok)                                                  # drain .10, slowdown 3.0, queue .80: all on the bound
    assert not R.reactive_stable(ok | {"endTime": 1100.5})
    assert not R.reactive_stable(ok | {"extra_seconds": 200.5})
    assert not R.reactive_stable(ok | {"queue_share": 0.81})
    assert not R.reactive_stable(ok | {"flow_seconds": None}) and not R.reactive_stable({})
    assert not R.reactive_stable(ok | {"extra_seconds": 300.0})                   # no solo time left: unknown


KS_ARMS = ["gnn", "wide", "mlp", "bar", "reactive", "random", "heft"]


def test_knative_rung_admission_and_holm_over_three(tmp_path):
    bad = {(40001, "g0"), (40001, "g1"), (40002, "g0")}
    xgrid(tmp_path, KS_ARMS, R.CD_SEEDS, ("f0.125",), charge_rung={"f0.125": "moderate"}, fields=stable_fields(bad))
    pairs = [("gnn", "reactive"), ("gnn", "bar"), ("gnn", "wide")]
    res = R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", arms=KS_ARMS, primary_pairs=pairs, rungs=("f0.125",),
                   charge_rung={"f0.125": "moderate"}, admit_reactive=True)
    assert set(res["labels"]) == {"gnn_vs_reactive|f0.125", "gnn_vs_bar|f0.125", "gnn_vs_wide|f0.125"}
    assert res["admission"]["excluded"] == ["40001:g0:f0.125", "40001:g1:f0.125", "40002:g0:f0.125"]
    assert res["admission"]["topologies_with_no_admitted_window"] == {"f0.125": [40001]}
    row = res["primary"]["no_runaway"]["gnn_vs_bar|f0.125"]
    assert row["n"] == 11 and row["holm_p"] == pytest.approx(min(1.0, 3 * row["p"]), rel=1e-9)
    assert res["options"]["admit_reactive"] is True and res["cells"] == 12 * 2 * (3 * 6 + 4)
    # a cell carrying the f-rung's own (unregistered) charge instead of the moderate copy fails loud
    for f in tmp_path.glob("*/*__ra_gnn_eng_*f0.125*.summary.json"):
        f.unlink()
    xgrid(tmp_path, ["gnn"], R.CD_SEEDS, ("f0.125",), env_of=lambda a, s, r: xenv(a, s, "heavy"))
    with pytest.raises(SystemExit, match="decision charge"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", arms=KS_ARMS, primary_pairs=pairs, rungs=("f0.125",),
                 charge_rung={"f0.125": "moderate"}, admit_reactive=True)


def test_scale_size_checks_failed_learned_block_and_underpowered(tmp_path):
    arms = ["gnn", "wide", "mlp", "bar", "heft"]
    gone = ("gnn", 2, 40003, "g1", "heavy")
    xgrid(tmp_path, arms, R.SEEDS, R.RUNGS, servers=64, clients=427, skip={gone})
    kind_gone = (R.CD_ARM_REGISTRY["gnn"], 2, 40003, "g1", "heavy")
    kw = dict(arms=arms, servers=64, clients=427, seeds=R.SEEDS, min_n=8)
    with pytest.raises(SystemExit, match="missing"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", **kw)
    res = R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", failed_learned=[kind_gone], **kw)
    assert res["gnn_blocked_by_failure"] == {"moderate": False, "heavy": True}
    assert res["labels"]["gnn_vs_bar|moderate"] == "CONFIRMED"
    assert res["labels"]["gnn_vs_bar|heavy"] == "DIRECTION" and res["primary"]["no_runaway"]["gnn_vs_bar|heavy"]["holm_p"] < 0.05   # blocked
    with pytest.raises(SystemExit, match="clients=427, not 160"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", failed_learned=[kind_gone], **(kw | {"clients": 160}))
    with pytest.raises(SystemExit, match="not abcd1234 clean at 24 servers"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", failed_learned=[kind_gone], **(kw | {"servers": 24}))
    with pytest.raises(SystemExit, match="only learned-arm cells"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", failed_learned=[("cd_ib0_inflightfull_k100", 0, 40001, "g0", "heavy")], **kw)


def test_cd_label_precedence():
    nr = lambda m, h, n: {"median_pct": m, "holm_p": h, "n": n}
    al = {"median_pct": -1.0}
    assert R.cd_label(nr(-5.0, 0.04, 7), al, min_n=8) == "CONFIRMED"              # CONFIRMED stands at any n
    assert R.cd_label(nr(-5.0, 0.2, 7), al, min_n=8) == "UNDERPOWERED"
    assert R.cd_label(nr(3.0, 0.2, 7), al, min_n=8) == "UNDERPOWERED"
    assert R.cd_label(nr(None, None, 0), {"median_pct": None}, min_n=8) == "UNDERPOWERED"
    assert R.cd_label(nr(-5.0, 0.01, 12), al, blocked=True, min_n=8) == "DIRECTION"
    assert R.cd_label(nr(-5.0, 0.01, 7), al, blocked=True, min_n=8) == "UNDERPOWERED"


def test_twin_arm_interim_tag_and_checkpoint_check(tmp_path):
    arms = ["gnn", "twin", "wide", "bar"]
    xgrid(tmp_path, arms, R.SEEDS, R.RUNGS)
    res = R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", arms=arms, primary_pairs=[("gnn", "twin")], seeds=R.SEEDS, min_n=8, tag="INTERIM")
    assert set(res["labels"]) == {"gnn_vs_twin|moderate", "gnn_vs_twin|heavy"} and all(v.startswith("INTERIM-") for v in res["labels"].values())
    row = res["primary"]["no_runaway"]["gnn_vs_twin|moderate"]
    assert row["holm_p"] >= row["p"] and res["options"]["tag"] == "INTERIM"
    f = next(tmp_path.glob("*/cc40s40001__g0moderate__ra_twin_eng_*_s2.summary.json"))
    d = json.loads(f.read_text()); d["env"]["GNN_MODEL_PATH"] = "/x/models/uncut-linkstatic-gnn-eng-e40-seed2.pt"; f.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="served"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", arms=arms, primary_pairs=[("gnn", "twin")], seeds=R.SEEDS)


def test_heft_charge_checked_per_rung(tmp_path):
    arms = ["gnn", "wide", "bar", "heft"]
    xgrid(tmp_path, arms, R.SEEDS, R.RUNGS, env_of=lambda a, s, r: xenv(a, s, "moderate" if a == "heft" else r))
    with pytest.raises(SystemExit, match="heft decision charge"):
        R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", arms=arms, seeds=R.SEEDS)


def test_charge_ratio_normalised_and_flag(tmp_path):
    arms = ["gnn", "wide", "bar"]

    def fields(arm, s, t, w, r):
        if arm not in ("gnn", "wide"):
            return {}
        x = {"gnn": 2.0, "wide": 1.41}[arm] if r == "moderate" else 1.0     # measured / charged
        wv = R.CD_CHARGE[(arm, r)]
        return {"decisionTiming": {"by_bucket": {b: {"total_s": x * wv[b] * 10 / 1000.0, "plan_evals": 10} for b in wv}}}
    xgrid(tmp_path, arms, R.SEEDS, R.RUNGS, servers=64, fields=fields)
    res = R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", arms=arms, seeds=R.SEEDS, servers=64, charge_ratio=True)
    cr = res["charge_ratio"]
    assert cr["ratios"]["gnn|moderate"]["median"] == pytest.approx(2.0) and cr["ratios"]["gnn|moderate"]["normalised"] == pytest.approx(2.0 / 1.45)
    assert cr["gnn_charge_limited"] == {"moderate": False, "heavy": False}       # 1.38 normalised: under the 1.5 bar
    res2_fields = lambda a, s, t, w, r: fields(a, s, t, w, r) if a != "gnn" or r != "moderate" else {"decisionTiming": {"by_bucket": {b: {"total_s": 2.5 * v * 10 / 1000.0, "plan_evals": 10} for b, v in R.CD_CHARGE[("gnn", "moderate")].items()}}}
    for f in tmp_path.glob("*/*.summary.json"):
        f.unlink()
    xgrid(tmp_path, arms, R.SEEDS, R.RUNGS, servers=64, fields=res2_fields)
    res2 = R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x", arms=arms, seeds=R.SEEDS, servers=64, charge_ratio=True)
    assert res2["charge_ratio"]["gnn_charge_limited"]["moderate"] is True        # 2.5 / 1.45 = 1.72 > 1.5 and > wide's 1.0


def test_default_cd_read_carries_no_extension_keys(tmp_path):
    cd_grid(tmp_path, cd_f)
    res = R.run_cd(str(tmp_path), RUL, CD_TOPOS, "abcd1234", None, "x")
    assert not {"options", "admission", "charge_ratio", "gnn_blocked_by_failure"} & set(res)


FILED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "lineages", "live_link_features_v1", "confirmation_reads", "read_primary_A_epoch40.json")


@pytest.mark.skipif(not (os.environ.get("LLF_PARENT_CELLS_A") and os.path.exists(FILED)), reason="needs the parent's 1,200 sealed cells (LLF_PARENT_CELLS_A) and the filed read")
def test_default_mode_reproduces_the_filed_primary_read(tmp_path):
    rul = json.load(open(os.path.join(os.path.dirname(FILED), "..", "confirm_rulings.json")))
    res = R.run(os.environ["LLF_PARENT_CELLS_A"], rul, R.IDS, "static=02668ecf,refs=0ec7cdcb,bar=0ec7cdcb", None, R.IDS_SHA256)
    want = json.load(open(FILED))
    got = json.loads(json.dumps(res, default=str))
    got.pop("reader_sha256"); want.pop("reader_sha256")
    assert got == want
