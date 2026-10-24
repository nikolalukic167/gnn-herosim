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
