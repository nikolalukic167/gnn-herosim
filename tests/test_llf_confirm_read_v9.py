"""outbound_link_features_v1: the v9 arm set of the cd-mode reader (8-test primary family, the v8 secondary pair with its own Holm, per-arm checkpoint and charge checks)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
import llf_confirm_read as R  # noqa: E402

TOPOS = tuple(range(40001, 40013))
RUL = {"primary_metric": "whole", "runaway_metric": "whole", "arm_median_pool": "all_seeds_pooled", "runaway_unit": "any_seed_flags_cell", "no_window_left": "drop_topology",
       "weaker_order": ["MP-ATTRIBUTED", "MP-DIRECTION", "TIE", "TWIN-FASTER"], "not_slower_than_mlp_pct": 0.0, "seed_clause_reading": "same_reading_as_label", "mp_direction_min_faster": 8}
BUCKETS = {"le10": 5.0, "11_50": 2.0, "gt50": 2.5}
CHARGES = {f"{a}|{r}": {k: v + 0.01 * i for k, v in BUCKETS.items()} for i, a in enumerate(R.V9_CHARGED) for r in R.RUNGS}
RATIO = {"gnn": 1.0, "twin": 1.04, "wide": 1.05, "mlp": 1.05, "v8": 1.02, "bar": 1.2}
PIN = "9a9a9a9a"


def base(t, w, rung):
    return (4.0 + 0.13 * (t % 7) + 0.2 * (w == "g1")) * (1.5 if rung == "heavy" else 1.0)


def env(arm, seed, rung, charges=CHARGES):
    if arm == "bar":
        return {"HEROSIM_DECISION_COST_MS": str(R.CD_BAR_CHARGE[rung])}
    charge = R.CD_CHARGE[("gnn", rung)] if arm == "v8" else charges[f"{arm}|{rung}"]
    return {"GNN_MODEL_PATH": f"/x/models/{R.V9_CKPT[arm]}{seed}.pt", "HEROSIM_DECISION_COST_MS_BY_BUCKET": json.dumps(charge)}


def grid(tmp, env_of=env, skip=(), f=None):
    n = 0
    for arm, kind in R.V9_ARM_REGISTRY.items():
        for s in ((0,) if arm in R.S0_ARMS else R.CD_SEEDS):
            for t in TOPOS:
                for w in R.WINDOWS:
                    for r in R.RUNGS:
                        if (arm, s, t, w, r) in skip:
                            continue
                        d = tmp / f"sub{n % 3}"
                        d.mkdir(parents=True, exist_ok=True)
                        v = (f or (lambda a, s_, t_, w_, r_: base(t_, w_, r_) * RATIO[a] * (1 + 0.01 * ((t_ * 7 + s_) % 5 - 2))))(arm, s, t, w, r)
                        (d / f"cc40s{t}__{w}{r}__{kind}_s{s}.summary.json").write_text(json.dumps(
                            {"averageElapsedTime": v, "steady_state": {"after_360s": {"mean": v * 0.9}}, "code": {"commit": PIN, "dirty": False}, "servers": 24, "env": env_of(arm, s, r)}))
                        n += 1
    return n


def read(tmp, **kw):
    return R.run_cd(str(tmp), RUL, TOPOS, PIN, None, "x", arm_set_name="v9", charges=CHARGES, **kw)


def test_the_primary_family_is_eight_tests_with_holm_over_eight_and_the_secondary_has_its_own_holm(tmp_path):
    assert grid(tmp_path) == 12 * 2 * 2 * (4 * 6 + 6 + 1)
    res = read(tmp_path, min_n=8)
    prim = {f"gnn_vs_{b}|{r}" for b in ("bar", "twin", "wide", "mlp") for r in R.RUNGS}
    assert set(res["primary"]["all"]) == prim and len(prim) == 8
    rows = res["primary"]["no_runaway"]
    assert min(v["holm_p"] for v in rows.values()) == pytest.approx(min(1.0, 8 * min(v["p"] for v in rows.values())), rel=1e-9)
    assert set(res["secondary_family"]["all"]) == {"gnn_vs_v8|moderate", "gnn_vs_v8|heavy"}
    sec = res["secondary_family"]["no_runaway"]
    p_min = min(v["p"] for v in sec.values())
    assert min(v["holm_p"] for v in sec.values()) == pytest.approx(min(1.0, 2 * p_min), rel=1e-9)
    assert all(res["labels"][f"secondary|gnn_vs_v8|{r}"] == "CONFIRMED" for r in R.RUNGS)
    assert all(res["labels"][f"gnn_vs_bar|{r}"] == "CONFIRMED" for r in R.RUNGS)
    assert res["charges"]["gnn|moderate"] == CHARGES["gnn|moderate"] and res["options"]["arm_set"] == "v9"


def test_a_gnn_failure_blocks_confirmed_at_its_rung_for_both_families(tmp_path):
    gone = ("gnn", 2, 40003, "g1", "heavy")
    grid(tmp_path, skip={gone})
    failed = [(R.V9_ARM_REGISTRY["gnn"], 2, 40003, "g1", "heavy")]
    with pytest.raises(SystemExit, match="missing"):
        read(tmp_path)
    res = read(tmp_path, failed_learned=failed)
    assert res["gnn_blocked_by_failure"] == {"moderate": False, "heavy": True}
    assert res["labels"]["gnn_vs_bar|heavy"] == "DIRECTION" and res["labels"]["secondary|gnn_vs_v8|heavy"] == "DIRECTION"
    assert res["labels"]["gnn_vs_bar|moderate"] == "CONFIRMED"


@pytest.mark.parametrize("bad", ["ckpt_v9_as_v8", "ckpt_seed", "charge_v8_value", "v8_charge", "pin"])
def test_checkpoint_charge_and_pin_checks_fail_loud(tmp_path, bad):
    def env_of(arm, seed, rung):
        e = env(arm, seed, rung)
        if arm == "twin" and seed == 3 and rung == "heavy":
            if bad == "ckpt_v9_as_v8":
                e["GNN_MODEL_PATH"] = f"/x/models/uncut-linkstatic-twin-e40-seed{seed}.pt"
            if bad == "ckpt_seed":
                e["GNN_MODEL_PATH"] = f"/x/models/{R.V9_CKPT['twin']}1.pt"
            if bad == "charge_v8_value":
                e["HEROSIM_DECISION_COST_MS_BY_BUCKET"] = json.dumps(R.CD_CHARGE[("twin", "heavy")])
        if arm == "v8" and seed == 3 and rung == "heavy" and bad == "v8_charge":
            e["HEROSIM_DECISION_COST_MS_BY_BUCKET"] = json.dumps(CHARGES["gnn|heavy"])
        return e
    grid(tmp_path, env_of=env_of)
    if bad == "pin":
        with pytest.raises(SystemExit, match="not 1111aaaa clean"):
            R.run_cd(str(tmp_path), RUL, TOPOS, "1111aaaa", None, "x", arm_set_name="v9", charges=CHARGES)
        return
    with pytest.raises(SystemExit, match="FAIL LOUD"):
        read(tmp_path)


def test_the_v9_arm_set_needs_its_own_measured_charges(tmp_path):
    grid(tmp_path)
    with pytest.raises(SystemExit, match="needs --charges"):
        R.run_cd(str(tmp_path), RUL, TOPOS, PIN, None, "x", arm_set_name="v9")
    with pytest.raises(SystemExit, match="--charges lacks"):
        R.run_cd(str(tmp_path), RUL, TOPOS, PIN, None, "x", arm_set_name="v9", charges={k: v for k, v in CHARGES.items() if k != "wide|heavy"})
    with pytest.raises(SystemExit, match="unknown arm set"):
        R.run_cd(str(tmp_path), RUL, TOPOS, PIN, None, "x", arm_set_name="v8")


def test_the_default_registry_and_tables_are_untouched():
    assert R.CD_ARM_REGISTRY["gnn"] == "ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100" and "v8" not in R.CD_ARM_REGISTRY
    assert R.CD_CKPT["gnn"] == "uncut-linkstatic-gnn-eng-e40-seed"
