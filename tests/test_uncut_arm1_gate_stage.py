import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_arm1_gate_stage as S  # noqa: E402


@pytest.fixture
def src(tmp_path):
    ig = tmp_path / "ig"
    for r in ("wf1_moderate", "wf1_heavy"):
        (ig / r).mkdir(parents=True)
    ck = tmp_path / "ck"
    ck.mkdir()
    want = {}
    for arm in S.ARMS:
        for seed in (1, 2):
            b = f"{arm}{seed}".encode()
            (ck / f"m-{arm.replace('_', '-')}-seed{seed}.pt").write_bytes(b)
            (ck / f"m-{arm.replace('_', '-')}-seed{seed}.contract.json").write_text(json.dumps({"physics_env": {"warmth_physics": "node_disk_v2"}}))
            want[f"{arm}|s{seed}"] = hashlib.md5(b).hexdigest()
    md5 = tmp_path / "md5.json"
    md5.write_text(json.dumps({**want, "_provenance": {k: {"job": "871420_0", "host": "h", "cpu": "AMD EPYC 9475F"} for k in want}}))
    split = tmp_path / "arm1_split.json"
    split.write_text("{}")
    return tmp_path, ck, ig, md5, split


def args(t, ck, ig, md5, split, dest="in"):
    return ["--md5", str(md5), "--src", str(ck / "m-{arm}-seed{seed}.pt"), "--inputs-group", str(ig), "--split", str(split), "--dest", str(t / dest)]


def test_stage_copies_under_gate_names(src):
    t, ck, ig, md5, split = src
    out = S.main(args(t, ck, ig, md5, split))
    assert len(out["checkpoints"]) == 6
    assert (t / "in" / "models" / "uncut-arm1-mlp-same-seed2.pt").read_bytes() == b"mlp_same2"
    assert (t / "in" / "models" / "uncut-arm1-twin-eng-seed1.contract.json").exists() and (t / "in" / "arm1_split.json").exists()
    assert os.path.islink(t / "in" / "wf1_heavy")


def test_md5_mismatch_fails(src):
    t, ck, ig, md5, split = src
    (ck / "m-gnn-eng-seed1.pt").write_bytes(b"other")
    with pytest.raises(SystemExit, match="training log"):
        S.main(args(t, ck, ig, md5, split))
    assert not (t / "in").exists()


def test_missing_contract_fails(src):
    t, ck, ig, md5, split = src
    os.remove(ck / "m-twin-eng-seed2.contract.json")
    with pytest.raises(SystemExit, match="missing"):
        S.main(args(t, ck, ig, md5, split))


def test_duplicate_md5_across_keys_fails(src):
    t, ck, ig, md5, split = src
    want = json.loads(md5.read_text())
    want["mlp_same|s2"] = want["mlp_same|s1"]
    md5.write_text(json.dumps(want))
    with pytest.raises(SystemExit, match="several keys"):
        S.main(args(t, ck, ig, md5, split))


@pytest.mark.parametrize("contract", [{"physics_env": {}}, {}, {"physics_env": {"warmth_physics": "platform_reuse_v1"}}])
def test_contract_warmth_must_be_node_disk_v2(src, contract):
    t, ck, ig, md5, split = src
    (ck / "m-mlp-same-seed2.contract.json").write_text(json.dumps(contract))
    with pytest.raises(SystemExit, match="warmth_physics"):
        S.main(args(t, ck, ig, md5, split))
    assert not (t / "in").exists()


def test_existing_dest_fails(src):
    t, ck, ig, md5, split = src
    (t / "in").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        S.main(args(t, ck, ig, md5, split))


def test_arm2_stage_by_arms_and_stem(src):
    t, ck, ig, md5, split = src
    want = {k: v for k, v in json.loads(md5.read_text()).items() if not k.startswith("mlp_same")}
    md5.write_text(json.dumps(want))
    out = S.main(args(t, ck, ig, md5, split) + ["--arms", "gnn_eng,twin_eng", "--stem", "uncut-arm2k0"])
    assert out["arms"] == ["gnn_eng", "twin_eng"] and len(out["checkpoints"]) == 4
    assert (t / "in" / "models" / "uncut-arm2k0-twin-eng-seed2.pt").read_bytes() == b"twin_eng2"


def test_arms_stage_in_turn_from_one_growing_md5_file(src):
    t, ck, ig, md5, split = src
    out = S.main(args(t, ck, ig, md5, split, dest="in_gt") + ["--arms", "gnn_eng,twin_eng"])
    assert sorted(out["checkpoints"]) == ["gnn_eng|s1", "gnn_eng|s2", "twin_eng|s1", "twin_eng|s2"]
    assert not (t / "in_gt" / "models" / "uncut-arm1-mlp-same-seed1.pt").exists()
    out = S.main(args(t, ck, ig, md5, split, dest="in_mlp") + ["--arms", "mlp_same"])
    assert sorted(out["checkpoints"]) == ["mlp_same|s1", "mlp_same|s2"]


def test_arms_scope_fails_loud(src):
    t, ck, ig, md5, split = src
    want = json.loads(md5.read_text())
    md5.write_text(json.dumps({**want, "raw|s1": "0" * 32}))
    with pytest.raises(SystemExit, match="keys outside"):
        S.main(args(t, ck, ig, md5, split) + ["--arms", "gnn_eng,twin_eng"])
    md5.write_text(json.dumps(want))
    with pytest.raises(SystemExit, match="not a subset"):
        S.main(args(t, ck, ig, md5, split) + ["--arms", "gnn_eng,raw"])
    md5.write_text(json.dumps({k: v for k, v in json.loads(md5.read_text()).items() if not k.startswith("twin_eng")}))
    with pytest.raises(SystemExit, match="no checkpoint for"):
        S.main(args(t, ck, ig, md5, split))
    assert not (t / "in").exists()


def test_gate_arm2_stems_match_staged_names():
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    assert set(G.ARM2_KINDS) == {f"ra_{a}_arm2" for a in ("gnn_eng", "twin_eng")}
    assert set(G.ARM2K0_KINDS) == {f"ra_{a}_arm2k0" for a in ("gnn_eng", "twin_eng")}


def test_gate_knows_arm1_kinds_outside_default_grid():
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    for k in G.ARM1_KINDS:
        assert k in G.RA_KINDS and f"{k}_nosplit" in G.R1A_ARMS and f"{k}_nosplit" in G.R1A_NOSPLIT
    assert "ra_twin_eng_arm1" in G.RA_MP_OFF and "ra_gnn_eng_arm1" not in G.RA_MP_OFF
    assert G.split_dc("ra_mlp_same_arm1_nosplit_k100") == ("ra_mlp_same_arm1_nosplit", 1.0)


def _set_btf(ck, value):
    for p in ck.glob("*.contract.json"):
        c = json.loads(p.read_text())
        c.pop("batched_teacher_forced", None)
        p.write_text(json.dumps(c if value == "absent" else {**c, "batched_teacher_forced": value}))


def _gt_only(md5):
    md5.write_text(json.dumps({k: v for k, v in json.loads(md5.read_text()).items() if not k.startswith("mlp_same")}))


@pytest.mark.parametrize("btf", ["absent", None, False])
def test_arm1_stems_take_unbatched_contracts_and_arm1b_refuses_them(src, btf):
    t, ck, ig, md5, split = src
    _set_btf(ck, btf)
    S.main(args(t, ck, ig, md5, split, dest="in1"))
    _gt_only(md5)
    with pytest.raises(SystemExit, match="batched_teacher_forced"):
        S.main(args(t, ck, ig, md5, split, dest="in1b") + ["--arms", "gnn_eng,twin_eng", "--stem", "uncut-arm1b"])
    assert not (t / "in1b").exists()


@pytest.mark.parametrize("btf", [True, 1, "true"])
def test_arm1b_stem_needs_batched_true_and_other_stems_refuse_it(src, btf):
    t, ck, ig, md5, split = src
    _gt_only(md5)
    _set_btf(ck, btf)
    for stem in ("uncut-arm1", "uncut-arm2", "uncut-arm2k0"):
        with pytest.raises(SystemExit, match="batched_teacher_forced"):
            S.main(args(t, ck, ig, md5, split, dest="in") + ["--arms", "gnn_eng,twin_eng", "--stem", stem])
    a1b = args(t, ck, ig, md5, split, dest="in1b") + ["--arms", "gnn_eng,twin_eng", "--stem", "uncut-arm1b"]
    if btf is not True:
        with pytest.raises(SystemExit, match="batched_teacher_forced"):
            S.main(a1b)
        return
    out = S.main(a1b)
    assert out["stem"] == "uncut-arm1b" and len(out["checkpoints"]) == 4
    assert (t / "in1b" / "models" / "uncut-arm1b-twin-eng-seed2.pt").read_bytes() == b"twin_eng2"


def test_arm1b_takes_the_mlp_comparator(src):
    """Arm 1b gained mlp_same (amendment of 2026-10-15): the stager stages all three arms under the 1b stem, and still refuses an arm outside ARMS."""
    t, ck, ig, md5, split = src
    _set_btf(ck, True)
    out = S.main(args(t, ck, ig, md5, split) + ["--stem", "uncut-arm1b"])
    assert len(out["checkpoints"]) == 6 and {k.split("|")[0] for k in out["checkpoints"]} == {"gnn_eng", "twin_eng", "mlp_same"}
    with pytest.raises(SystemExit, match="not a subset"):
        S.main(args(t, ck, ig, md5, split) + ["--stem", "uncut-arm1b", "--arms", "gnn_eng,set_transformer"])


def test_md5_file_names_the_checkpoint_dir(src, tmp_path):
    t, ck, ig, md5, split = src
    want = json.loads(md5.read_text())
    md5.write_text(json.dumps({**want, "_ckpt_dir": str(ck)}))
    out = S.main(args(t, ck, ig, md5, split, dest="a")[:2] + ["--src", "{dir}/m-{arm}-seed{seed}.pt"] + args(t, ck, ig, md5, split, dest="a")[4:])
    assert out["ckpt_dir"] == str(ck) and len(out["checkpoints"]) == 6
    S.main(args(t, ck, ig, md5, split, dest="b"))
    other = tmp_path / "other"
    other.mkdir()
    md5.write_text(json.dumps({**want, "_ckpt_dir": str(other)}))
    with pytest.raises(SystemExit, match="not in the md5 file's _ckpt_dir"):
        S.main(args(t, ck, ig, md5, split, dest="c"))
    for bad, match in (({**want, "_ckpt_dir": str(tmp_path / "nope")}, "existing directory"), ({**want, "_note": "x"}, "metadata keys"), ({k: v for k, v in want.items() if k != "_ckpt_dir"}, "names no _ckpt_dir")):
        md5.write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match=match):
            S.main(args(t, ck, ig, md5, split, dest="d")[:2] + ["--src", "{dir}/m-{arm}-seed{seed}.pt"] + args(t, ck, ig, md5, split, dest="d")[4:])
    assert not (t / "c").exists() and not (t / "d").exists()


def test_reader_skips_md5_metadata(tmp_path):
    import uncut_arm_gate_read as R
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"gnn_eng|s1": "x", "_ckpt_dir": "/share/x"}))
    assert R.ckpt_md5s(str(f)) == {"gnn_eng|s1": "x"} and R.ckpt_md5s(f"arm1b={f}") == {"gnn_eng@arm1b|s1": "x"}


def _prov(md5, edit):
    d = json.loads(md5.read_text())
    edit(d)
    md5.write_text(json.dumps(d))


def test_provenance_hedge_rule(src):
    t, ck, ig, md5, split = src
    _prov(md5, lambda d: d["_provenance"].__setitem__("twin_eng|s2", {"job": "869803_3", "host": "h", "cpu": "AMD EPYC 9335"}))
    with pytest.raises(SystemExit, match="different jobs"):
        S.main(args(t, ck, ig, md5, split))
    _prov(md5, lambda d: d["_provenance"].__setitem__("twin_eng|s2", {"job": "871420_5", "host": "h2", "cpu": "AMD EPYC 9475F"}))
    out = S.main(args(t, ck, ig, md5, split))
    assert out["provenance"]["twin_eng|s2"]["job"] == "871420_5"
    for bad in ({"job": "871420_5", "host": "h"}, {"job": "", "host": "h", "cpu": "c"}, None):
        _prov(md5, lambda d: d["_provenance"].__setitem__("gnn_eng|s1", bad))
        with pytest.raises(SystemExit, match="needs non-empty job, host and cpu"):
            S.main(args(t, ck, ig, md5, split, dest="x"))
    _prov(md5, lambda d: d.pop("_provenance"))
    with pytest.raises(SystemExit, match="no _provenance"):
        S.main(args(t, ck, ig, md5, split, dest="x"))
    S.main(args(t, ck, ig, md5, split, dest="arm2") + ["--arms", "gnn_eng,twin_eng", "--stem", "uncut-arm2"])


def _chain(job, sub="arm2-gpu@0123abcd", host="a-l40s-o-1", cpu="AMD EPYC 9355 32-Core Processor"):
    links = [{"job": j, "host": host, "cpu": cpu} for j in job.split(",")]
    e = {"job": job, "host": ",".join(l["host"] for l in links), "cpu": "; ".join(l["cpu"] for l in links), "links": links}
    if sub is not None:
        e["submission"] = sub
    return e


def test_chained_provenance_shares_a_submission_label(src):
    t, ck, ig, md5, split = src
    arm2 = ["--arms", "gnn_eng,twin_eng", "--stem", "uncut-arm2"]

    def setp(d, g1, g2):
        d["_provenance"]["gnn_eng|s1"], d["_provenance"]["gnn_eng|s2"] = g1, g2
    # the GPU gnn seeds: separate chains (869961, 869963), one submission label -> staged
    _prov(md5, lambda d: setp(d, _chain("869961,870101"), _chain("869963")))
    out = S.main(args(t, ck, ig, md5, split, dest="ok") + arm2)
    assert out["provenance"]["gnn_eng|s1"]["links"][1]["job"] == "870101"
    _prov(md5, lambda d: setp(d, _chain("869961"), _chain("869963", sub="arm2-gpu@ffff0000")))
    with pytest.raises(SystemExit, match="different submissions"):
        S.main(args(t, ck, ig, md5, split, dest="x") + arm2)
    _prov(md5, lambda d: setp(d, _chain("869961", sub=None), _chain("869963", sub=None)))
    with pytest.raises(SystemExit, match="no submission label"):
        S.main(args(t, ck, ig, md5, split, dest="x") + arm2)
    _prov(md5, lambda d: setp(d, _chain("869961"), {"job": "869963", "host": "h", "cpu": "c"}))
    with pytest.raises(SystemExit, match="mixes chained and single-job"):
        S.main(args(t, ck, ig, md5, split, dest="x") + arm2)
    bad = _chain("869961")
    bad["links"] = [{"job": "869961", "host": "h"}]
    _prov(md5, lambda d: setp(d, bad, _chain("869963")))
    with pytest.raises(SystemExit, match="links must be non-empty"):
        S.main(args(t, ck, ig, md5, split, dest="x") + arm2)
    # single-job provenance outside Arm 1 keeps the one-job rule
    _prov(md5, lambda d: setp(d, {"job": "869961", "host": "h", "cpu": "c"}, {"job": "869963", "host": "h", "cpu": "c"}))
    with pytest.raises(SystemExit, match="different jobs"):
        S.main(args(t, ck, ig, md5, split, dest="x") + arm2)


def test_arm1_refuses_chained_provenance(src):
    t, ck, ig, md5, split = src
    _prov(md5, lambda d: d["_provenance"].update({"gnn_eng|s1": _chain("871420_0"), "gnn_eng|s2": _chain("871420_1")}))
    with pytest.raises(SystemExit, match="Arm 1's hedge rule is one Slurm job per arm"):
        S.main(args(t, ck, ig, md5, split))


def test_s7_schema_per_arm_dirs_and_files(src, tmp_path):
    t, ck, ig, md5, split = src
    ck2 = tmp_path / "ck2"
    ck2.mkdir()
    for p in ck.glob("m-mlp-same-*"):
        p.rename(ck2 / p.name)

    def edit(d):
        d["_ckpt_dir"] = {"gnn_eng": str(ck), "twin_eng": str(ck), "mlp_same": str(ck2)}
        for k, e in d["_provenance"].items():
            e.update(file=f"m-{k.split('|')[0].replace('_', '-')}-seed{k[-1]}.pt", wandb_id="w")
    _prov(md5, edit)
    base = ["--md5", str(md5), "--inputs-group", str(ig), "--split", str(split)]
    out = S.main(base + ["--dest", str(t / "a")])
    assert out["checkpoints"]["mlp_same|s1"]["src"] == str(ck2 / "m-mlp-same-seed1.pt") and len(out["checkpoints"]) == 6
    _prov(md5, lambda d: d["_provenance"]["gnn_eng|s1"].__setitem__("file", "other.pt"))
    with pytest.raises(SystemExit, match="missing|not the _provenance file"):
        S.main(base + ["--dest", str(t / "b")])
    with pytest.raises(SystemExit, match="not the _provenance file"):
        S.main(args(t, ck, ig, md5, split, dest="c")[:2] + ["--src", "{dir}/m-{arm}-seed{seed}.pt"] + args(t, ck, ig, md5, split, dest="c")[4:])
    _prov(md5, lambda d: d["_ckpt_dir"].pop("mlp_same"))
    with pytest.raises(SystemExit, match="every arm"):
        S.main(base + ["--dest", str(t / "d")])



def test_g8b_fast_kinds_check_lambda_and_the_recomputed_table(tmp_path):
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    for k in G.COMB_FAST:
        assert k not in G.COMB_KINDS and k in G.RA_KINDS and f"{k}_nosplit" in G.R1A_NOSPLIT and f"{k}_nosplit" in G.R1A_ARMS
        assert G.comb_contract(k) == "partial_state_v5" and G.COMB_CHECK[k] == "ra_gnn_eng" and G.split_dc(f"{k}_nosplit_k100") == (f"{k}_nosplit", 1.0)
    assert G.COMB_STEMS["ra_gnn_eng_a3l"] == "uncut-a3l-gnn-eng" and G.COMB_STEMS["ra_gnn_eng_a3lft"] == "uncut-a3lft-gnn-eng"
    good = {"batched_teacher_forced": True, "partial_state_contract": "partial_state_v5", "regret_lambda": 2.0,
             "move_regret": {"sha256": G.COMB_FAST_TABLE_SHA256, "lambda": 2.0}}
    p = tmp_path / "c.contract.json"
    p.write_text(json.dumps(good))
    G.comb_trainer_check("ra_gnn_eng_a3l", str(p))
    for bad in ({"regret_lambda": 0.5}, {"move_regret": {"sha256": "17a7387271eeea6f961371e7a4f93603efb58bc0d368af9f84f410af0af166be", "lambda": 2.0}}, {"regret_lambda": None}):
        p.write_text(json.dumps({**good, **bad}))
        with pytest.raises(SystemExit, match="FAIL LOUD"):
            G.comb_trainer_check("ra_gnn_eng_a3lft", str(p))
    p.write_text(json.dumps(good))
    with pytest.raises(SystemExit, match="not a regret-lever run"):
        G.comb_trainer_check("ra_gnn_eng_combf", str(p))


def test_comb_kr_serves_fl_gnn_and_the_fl_mlp_at_the_frozen_cotenancy_k(tmp_path, monkeypatch):
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    assert G.COMB_KR_KINDS == ("ra_gnn_eng_combfl_nosplit", "ra_mlp_same_combfl_nosplit")
    for k in G.COMB_KR_KINDS:
        assert G.split_dc(f"{k}_k100") == (k, 1.0)
    w = json.load(open(G.DC_W_FILE))
    f = tmp_path / "w.json"
    monkeypatch.setattr(G, "DC_W_FILE", str(f))
    for bad in (None, 0.9, True, "1.2"):
        f.write_text(json.dumps({**w, **({} if bad is None else {"k_cotenancy": bad})}))
        with pytest.raises(SystemExit, match="k_cotenancy"):
            G.split_dc("ra_gnn_eng_combfl_nosplit_kr")
    w["w_ms_per_plan_by_bucket"]["ra_mlp_same_combfl_nosplit"] = {"heavy": {"le10": 2.0, "11_50": 1.0, "gt50": 1.5}}
    f.write_text(json.dumps({**w, "k_cotenancy": 1.468}))
    kind, k = G.split_dc("ra_mlp_same_combfl_nosplit_kr")
    assert (kind, k) == ("ra_mlp_same_combfl_nosplit", 1.468)
    env, val = G.dc_cost_ms(kind, k, "heavy")
    assert env == G.DC_BUCKET_ENV and json.loads(val) == {"le10": 2.936, "11_50": 1.468, "gt50": 2.202}
    # every other _kr stays an unknown arm, as before this change (the listing proof relies on it)
    for other in ("ra_gnn_eng_combf_nosplit_kr", "ra_gnn_eng_combl_nosplit_kr", "ra_gnn_eng_a3l_nosplit_kr", "cd_ib0_inflightfull_kr", "ra_gnn_eng_arm1_nosplit_kr"):
        assert G.split_dc(other) == (other, None)


def test_g8c_flreg_kinds_check_lambda_the_table_and_v6_and_serve_kr(tmp_path, monkeypatch):
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    assert G.COMB_FLREG == ("ra_gnn_eng_flreg", "ra_mlp_same_flreg")
    for k in G.COMB_FLREG:
        assert k not in G.COMB_KINDS and k not in G.COMB_FAST and k in G.RA_KINDS
        assert f"{k}_nosplit" in G.R1A_NOSPLIT and f"{k}_nosplit" in G.R1A_ARMS
        assert G.comb_contract(k) == "partial_state_v6" and G.split_dc(f"{k}_nosplit_k100") == (f"{k}_nosplit", 1.0)
    assert G.COMB_CHECK["ra_gnn_eng_flreg"] == "ra_gnn_eng_v6" and G.COMB_CHECK["ra_mlp_same_flreg"] == "ra_mlp_same_v6"
    assert G.COMB_STEMS["ra_gnn_eng_flreg"] == "uncut-flreg-gnn-eng" and G.COMB_STEMS["ra_mlp_same_flreg"] == "uncut-flreg-mlp-same"
    good = {"batched_teacher_forced": True, "partial_state_contract": "partial_state_v6", "regret_lambda": 2.0,
            "move_regret": {"sha256": G.COMB_FAST_TABLE_SHA256, "lambda": 2.0}}
    p = tmp_path / "c.contract.json"
    p.write_text(json.dumps(good))
    for k in G.COMB_FLREG:
        G.comb_trainer_check(k, str(p))
    for bad in ({"regret_lambda": None}, {"regret_lambda": 0.5}, {"move_regret": None},
                {"move_regret": {"sha256": "17a7387271eeea6f961371e7a4f93603efb58bc0d368af9f84f410af0af166be", "lambda": 2.0}},
                {"partial_state_contract": "partial_state_v5"}, {"batched_teacher_forced": None}):
        p.write_text(json.dumps({**good, **bad}))
        for k in G.COMB_FLREG:
            with pytest.raises(SystemExit, match="FAIL LOUD"):
                G.comb_trainer_check(k, str(p))
    # the filed FL kinds still refuse a regret-lever sidecar
    p.write_text(json.dumps(good))
    with pytest.raises(SystemExit, match="not a regret-lever run"):
        G.comb_trainer_check("ra_gnn_eng_combfl", str(p))
    # _kr: the two FL + lambda kinds serve at the frozen k_cotenancy; COMB_KR_KINDS (the FL confirmation's) is unchanged
    assert G.COMB_KR_KINDS == ("ra_gnn_eng_combfl_nosplit", "ra_mlp_same_combfl_nosplit")
    w = json.load(open(G.DC_W_FILE))
    f = tmp_path / "w.json"
    f.write_text(json.dumps({**w, "k_cotenancy": 1.468}))
    monkeypatch.setattr(G, "DC_W_FILE", str(f))
    for k in G.FLREG_KR_KINDS:
        assert G.split_dc(f"{k}_kr") == (k, 1.468)
    assert G.split_dc("ra_gnn_eng_flreg_kr") == ("ra_gnn_eng_flreg_kr", None)
