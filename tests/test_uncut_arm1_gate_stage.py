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
    md5.write_text(json.dumps(want))
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


def test_arm1b_has_no_mlp(src):
    t, ck, ig, md5, split = src
    _set_btf(ck, True)
    with pytest.raises(SystemExit, match="Arm 1b"):
        S.main(args(t, ck, ig, md5, split) + ["--stem", "uncut-arm1b"])
