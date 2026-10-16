import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_arm1_gate_stage as S  # noqa: E402

SHA = S.ARM3_TABLE_SHA256


def _contract(lam=0.5, sha=SHA, btf=True, **kw):
    c = {"physics_env": {"warmth_physics": "node_disk_v2"}, "batched_teacher_forced": btf}
    if lam is not None:
        c["regret_lambda"] = lam
        c["move_regret"] = {"sha256": sha, "lambda": lam}
    c.update(kw)
    return c


@pytest.fixture
def arm3(tmp_path):
    ig = tmp_path / "ig"
    for r in ("wf1_moderate", "wf1_heavy"):
        (ig / r).mkdir(parents=True)
    ck = tmp_path / "ck"
    ck.mkdir()
    want = {}
    for arm in ("gnn_eng", "twin_eng"):
        for seed in (1, 2):
            b = f"{arm}{seed}".encode()
            stem = f"l-{arm.replace('_', '-')}-seed{seed}"
            (ck / f"{stem}.pt").write_bytes(b)
            (ck / f"{stem}.contract.json").write_text(json.dumps(_contract()))
            want[f"{arm}|s{seed}"] = hashlib.md5(b).hexdigest()
    md5 = tmp_path / "md5.json"
    md5.write_text(json.dumps(want))
    split = tmp_path / "split.json"
    split.write_text("{}")
    return tmp_path, ck, ig, md5, split


def argv(t, ck, ig, md5, split, dest="in"):
    return ["--md5", str(md5), "--src", str(ck / "l-{arm}-seed{seed}.pt"), "--inputs-group", str(ig), "--split", str(split), "--dest", str(t / dest),
            "--stem", "uncut-arm3", "--arms", "gnn_eng,twin_eng"]


def rewrite(ck, arm, seed, **kw):
    (ck / f"l-{arm.replace('_', '-')}-seed{seed}.contract.json").write_text(json.dumps(_contract(**kw)))


def test_arm3_stages_one_lambda_under_the_gate_names(arm3):
    t, ck, ig, md5, split = arm3
    out = S.main(argv(t, ck, ig, md5, split))
    assert out["stem"] == "uncut-arm3" and out["regret_lambda"] == 0.5 and len(out["checkpoints"]) == 4
    assert (t / "in" / "models" / "uncut-arm3-gnn-eng-seed1.pt").read_bytes() == b"gnn_eng1"
    assert (t / "in" / "models" / "uncut-arm3-twin-eng-seed2.contract.json").exists()


@pytest.mark.parametrize("kw,match", [(dict(lam=None), "regret_lambda"), (dict(lam=0.0), "regret_lambda"), (dict(lam="0.5"), "regret_lambda"),
                                      (dict(sha="0" * 64), "frozen table"), (dict(btf=False), "batched_teacher_forced"), (dict(btf=None), "batched_teacher_forced")])
def test_arm3_refuses_a_checkpoint_that_is_not_a_lever_run(arm3, kw, match):
    t, ck, ig, md5, split = arm3
    rewrite(ck, "gnn_eng", 1, **kw)
    with pytest.raises(SystemExit, match=match):
        S.main(argv(t, ck, ig, md5, split))
    assert not (t / "in").exists()


def test_arm3_refuses_two_lambdas(arm3):
    t, ck, ig, md5, split = arm3
    rewrite(ck, "twin_eng", 2, lam=2.0)
    with pytest.raises(SystemExit, match="several lambdas"):
        S.main(argv(t, ck, ig, md5, split))
    assert not (t / "in").exists()


def test_arm3_refuses_a_lambda_that_disagrees_with_its_table_record(arm3):
    t, ck, ig, md5, split = arm3
    (ck / "l-gnn-eng-seed1.contract.json").write_text(json.dumps(_contract(move_regret={"sha256": SHA, "lambda": 2.0})))
    with pytest.raises(SystemExit, match="frozen table"):
        S.main(argv(t, ck, ig, md5, split))


def test_arm3_has_no_mlp_and_other_stems_refuse_a_lever_checkpoint(arm3):
    t, ck, ig, md5, split = arm3
    with pytest.raises(SystemExit, match="only"):
        S.main(argv(t, ck, ig, md5, split)[:-2] + ["--arms", "gnn_eng,mlp_same"])
    for stem in ("uncut-arm1", "uncut-arm1b", "uncut-arm2", "uncut-arm2k0"):
        a = argv(t, ck, ig, md5, split, dest="x" + stem)
        a[a.index("--stem") + 1] = stem
        with pytest.raises(SystemExit, match="batched_teacher_forced|not a lever|_provenance"):
            S.main(a)
