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
            (ck / f"m-{arm.replace('_', '-')}-seed{seed}.contract.json").write_text("{}")
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


def test_existing_dest_fails(src):
    t, ck, ig, md5, split = src
    (t / "in").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        S.main(args(t, ck, ig, md5, split))


def test_gate_knows_arm1_kinds_outside_default_grid():
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    for k in G.ARM1_KINDS:
        assert k in G.RA_KINDS and f"{k}_nosplit" in G.R1A_ARMS and f"{k}_nosplit" in G.R1A_NOSPLIT
    assert "ra_twin_eng_arm1" in G.RA_MP_OFF and "ra_gnn_eng_arm1" not in G.RA_MP_OFF
    assert G.split_dc("ra_mlp_same_arm1_nosplit_k100") == ("ra_mlp_same_arm1_nosplit", 1.0)


def test_kr_serves_arm1_learned_kinds_at_the_frozen_cotenancy_k(tmp_path, monkeypatch):
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    for tag, k in G.DC_K.items():
        assert G.split_dc(f"cd_ib0_{tag}") == ("cd_ib0", k) and G.split_dc(f"ra_gnn_eng_arm1_nosplit_{tag}") == ("ra_gnn_eng_arm1_nosplit", k)
    assert G.split_dc("cd_ib0") == ("cd_ib0", None)
    w = json.load(open(G.DC_W_FILE))
    f = tmp_path / "w.json"
    monkeypatch.setattr(G, "DC_W_FILE", str(f))
    for bad in (None, 0.9, True, "1.2"):
        f.write_text(json.dumps({**w, **({} if bad is None else {"k_cotenancy": bad})}))
        with pytest.raises(SystemExit, match="k_cotenancy"):
            G.split_dc("ra_twin_eng_arm1_nosplit_kr")
    w["w_ms_per_plan_by_bucket"]["ra_twin_eng_arm1_nosplit"] = {"heavy": {"le10": 2.0, "11_50": 1.0, "gt50": 1.5}}
    f.write_text(json.dumps({**w, "k_cotenancy": 1.25}))
    kind, k = G.split_dc("ra_twin_eng_arm1_nosplit_kr")
    assert (kind, k) == ("ra_twin_eng_arm1_nosplit", 1.25)
    env, val = G.dc_cost_ms(kind, k, "heavy")
    assert env == G.DC_BUCKET_ENV and json.loads(val) == {"le10": 2.5, "11_50": 1.25, "gt50": 1.875}
    for hand in ("cd_ib0_kr", "cd_kr", "ra_gnn_eng_nosplit_kr"):
        with pytest.raises(SystemExit, match="only the Arm 1"):
            G.split_dc(hand)


def test_gate_knows_arm1b_kinds_and_their_kr_cells(tmp_path, monkeypatch):
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    assert G.ARM1B_KINDS == ("ra_gnn_eng_arm1b", "ra_twin_eng_arm1b")
    for k in G.ARM1B_KINDS:
        assert k in G.RA_KINDS and f"{k}_nosplit" in G.R1A_ARMS and f"{k}_nosplit" in G.R1A_NOSPLIT
    assert "ra_twin_eng_arm1b" in G.RA_MP_OFF and "ra_gnn_eng_arm1b" not in G.RA_MP_OFF
    assert G.split_dc("ra_gnn_eng_arm1b_nosplit_k100") == ("ra_gnn_eng_arm1b_nosplit", 1.0)
    f = tmp_path / "w.json"
    f.write_text(json.dumps({**json.load(open(G.DC_W_FILE)), "k_cotenancy": 1.5}))
    monkeypatch.setattr(G, "DC_W_FILE", str(f))
    assert G.split_dc("ra_twin_eng_arm1b_nosplit_kr") == ("ra_twin_eng_arm1b_nosplit", 1.5)
    with pytest.raises(SystemExit, match="only the Arm 1"):
        G.split_dc("ra_mlp_same_arm1b_nosplit_kr")


@pytest.mark.parametrize("btf,arm1_ok,arm1b_ok", [("absent", True, False), (None, True, False), (False, True, False),
                                                  (True, False, True), (1, False, False), (0, False, False), ("true", False, False)])
def test_trainer_check_tells_arm1b_from_arm1(tmp_path, btf, arm1_ok, arm1b_ok):
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    side = tmp_path / "c.contract.json"
    side.write_text(json.dumps({} if btf == "absent" else {"batched_teacher_forced": btf}))
    for kinds, ok in ((G.ARM1_KINDS, arm1_ok), (G.ARM1B_KINDS, arm1b_ok)):
        for k in kinds:
            if ok:
                G.arm1_trainer_check(k, str(side))
            else:
                with pytest.raises(SystemExit, match="batched_teacher_forced"):
                    G.arm1_trainer_check(k, str(side))
    G.arm1_trainer_check("ra_gnn_eng", str(side))


def _fake_sys(tmp_path, smt="off", cores=4, tpc=1):
    d = tmp_path / "cpu"
    for c in range(cores * tpc):
        (d / f"cpu{c}" / "topology").mkdir(parents=True)
        (d / f"cpu{c}" / "topology" / "physical_package_id").write_text("0\n")
        (d / f"cpu{c}" / "topology" / "core_id").write_text(f"{c % cores}\n")
    (d / "smt").mkdir()
    (d / "smt" / "control").write_text(smt + "\n")
    (d / "smt" / "active").write_text(("1" if smt == "on" else "0") + "\n")
    (d / "online").write_text(f"0-{cores * tpc - 1}\n")
    info = tmp_path / "cpuinfo"
    info.write_text("processor\t: 0\nmodel name\t: AMD EPYC 9335 32-Core Processor\n")
    return str(d), str(info)


def test_host_block_and_checks(tmp_path, monkeypatch):
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    d, info = _fake_sys(tmp_path, smt="on", cores=4, tpc=2)
    monkeypatch.setattr(G.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4, 5, 6, 7})
    monkeypatch.setattr(G, "PARALLEL", 5)
    h = G.host_block(d, info)
    assert h["nproc"] == 8 and h["physical_cores"] == 4 and h["cpuset"] == "0-7" and h["smt_control"] == "on" and h["smt_active"] == "1"
    assert h["cpu_model"] == "AMD EPYC 9335 32-Core Processor" and h["parallel"] == 5 and h["cpus_online"] == "0-7"
    with pytest.raises(SystemExit, match="physical cores"):
        G.check_parallel(h)
    G.check_parallel({**h, "parallel": 4})
    G.check_host(h)
    monkeypatch.setenv("R1A_EXPECT_SMT", "off")
    with pytest.raises(SystemExit, match="R1A_EXPECT_SMT"):
        G.check_host(h)
    monkeypatch.setenv("R1A_EXPECT_SMT", "on")
    monkeypatch.setenv("R1A_EXPECT_CPU_MODEL", "AMD EPYC 9475F 48-Core Processor")
    with pytest.raises(SystemExit, match="R1A_EXPECT_CPU_MODEL"):
        G.check_host(h)
    assert G._cpulist([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"


def test_host_block_on_this_machine():
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    h = G.host_block()
    assert h["nproc"] == len(os.sched_getaffinity(0)) and 1 <= h["physical_cores"] <= h["nproc"] and h["hostname"]


def test_g7_kr_serves_arm2_and_k0_kinds_and_nothing_else(tmp_path, monkeypatch):
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    f = tmp_path / "w.json"
    monkeypatch.setattr(G, "DC_W_FILE", str(f))
    f.write_text(json.dumps({**json.load(open(os.path.join(os.path.dirname(G.__file__), "decision_cost_w.json"))), "k_cotenancy": 1.5}))
    for base in G.ARM2_ALL:
        k = f"{base}_nosplit_kr"
        assert G.split_dc(k) == (f"{base}_nosplit", 1.5)
        assert G.split_dc(f"{base}_nosplit_k100") == (f"{base}_nosplit", 1.0)
    for other in ("cd_ib0_kr", "ra_gnn_eng_nosplit_kr", "ra_gnn_eng_arm2_kr", "ra_twin_eng_arm2k0_kr"):
        with pytest.raises(SystemExit, match="only the Arm 1"):
            G.split_dc(other)
