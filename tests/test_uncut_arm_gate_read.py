import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_arm_gate_read as G  # noqa: E402

TOPOS = list(range(1, 13))
TOPS = ",".join(map(str, TOPOS))
P, T, M = "uncut_arm1_gnn_eng", "uncut_arm1_twin_eng", "ra_mlp_same"


def cell(d, arm, seed, t, w, r, mean, ckpt=None, lat_big=None, busy=1.0):
    name = f"cc40s{t}__{w}{r}__{arm}_k100_s{seed}"
    s = {"averageElapsedTime": mean, "steady_state": {"after_360s": {"mean": mean}}, "num_tasks": 4,
         "endTime": 100.0, "arrival_end": {"last_arrival_s": 99.0}, "code": {"commit": "pin1234", "dirty": False},
         "schedulerCounters": {"sched_queue_wait_seconds": 0.0, "sched_busy_seconds": busy}, "env": {}}
    if ckpt:
        s["env"]["GNN_MODEL_PATH"] = ckpt
    with open(os.path.join(d, name + ".summary.json"), "w") as fh:
        json.dump(s, fh)
    big = mean if lat_big is None else lat_big
    rows = [(0, 10.0, 10.0 + mean, 2, 2), (1, 400.0, 400.0 + mean, 2, 2), (2, 20.0, 20.0 + big, 20, 20), (3, 500.0, 500.0 + big, 20, 20)]
    np.save(os.path.join(d, name + ".tasks_bsz.npy"), np.asarray(rows, dtype=np.float64))
    np.save(os.path.join(d, name + ".tasks_lock.npy"), np.asarray([(r[0], r[1], r[1] + 0.1, 0.1) for r in rows], dtype=np.float64))
    with open(os.path.join(d, name + ".log"), "w") as fh:
        fh.write("INFO start\n[CODE] commit=pin1234abcd branch=HEAD dirty=False diff=e3b0\n")


@pytest.fixture
def gate(tmp_path):
    d = tmp_path / "g"
    d.mkdir()
    ck = tmp_path / "m.pt"
    ck.write_bytes(b"weights")
    md5 = hashlib.md5(b"weights").hexdigest()
    cm = tmp_path / "ck.json"
    cm.write_text(json.dumps({f"{a}|s{s}": md5 for a in (P, T, M) for s in (1, 2)}))
    return str(d), str(ck), str(cm)


def fill(d, ck, primary_mean, twin_mean=2.0, mlp_mean=2.0, bar_mean=2.0, primary_big=None, busy_primary=1.0, alternate=False):
    for i, t in enumerate(TOPOS):
        for w in G.WINDOWS:
            for r in G.RUNGS:
                jit = 0.001 * (i + 1) * (-1 if alternate and i % 2 else 1)
                cell(d, "cd_ib0", 0, t, w, r, bar_mean, busy=0.1)
                cell(d, "cd", 0, t, w, r, bar_mean, busy=0.1)
                for sd in (1, 2):
                    cell(d, P, sd, t, w, r, primary_mean - jit, ck, lat_big=primary_big, busy=busy_primary)
                    cell(d, T, sd, t, w, r, twin_mean, ck)
                    cell(d, M, sd, t, w, r, mlp_mean, ck)


def run(d, cm, **kw):
    args = ["--gate", d, "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", "pin1234",
            "--primary", P, "--twin", T, "--mlp", M, "--checkpoint-md5", cm]
    for k, v in kw.items():
        args += [f"--{k}", v]
    return G.main(args)


def test_program_win_and_mp_attributed(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8, primary_big=1.6)
    res = run(d, cm)
    assert res["F1"]["label"] == "WIN" and res["F2"]["holm_family"] == 8
    assert res["F2"]["MP"] == "MP-ATTRIBUTED" and res["F2"]["MODEL-CLASS"].startswith("MODEL-CLASS-ATTRIBUTED")
    g = res["descriptive"]["gap_attribution"]["heavy"]
    assert g["label"] == "UNCUT-ATTRIBUTED" and g["share_gt10"] > 0.5


def test_ties(gate):
    d, ck, cm = gate
    fill(d, ck, 2.0, alternate=True)
    assert run(d, cm)["F1"]["label"] == "TIES"


def test_small_significant_gain_is_mixed(gate):
    d, ck, cm = gate
    fill(d, ck, 2.0)
    res = run(d, cm)
    assert res["F1"]["label"] == "MIXED" and "passing" in res["F1"]["why"]


def test_anchor_blocks_win(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8, busy_primary=0.0)
    assert run(d, cm)["F1"]["label"] == "F3 at k = 0.25 required"


def test_wrong_topo_hash_fails(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    with pytest.raises(SystemExit):
        G.main(["--gate", d, "--topos", TOPS, "--topos-sha256", "0" * 64, "--pin", "pin1234", "--primary", P, "--twin", T, "--checkpoint-md5", cm])


def test_missing_bsz_sidecar_fails(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    os.remove(os.path.join(d, f"cc40s1__g0heavy__{P}_k100_s1.tasks_bsz.npy"))
    with pytest.raises(SystemExit):
        run(d, cm)


def test_checkpoint_md5_mismatch_fails(gate, tmp_path):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    open(ck, "wb").write(b"other")
    with pytest.raises(SystemExit):
        run(d, cm)


def test_one_seed_missing_drops_the_cell(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    os.remove(os.path.join(d, f"cc40s1__g0heavy__{P}_k100_s2.summary.json"))
    os.remove(os.path.join(d, f"cc40s1__g0heavy__{P}_k100_s2.tasks_bsz.npy"))
    res = run(d, cm)
    assert "1g0" in res["health"][P]["heavy"]["missing"]


def test_one_collapsing_seed_counts(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    cell(d, P, 2, 1, "g0", "heavy", 9.0, ck)
    h = run(d, cm)["health"][P]["heavy"]
    assert h["per_seed"][2]["collapse_any"] == 1 and h["per_seed"][1]["collapse_any"] == 0 and h["collapse_any"] == 1


def test_unstamped_lock_row_fails(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    p = os.path.join(d, f"cc40s1__g0heavy__{P}_k100_s1.tasks_lock.npy")
    a = np.load(p)
    a[2, 2] = np.nan
    np.save(p, a)
    with pytest.raises(SystemExit, match="unstamped"):
        run(d, cm)


def test_missing_lock_sidecar_fails(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    os.remove(os.path.join(d, "cc40s1__g0heavy__cd_ib0_k100_s0.tasks_lock.npy"))
    with pytest.raises(SystemExit, match="lock-wait sidecar"):
        run(d, cm)


@pytest.mark.parametrize("line", ["[CODE] commit=deadbeef0000 branch=HEAD dirty=False", "[CODE] commit=pin1234abcd branch=HEAD dirty=True", "no banner"])
def test_log_banner_must_show_pin_clean(gate, line):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    with open(os.path.join(d, "cc40s1__g0heavy__cd_ib0_k100_s0.log"), "w") as fh:
        fh.write(line + "\n")
    with pytest.raises(SystemExit, match="log banner"):
        run(d, cm)


def test_ckpt_arm_maps_gate_names_to_training_log():
    assert G.ckpt_arm("ra_gnn_eng_arm1_nosplit") == "gnn_eng" and G.ckpt_arm("ra_mlp_same_arm1_nosplit") == "mlp_same"
    assert G.ckpt_arm("uncut_arm1_twin_eng") == "uncut_arm1_twin_eng"


def test_training_log_keys_and_merged_files(gate, tmp_path):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    md5 = json.load(open(cm))
    log = tmp_path / "log.json"
    log.write_text(json.dumps({k: v for k, v in md5.items() if not k.startswith(P)}))
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({k: v for k, v in md5.items() if k.startswith(P)}))
    assert run(d, f"{log},{ref}")["F1"]["label"] == "WIN"
    ref.write_text(json.dumps({f"{T}|s1": "0" * 32}))
    with pytest.raises(SystemExit, match="disagree"):
        run(d, f"{log},{ref}")
