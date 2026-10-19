import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_arm_gate_read as G  # noqa: E402

TOPOS = list(range(1, 13))
TOPS = ",".join(map(str, TOPOS))
P, T, M = "uncut_arm1_gnn_eng", "uncut_arm1_twin_eng", "ra_mlp_same"


def cell(d, arm, seed, t, w, r, mean, ckpt=None, lat_big=None, busy=1.0, warmth="node_disk_v2", commit="pin1234abcd", sfx="_k100", env=None, servers=None):
    name = f"cc40s{t}__{w}{r}__{arm}{sfx}_s{seed}"
    s = {"averageElapsedTime": mean, "steady_state": {"after_360s": {"mean": mean}}, "num_tasks": 4,
         "endTime": 100.0, "arrival_end": {"last_arrival_s": 99.0}, "code": {"commit": commit, "dirty": False},
         "schedulerCounters": {"sched_queue_wait_seconds": 0.0, "sched_busy_seconds": busy}, "env": {"HEROSIM_WARMTH_PHYSICS": warmth},
         "python_env": {"torch_num_threads": 1}, "host": {"cpu_model": "AMD EPYC 9335", "smt_control": "off", "nproc": 64}}
    if ckpt:
        s["env"]["GNN_MODEL_PATH"] = ckpt
    if servers is not None:
        s["servers"] = servers
    s["env"].update(env or {})
    with open(os.path.join(d, name + ".summary.json"), "w") as fh:
        json.dump(s, fh)
    big = mean if lat_big is None else lat_big
    rows = [(0, 10.0, 10.0 + mean, 2, 2), (1, 400.0, 400.0 + mean, 2, 2), (2, 20.0, 20.0 + big, 20, 20), (3, 500.0, 500.0 + big, 20, 20)]
    np.save(os.path.join(d, name + ".tasks_bsz.npy"), np.asarray(rows, dtype=np.float64))
    np.save(os.path.join(d, name + ".tasks_lock.npy"), np.asarray([(r[0], r[1], r[1] + 0.1, 0.1) for r in rows], dtype=np.float64))
    with open(os.path.join(d, name + ".log"), "w") as fh:
        fh.write(f"INFO start\n[CODE] commit={commit[:8]} branch=HEAD dirty=False diff=e3b0\n")


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
    kw.setdefault("cotenancy", "off")
    args = ["--gate", d, "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", kw.pop("pin", "pin1234"),
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


@pytest.mark.parametrize("warmth", ["platform_reuse_v1", None])
def test_warmth_physics_must_agree_across_bar_and_learned(gate, warmth):
    d, ck, cm = gate
    fill(d, ck, 2.0)
    assert run(d, cm)["warmth_physics"] == "node_disk_v2"
    cell(d, P, 1, TOPOS[0], G.WINDOWS[0], G.RUNGS[0], 2.0, ck, warmth=warmth)
    with pytest.raises(SystemExit, match="HEROSIM_WARMTH_PHYSICS"):
        run(d, cm)


W0 = {"w_ms_per_plan": {"cd_ib0": {"heavy": 1.0}}, "w_ms_per_plan_by_bucket": {"ra_gnn_eng_nosplit": {"heavy": {"le10": 6.0}}},
      "_source": "s", "_source_by_bucket": "b"}
GT = ("ra_gnn_eng_arm1_nosplit", "ra_twin_eng_arm1_nosplit")


def commit(repo, files, msg):
    for path, body in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(body if isinstance(body, str) else json.dumps(body, indent=1))
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", msg], check=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def w_with(base, kinds, by_bucket=False):
    w = json.loads(json.dumps(base))
    for k in kinds:
        w["w_ms_per_plan"][k] = {"heavy": 6.5, "moderate": 6.9}
        if by_bucket:
            w["w_ms_per_plan_by_bucket"][k] = {"heavy": {"le10": 6.5, "11_50": 2.0}}
        w[f"_source_{k}"] = "own le10, single cell"
    return w


@pytest.fixture
def chain(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    base = commit(repo, {G.W_FILE: W0, "scripts_cosim/gate.py": "x = 1\n"}, "base")
    w1 = commit(repo, {G.W_FILE: w_with(W0, GT, by_bucket=True)}, "gnn+twin w")
    w2 = commit(repo, {G.W_FILE: w_with(w_with(W0, GT, by_bucket=True), ("ra_mlp_same_arm1_nosplit",))}, "mlp w")
    return str(repo), base, w1, w2


def test_w_pin_chain_accepts_w_only_chain(chain):
    repo, base, w1, w2 = chain
    ws = G.w_pin_chain(repo, base[:8], [w1[:8], w2[:8]])
    assert list(ws) == [base, w1, w2] and "ra_mlp_same_arm1_nosplit" in ws[w2]["w_ms_per_plan"]


@pytest.mark.parametrize("bad, match", [
    ("other_file", "not only"), ("changed_key", "frozen"), ("recalibrated", "frozen"), ("foreign_key", "outside the learned arms"),
    ("foreign_top", "top-level"), ("source_edit", "outside the learned arms"), ("not_chain", "not an ancestor")])
def test_w_pin_chain_fails_loud(chain, bad, match):
    repo, base, w1, w2 = chain
    w = w_with(w_with(W0, GT, by_bucket=True), ("ra_mlp_same_arm1_nosplit",))
    pins = [w1]
    if bad == "other_file":
        pins.append(commit(__import__("pathlib").Path(repo), {G.W_FILE: w, "scripts_cosim/gate.py": "x = 2\n"}, "w + code"))
    elif bad == "changed_key":
        w["w_ms_per_plan"]["cd_ib0"]["heavy"] = 1.1
    elif bad == "recalibrated":
        w["w_ms_per_plan"]["ra_gnn_eng_arm1_nosplit"]["heavy"] = 6.6
    elif bad == "foreign_key":
        w["w_ms_per_plan"]["cd_new"] = {"heavy": 1.0}
    elif bad == "foreign_top":
        w["_note"] = "n"
    elif bad == "source_edit":
        w["_source_by_bucket"] = "b, edited"
    elif bad == "not_chain":
        pins = [w2, w1]
    if bad not in ("other_file", "not_chain"):
        pins.append(commit(__import__("pathlib").Path(repo), {G.W_FILE: w}, bad))
    with pytest.raises(SystemExit, match=match):
        G.w_pin_chain(repo, base, pins)


def test_learned_cells_at_w_pin_hand_cells_at_base(gate, chain):
    d, ck, cm = gate
    repo, base, w1, w2 = chain
    learned = {P, T, M}
    for i, t in enumerate(TOPOS):
        for w in G.WINDOWS:
            for r in G.RUNGS:
                for arm, sd in [("cd_ib0", 0), ("cd", 0), (P, 1), (P, 2), (T, 1), (T, 2), (M, 1), (M, 2)]:
                    cell(d, arm, sd, t, w, r, 2.0, ck if arm in learned else None, commit=(w1 if arm in learned else base))
    res = run(d, cm, pin=base[:8], repo=repo, **{"w-pins": f"{w1[:8]},{w2[:8]}"})
    assert set(res["w_pins"]) == {base, w1, w2}
    cell(d, "cd_ib0", 0, TOPOS[0], G.WINDOWS[0], G.RUNGS[0], 2.0, commit=w1)
    with pytest.raises(SystemExit, match="is not the gate pin"):
        run(d, cm, pin=base[:8], repo=repo, **{"w-pins": f"{w1[:8]},{w2[:8]}"})


def test_arm1_cell_needs_its_w_at_its_pin(gate, chain):
    d, ck, cm = gate
    repo, base, w1, w2 = chain
    cells = {("ra_mlp_same_arm1_nosplit", 1, 19537, "g0", "heavy"): {"code": {"commit": w1, "dirty": False}, "_path": "x.summary.json"}}
    with pytest.raises(SystemExit, match="no calibrated w"):
        G.asserts(cells, base, ["ra_mlp_same_arm1_nosplit"], {}, G.w_pin_chain(repo, base, [w1, w2]))


def fill_kr(d, ck, primary_mean, twin_mean=2.0, mlp_mean=2.0):
    for i, t in enumerate(TOPOS):
        for w in G.WINDOWS:
            for r in G.RUNGS:
                for sd in (1, 2):
                    cell(d, P, sd, t, w, r, primary_mean - 0.001 * (i + 1), ck, sfx="_kr")
                    cell(d, T, sd, t, w, r, twin_mean, ck, sfx="_kr")
                    cell(d, M, sd, t, w, r, mlp_mean, ck, sfx="_kr")


@pytest.mark.parametrize("kr_mean", [1.8, 2.0])
def test_cotenancy_verdict_must_also_win(gate, kr_mean):
    d, ck, cm = gate
    fill(d, ck, 1.8, primary_big=1.6)
    fill_kr(d, ck, kr_mean)
    res = run(d, cm, cotenancy="required")
    assert res["F1"]["label"] == "WIN"
    if kr_mean < 2:
        assert res["combined"]["F1"] == "WIN" and res["combined"]["flags"] == []
    else:
        assert not res["combined"]["F1"].startswith("WIN") and res["combined"]["F1"] == res["cotenancy"]["F1"]["label"]
        assert res["combined"]["flags"] == ["NOT-COTENANCY-ROBUST"] and res["combined"]["MP"] != "MP-ATTRIBUTED"


def test_cotenancy_anomaly_flagged_label_unchanged(gate):
    d, ck, cm = gate
    fill(d, ck, 2.0, alternate=True)
    fill_kr(d, ck, 1.8)
    res = run(d, cm, cotenancy="required")
    assert res["cotenancy"]["F1"]["label"].startswith("WIN") and not res["F1"]["label"].startswith("WIN")
    assert res["combined"]["F1"] == res["F1"]["label"] and res["combined"]["flags"] == ["COTENANCY-ANOMALY"]
    assert "COTENANCY-ANOMALY" in res["combined"]["why"]


F1_LABELS = ["WIN", "WIN-ROBUST", "WIN-OVER-BRITTLE-BAR", "WIN-ARM-ONLY", "WIN-ROBUST (ARM-ONLY)", "WIN-OVER-BRITTLE-BAR (ARM-ONLY)",
             "F3 at k = 0.25 required", "MIXED", "TIES", "E-FASTER"]
WIN_AXES = {"WIN": (1, 0), "WIN-ROBUST": (1, 1), "WIN-OVER-BRITTLE-BAR": (1, 2), "WIN-ARM-ONLY": (0, 0), "WIN-ROBUST (ARM-ONLY)": (0, 1),
            "WIN-OVER-BRITTLE-BAR (ARM-ONLY)": (0, 2), "WIN-ARM-ONLY; ROBUST": (0, 1), "WIN-ARM-ONLY; OVER-BRITTLE-BAR": (0, 2)}
CANON = {(1, 0): "WIN", (1, 1): "WIN-ROBUST", (1, 2): "WIN-OVER-BRITTLE-BAR", (0, 0): "WIN-ARM-ONLY", (0, 1): "WIN-ARM-ONLY; ROBUST",
         (0, 2): "WIN-ARM-ONLY; OVER-BRITTLE-BAR"}


@pytest.mark.parametrize("x", F1_LABELS)
@pytest.mark.parametrize("y", F1_LABELS)
def test_weaker_f1_every_label_pair(x, y):
    got = G.weaker_f1(x, y)
    assert got == G.weaker_f1(y, x) or (x not in WIN_AXES and y not in WIN_AXES and x == y)
    if x in WIN_AXES and y in WIN_AXES:
        (px, qx), (py, qy) = WIN_AXES[x], WIN_AXES[y]
        assert got == CANON[(min(px, py), max(qx, qy))]
    elif x in WIN_AXES or y in WIN_AXES:
        assert got == (y if x in WIN_AXES else x)
    else:
        assert got == G.NON_WIN[max(G.NON_WIN.index(x), G.NON_WIN.index(y))]
    assert G.prog_win(got) == (got in ("WIN", "WIN-ROBUST"))


def test_brittle_bar_qualifier_survives_arm_only():
    assert G.weaker_f1("WIN-OVER-BRITTLE-BAR", "WIN-ARM-ONLY") == "WIN-ARM-ONLY; OVER-BRITTLE-BAR"
    assert G.weaker_f1("WIN-ARM-ONLY; OVER-BRITTLE-BAR", "WIN") == "WIN-ARM-ONLY; OVER-BRITTLE-BAR"
    with pytest.raises(SystemExit, match="unknown F1 label"):
        G.weaker_f1("WIN-SOMETHING", "WIN")


def test_cotenancy_cells_required_and_learned_only(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    with pytest.raises(SystemExit, match="no co-tenancy"):
        run(d, cm, cotenancy="required")
    cell(d, "cd_ib0", 0, TOPOS[0], G.WINDOWS[0], G.RUNGS[0], 2.0, sfx="_kr")
    with pytest.raises(SystemExit, match="not of a learned arm"):
        run(d, cm, cotenancy="required")


GATE = G.GATE_CHANGE_FILES[0]


@pytest.fixture
def gchain(tmp_path):
    from pathlib import Path
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    base = commit(repo, {G.W_FILE: W0, GATE: "x = 1\n", G.GATE_CHANGE_FILES[1]: "t = 1\n", "other.py": "o = 1\n"}, "base")
    gc = commit(repo, {GATE: "x = 2\n", G.GATE_CHANGE_FILES[1]: "t = 2\n"}, "kr tag")
    w1 = commit(repo, {G.W_FILE: {**w_with(W0, GT, by_bucket=True), "k_cotenancy": 1.3, "_source_k_cotenancy": "r"}}, "gnn+twin w, r")
    w2 = commit(repo, {G.W_FILE: {**w_with(w_with(W0, GT, by_bucket=True), ("ra_mlp_same_arm1_nosplit",)), "k_cotenancy": 1.3,
                                  "_source_k_cotenancy": "r"}}, "mlp w")
    return Path(repo), base, gc, w1, w2


def test_gate_change_chain_accepts(gchain):
    repo, base, gc, w1, w2 = gchain
    ws = G.w_pin_chain(str(repo), base, [w1, w2], gc)
    assert list(ws) == [base, gc, w1, w2] and ws[w2]["k_cotenancy"] == 1.3


@pytest.mark.parametrize("bad", ["gate_touches_other", "w_pin_touches_gate", "k_below_one", "k_changed", "gc_not_child"])
def test_gate_change_chain_fails_loud(gchain, bad):
    repo, base, gc, w1, w2 = gchain
    pins, g = [w1, w2], gc
    w2doc = json.loads(subprocess.run(["git", "-C", str(repo), "show", f"{w2}:{G.W_FILE}"], capture_output=True, text=True).stdout)
    if bad == "gate_touches_other":
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", base], check=True)
        g = commit(repo, {GATE: "x = 2\n", G.GATE_CHANGE_FILES[1]: "t = 2\n", "other.py": "o = 2\n"}, "bad gc")
        pins = [commit(repo, {G.W_FILE: w2doc}, "w")]
        match = "not exactly"
    elif bad == "w_pin_touches_gate":
        pins = [w1, commit(repo, {G.W_FILE: {**w2doc, "_source_ra_mlp_same_arm1_nosplit": "x"}, GATE: "x = 3\n"}, "w + gate")]
        match = "not only"
    elif bad == "k_below_one":
        pins = [w1, commit(repo, {G.W_FILE: {**w2doc, "k_cotenancy": 0.9}}, "k<1")]
        match = "frozen|k_cotenancy"
    elif bad == "k_changed":
        pins = [w1, commit(repo, {G.W_FILE: {**w2doc, "k_cotenancy": 1.4}}, "k changed")]
        match = "changes or adds"
    else:
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", gc], check=True)
        mid = commit(repo, {"other.py": "o = 9\n"}, "between")
        g = commit(repo, {GATE: "x = 4\n"}, "gc on top")
        pins = [commit(repo, {G.W_FILE: w2doc}, "w")]
        match = "not a child"
    with pytest.raises(SystemExit, match=match):
        G.w_pin_chain(str(repo), base, pins, g)


def test_charge_check_k_times_frozen_w():
    w = w_with(W0, GT, by_bucket=True)
    key = ("ra_gnn_eng_arm1_nosplit", 1, 19537, "g0", "heavy")
    ok = {"HEROSIM_DECISION_COST_MS_BY_BUCKET": json.dumps({"le10": 6.5 * 1.3, "11_50": 2.0 * 1.3})}
    G.charge_check(key, {"env": ok}, w, 1.3)
    with pytest.raises(SystemExit, match="not k = 1.0"):
        G.charge_check(key, {"env": ok}, w, 1.0)
    with pytest.raises(SystemExit, match="served decision cost"):
        G.charge_check(("ra_gnn_eng_arm1_nosplit", 1, 19537, "g0", "moderate"), {"env": {"HEROSIM_DECISION_COST_MS": "6.0"}}, w, 1.0)
    G.charge_check(("ra_gnn_eng_arm1_nosplit", 1, 19537, "g0", "moderate"), {"env": {"HEROSIM_DECISION_COST_MS": "6.9"}}, w, 1.0)


B1, B1T, A1 = "ra_gnn_eng_arm1b_nosplit", "ra_twin_eng_arm1b_nosplit", "ra_gnn_eng_arm1_nosplit"
B1M, A1M = "ra_mlp_same_arm1b_nosplit", "ra_mlp_same_arm1_nosplit"


def w1b(w, x=1.2, buckets=True):
    w = json.loads(json.dumps(w))
    for a in ("gnn_eng", "twin_eng"):
        src, k = f"ra_{a}_arm1_nosplit", f"ra_{a}_arm1b_nosplit"
        w["w_ms_per_plan"][k] = {r: v * x for r, v in w["w_ms_per_plan"][src].items()}
        if buckets and src in w["w_ms_per_plan_by_bucket"]:
            w["w_ms_per_plan_by_bucket"][k] = {r: {b: v * x for b, v in bs.items()} for r, bs in w["w_ms_per_plan_by_bucket"][src].items()}
        w[f"_source_{k}"] = "own le10, exclusive"
    return w


@pytest.mark.parametrize("x, ok", [(1.2, True), (1.5, True), (1 / 1.5, True), (1.51, False), (0.6, False)])
def test_arm1b_w_guard(x, ok):
    w = w1b(w_with(W0, GT, by_bucket=True), x)
    if ok:
        G.w_guard(w, "sha")
    else:
        with pytest.raises(SystemExit, match="stop"):
            G.w_guard(w, "sha")


def test_arm1b_w_guard_needs_arm1_and_matching_buckets():
    w = w1b(w_with(W0, GT, by_bucket=True))
    G.w_guard(w, "sha")
    w["w_ms_per_plan_by_bucket"][B1]["heavy"]["le10"] *= 1.6
    with pytest.raises(SystemExit, match="stop"):
        G.w_guard(w, "sha")
    with pytest.raises(SystemExit, match="buckets"):
        G.w_guard(w1b(w_with(W0, GT, by_bucket=True), buckets=False), "sha")
    lone = w1b(w_with(W0, GT, by_bucket=True))
    del lone["w_ms_per_plan"]["ra_twin_eng_arm1_nosplit"]
    with pytest.raises(SystemExit, match="without Arm 1"):
        G.w_guard(lone, "sha")


def test_two_gate_changes_then_w_pins_with_arm1b(gchain):
    repo, base, gc, _, _ = gchain
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "b1", gc], check=True)
    gc1b = commit(repo, {GATE: "x = 3\n", G.GATE_CHANGE_FILES[1]: "t = 3\n"}, "arm1b kinds")
    w1 = commit(repo, {G.W_FILE: {**w1b(w_with(W0, GT, by_bucket=True)), "k_cotenancy": 1.4, "_source_k_cotenancy": "r"}}, "w1")
    ws = G.w_pin_chain(str(repo), base, [w1], f"{gc},{gc1b}")
    assert list(ws) == [base, gc, gc1b, w1] and B1 in ws[w1]["w_ms_per_plan"]
    with pytest.raises(SystemExit, match="not a child"):
        G.w_pin_chain(str(repo), base, [w1], gc1b)
    bad = commit(repo, {G.W_FILE: {**w1b(w_with(W0, GT, by_bucket=True), 1.6), "k_cotenancy": 1.4, "_source_k_cotenancy": "r"}}, "w far")
    with pytest.raises(SystemExit, match="changes frozen|stop"):
        G.w_pin_chain(str(repo), base, [w1, bad], f"{gc},{gc1b}")


def test_arm1b_md5_namespace(tmp_path):
    assert G.ckpt_tag(B1) == "arm1b" and G.ckpt_tag(B1T) == "arm1b" and G.ckpt_tag(A1) == "" and G.ckpt_arm(B1) == "gnn_eng"
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"gnn_eng|s1": "x"}))
    b.write_text(json.dumps({"gnn_eng|s1": "y"}))
    assert G.ckpt_md5s(f"{a},arm1b={b}") == {"gnn_eng|s1": "x", "gnn_eng@arm1b|s1": "y"}
    with pytest.raises(SystemExit, match="disagree"):
        G.ckpt_md5s(f"{a},{b}")


def fill1b(d, ck, b_mean, a_mean=1.9):
    for i, t in enumerate(TOPOS):
        for w in G.WINDOWS:
            for r in G.RUNGS:
                cell(d, "cd_ib0", 0, t, w, r, 2.0, busy=0.1)
                cell(d, "cd", 0, t, w, r, 2.0, busy=0.1)
                for sfx in ("_k100", "_kr"):
                    for sd in (1, 2):
                        cell(d, B1, sd, t, w, r, b_mean - 0.001 * (i + 1), ck, sfx=sfx)
                        cell(d, B1T, sd, t, w, r, 2.0, ck, sfx=sfx)
                        cell(d, A1, sd, t, w, r, a_mean, ck, sfx=sfx)


def run1b(d, cm1, cm1b, after, **kw):
    args = ["--gate", d, "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", "pin1234",
            "--primary", B1, "--twin", B1T, "--checkpoint-md5", f"{cm1},arm1b={cm1b}"]
    if after:
        args += ["--after-read", after]
    kw.setdefault("compare", A1)
    for k, v in kw.items():
        args += [f"--{k}", v]
    return G.main(args)


@pytest.fixture
def arm1b(tmp_path, gate):
    d, ck, _ = gate
    m = hashlib.md5(b"weights").hexdigest()
    cm1, cm1b = tmp_path / "ck1.json", tmp_path / "ck1b.json"
    cm1.write_text(json.dumps({f"{a}|s{s}": m for a in ("gnn_eng", "twin_eng") for s in (1, 2)}))
    cm1b.write_text(json.dumps({f"{a}|s{s}": m for a in ("gnn_eng", "twin_eng") for s in (1, 2)}))
    fill1b(d, ck, 1.8)

    def after(label, primary=A1):
        p = tmp_path / f"arm1_{len(label)}_{primary}.json"
        p.write_text(json.dumps({"primary": primary, "combined": {"F1": label}}))
        return str(p)
    return d, str(cm1), str(cm1b), after


@pytest.mark.parametrize("arm1_label, want", [("WIN", "WIN"), ("WIN-ROBUST", "WIN"), ("WIN-ARM-ONLY", "WIN-ARM-ONLY"),
                                               ("WIN-OVER-BRITTLE-BAR", "WIN-ARM-ONLY"), ("TIES", "WIN-ARM-ONLY")])
def test_arm1b_fixed_sequence(arm1b, arm1_label, want):
    d, cm1, cm1b, after = arm1b
    res = run1b(d, cm1, cm1b, after(arm1_label))
    assert res["F2"]["holm_family"] == 4 and res["sequence"]["program_label_open"] == G.prog_win(arm1_label)
    assert res["F1"]["label"] == want and res["cotenancy"]["F1"]["label"] == want and res["combined"]["F1"] == want
    if want != "WIN":
        assert "fixed sequence" in res["F1"]["why"] and res["combined"]["MP"] != "MP-ATTRIBUTED"


def test_arm1b_direction_only_vs_arm1(arm1b):
    d, cm1, cm1b, after = arm1b
    res = run1b(d, cm1, cm1b, after("WIN"), compare=A1)
    for v in (res, res["cotenancy"]):
        dv = v["descriptive"]["direction_vs_compare"]
        assert len(dv) == 4 and all(e["direction"] == "a faster" and e["a_faster"] == 12 for e in dv)
        assert all("median_pct" not in e and "p_holm" not in e for e in dv)


def test_arm1b_needs_arm1_read_first_and_its_own_md5(arm1b, tmp_path):
    d, cm1, cm1b, after = arm1b
    with pytest.raises(SystemExit, match="after-read"):
        run1b(d, cm1, cm1b, "")
    with pytest.raises(SystemExit, match="not Arm 1's gate read"):
        run1b(d, cm1, cm1b, after("WIN", primary="ra_gnn_eng_arm2_nosplit"))
    with pytest.raises(SystemExit, match="both verdicts"):
        run1b(d, cm1, cm1b, after("WIN"), cotenancy="off")
    args = ["--gate", d, "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", "pin1234",
            "--primary", B1, "--twin", B1T, "--checkpoint-md5", cm1, "--after-read", after("WIN"), "--compare", A1]
    with pytest.raises(SystemExit, match="md5 does not match"):
        G.main(args)
    with pytest.raises(SystemExit, match="only with it"):
        run(d, cm1, **{"after-read": after("WIN")})


@pytest.mark.parametrize("bad", ["no_host", "threads"])
def test_learned_cells_need_host_block_and_one_thread(gate, bad):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    p = os.path.join(d, f"cc40s1__g0heavy__{P}_k100_s1.summary.json")
    s = json.load(open(p))
    if bad == "no_host":
        s.pop("host")
    else:
        s["python_env"]["torch_num_threads"] = 4
    json.dump(s, open(p, "w"))
    with pytest.raises(SystemExit, match="host block and 1 torch thread"):
        run(d, cm)


def test_hosts_reported_per_arm(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    p = os.path.join(d, "cc40s1__g0heavy__cd_ib0_k100_s0.summary.json")
    s = json.load(open(p))
    s.pop("host")
    json.dump(s, open(p, "w"))
    h = run(d, cm)["hosts"]
    assert h[P] == {"AMD EPYC 9335 | smt off | nproc 64": 192} and h["cd_ib0"]["unrecorded | smt ? | nproc ?"] == 1


def w1b_mlp(w, x=1.2):
    w = w1b(w, 1.2)  # gnn and twin inside the guard; only the mlp's ratio is varied
    w["w_ms_per_plan"][B1M] = {r: v * x for r, v in w["w_ms_per_plan"][A1M].items()}
    if A1M in w.get("w_ms_per_plan_by_bucket", {}):
        w["w_ms_per_plan_by_bucket"][B1M] = {r: {b: v * x for b, v in bs.items()} for r, bs in w["w_ms_per_plan_by_bucket"][A1M].items()}
    w[f"_source_{B1M}"] = "own le10, exclusive"
    return w


@pytest.mark.parametrize("x, ok", [(1.2, True), (1.5, True), (1 / 1.5, True), (1.51, False), (0.6, False)])
def test_arm1b_mlp_w_guard_is_against_arm1s_mlp(x, ok):
    w = w1b_mlp(w_with(W0, list(GT) + [A1M], by_bucket=True), x)
    assert B1M in G.ARM1B_W_KINDS and B1M in G.LEARNED_W_KINDS
    if ok:
        G.w_guard(w, "sha")
    else:
        with pytest.raises(SystemExit, match="stop"):
            G.w_guard(w, "sha")
    del w["w_ms_per_plan"][A1M]
    with pytest.raises(SystemExit, match="without Arm 1"):
        G.w_guard(w, "sha")


def test_arm1b_without_an_mlp_w_is_guarded_as_before():
    w = w1b(w_with(W0, GT, by_bucket=True))
    assert B1M not in w["w_ms_per_plan"]
    G.w_guard(w, "sha")


@pytest.fixture
def arm1b_mlp(tmp_path, gate, arm1b):
    d, ck, _ = gate
    _, cm1, _, after = arm1b
    for t in TOPOS:
        for w in G.WINDOWS:
            for r in G.RUNGS:
                for sfx in ("_k100", "_kr"):
                    for sd in (1, 2):
                        cell(d, B1M, sd, t, w, r, 2.0, ck, sfx=sfx)
    m = hashlib.md5(b"weights").hexdigest()
    cm = tmp_path / "ck1b_mlp.json"
    cm.write_text(json.dumps({f"{a}|s{s}": m for a in ("gnn_eng", "twin_eng", "mlp_same") for s in (1, 2)}))
    return d, cm1, str(cm), after


def test_arm1b_mlp_read_is_a_holm_family_of_eight(arm1b_mlp):
    d, cm1, cm1b, after = arm1b_mlp
    res = run1b(d, cm1, cm1b, after("WIN"), mlp=B1M)
    assert res["F2"]["holm_family"] == 8 and "MODEL-CLASS" in res["F2"]
    assert res["combined"]["MODEL-CLASS"].startswith("MODEL-CLASS")


def test_arm1b_mlp_flag_is_the_same_trainers_mlp(arm1b_mlp):
    d, cm1, cm1b, after = arm1b_mlp
    with pytest.raises(SystemExit, match="same trainer"):
        run1b(d, cm1, cm1b, after("WIN"), mlp=A1M)
    with pytest.raises(SystemExit, match="same trainer"):
        run(d, cm1, mlp=B1M)


C1, C1T = "ra_gnn_eng_arm3_nosplit", "ra_twin_eng_arm3_nosplit"


def w3(w, x=1.2, buckets=True):
    """Arm 3's w on top of w1b's: gnn_eng and twin_eng, each x times Arm 1b's."""
    w = w1b(w, 1.2, buckets)
    for a in ("gnn_eng", "twin_eng"):
        src, k = f"ra_{a}_arm1b_nosplit", f"ra_{a}_arm3_nosplit"
        w["w_ms_per_plan"][k] = {r: v * x for r, v in w["w_ms_per_plan"][src].items()}
        if buckets and src in w.get("w_ms_per_plan_by_bucket", {}):
            w["w_ms_per_plan_by_bucket"][k] = {r: {b: v * x for b, v in bs.items()} for r, bs in w["w_ms_per_plan_by_bucket"][src].items()}
        w[f"_source_{k}"] = "own le10, exclusive"
    return w


@pytest.mark.parametrize("x, ok", [(1.2, True), (1.5, True), (1 / 1.5, True), (1.51, False), (0.6, False)])
def test_arm3_w_guard_is_against_arm1b(x, ok):
    w = w3(w_with(W0, GT, by_bucket=True), x)
    if ok:
        G.w_guard(w, "sha")
    else:
        with pytest.raises(SystemExit, match="Arm 3 w .*Arm 1b's"):
            G.w_guard(w, "sha")
    lone = w3(w_with(W0, GT, by_bucket=True))
    del lone["w_ms_per_plan"]["ra_twin_eng_arm1b_nosplit"]
    with pytest.raises(SystemExit, match="without Arm 1b's"):
        G.w_guard(lone, "sha")


def test_arm3_md5_namespace_and_w_kinds(tmp_path):
    assert G.ckpt_tag(C1) == "arm3" and G.ckpt_tag(C1T) == "arm3" and G.ckpt_tag(B1) == "arm1b" and G.ckpt_tag(A1) == "" and G.ckpt_arm(C1) == "gnn_eng"
    assert set(G.ARM3_W_KINDS) == {C1, C1T} and set(G.ARM3_W_KINDS) <= set(G.LEARNED_W_KINDS)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"gnn_eng|s1": "x"}))
    b.write_text(json.dumps({"gnn_eng|s1": "y"}))
    assert G.ckpt_md5s(f"{a},arm3={b}") == {"gnn_eng|s1": "x", "gnn_eng@arm3|s1": "y"}


def fill3(d, ck, c_mean, b_mean=1.9):
    for i, t in enumerate(TOPOS):
        for w in G.WINDOWS:
            for r in G.RUNGS:
                cell(d, "cd_ib0", 0, t, w, r, 2.0, busy=0.1)
                cell(d, "cd", 0, t, w, r, 2.0, busy=0.1)
                for sfx in ("_k100", "_kr"):
                    for sd in (1, 2):
                        cell(d, C1, sd, t, w, r, c_mean - 0.001 * (i + 1), ck, sfx=sfx)
                        cell(d, C1T, sd, t, w, r, 2.0, ck, sfx=sfx)
                        cell(d, B1, sd, t, w, r, b_mean, ck, sfx=sfx)


@pytest.fixture
def arm3(tmp_path, gate):
    d, ck, _ = gate
    m = hashlib.md5(b"weights").hexdigest()
    cm1b, cm3 = tmp_path / "ck1b.json", tmp_path / "ck3.json"
    for f in (cm1b, cm3):
        f.write_text(json.dumps({f"{a}|s{s}": m for a in ("gnn_eng", "twin_eng") for s in (1, 2)}))
    fill3(d, ck, 1.8)

    def after(label, primary=B1):
        p = tmp_path / f"arm1b_{len(label)}_{primary}.json"
        p.write_text(json.dumps({"primary": primary, "combined": {"F1": label}}))
        return str(p)
    return d, str(cm1b), str(cm3), after


def run3(d, cm1b, cm3, after, **kw):
    args = ["--gate", d, "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", "pin1234",
            "--primary", C1, "--twin", C1T, "--checkpoint-md5", f"arm1b={cm1b},arm3={cm3}"]
    if after:
        args += ["--after-read", after]
    kw.setdefault("compare", B1)
    for k, v in kw.items():
        args += [f"--{k}", v]
    return G.main(args)


@pytest.mark.parametrize("arm1b_label", ["WIN", "WIN-ROBUST", "WIN-ARM-ONLY", "TIES"])
def test_arm3_is_never_program_level(arm3, arm1b_label):
    """B6: the lever was conceived after interim looks and gets no share of the program alpha: at best WIN-ARM-ONLY, whatever Arm 1b read."""
    d, cm1b, cm3, after = arm3
    res = run3(d, cm1b, cm3, after(arm1b_label))
    assert res["sequence"]["program_label_open"] is False and res["sequence"]["arm1b_combined_F1"] == arm1b_label
    assert res["F2"]["holm_family"] == 4
    assert res["F1"]["label"] == "WIN-ARM-ONLY" and res["cotenancy"]["F1"]["label"] == "WIN-ARM-ONLY" and res["combined"]["F1"] == "WIN-ARM-ONLY"
    assert res["combined"]["MP"] != "MP-ATTRIBUTED"


def test_arm3_needs_the_arm1b_read_first_no_mlp_and_both_verdicts(arm3):
    d, cm1b, cm3, after = arm3
    with pytest.raises(SystemExit, match="after-read"):
        run3(d, cm1b, cm3, "")
    with pytest.raises(SystemExit, match="not Arm 1b's gate read"):
        run3(d, cm1b, cm3, after("WIN", primary=A1))
    with pytest.raises(SystemExit, match="both verdicts"):
        run3(d, cm1b, cm3, after("WIN"), cotenancy="off")
    with pytest.raises(SystemExit, match="no MLP"):
        run3(d, cm1b, cm3, after("WIN"), mlp=B1M)
    with pytest.raises(SystemExit, match="one trainer's"):
        run3(d, cm1b, cm3, after("WIN"), twin=B1T)


def test_arm3_without_a_twin_skips_f2_and_says_so(arm3):
    """No MP-OFF twin is trained (wave 2 held): F1 against the bar is read as filed, F2 is skipped and named, no twin cells are needed."""
    d, cm1b, cm3, after = arm3
    for f in os.listdir(d):
        if C1T in f:
            os.remove(os.path.join(d, f))
    res = run3(d, cm1b, cm3, after("WIN"), twin="")
    assert res["twin"] == "" and C1T not in res["health"] and C1T not in res["decision_share_pct"]
    assert res["F2"] == {"holm_family": 0, "tests": [], "MP": G.NO_TWIN} and res["cotenancy"]["F2"]["MP"] == G.NO_TWIN
    assert res["combined"]["MP"] == G.NO_TWIN and "no MP attribution" in G.NO_TWIN
    assert res["F1"]["label"] == "WIN-ARM-ONLY" and res["combined"]["F1"] == "WIN-ARM-ONLY"
    assert len(res["F1"]["tests"]) == 4 and res["sequence"]["program_label_open"] is False


def test_only_an_arm3_read_may_go_without_a_twin(arm1b):
    d, cm1, cm1b, after = arm1b
    with pytest.raises(SystemExit, match="--twin is required"):
        run1b(d, cm1, cm1b, after("WIN"), twin="")


SIDE, T2 = "scripts_cosim/joint_burst_v2_sidecheck.py", "tests/test_arm2_gate.py"
A2 = {"ra_gnn_eng_arm2_nosplit": {"moderate": 0.7, "heavy": 0.7}}


def test_g4_file_set_is_admitted_only_for_the_g4_commit_and_arm2_w_only_on_its_chain(gchain, monkeypatch):
    repo, base, _, _, _ = gchain
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "g4", base], check=True)
    g4 = commit(repo, {GATE: "x = 7\n", SIDE: "s = 1\n", T2: "t2 = 1\n"}, "G4-like: gate, sidecheck, its test")
    w = commit(repo, {G.W_FILE: {**W0, "w_ms_per_plan": {**W0["w_ms_per_plan"], **A2}, "_source_ra_gnn_eng_arm2_nosplit": "x"}}, "arm2 w")
    with pytest.raises(SystemExit, match="not exactly"):
        G.w_pin_chain(str(repo), base, [w], g4)  # not the G4 commit: the exact two-file rule
    monkeypatch.setattr(G, "G4", g4)
    monkeypatch.setattr(G, "GATE_CHANGE_FILES_BY_COMMIT", {g4: (GATE, SIDE, T2)})
    ws = G.w_pin_chain(str(repo), base, [w], g4)
    assert "ra_gnn_eng_arm2_nosplit" in ws[w]["w_ms_per_plan"]
    # the same Arm 2 w key on a chain without G4 is refused
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "nog4", base], check=True)
    gc = commit(repo, {GATE: "x = 8\n", G.GATE_CHANGE_FILES[1]: "t = 8\n"}, "plain gate change")
    w2 = commit(repo, {G.W_FILE: {**W0, "w_ms_per_plan": {**W0["w_ms_per_plan"], **A2}, "_source_ra_gnn_eng_arm2_nosplit": "x"}}, "arm2 w off-chain")
    with pytest.raises(SystemExit, match="outside the learned arms|adds top-level"):
        G.w_pin_chain(str(repo), base, [w2], gc)
    # and the G4 file set on any other commit still fails
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "other", base], check=True)
    g = commit(repo, {GATE: "x = 9\n", SIDE: "s = 2\n", T2: "t2 = 2\n"}, "not G4")
    with pytest.raises(SystemExit, match="not exactly"):
        G.w_pin_chain(str(repo), base, [], g)


def test_ckpt_arm_maps_arm2_and_k0():
    assert G.ckpt_arm("ra_gnn_eng_arm2_nosplit") == "gnn_eng" and G.ckpt_arm("ra_twin_eng_arm2k0_nosplit") == "twin_eng"
    assert G.ckpt_tag("ra_gnn_eng_arm2_nosplit") == "" and G.G4 in G.GATE_CHANGE_FILES_BY_COMMIT and set(G.GATE_CHANGE_FILES_BY_COMMIT) == {G.G4, G.G8D, G.P2}


def fill_seeds(d, ck, seeds, primary_mean, twin_mean=2.0):
    for t in TOPOS:
        for w in G.WINDOWS:
            for r in G.RUNGS:
                cell(d, "cd_ib0", 0, t, w, r, 2.0, busy=0.1)
                cell(d, "cd", 0, t, w, r, 2.0, busy=0.1)
                for sd in seeds:
                    cell(d, P, sd, t, w, r, primary_mean - 0.001 * t, ck)
                    cell(d, T, sd, t, w, r, twin_mean, ck)


def seeds_md5(tmp_path, seeds):
    cm = tmp_path / "ck_seeds.json"
    cm.write_text(json.dumps({f"{a}|s{s}": hashlib.md5(b"weights").hexdigest() for a in (P, T) for s in seeds}))
    return str(cm)


def seeds_run(d, cm, seeds=None):
    args = ["--gate", d, "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", "pin1234",
            "--primary", P, "--twin", T, "--checkpoint-md5", cm, "--cotenancy", "off"]
    return G.main(args + (["--seeds", seeds] if seeds is not None else []))


def test_seeds_flag_reads_the_extension_seeds_and_records_them(gate, tmp_path):
    d, ck, _ = gate
    fill_seeds(d, ck, (11, 12, 13), 1.8)
    res = seeds_run(d, seeds_md5(tmp_path, (11, 12, 13)), "11,12,13")
    assert res["seeds"] == [11, 12, 13] and set(res["F1"]["seeds_alone"]) == {11, 12, 13}
    assert res["F1"]["tests"][0]["a_seeds"] == [11, 12, 13] and res["F1"]["tests"][0]["n_topologies"] == 12
    assert set(res["health"][P]["heavy"]["per_seed"]) == {11, 12, 13}


def test_default_seeds_add_no_key(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    res = run(d, cm)
    assert "seeds" not in res and set(res["F1"]["seeds_alone"]) == {1, 2}


def test_learned_cell_outside_seeds_fails_loud(gate, tmp_path):
    d, ck, _ = gate
    fill_seeds(d, ck, (1, 2, 11), 1.8)
    with pytest.raises(SystemExit, match="outside --seeds"):
        seeds_run(d, seeds_md5(tmp_path, (1, 2, 11)), "11")
    with pytest.raises(SystemExit, match="outside --seeds"):
        seeds_run(d, seeds_md5(tmp_path, (1, 2, 11)))


@pytest.mark.parametrize("bad", ["", "0", "11,11", "x"])
def test_bad_seeds_fail_loud(gate, bad):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    with pytest.raises(SystemExit, match="--seeds"):
        seeds_run(d, cm, bad)


def test_extension_read_may_omit_the_twin_but_the_filed_read_may_not(gate, tmp_path):
    d, ck, _ = gate
    fill_seeds(d, ck, (11, 12), 1.8)
    cm = seeds_md5(tmp_path, (11, 12))
    args = ["--gate", d, "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", "pin1234",
            "--primary", P, "--checkpoint-md5", cm, "--cotenancy", "off"]
    for f in os.listdir(d):
        if f"__{T}_" in f:
            os.remove(os.path.join(d, f))
    res = G.main(args + ["--seeds", "11,12"])
    assert res["F2"]["MP"] == G.NO_TWIN_EXT and res["F2"]["holm_family"] == 0


def test_filed_read_still_needs_the_twin(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8)
    args = ["--gate", d, "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", "pin1234",
            "--primary", P, "--checkpoint-md5", cm, "--cotenancy", "off"]
    with pytest.raises(SystemExit, match="--twin is required"):
        G.main(args)


def test_bar_flag_reads_against_the_inflight_bar(gate):
    d, ck, cm = gate
    fill(d, ck, 1.8, bar_mean=3.0)
    for t in TOPOS:
        for w in G.WINDOWS:
            for r in G.RUNGS:
                cell(d, "cd_ib0_inflightfull", 0, t, w, r, 2.0, busy=0.1)
    res = run(d, cm, bar="cd_ib0_inflightfull")
    assert res["bar"] == "cd_ib0_inflightfull" and res["F1"]["tests"][0]["b"] == "cd_ib0_inflightfull"
    assert -11.0 < res["F1"]["tests"][0]["median_pct"] < -9.0  # 1.8 vs 2.0, not vs cd_ib0's 3.0
    assert "bar" not in run(d, cm)  # default: the filed bar, no new key
    with pytest.raises(SystemExit, match="--bar"):
        run(d, cm, bar="cd")


def test_comb_kinds_are_w_charged_and_may_be_added_by_a_w_only_pin(tmp_path):
    assert set(G.COMB_W_KINDS) <= set(G.LEARNED_W_KINDS)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    base = commit(repo, {G.W_FILE: W0, "scripts_cosim/gate.py": "x = 1\n"}, "base")
    w1 = commit(repo, {G.W_FILE: w_with(W0, ("ra_gnn_eng_combfl_nosplit", "ra_gnn_eng_a3lft_nosplit"))}, "comb w")
    ws = G.w_pin_chain(str(repo), base[:8], [w1[:8]])
    assert "ra_gnn_eng_combfl_nosplit" in ws[w1]["w_ms_per_plan"]
    cells = {("ra_gnn_eng_combf_nosplit", 1, 19537, "g0", "heavy"): {"code": {"commit": w1, "dirty": False}, "_path": "x.summary.json"}}
    with pytest.raises(SystemExit, match="no calibrated w"):
        G.asserts(cells, base, ["ra_gnn_eng_combf_nosplit"], {}, ws)


def test_other_learned_kinds_in_gate_are_checked_at_their_w_pin_and_left_out_of_the_read(tmp_path):
    """FL read from a directory that also holds F (both at the w-only pin) writes the same bytes as from an FL-only directory; F's cells
    still pass the learned-cell asserts (a wrong checkpoint or a failed F cell fails loud) instead of being held to --pin as hand cells."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    fl, f = "ra_gnn_eng_combfl_nosplit", "ra_gnn_eng_combf_nosplit"
    base = commit(repo, {G.W_FILE: W0, "scripts_cosim/gate.py": "x = 1\n"}, "base")
    w1 = commit(repo, {G.W_FILE: w_with(W0, (fl, f))}, "comb w")
    ck = tmp_path / "m.pt"
    ck.write_bytes(b"weights")
    cm = tmp_path / "ck.json"
    cm.write_text(json.dumps({f"{k}|s{s}": hashlib.md5(b"weights").hexdigest() for k in (fl, f) for s in (1, 2)}))
    only, both = tmp_path / "only", tmp_path / "both"
    only.mkdir()
    both.mkdir()
    for d in (only, both):
        for t in TOPOS:
            for w in G.WINDOWS:
                for r in G.RUNGS:
                    cell(str(d), "cd_ib0", 0, t, w, r, 2.0, busy=0.1, commit=base)
                    cell(str(d), "cd", 0, t, w, r, 2.0, busy=0.1, commit=base)
                    wr = {"heavy": "6.5", "moderate": "6.9"}[r]
                    for sd in (1, 2):
                        cell(str(d), fl, sd, t, w, r, 1.8 - 0.001 * t, str(ck), commit=w1, env={"HEROSIM_DECISION_COST_MS": wr})
                        if d == both:
                            cell(str(d), f, sd, t, w, r, 2.5, str(ck), commit=w1, env={"HEROSIM_DECISION_COST_MS": wr})

    def read(d, out, ckm=cm):
        return G.main(["--gate", str(d), "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", base[:8],
                       "--repo", str(repo), "--w-pins", w1[:8], "--primary", fl, "--checkpoint-md5", str(ckm), "--cotenancy", "off",
                       "--out", str(tmp_path / out)])

    read(only, "only.json")
    read(both, "both.json")
    assert (tmp_path / "only.json").read_bytes() == (tmp_path / "both.json").read_bytes()
    bad = tmp_path / "ck_bad.json"
    bad.write_text(json.dumps({**json.loads(cm.read_text()), f"{f}|s1": "0" * 32}))
    with pytest.raises(SystemExit, match="checkpoint"):
        read(both, "bad.json", bad)
    with open(both / f"cc40s{TOPOS[0]}__g0heavy__{f}_k100_s1.failed.json", "w") as fh:
        json.dump({"why": "x"}, fh)
    os.remove(both / f"cc40s{TOPOS[0]}__g0heavy__{f}_k100_s1.summary.json")
    with pytest.raises(SystemExit, match="failed cells of the other learned kinds"):
        read(both, "failed.json")


def test_comb_primary_reads_without_a_twin(gate, tmp_path):
    d, ck, _ = gate
    k = "ra_gnn_eng_a3lft_nosplit"
    for t in TOPOS:
        for w in G.WINDOWS:
            for r in G.RUNGS:
                cell(d, "cd_ib0", 0, t, w, r, 2.0, busy=0.1)
                cell(d, "cd", 0, t, w, r, 2.0, busy=0.1)
                for sd in (1, 2):
                    cell(d, k, sd, t, w, r, 1.8 - 0.001 * t, ck)
    cm = tmp_path / "ck_comb.json"
    cm.write_text(json.dumps({f"{k}|s{s}": hashlib.md5(b"weights").hexdigest() for s in (1, 2)}))
    res = G.main(["--gate", d, "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", "pin1234",
                  "--primary", k, "--checkpoint-md5", str(cm), "--cotenancy", "off"])
    assert res["F2"]["MP"] == G.NO_TWIN_COMB and res["F1"]["tests"][0]["n_topologies"] == 12


def comb_sweep(d, ck, tmp_path, k):
    for t in TOPOS:
        for w in G.WINDOWS:
            for r in G.RUNGS:
                cell(d, "cd_ib0", 0, t, w, r, 2.0, busy=0.1)
                cell(d, "cd", 0, t, w, r, 2.0, busy=0.1)
                for sd in (1, 2):
                    cell(d, k, sd, t, w, r, 1.8 - 0.001 * t, ck)
    cm = tmp_path / "ck_comb.json"
    cm.write_text(json.dumps({f"{k}|s{s}": hashlib.md5(b"weights").hexdigest() for s in (1, 2)}))
    return G.main(["--gate", d, "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", "pin1234",
                   "--primary", k, "--checkpoint-md5", str(cm), "--cotenancy", "off"])


@pytest.mark.parametrize("k", G.COMB_W_KINDS)
def test_comb_clean_sweep_is_capped_at_win_arm_only(gate, tmp_path, k):
    """B6: a clean sweep (every topology ahead at both rungs, both statistics) for any combined kind prints WIN-ARM-ONLY, never WIN."""
    d, ck, _ = gate
    res = comb_sweep(d, ck, tmp_path, k)
    assert all(e["p_holm"] < G.PROG_ALPHA and e["faster"] == e["n_topologies"] == 12 for e in res["F1"]["tests"])
    assert res["F1"]["label"] == "WIN-ARM-ONLY" and "combined" not in res
    assert res["F1"]["why"] == G.COMB_B6_WHY


def test_b6_cap_reasons_and_kinds():
    """The cap is keyed on COMB_W_KINDS membership; the Arm 1b / Arm 3 fixed-sequence reason is unchanged."""
    assert G.FIXED_SEQ_WHY.startswith("fixed sequence") and "B6" in G.COMB_B6_WHY
    assert not set(G.COMB_W_KINDS) & set(G.ARM1_W_KINDS + G.ARM1B_W_KINDS + G.ARM3_W_KINDS)


def test_combine_keeps_a_no_twin_label_for_a_seed_extension_arm_under_both_verdicts():
    """The seed extension reads an arm with no new-seed twin under --cotenancy required: both verdicts carry NO_TWIN_EXT and combine() must keep it (it raised StopIteration)."""
    import types

    f1 = {"label": "TIES"}
    for lab in (G.NO_TWIN_EXT, G.NO_TWIN, G.NO_TWIN_COMB):
        side = {"F1": f1, "F2": {"MP": lab}}
        out = G.combine(side, dict(side), types.SimpleNamespace(mlp=""))
        assert out["MP"] == lab
    mixed = G.combine({"F1": f1, "F2": {"MP": "MP-NOT-SEPARATED"}}, {"F1": f1, "F2": {"MP": "MP-DIRECTION"}}, types.SimpleNamespace(mlp=""))
    assert "NOT-SEPARATED" in mixed["MP"]          # the twin-bearing path is unchanged: the weaker of the two


@pytest.fixture
def conf(tmp_path, monkeypatch):
    """An FL confirmation gate: FL gnn and the FL MLP (_k100 and _kr) at a w-only pin carrying k_cotenancy, the bars at the base pin."""
    monkeypatch.setattr(G, "BAR", G.BAR)  # main() sets these module globals; restore them for the tests that follow
    monkeypatch.setattr(G, "SEEDS", G.SEEDS)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    fl, mlp = G.CONFIRM_PRIMARY, G.CONFIRM_MLP
    base = commit(repo, {G.W_FILE: W0, "scripts_cosim/gate.py": "x = 1\n", "docs/lineages/uncut_groups_s0_v1.md": "# node\n"}, "base")
    w1 = commit(repo, {G.W_FILE: {**w_with(W0, (fl, mlp)), "k_cotenancy": 1.468}}, "conf w")
    ck = tmp_path / "m.pt"
    ck.write_bytes(b"weights")
    cm = tmp_path / "ck.json"
    cm.write_text(json.dumps({f"{k}|s{s}": hashlib.md5(b"weights").hexdigest() for k in (fl, mlp) for s in (1, 2)}))
    d = tmp_path / "cells"
    d.mkdir()

    def fill(fl_mean=1.8, mlp_mean=1.9, fl_kr=None):
        for t in TOPOS:
            for w in G.WINDOWS:
                for r in G.RUNGS:
                    cell(str(d), "cd_ib0_inflightfull", 0, t, w, r, 2.0, busy=0.1, commit=base)
                    cell(str(d), "cd", 0, t, w, r, 2.0, busy=0.1, commit=base)
                    wr = {"heavy": 6.5, "moderate": 6.9}[r]
                    for sd in (1, 2):
                        for arm, m in ((fl, fl_mean), (mlp, mlp_mean)):
                            jit = 0.001 * t
                            cell(str(d), arm, sd, t, w, r, m - jit, str(ck), commit=w1, env={"HEROSIM_DECISION_COST_MS": repr(wr)})
                            mk = (fl_kr if (arm == fl and fl_kr is not None) else m) - jit
                            cell(str(d), arm, sd, t, w, r, mk, str(ck), commit=w1, sfx="_kr", env={"HEROSIM_DECISION_COST_MS": repr(1.468 * wr)})

    def read(stage, out, **kw):
        args = ["--gate", str(d), "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", base[:8],
                "--repo", str(repo), "--w-pins", w1[:8], "--primary", fl, "--checkpoint-md5", str(cm), "--bar", "cd_ib0_inflightfull",
                "--confirm", "--stage", stage, "--out", str(tmp_path / out)]
        for k, v in kw.items():
            args += [f"--{k}", v]
        return G.main(args)

    return {"repo": repo, "d": d, "fill": fill, "read": read, "tmp": tmp_path, "cm": cm, "fl": fl, "mlp": mlp}


def test_confirm_primary_stage_reads_f1_only_and_lifts_b6(conf):
    conf["fill"]()
    res = conf["read"]("primary", "p.json", mlp=G.CONFIRM_MLP)
    assert res["confirmation"]["label"] == "CONFIRMED" and res["combined"]["F1"] == "WIN"
    assert "MODEL-CLASS" not in res["combined"] and "MODEL-CLASS" not in res["F2"] and "descriptive" not in res
    assert G.CONFIRM_MLP not in res["health"] and G.CONFIRM_MLP not in json.dumps(res["F1"])
    with pytest.raises(SystemExit, match="written once"):
        conf["read"]("primary", "p.json")


def test_confirm_not_confirmed_when_the_cotenancy_verdict_fails(conf):
    conf["fill"](fl_kr=2.05)
    res = conf["read"]("primary", "p.json")
    assert res["confirmation"]["label"] == "NOT-CONFIRMED" and res["F1"]["label"] == "WIN" and res["cotenancy"]["F1"]["label"] != "WIN"


def test_confirm_primary_failure_writes_nothing(conf):
    conf["fill"]()
    bad = conf["tmp"] / "bad.json"
    bad.write_text(json.dumps({**json.loads(conf["cm"].read_text()), f"{G.CONFIRM_MLP}|s2": "0" * 32}))
    with pytest.raises(SystemExit, match="checkpoint"):
        conf["read"]("primary", "p.json", **{"checkpoint-md5": str(bad)})
    assert not (conf["tmp"] / "p.json").exists() and not (conf["tmp"] / "p.json.tmp").exists()


def test_confirm_secondary_needs_the_primary_filed_and_reproduces_it(conf):
    conf["fill"]()
    conf["read"]("primary", "p.json", mlp=G.CONFIRM_MLP)
    p = str(conf["tmp"] / "p.json")
    repo = conf["repo"]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sec = dict(mlp=G.CONFIRM_MLP, **{"primary-read": p, "record": head, "filed-ref": "HEAD"})
    with pytest.raises(SystemExit, match="not filed"):
        conf["read"]("secondary", "s.json", **sec)
    rec = commit(repo, {"docs/lineages/uncut_groups_s0_v1.md": f"# node\n- primary read md5 `{G.md5(p)}`\n"}, "file the primary")
    sec["record"] = rec
    res = conf["read"]("secondary", "s.json", **sec)
    assert res["combined"]["MODEL-CLASS"] == f"MODEL-CLASS-ATTRIBUTED (corpus: {G.CONFIRM_CORPUS})"
    assert res["F2"]["holm_family"] == 4 and res["confirmation"]["primary_read"]["md5"] == G.md5(p) and "descriptive" in res
    q = json.loads(open(p).read())
    q["combined"]["F1"] = "MIXED"
    open(p, "w").write(json.dumps(q))
    rec2 = commit(repo, {"docs/lineages/uncut_groups_s0_v1.md": f"# node\n- primary read md5 `{G.md5(p)}`\n"}, "refile")
    with pytest.raises(SystemExit, match="differ from the filed primary"):
        conf["read"]("secondary", "s2.json", **dict(sec, record=rec2))


def test_confirm_arguments_fail_loud(conf):
    conf["fill"]()
    with pytest.raises(SystemExit, match="fresh"):
        G.main(["--gate", str(conf["d"]), "--topos", "19537,1", "--topos-sha256", hashlib.sha256(b"19537,1").hexdigest(), "--pin", "x",
                "--primary", G.CONFIRM_PRIMARY, "--checkpoint-md5", str(conf["cm"]), "--bar", "cd_ib0_inflightfull", "--confirm", "--stage", "primary"])
    for kw in ({"cotenancy": "off"}, {"twin": "ra_twin_eng_arm1_nosplit"}, {"primary": "ra_gnn_eng_combf_nosplit"}):
        with pytest.raises(SystemExit, match="--confirm"):
            conf["read"]("primary", "x.json", **kw)
    with pytest.raises(SystemExit, match="--stage secondary needs"):
        conf["read"]("secondary", "x.json")
    with pytest.raises(SystemExit, match="FL confirmation's"):
        G.main(["--gate", str(conf["d"]), "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", "x",
                "--primary", G.CONFIRM_PRIMARY, "--checkpoint-md5", str(conf["cm"]), "--stage", "primary"])


@pytest.fixture
def conf2(tmp_path, monkeypatch):
    """An FL confirmation-2 gate: FL gnn seeds 3-6 (_k100 and _kr) in --gate with the bars; FL gnn and FL MLP seeds 1-2 in --desc-gate."""
    for g in ("BAR", "SEEDS", "PROG_ALPHA"):  # main() sets these module globals; restore them for the tests that follow
        monkeypatch.setattr(G, g, getattr(G, g))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    fl, mlp = G.CONFIRM_PRIMARY, G.CONFIRM_MLP
    base = commit(repo, {G.W_FILE: W0, "scripts_cosim/gate.py": "x = 1\n", "docs/lineages/uncut_groups_s0_v1.md": "# node\n"}, "base")
    w1 = commit(repo, {G.W_FILE: {**w_with(W0, (fl, mlp)), "k_cotenancy": 1.468}}, "conf w")
    ck = tmp_path / "m.pt"
    ck.write_bytes(b"weights")
    cm = tmp_path / "ck.json"
    cm.write_text(json.dumps({**{f"{fl}|s{s}": hashlib.md5(b"weights").hexdigest() for s in range(1, 7)},
                              **{f"{mlp}|s{s}": hashlib.md5(b"weights").hexdigest() for s in (1, 2)}}))
    d, dd = tmp_path / "cells", tmp_path / "desc"
    d.mkdir()
    dd.mkdir()

    def put(where, arm, sd, t, w, r, m):
        wr = {"heavy": 6.5, "moderate": 6.9}[r]
        cell(str(where), arm, sd, t, w, r, m, str(ck), commit=w1, env={"HEROSIM_DECISION_COST_MS": repr(wr)})
        cell(str(where), arm, sd, t, w, r, m, str(ck), commit=w1, sfx="_kr", env={"HEROSIM_DECISION_COST_MS": repr(1.468 * wr)})

    def fill(seed_mean=None, desc=True):
        seed_mean = seed_mean or {}
        for t in TOPOS:
            for w in G.WINDOWS:
                for r in G.RUNGS:
                    cell(str(d), "cd_ib0_inflightfull", 0, t, w, r, 2.0, busy=0.1, commit=base)
                    cell(str(d), "cd", 0, t, w, r, 2.0, busy=0.1, commit=base)
                    jit = 0.001 * t
                    for sd in G.CONFIRM2_SEEDS:
                        put(d, fl, sd, t, w, r, seed_mean.get(sd, 1.8) - jit)
                    if desc:
                        for sd in (1, 2):
                            put(dd, fl, sd, t, w, r, 1.85 - jit)
                            put(dd, mlp, sd, t, w, r, 1.9 - jit)

    def read(stage, out, gate=None, **kw):
        args = ["--gate", str(gate or d), "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", base[:8],
                "--repo", str(repo), "--w-pins", w1[:8], "--primary", fl, "--checkpoint-md5", str(cm), "--bar", "cd_ib0_inflightfull",
                "--seeds", "3,4,5,6", "--confirm2", "--stage", stage, "--out", str(tmp_path / out)]
        for k, v in kw.items():
            args += [f"--{k}"] + ([] if v is None else [v])
        return G.main(args)

    def file_primary(p):
        return commit(repo, {"docs/lineages/uncut_groups_s0_v1.md": f"# node\n- primary read md5 `{G.md5(p)}`\n"}, "file the primary")

    return {"repo": repo, "d": d, "dd": dd, "fill": fill, "read": read, "tmp": tmp_path, "cm": cm, "fl": fl, "mlp": mlp, "file": file_primary}


def test_confirm2_primary_reads_seeds_3_to_6_at_the_split_alpha(conf2):
    conf2["fill"]()
    res = conf2["read"]("primary", "p.json", mlp=G.CONFIRM_MLP)
    assert res["confirmation"]["label"] == G.CONFIRM2_YES and res["confirmation"]["alpha"] == G.CONFIRM2_ALPHA_LINE
    assert res["seeds"] == [3, 4, 5, 6] and sorted(res["F1"]["seeds_alone"]) == [3, 4, 5, 6]
    assert G.PROG_ALPHA == 0.0125 and "descriptive" not in res and "descriptive" not in res["cotenancy"]
    assert G.CONFIRM_MLP not in res["health"] and res["F2"]["tests"] == []


def test_confirm2_one_seed_above_minus_one_percent_is_not_confirmed(conf2):
    conf2["fill"](seed_mean={5: 1.99})
    res = conf2["read"]("primary", "p.json")
    assert res["confirmation"]["label"] == "NOT-CONFIRMED" and "seeds_alone_ok=False" in res["F1"]["why"]


def test_split_alpha_turns_a_holm_p_between_0125_and_025_into_arm_only(monkeypatch):
    f1 = [{"median_pct": -3.0, "p_holm": 0.02} for _ in range(4)]
    h = {r: {"collapse_any": 0} for r in G.RUNGS}
    args = (f1, {3: f1}, h, h, {r: True for r in G.RUNGS}, f1)
    assert G.f1_label(*args)[0] == "WIN"
    monkeypatch.setattr(G, "PROG_ALPHA", G.CONFIRM2_ALPHA)
    lab, why = G.f1_label(*args)
    assert lab == "WIN-ARM-ONLY" and "0.0125" in why


@pytest.mark.parametrize("seeds", ["1,2", "3,4,5", "3,4,5,6,7", "4,3,5,6", "3,4,5,7"])
def test_confirm2_refuses_any_seed_set_but_3_to_6(conf2, seeds):
    conf2["fill"](desc=False)
    with pytest.raises(SystemExit, match="seeds"):
        conf2["read"]("primary", "x.json", seeds=seeds)


@pytest.mark.parametrize("t", [19707, 19730, 19461, 19537, 19560])
def test_confirm2_refuses_every_guarded_topology(conf2, t):
    tops = f"{t},1"
    with pytest.raises(SystemExit, match="fresh"):
        G.main(["--gate", str(conf2["d"]), "--topos", tops, "--topos-sha256", hashlib.sha256(tops.encode()).hexdigest(), "--pin", "x",
                "--primary", G.CONFIRM_PRIMARY, "--checkpoint-md5", str(conf2["cm"]), "--bar", "cd_ib0_inflightfull", "--seeds", "3,4,5,6",
                "--confirm2", "--stage", "primary"])


def test_confirm_still_needs_seeds_1_2_and_confirmation_1_topologies_stay_legal_for_it(conf2):
    conf2["fill"](desc=False)
    with pytest.raises(SystemExit, match="--seeds 1,2"):
        G.main(["--gate", str(conf2["d"]), "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", "x",
                "--primary", G.CONFIRM_PRIMARY, "--checkpoint-md5", str(conf2["cm"]), "--bar", "cd_ib0_inflightfull", "--seeds", "3,4,5,6",
                "--confirm", "--stage", "primary"])
    assert not (set(range(19707, 19731)) & G.CONFIRM_USED_TOPOS)
    with pytest.raises(SystemExit, match="one"):
        conf2["read"]("primary", "x.json", confirm=None)


def test_confirm2_keeps_mlp_cells_out_of_the_gate(conf2):
    conf2["fill"]()
    with pytest.raises(SystemExit, match="desc-gate"):
        conf2["read"]("primary", "x.json", gate=f"{conf2['d']},{conf2['dd']}")
    with pytest.raises(SystemExit, match="C3"):
        conf2["read"]("primary", "x.json", **{"desc-gate": str(conf2["dd"])})


def test_confirm2_secondary_opens_the_desc_gate_only_after_the_primary_is_filed(conf2):
    conf2["fill"]()
    conf2["read"]("primary", "p.json", mlp=G.CONFIRM_MLP)
    p = str(conf2["tmp"] / "p.json")
    head = subprocess.run(["git", "-C", str(conf2["repo"]), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    broken = conf2["tmp"] / "broken"
    broken.mkdir()
    (broken / "cc40s1__g0heavy__ra_mlp_same_combfl_nosplit_k100_s9.summary.json").write_text("{}")
    sec = dict(mlp=G.CONFIRM_MLP, **{"primary-read": p, "record": head, "filed-ref": "HEAD", "desc-gate": str(broken)})
    with pytest.raises(SystemExit, match="not filed"):  # the unfiled primary stops the read before the (broken) desc gate is opened
        conf2["read"]("secondary", "s.json", **sec)
    sec["record"] = conf2["file"](p)
    with pytest.raises(SystemExit, match="outside FL gnn / FL MLP seeds 1-2"):
        conf2["read"]("secondary", "s.json", **sec)
    with pytest.raises(SystemExit, match="needs --desc-gate"):
        conf2["read"]("secondary", "s.json", **{k: v for k, v in sec.items() if k != "desc-gate"})
    sec["desc-gate"] = str(conf2["dd"])
    res = conf2["read"]("secondary", "s.json", **sec)
    for v in (res, res["cotenancy"]):
        mc, s12 = v["descriptive"]["F2_MC"], v["descriptive"]["seeds_1_2_vs_bar"]
        assert mc["label"] == G.CONFIRM2_MC and len(mc["tests"]) == 4 and all(e["a_seeds"] == [3, 4, 5, 6] for e in mc["tests"])
        assert all(e["median_pct"] < 0 and "p_holm" not in e for e in mc["tests"])
        assert len(s12["tests"]) == 4 and all(e["a_seeds"] == [1, 2] and e["median_pct"] < 0 for e in s12["tests"])
    assert "MODEL-CLASS" not in res["combined"] and res["confirmation"]["label"] == G.CONFIRM2_YES
    g = res["descriptive"]["gap_attribution"]["heavy"]  # mean over the four served seeds: (1.8 - jit) - 2.0 per task
    assert abs(g["gap_seconds"] - sum(4 * 4 * (1.8 - 0.001 * t - 2.0) for t in TOPOS)) < 1e-9


def test_gap_attribution_two_seed_mean_is_unchanged_bit_for_bit():
    a, b = 1.2345678901234567, 2.3456789012345678
    assert sum([a, b]) / len([a, b]) == (a + b) / 2.0


def _drop(d, pattern):
    hits = [f for f in os.listdir(d) if f.startswith(pattern)]
    assert hits, pattern
    for f in hits:
        os.remove(os.path.join(d, f))


@pytest.mark.parametrize("case", ["missing_seed", "missing_window", "failed_cell", "extra_cell"])
def test_confirm2_is_void_unless_the_gate_is_the_full_registered_grid(conf2, case):
    conf2["fill"](desc=False)
    d, fl = str(conf2["d"]), G.CONFIRM_PRIMARY
    if case == "missing_seed":
        _drop(d, f"cc40s3__g1heavy__{fl}_kr_s5.")
    elif case == "missing_window":
        _drop(d, "cc40s7__g2moderate__cd_ib0_inflightfull_k100_s0.")
    elif case == "failed_cell":
        _drop(d, f"cc40s4__g0heavy__{fl}_k100_s6.summary.json")
        (conf2["d"] / f"cc40s4__g0heavy__{fl}_k100_s6.failed.json").write_text(json.dumps({"why": "watchdog: projected wall"}))
    else:
        cell(d, "cd", 0, 13, "g0", "heavy", 2.0)
    with pytest.raises(SystemExit, match="VOID") as e:
        conf2["read"]("primary", "p.json")
    assert not (conf2["tmp"] / "p.json").exists()
    if case == "failed_cell":
        assert "1 failed" in str(e.value)


def test_confirm2_grid_counts_and_desc_gate_must_be_complete(conf2):
    conf2["fill"]()
    res = conf2["read"]("primary", "p.json", mlp=G.CONFIRM_MLP)
    assert res["confirmation"]["grid_cells"] == len(TOPOS) * 4 * 2 * (4 * 2 + 2)  # seeds 3-6 x both charges, plus the two bars
    p = str(conf2["tmp"] / "p.json")
    _drop(str(conf2["dd"]), f"cc40s9__g3heavy__{G.CONFIRM_MLP}_kr_s2.")
    sec = dict(mlp=G.CONFIRM_MLP, **{"primary-read": p, "record": conf2["file"](p), "filed-ref": "HEAD", "desc-gate": str(conf2["dd"])})
    with pytest.raises(SystemExit, match="VOID: --desc-gate"):
        conf2["read"]("secondary", "s.json", **sec)


@pytest.fixture
def s64(tmp_path, monkeypatch):
    """An FL size-64 gate: FL gnn seeds 1-2 (_k100 and _kr) with the bars in --gate; the uncharged k = 0 cells (bare kinds) of FL gnn s1-2 and the
    bar plus the FL MLP s1-2 _k100 in --desc-gate; reactive s0 in --reactive-gate."""
    for g in ("BAR", "SEEDS", "PROG_ALPHA"):
        monkeypatch.setattr(G, g, getattr(G, g))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    fl, mlp = G.CONFIRM_PRIMARY, G.CONFIRM_MLP
    base = commit(repo, {G.W_FILE: W0, "scripts_cosim/gate.py": "x = 1\n", "docs/lineages/uncut_groups_s0_v1.md": "# node\n"}, "base")
    w1 = commit(repo, {G.W_FILE: {**w_with(W0, (fl, mlp)), "k_cotenancy": 1.468}}, "conf w")
    ck = tmp_path / "m.pt"
    ck.write_bytes(b"weights")
    cm = tmp_path / "ck.json"
    cm.write_text(json.dumps({f"{k}|s{s}": hashlib.md5(b"weights").hexdigest() for k in (fl, mlp) for s in (1, 2)}))
    d, dd, rd = tmp_path / "cells", tmp_path / "desc", tmp_path / "reactive"
    for x in (d, dd, rd):
        x.mkdir()

    def fill(fl_mean=1.8, k0_charged=False):
        for t in TOPOS:
            for w in G.WINDOWS:
                for r in G.RUNGS:
                    jit = 0.001 * t
                    wr = {"heavy": 6.5, "moderate": 6.9}[r]
                    cell(str(d), "cd_ib0_inflightfull", 0, t, w, r, 2.0, busy=0.1, commit=base, servers=64)
                    cell(str(d), "cd", 0, t, w, r, 2.0, busy=0.1, commit=base, servers=64)
                    cell(str(dd), "cd_ib0_inflightfull", 0, t, w, r, 1.99, busy=0.1, commit=base, sfx="", servers=64)
                    cell(str(rd), "reactive", 0, t, w, r, 5.0, busy=0.1, commit=base, sfx="", servers=64)
                    for sd in (1, 2):
                        cell(str(d), fl, sd, t, w, r, fl_mean - jit, str(ck), commit=w1, env={"HEROSIM_DECISION_COST_MS": repr(wr)}, servers=64)
                        cell(str(d), fl, sd, t, w, r, fl_mean - jit, str(ck), commit=w1, sfx="_kr", env={"HEROSIM_DECISION_COST_MS": repr(1.468 * wr)}, servers=64)
                        cell(str(dd), fl, sd, t, w, r, fl_mean - 0.01 - jit, str(ck), commit=w1, sfx="",
                             env={"HEROSIM_DECISION_COST_MS": repr(wr)} if k0_charged else None, servers=64)
                        cell(str(dd), mlp, sd, t, w, r, 1.9 - jit, str(ck), commit=w1, env={"HEROSIM_DECISION_COST_MS": repr(wr)}, servers=64)

        for f in rd.glob("*.tasks_bsz.npy"):  # the per-arrival reactive path writes no groupSize / servedBatchSize (P5.1)
            _unset_batch_fields(f)

    def read(stage, out, **kw):
        args = ["--gate", str(d), "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", base[:8],
                "--repo", str(repo), "--w-pins", w1[:8], "--primary", fl, "--checkpoint-md5", str(cm), "--bar", "cd_ib0_inflightfull",
                "--confirm-size64", "--stage", stage, "--out", str(tmp_path / out)]
        for k, v in kw.items():
            args += [f"--{k}"] + ([] if v is None else [v])
        return G.main(args)

    def file_primary(p):
        return commit(repo, {"docs/lineages/uncut_groups_s0_v1.md": f"# node\n- primary read md5 `{G.md5(p)}`\n"}, "file the primary")

    return {"d": d, "dd": dd, "rd": rd, "fill": fill, "read": read, "tmp": tmp_path, "cm": cm, "file": file_primary, "repo": repo}


def test_size64_primary_reads_seeds_1_2_at_the_program_alpha_and_lifts_b6(s64):
    s64["fill"]()
    res = s64["read"]("primary", "p.json", mlp=G.CONFIRM_MLP)
    assert res["confirmation"]["label"] == G.SIZE64_YES and res["confirmation"]["family"] == "FL size-64"
    assert G.PROG_ALPHA == G.PROG_ALPHA_FILED and sorted(res["F1"]["seeds_alone"]) == [1, 2]
    assert res["confirmation"]["grid_cells"] == len(TOPOS) * 4 * 2 * (2 * 2 + 2)
    assert "descriptive" not in res and "descriptive" not in res["cotenancy"]


def test_size64_not_confirmed_when_fl_does_not_clear_the_line(s64):
    s64["fill"](fl_mean=1.99)
    assert s64["read"]("primary", "p.json")["confirmation"]["label"] == "NOT-CONFIRMED"


@pytest.mark.parametrize("t", [19953, 19976, 30011, 30123, 20021, 20032, 19707, 19461, 19537])
def test_size64_refuses_every_guarded_topology(s64, t):
    tops = f"{t},1"
    with pytest.raises(SystemExit, match="fresh"):
        G.main(["--gate", str(s64["d"]), "--topos", tops, "--topos-sha256", hashlib.sha256(tops.encode()).hexdigest(), "--pin", "x",
                "--primary", G.CONFIRM_PRIMARY, "--checkpoint-md5", str(s64["cm"]), "--bar", "cd_ib0_inflightfull", "--confirm-size64",
                "--stage", "primary"])


def test_size64_arguments_fail_loud(s64):
    s64["fill"]()
    with pytest.raises(SystemExit, match="--seeds 1,2"):
        s64["read"]("primary", "x.json", seeds="3,4,5,6")
    with pytest.raises(SystemExit, match="separate families"):
        s64["read"]("primary", "x.json", confirm2=None)
    with pytest.raises(SystemExit, match="seal"):
        s64["read"]("primary", "x.json", **{"desc-gate": str(s64["dd"])})
    with pytest.raises(SystemExit, match="reactive-gate belongs"):
        s64["read"]("primary", "x.json", **{"reactive-gate": str(s64["rd"])})


def test_size64_is_void_unless_the_gate_is_the_registered_grid(s64):
    s64["fill"]()
    _drop(str(s64["d"]), f"cc40s5__g2moderate__{G.CONFIRM_PRIMARY}_kr_s2.")
    with pytest.raises(SystemExit, match="VOID: --gate"):
        s64["read"]("primary", "p.json")


def test_size64_secondary_reads_k0_mlp_and_reactive_only_after_the_primary_is_filed(s64):
    s64["fill"]()
    s64["read"]("primary", "p.json", mlp=G.CONFIRM_MLP)
    p = str(s64["tmp"] / "p.json")
    head = subprocess.run(["git", "-C", str(s64["repo"]), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    broken = s64["tmp"] / "broken"
    broken.mkdir()
    sec = dict(mlp=G.CONFIRM_MLP, **{"primary-read": p, "record": head, "filed-ref": "HEAD", "desc-gate": str(broken), "reactive-gate": str(broken)})
    with pytest.raises(SystemExit, match="not filed"):
        s64["read"]("secondary", "s.json", **sec)
    sec["record"] = s64["file"](p)
    with pytest.raises(SystemExit, match="needs --desc-gate"):
        s64["read"]("secondary", "s.json", **{k: v for k, v in sec.items() if k != "reactive-gate"})
    sec.update({"desc-gate": str(s64["dd"]), "reactive-gate": str(s64["rd"])})
    res = s64["read"]("secondary", "s.json", **sec)
    d = res["descriptive"]
    assert res["confirmation"]["desc_grid_cells"] == len(TOPOS) * 4 * 2 * (2 + 1 + 2)
    assert res["confirmation"]["reactive_grid_cells"] == len(TOPOS) * 4 * 2
    for k in ("k0_fl_vs_bar", "mlp_vs_bar", "fl_vs_mlp", "reactive_vs_bar"):
        assert len(d[k]["tests"]) == 4 and all("p_holm" not in e for e in d[k]["tests"])
    assert all(e["median_pct"] < 0 for e in d["k0_fl_vs_bar"]["tests"] + d["fl_vs_mlp"]["tests"])
    assert all(e["median_pct"] > 0 for e in d["reactive_vs_bar"]["tests"])
    assert res["confirmation"]["label"] == G.SIZE64_YES and "MODEL-CLASS" not in res["combined"]


@pytest.mark.parametrize("case", ["k0_charged", "desc_missing", "reactive_missing", "mlp_bare"])
def test_size64_secondary_gates_fail_loud(s64, case):
    s64["fill"](k0_charged=(case == "k0_charged"))
    s64["read"]("primary", "p.json")
    p = str(s64["tmp"] / "p.json")
    if case == "desc_missing":
        _drop(str(s64["dd"]), f"cc40s2__g1heavy__cd_ib0_inflightfull_s0.")
    elif case == "reactive_missing":
        _drop(str(s64["rd"]), "cc40s8__g0moderate__reactive_s0.")
    elif case == "mlp_bare":
        cell(str(s64["dd"]), G.CONFIRM_MLP, 3, 3, "g0", "heavy", 1.9, sfx="", servers=64)  # a bare MLP cell (a distinct key, seed 3)
    sec = dict(mlp=G.CONFIRM_MLP, **{"primary-read": p, "record": s64["file"](p), "filed-ref": "HEAD", "desc-gate": str(s64["dd"]),
                                     "reactive-gate": str(s64["rd"])})
    match = {"k0_charged": "k = 0 cell but served", "desc_missing": "VOID: --desc-gate", "reactive_missing": "VOID: --reactive-gate",
             "mlp_bare": "bare kinds and the MLP"}[case]
    with pytest.raises(SystemExit, match=match):
        s64["read"]("secondary", "s.json", **sec)


@pytest.mark.parametrize("where,servers", [("gate", 24), ("gate", None), ("desc", 24), ("reactive", 40)])
def test_size64_fails_loud_on_any_cell_not_at_64_servers(s64, where, servers):
    s64["fill"]()
    fl = G.CONFIRM_PRIMARY
    d, name = {"gate": (s64["d"], f"cc40s6__g3heavy__{fl}_kr_s1"), "desc": (s64["dd"], f"cc40s6__g3heavy__{G.CONFIRM_MLP}_k100_s2"),
               "reactive": (s64["rd"], "cc40s6__g3heavy__reactive_s0")}[where]
    p = d / f"{name}.summary.json"
    x = json.loads(p.read_text())
    x.pop("servers") if servers is None else x.update(servers=servers)
    p.write_text(json.dumps(x))
    if where == "gate":
        with pytest.raises(SystemExit, match="servers = 64"):
            s64["read"]("primary", "p.json")
        return
    s64["read"]("primary", "p.json")
    pr = str(s64["tmp"] / "p.json")
    sec = dict(mlp=G.CONFIRM_MLP, **{"primary-read": pr, "record": s64["file"](pr), "filed-ref": "HEAD", "desc-gate": str(s64["dd"]),
                                     "reactive-gate": str(s64["rd"])})
    with pytest.raises(SystemExit, match=f"--{where}-gate: 1 cell"):
        s64["read"]("secondary", "s.json", **sec)


def _unset_batch_fields(p, rows=None):
    a = np.load(p)
    a[:, 3] = -1
    a[:, 4] = -1
    np.save(p, a if rows is None else a[:rows])


@pytest.mark.parametrize("case", ["reactive_unset_passes", "reactive_rowcount_fails", "reactive_no_sidecar_fails", "reactive_filled_fails",
                                  "mlp_unset_fails", "bar_unset_fails"])
def test_size64_per_arrival_reactive_row_and_batched_cells(s64, case):
    """P5.1 (job 885487 failed on the reactive sidecar): reactive places per arrival and writes no groupSize / servedBatchSize; batched
    learned and bar cells are still held to both under every gate."""
    s64["fill"]()
    rd, dd = s64["rd"], s64["dd"]
    r = rd / "cc40s4__g1heavy__reactive_s0.tasks_bsz.npy"
    if case == "reactive_rowcount_fails":
        _unset_batch_fields(r, rows=3)
    elif case == "reactive_no_sidecar_fails":
        r.unlink()
    elif case == "reactive_filled_fails":
        a = np.load(r)
        a[1, 4] = 3
        np.save(r, a)
    elif case == "mlp_unset_fails":
        _unset_batch_fields(dd / f"cc40s4__g1heavy__{G.CONFIRM_MLP}_k100_s1.tasks_bsz.npy")
    elif case == "bar_unset_fails":
        _unset_batch_fields(s64["d"] / "cc40s4__g1heavy__cd_ib0_inflightfull_k100_s0.tasks_bsz.npy")
        with pytest.raises(SystemExit, match="groupSize/servedBatchSize"):
            s64["read"]("primary", "p.json")
        return
    s64["read"]("primary", "p.json")
    p = str(s64["tmp"] / "p.json")
    sec = dict(mlp=G.CONFIRM_MLP, **{"primary-read": p, "record": s64["file"](p), "filed-ref": "HEAD", "desc-gate": str(dd), "reactive-gate": str(rd)})
    if case == "reactive_unset_passes":
        assert len(s64["read"]("secondary", "s.json", **sec)["descriptive"]["reactive_vs_bar"]["tests"]) == 4
        return
    match = {"reactive_rowcount_fails": "one row per task", "reactive_no_sidecar_fails": "no per-task batch-size sidecar",
             "reactive_filled_fails": "a batched path served it",
             "mlp_unset_fails": "groupSize/servedBatchSize"}[case]
    with pytest.raises(SystemExit, match=match):
        s64["read"]("secondary", "s.json", **sec)


@pytest.fixture
def fll(tmp_path, monkeypatch):
    """An FL + lambda gate: the bars, FL + lambda and FL (seeds 1-2, _k100 and _kr) in --gate at 24 servers; FL MLP + lambda and the FL MLP
    (seeds 1-2, both charges) in --desc-gate. The w chain: base -> W-CONF (FL and FL MLP w, k_cotenancy) -> P7 (the two flreg kinds' w)."""
    for g in ("BAR", "SEEDS", "PROG_ALPHA"):
        monkeypatch.setattr(G, g, getattr(G, g))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    p, c, mp, mc = G.FLL_PRIMARY, G.FLL_CONTROL, G.FLL_MLP, G.FLL_MLP_CONTROL
    base = commit(repo, {G.W_FILE: W0, "scripts_cosim/gate.py": "x = 1\n", "docs/lineages/uncut_groups_s0_v1.md": "# node\n"}, "base")
    w1 = commit(repo, {G.W_FILE: {**w_with(W0, (c, mc)), "k_cotenancy": 1.468}}, "conf w")
    w2 = commit(repo, {G.W_FILE: {**w_with(w_with(W0, (c, mc)), (p, mp)), "k_cotenancy": 1.468}}, "flreg w")
    ck = tmp_path / "m.pt"
    ck.write_bytes(b"weights")
    cm = tmp_path / "ck.json"
    cm.write_text(json.dumps({f"{k}|s{s}": hashlib.md5(b"weights").hexdigest() for k in (p, c, mp, mc) for s in (1, 2)}))
    d, dd = tmp_path / "cells", tmp_path / "desc"
    for x in (d, dd):
        x.mkdir()

    def fill(lam=(1.8, 1.8), ctl=(2.0, 2.0), servers=24, mlp_in_gate=False, big=None):
        for t in TOPOS:
            for w in G.WINDOWS:
                for r in G.RUNGS:
                    jit = 0.001 * t
                    wr = {"heavy": 6.5, "moderate": 6.9}[r]
                    cell(str(d), "cd_ib0_inflightfull", 0, t, w, r, 2.1, busy=0.1, commit=base, servers=servers)
                    cell(str(d), "cd", 0, t, w, r, 2.2, busy=0.1, commit=base, servers=servers)
                    for i, sd in enumerate((1, 2)):
                        for kind, mean, out in ((p, lam[i], d), (c, ctl[i], d), (mp, 1.95, d if mlp_in_gate else dd), (mc, 2.05, d if mlp_in_gate else dd)):
                            m = mean - jit if not (big and (kind, sd, t, w, r) == big) else 50.0
                            cell(str(out), kind, sd, t, w, r, m, str(ck), commit=w2, env={"HEROSIM_DECISION_COST_MS": repr(wr)}, servers=servers)
                            cell(str(out), kind, sd, t, w, r, m, str(ck), commit=w2, sfx="_kr", env={"HEROSIM_DECISION_COST_MS": repr(1.468 * wr)}, servers=servers)

    def read(stage, out, **kw):
        args = ["--gate", str(d), "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", base[:8],
                "--repo", str(repo), "--w-pins", f"{w1[:8]},{w2[:8]}", "--primary", p, "--checkpoint-md5", str(cm), "--bar", "cd_ib0_inflightfull",
                "--confirm-fl-lambda", "--stage", stage, "--out", str(tmp_path / out)]
        for k, v in kw.items():
            args += [f"--{k}"] + ([] if v is None else [v])
        return G.main(args)

    def file_primary(path):
        return commit(repo, {"docs/lineages/uncut_groups_s0_v1.md": f"# node\n- primary read md5 `{G.md5(path)}`\n"}, "file the primary")

    return {"d": d, "dd": dd, "fill": fill, "read": read, "tmp": tmp_path, "cm": cm, "file": file_primary, "repo": repo}


def test_fll_primary_win_is_lambda_vs_its_control_under_both_charges(fll):
    fll["fill"]()
    res = fll["read"]("primary", "p.json")
    assert res["confirmation"]["label"] == G.FLL_YES and res["combined"]["F1"] == "WIN"
    assert res["F1"]["tests"][0]["a"] == G.FLL_PRIMARY and res["F1"]["tests"][0]["b"] == G.FLL_CONTROL
    assert res["confirmation"]["grid_cells"] == len(TOPOS) * 4 * 2 * (2 * 2 * 2 + 2) and res["servers"] == 24
    assert all(e["median_pct"] < -2 and e["p_holm"] < G.PROG_ALPHA_FILED for e in res["F1"]["tests"] + res["cotenancy"]["F1"]["tests"])
    assert "descriptive" not in res and G.PROG_ALPHA == G.PROG_ALPHA_FILED


def test_fll_per_seed_clause_is_seed_matched(fll):
    # pooled clears the line (-2.75 %) but lambda seed 2 is only -0.5 % against FL seed 2
    fll["fill"](lam=(1.9, 1.99), ctl=(2.0, 2.0))
    res = fll["read"]("primary", "p.json")
    assert all(e["median_pct"] <= -2 for e in res["F1"]["tests"])
    assert res["F1"]["label"] == "MIXED" and "seeds_alone_ok=False" in res["F1"]["why"] and res["confirmation"]["label"] == "NOT-WIN"
    # the same seed-2 cells against a seed mean would pass; seed-matched they do not
    assert all(e["median_pct"] > -1 for e in res["F1"]["seeds_alone"][2]) and all(e["b"] == G.FLL_CONTROL for e in res["F1"]["seeds_alone"][2])


def test_fll_collapse_clause_is_seed_matched(fll):
    fll["fill"](big=(G.FLL_PRIMARY, 1, 3, "g1", "heavy"))
    res = fll["read"]("primary", "p.json")
    h = res["F1"]
    assert res["health"][G.FLL_PRIMARY]["heavy"]["per_seed"][1]["collapse_any"] == 1 and "collapse_ok=False" in h["why"]
    assert res["confirmation"]["label"] == "NOT-WIN"


def test_fll_lambda_slower_is_fl_faster_and_not_win(fll):
    fll["fill"](lam=(2.3, 2.3))
    res = fll["read"]("primary", "p.json")
    assert res["F1"]["label"] == "FL-FASTER" and res["combined"]["F1"] == "FL-FASTER" and res["confirmation"]["label"] == "NOT-WIN"


@pytest.mark.parametrize("t", [30129, 30140, 19953, 30011, 19707, 19461, 19537, 20021])
def test_fll_refuses_every_guarded_topology(fll, t):
    tops = f"{t},1"
    with pytest.raises(SystemExit, match="fresh"):
        G.main(["--gate", str(fll["d"]), "--topos", tops, "--topos-sha256", hashlib.sha256(tops.encode()).hexdigest(), "--pin", "x", "--repo", "r",
                "--w-pins", "w", "--primary", G.FLL_PRIMARY, "--checkpoint-md5", str(fll["cm"]), "--bar", "cd_ib0_inflightfull",
                "--confirm-fl-lambda", "--stage", "primary"])


@pytest.mark.parametrize("kw, match", [
    ({"seeds": "1,2,3"}, "--seeds 1,2"), ({"confirm2": None}, "its own family"), ({"twin": "x"}, "no twin"),
    ({"mlp": G.FLL_MLP}, "seal"), ({"cotenancy": "off"}, "--cotenancy required"), ({"stage": "full"}, "stage primary or secondary")])
def test_fll_arguments_fail_loud(fll, kw, match):
    fll["fill"]()
    stage = kw.pop("stage", "primary")
    with pytest.raises(SystemExit, match=match):
        fll["read"](stage, "x.json", **kw)


@pytest.mark.parametrize("case", ["missing", "mlp_in_gate", "servers"])
def test_fll_primary_is_void_or_fails_loud_unless_the_registered_24_server_grid(fll, case):
    fll["fill"](mlp_in_gate=(case == "mlp_in_gate"), servers=64 if case == "servers" else 24)
    if case == "missing":
        _drop(str(fll["d"]), f"cc40s5__g2moderate__{G.FLL_PRIMARY}_kr_s2.")
    with pytest.raises(SystemExit, match={"missing": "VOID: --gate", "mlp_in_gate": "belong in --desc-gate", "servers": "servers = 24"}[case]):
        fll["read"]("primary", "p.json")


def test_fll_secondary_opens_the_desc_gate_only_after_the_primary_is_filed(fll):
    fll["fill"]()
    fll["read"]("primary", "p.json")
    p = str(fll["tmp"] / "p.json")
    head = subprocess.run(["git", "-C", str(fll["repo"]), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sec = {"mlp": G.FLL_MLP, "primary-read": p, "record": head, "filed-ref": "HEAD", "desc-gate": str(fll["dd"])}
    with pytest.raises(SystemExit, match="not filed"):
        fll["read"]("secondary", "s.json", **sec)
    sec["record"] = fll["file"](p)
    with pytest.raises(SystemExit, match="needs --desc-gate"):
        fll["read"]("secondary", "s.json", **{k: v for k, v in sec.items() if k != "desc-gate"})
    res = fll["read"]("secondary", "s.json", **sec)
    assert res["confirmation"]["desc_grid_cells"] == len(TOPOS) * 4 * 2 * 8 and res["confirmation"]["label"] == G.FLL_YES
    for v in (res, res["cotenancy"]):
        for k in ("fl_lambda_vs_bar", "fl_lambda_vs_cd", "mlp_lambda_vs_mlp", "fl_lambda_vs_mlp_lambda"):
            assert len(v["descriptive"][k]["tests"]) == 4 and all("p_holm" not in e for e in v["descriptive"][k]["tests"])
    assert all(e["median_pct"] < 0 for e in res["descriptive"]["mlp_lambda_vs_mlp"]["tests"])
    d = res["descriptive"]["mlp_lambda_vs_fl_gnn"]["tests"]
    assert len(d) == 4 and d[0]["a"] == G.FLL_MLP and d[0]["b"] == G.FLL_CONTROL and "mlp_lambda_vs_fl_gnn" not in res["cotenancy"]["descriptive"]


def test_fll_secondary_desc_gate_must_be_the_exact_grid(fll):
    fll["fill"]()
    fll["read"]("primary", "p.json")
    p = str(fll["tmp"] / "p.json")
    rec = fll["file"](p)
    _drop(str(fll["dd"]), f"cc40s7__g0heavy__{G.FLL_MLP_CONTROL}_kr_s1.")
    with pytest.raises(SystemExit, match="VOID: --desc-gate"):
        fll["read"]("secondary", "s.json", mlp=G.FLL_MLP, **{"primary-read": p, "record": rec, "filed-ref": "HEAD", "desc-gate": str(fll["dd"])})


def test_fll_kinds_are_w_charged_learned_kinds_outside_the_combined_b6_set():
    assert set(G.FLREG_W_KINDS) <= set(G.LEARNED_W_KINDS) and not set(G.FLREG_W_KINDS) & set(G.COMB_W_KINDS)
    assert G.FLL_CONTROL == G.CONFIRM_PRIMARY and G.FLL_MLP_CONTROL == G.CONFIRM_MLP


def test_g8d_file_set_is_admitted_for_that_commit_only(gchain, monkeypatch):
    repo, base, gc = gchain[0], gchain[1], gchain[2]
    assert G.GATE_CHANGE_FILES_BY_COMMIT[G.G8D] == (GATE, "scripts_cosim/joint_burst_v2_sidecheck.py", G.GATE_CHANGE_FILES[1])
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", gc], check=True)
    g8d = commit(repo, {GATE: "x = 3\n", "scripts_cosim/joint_burst_v2_sidecheck.py": "y = 1\n", G.GATE_CHANGE_FILES[1]: "t = 3\n"}, "g8d")
    with pytest.raises(SystemExit, match="not exactly"):
        G.w_pin_chain(str(repo), base, [], f"{gc},{g8d}")
    monkeypatch.setitem(G.GATE_CHANGE_FILES_BY_COMMIT, g8d, G.GATE_CHANGE_FILES_BY_COMMIT[G.G8D])
    G.w_pin_chain(str(repo), base, [], f"{gc},{g8d}")


def test_p2_w_only_link_is_admitted_inside_the_gate_change_sequence_for_that_commit_only(gchain, monkeypatch):
    """P8.1: the FL + lambda chain has the P2 w-only pin between gate changes (... -> P1 -> P2 (w) -> G8c -> G8d -> P7). The link is admitted
    with the w file as its only file, for that commit only, and w_frozen still checks it."""
    repo, base, gc, w1 = gchain[0], gchain[1], gchain[2], gchain[3]
    assert G.GATE_CHANGE_FILES_BY_COMMIT[G.P2] == (G.W_FILE,) and G.P2.startswith("74ddf70e")
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", w1], check=True)
    g2 = commit(repo, {GATE: "x = 9\n", G.GATE_CHANGE_FILES[1]: "t = 9\n"}, "gate change on the w-only pin")
    w = json.loads((repo / G.W_FILE).read_text())
    w3 = commit(repo, {G.W_FILE: w_with(w, ("ra_gnn_eng_flreg_nosplit",))}, "P7-like w-only pin")
    with pytest.raises(SystemExit, match="not exactly"):
        G.w_pin_chain(str(repo), base, [w3], f"{gc},{w1},{g2}")
    with pytest.raises(SystemExit, match="not a child"):
        G.w_pin_chain(str(repo), base, [w1, w3], f"{gc},{g2}")
    monkeypatch.setitem(G.GATE_CHANGE_FILES_BY_COMMIT, w1, (G.W_FILE,))
    ws = G.w_pin_chain(str(repo), base, [w3], f"{gc},{w1},{g2}")
    assert list(ws)[-1] == w3 and "ra_gnn_eng_flreg_nosplit" in ws[w3]["w_ms_per_plan"]
    # the admitted link is still a w-only commit under w_frozen: a link that changes a frozen key fails
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", gc], check=True)
    bad = json.loads(json.dumps(w))
    bad["w_ms_per_plan"]["cd_ib0"]["heavy"] = 9.0
    wb = commit(repo, {G.W_FILE: bad}, "a link that rewrites a frozen key")
    monkeypatch.setitem(G.GATE_CHANGE_FILES_BY_COMMIT, wb, (G.W_FILE,))
    with pytest.raises(SystemExit, match="changes frozen"):
        G.w_pin_chain(str(repo), base, [], f"{gc},{wb}")
