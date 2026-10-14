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


def cell(d, arm, seed, t, w, r, mean, ckpt=None, lat_big=None, busy=1.0, warmth="node_disk_v2", commit="pin1234abcd", sfx="_k100", env=None):
    name = f"cc40s{t}__{w}{r}__{arm}{sfx}_s{seed}"
    s = {"averageElapsedTime": mean, "steady_state": {"after_360s": {"mean": mean}}, "num_tasks": 4,
         "endTime": 100.0, "arrival_end": {"last_arrival_s": 99.0}, "code": {"commit": commit, "dirty": False},
         "schedulerCounters": {"sched_queue_wait_seconds": 0.0, "sched_busy_seconds": busy}, "env": {"HEROSIM_WARMTH_PHYSICS": warmth},
         "python_env": {"torch_num_threads": 1}, "host": {"cpu_model": "AMD EPYC 9335", "smt_control": "off", "nproc": 64}}
    if ckpt:
        s["env"]["GNN_MODEL_PATH"] = ckpt
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
