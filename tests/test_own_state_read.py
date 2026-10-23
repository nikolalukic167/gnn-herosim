"""--confirm-own-state (uncut_arm_gate_read.py): the own-state round's frozen reader, on a synthetic repo and cell tree. The real pins are
monkeypatched to this repo's commits and the 24 retest ids to 10 synthetic ones (an exact Wilcoxon on 10 topologies can reach Holm < .025)."""
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_arm_gate_read as G  # noqa: E402
from test_uncut_arm_gate_read import W0, cell, commit, w_with  # noqa: E402

TOPOS = list(range(40001, 40011))
TOPS = ",".join(map(str, TOPOS))
GNN, TWN, OGNN, OTWN, BAR = G.OS_GNN, G.OS_TWIN, G.CONFIRM_PRIMARY, G.TWIN, "cd_ib0_inflightfull"
NOMP = {"GNN_DISABLE_MESSAGE_PASSING": "1"}


def react_cell(d, t, w, r, commit_, share=0.1):
    cell(d, "reactive", 0, t, w, r, 3.0, busy=0.0, commit=commit_, sfx="", servers=24)
    stem = os.path.join(d, f"cc40s{t}__{w}{r}__reactive_s0")
    s = json.load(open(stem + ".summary.json"))
    s["queue_share"] = share
    json.dump(s, open(stem + ".summary.json", "w"))
    b = np.load(stem + ".tasks_bsz.npy")
    b[:, 3:] = -1
    np.save(stem + ".tasks_bsz.npy", b)


@pytest.fixture
def os_(tmp_path, monkeypatch):
    for g in ("BAR", "SEEDS", "PROG_ALPHA", "CD"):
        monkeypatch.setattr(G, g, getattr(G, g))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    base = commit(repo, {G.W_FILE: W0, "scripts_cosim/fresh_topo_burst_v1_gate.py": "x = 1\n", "tests/test_dagger_full_kinds.py": "y = 1\n",
                         "docs/lineages/uncut_groups_s0_v1.md": "# node\n"}, "base (G8)")
    w7_body = {**w_with(W0, (OGNN, G.CONFIRM_MLP)), "k_cotenancy": 1.468}
    w7 = commit(repo, {G.W_FILE: w7_body}, "P7")
    w7b_body = json.loads(json.dumps(w7_body))
    w7b_body["w_ms_per_plan"][OTWN] = {"heavy": 5.5, "moderate": 5.9}
    w7b_body["w_ms_per_plan_by_bucket"][OTWN] = {"heavy": {"le10": 5.5, "11_50": 2.0, "gt50": 2.2}, "moderate": {"le10": 5.9, "11_50": 2.1, "gt50": 2.3}}
    w7b_body[f"_source_{OTWN}"] = "twin w"
    w7b = commit(repo, {G.W_FILE: w7b_body}, "P7b")
    commit(repo, {"scripts_cosim/other.py": "z = 1\n"}, "unrelated work")
    g1 = commit(repo, {"scripts_cosim/fresh_topo_burst_v1_gate.py": "x = 2\n"}, "G1 kinds")
    g2 = commit(repo, {"tests/test_dagger_full_kinds.py": "y = 2\n"}, "G2 test")
    wr_body = json.loads(json.dumps(w7b_body))
    for new, old in ((GNN, OGNN), (TWN, OTWN)):
        for top in ("w_ms_per_plan", "w_ms_per_plan_by_bucket"):
            if old in wr_body[top]:
                wr_body[top][new] = json.loads(json.dumps(wr_body[top][old]))
        wr_body[f"_source_{new}"] = f"copy of {old}"
    r = commit(repo, {G.W_FILE: wr_body}, "R w-only")
    for k, v in (("OS_BASE", base), ("OS_G1", g1), ("OS_G2", g2), ("OS_R", r), ("OS_SERVE", r), ("TW_P7", w7), ("TW_P7B", w7b),
                 ("OS_TOPOS", tuple(TOPOS)), ("OS_TOPOS_SHA256", hashlib.sha256(TOPS.encode()).hexdigest()),
                 ("OS_CELLS", {"new": 24 * 80, "old": 4 * 80, "bar": 80, "react": 80})):
        monkeypatch.setattr(G, k, v)
    ck = tmp_path / "m.pt"
    ck.write_bytes(b"weights")
    h = hashlib.md5(b"weights").hexdigest()
    cm = tmp_path / "ck.json"
    cm.write_text(json.dumps({**{f"{k}|s{s}": h for k in (GNN, TWN) for s in range(1, 7)}, **{f"{k}|s{s}": h for k in (OGNN, OTWN) for s in (1, 2)}}))
    logs = tmp_path / "md5_dgu.json"
    logs.write_text(json.dumps({"_snapshot_time": "t", "_provenance": {}, **{f"{k}|s{s}": h for k in (GNN, TWN) for s in range(1, 7)}}, indent=1))
    monkeypatch.setattr(G, "OS_CKPT_SNAPSHOT_MD5", G.md5(str(logs)))
    ig = tmp_path / "inputs_group"
    for rr in G.RUNGS:
        (ig / f"wf1_{rr}" / "wl").mkdir(parents=True)
        (ig / f"wf1_{rr}" / "wl" / "grounded_g0_n50000.json").write_text(f"{rr} wl\n")
    (ig / "MD5SUMS").write_text("".join(f"{hashlib.md5(f'{rr} wl\n'.encode()).hexdigest()}  wf1_{rr}/wl/grounded_g0_n50000.json\n" for rr in G.RUNGS))
    monkeypatch.setattr(G, "OS_INPUTS_MD5SUMS", G.md5(str(ig / "MD5SUMS")))
    ci = tmp_path / "in_new"
    ci.mkdir()
    for rr in G.RUNGS:
        os.symlink(ig / f"wf1_{rr}", ci / f"wf1_{rr}")
    dirs = {x: tmp_path / x for x in ("new", "old", "bar", "react", "ident", "fl3_bar", "fl3_learned", "fl3_twin")}
    for x in dirs.values():
        x.mkdir()
    wr = lambda rung: {"heavy": 6.5, "moderate": 6.9}[rung]
    tbyb = lambda rung, k: json.dumps({b: k * v for b, v in w7b_body["w_ms_per_plan_by_bucket"][OTWN][rung].items()}, sort_keys=True)
    env_g = lambda rung, k: {"HEROSIM_DECISION_COST_MS": repr(k * wr(rung))}
    env_t = lambda rung, k: {"HEROSIM_DECISION_COST_MS_BY_BUCKET": tbyb(rung, k), **NOMP}
    # the filed fl3 cells and their identity reruns at R (same numbers, other pin), each with its per-task array
    def tasks(d_):
        for f in d_.glob("*.summary.json"):
            np.save(str(f).replace(".summary.json", ".tasks.npy"), np.asarray([[0.1, 2.0], [0.2, 2.1]], dtype=np.float32))
    for d_, at in ((dirs["fl3_bar"], base), (dirs["ident"], r)):
        cell(str(d_), BAR, 0, 30149, "g0", "heavy", 2.0, busy=0.1, commit=at, servers=24)
    for d_, at in ((dirs["fl3_learned"], w7), (dirs["ident"], r)):
        cell(str(d_), OGNN, 1, 30149, "g0", "heavy", 1.9, str(ck), commit=at, env=env_g("heavy", 1.0), servers=24)
    for d_, at in ((dirs["fl3_twin"], w7b), (dirs["ident"], r)):
        cell(str(d_), OTWN, 1, 30149, "g0", "heavy", 1.95, str(ck), commit=at, env=env_t("heavy", 1.0), servers=24)
    for x in ("fl3_bar", "fl3_learned", "fl3_twin", "ident"):
        tasks(dirs[x])

    def fill(gnn=1.8, twn=2.0, seed6_gnn=None, twin_busy=1.0, new_env=None, new_at=None):
        for i, t in enumerate(TOPOS):
            jit = 0.0007 * (i + 1)
            for w in G.WINDOWS:
                for rung in G.RUNGS:
                    cell(str(dirs["bar"]), BAR, 0, t, w, rung, 2.0, busy=0.1, commit=base, servers=24)
                    react_cell(str(dirs["react"]), t, w, rung, base)
                    for sd in (1, 2):
                        cell(str(dirs["old"]), OGNN, sd, t, w, rung, 1.9 - jit, str(ck), commit=r, env=env_g(rung, 1.0), servers=24)
                        cell(str(dirs["old"]), OTWN, sd, t, w, rung, 1.95 - jit, str(ck), commit=r, env=env_t(rung, 1.0), servers=24)
                    for sd in range(1, 7):
                        g = (seed6_gnn if (sd == 6 and seed6_gnn is not None) else gnn) - jit
                        for sfx, k in (("_k100", 1.0), ("_kr", 1.468)):
                            cell(str(dirs["new"]), GNN, sd, t, w, rung, g, str(ck), commit=new_at or r, sfx=sfx, env={**env_g(rung, k), **(new_env or {})}, servers=24)
                            cell(str(dirs["new"]), TWN, sd, t, w, rung, twn - jit, str(ck), commit=r, sfx=sfx, env=env_t(rung, k), servers=24, busy=twin_busy)

    def read(stage="primary", out="p.json", **kw):
        args = ["--gate", str(dirs["new"]), "--topos", TOPS, "--topos-sha256", hashlib.sha256(TOPS.encode()).hexdigest(), "--pin", base[:8],
                "--repo", str(repo), "--primary", GNN, "--twin", TWN, "--checkpoint-md5", str(cm), "--bar", BAR, "--seeds", "1,2,3,4,5,6",
                "--confirm-own-state", "--old-gate", str(dirs["old"]), "--bar-gate", str(dirs["bar"]), "--react-gate", str(dirs["react"]),
                "--ident-gate", str(dirs["ident"]), "--ident-ref", ",".join(str(dirs[x]) for x in ("fl3_bar", "fl3_learned", "fl3_twin")),
                "--ckpt-snapshot", str(logs), "--inputs-group", str(ig), "--cell-inputs", str(ci), "--filed-ref", "HEAD", "--stage", stage,
                "--out", str(tmp_path / out)]
        for k, v in kw.items():
            args += [f"--{k}"] + ([] if v is None else [v])
        return G.main(args)

    def secondary(p):
        rec = commit(repo, {"docs/lineages/uncut_groups_s0_v1.md": f"# node\n- own-state primary md5 `{G.md5(p)}`\n"}, "file the primary")
        return read("secondary", "s.json", **{"primary-read": p, "record": rec})

    return {"fill": fill, "read": read, "secondary": secondary, "tmp": tmp_path, "repo": repo, "dirs": dirs, "logs": logs, "cm": cm, "ig": ig,
            "base": base, "r": r, "g1": g1, "g2": g2, "w7": w7, "w7b": w7b, "wr_body": wr_body}


@pytest.mark.parametrize("gnn, twn, seed6, label", [(1.8, 2.0, None, "MP-ATTRIBUTED"), (2.0, 2.0, None, "TIE"), (2.0, 1.8, None, "TWIN-FASTER"),
                                                     (1.985, 2.0, None, "MP-DIRECTION"), (1.8, 2.0, 1.995, "MP-DIRECTION")])
def test_own_state_labels_six_seeds_both_charges(os_, gnn, twn, seed6, label):
    os_["fill"](gnn=gnn, twn=twn, seed6_gnn=seed6)
    res = os_["read"]()
    assert res["combined"]["F2"] == res["F2"]["label"] == res["cotenancy"]["F2"]["label"] == label
    assert res["confirmation"]["label"].startswith(label + " (DAgger union corpus")
    assert sorted(res["F2"]["seeds_alone"]) == [1, 2, 3, 4, 5, 6] and res["F2"]["collapse_reference"] == BAR
    assert res["confirmation"]["grid_cells"] == G.OS_CELLS and "descriptive" not in res
    assert res["admission"]["admitted"] == len(TOPOS) and res["admission"]["unknown"] == 0
    assert set(res["checkpoint_snapshot_md5"]) == {f"{k}|s{s}" for k in (GNN, TWN) for s in range(1, 7)} and res["identity"].startswith("PASS")


def test_own_state_anchor_and_seed6_clause(os_):
    os_["fill"](twin_busy=5.0)
    res = os_["read"]()
    assert res["F2"]["anchor_ok"] == {"moderate": False, "heavy": False} and res["F2"]["label"] == "MP-DIRECTION"


def test_own_state_collapse_against_the_bar(os_):
    os_["fill"]()
    sp = os_["dirs"]["new"] / f"cc40s{TOPOS[0]}__g1heavy__{GNN}_k100_s6.summary.json"
    x = json.loads(sp.read_text())
    x["averageElapsedTime"] = 7.0  # > 3 x the bar's 2.0: seed 6's gnn collapses, the twin's seed 6 does not
    sp.write_text(json.dumps(x))
    res = os_["read"]()
    assert res["health"][GNN]["heavy"]["per_seed"][6]["collapse_any"] == 1 and res["F2"]["label"] != "MP-ATTRIBUTED"


@pytest.mark.parametrize("case, match", [
    ("missing", "VOID: --gate"), ("old_missing", "VOID: --old-gate"), ("react_missing", "VOID: --react-gate"),
    ("new_at_base", "not the serving pin"), ("charge", "served decision cost"), ("cellular", "outside the registered recipe"),
    ("tcp_block", "outside the registered recipe"), ("mp_on_twin", "env differs"), ("log_md5", "is not the filed"),
    ("inputs", "does not match MD5SUMS"), ("cell_inputs", "is not the minted"), ("ident", "misses its filed fl3 cell"), ("servers", "servers = 24")])
def test_own_state_fails_loud(os_, case, match):
    d = os_["dirs"]
    if case == "new_at_base":
        os_["fill"](new_at=os_["base"])
    elif case == "charge":
        os_["fill"](new_env={"HEROSIM_DECISION_COST_MS": "9.9"})
    elif case == "cellular":
        os_["fill"](new_env={"HEROSIM_ACCESS_CELLULAR_OUT_MBPS": "2.0"})
    else:
        os_["fill"]()
    t0 = TOPOS[0]
    if case == "missing":
        for f in d["new"].glob(f"cc40s{t0}__g2heavy__{TWN}_kr_s5.*"):
            f.unlink()
    elif case == "old_missing":
        for f in d["old"].glob(f"cc40s{t0}__g0moderate__{OTWN}_k100_s2.*"):
            f.unlink()
    elif case == "react_missing":
        for f in d["react"].glob(f"cc40s{t0}__g3heavy__reactive_s0.*"):
            f.unlink()
    elif case in ("tcp_block", "mp_on_twin", "ident", "servers"):
        name = {"tcp_block": d["new"] / f"cc40s{t0}__g0heavy__{GNN}_kr_s3.summary.json", "mp_on_twin": d["new"] / f"cc40s{t0}__g0heavy__{TWN}_k100_s4.summary.json",
                "ident": d["ident"] / f"cc40s30149__g0heavy__{OTWN}_k100_s1.summary.json", "servers": d["bar"] / f"cc40s{t0}__g0heavy__{BAR}_k100_s0.summary.json"}[case]
        x = json.loads(name.read_text())
        if case == "tcp_block":
            x["peerExchangeTcp"] = {"on": True}
        elif case == "mp_on_twin":
            x["env"].pop("GNN_DISABLE_MESSAGE_PASSING")
        elif case == "ident":
            x["totalPeerExchangeTime"] = 1.0
        else:
            x["servers"] = 64
        name.write_text(json.dumps(x))
    elif case == "log_md5":
        os_["logs"].write_text(os_["logs"].read_text().replace(f"{TWN}|s6", f"{TWN}|s7"))
    elif case == "inputs":
        (os_["ig"] / "wf1_heavy" / "wl" / "grounded_g0_n50000.json").write_text("capped payloads\n")
    elif case == "cell_inputs":
        ci = os_["tmp"] / "in_new" / "wf1_heavy"
        ci.unlink()
        ci.mkdir()
    with pytest.raises(SystemExit, match=match):
        os_["read"]()


def test_own_state_md5_file_must_equal_the_checkpoint_snapshot(os_):
    os_["fill"]()
    cm = json.loads(os_["cm"].read_text())
    cm[f"{GNN}|s3"] = "0" * 32
    os_["cm"].write_text(json.dumps(cm))
    with pytest.raises(SystemExit, match="the checkpoint snapshot"):
        os_["read"]()


def test_own_state_chain_refusals(os_, monkeypatch):
    repo, r = os_["repo"], os_["r"]
    assert set(G.os_chain(str(repo))) == {os_["base"], r}  # OS_SERVE == OS_R in the fixture
    # a w link that also moves a frozen key
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "bad", os_["g2"]], check=True)
    w = json.loads(json.dumps(os_["wr_body"]))
    w["w_ms_per_plan"]["cd_ib0"]["heavy"] = 1.5
    monkeypatch.setattr(G, "OS_R", commit(repo, {G.W_FILE: w}, "bad R"))
    with pytest.raises(SystemExit, match="changes frozen"):
        G.os_chain(str(repo))
    # a new kind whose w is not a copy of its filed arm's
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "bad2", os_["g2"]], check=True)
    w = json.loads(json.dumps(os_["wr_body"]))
    w["w_ms_per_plan"][GNN] = {"heavy": 6.6, "moderate": 6.9}
    monkeypatch.setattr(G, "OS_R", commit(repo, {G.W_FILE: w}, "R not a copy"))
    with pytest.raises(SystemExit, match="is not a copy"):
        G.os_chain(str(repo))
    # the w link touching another file
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "bad3", os_["g2"]], check=True)
    monkeypatch.setattr(G, "OS_R", commit(repo, {G.W_FILE: os_["wr_body"], "scripts_cosim/x.py": "1\n"}, "R with code"))
    with pytest.raises(SystemExit, match="not only"):
        G.os_chain(str(repo))
    # a gate link touching more than its file
    monkeypatch.setattr(G, "OS_G1_FILES", ("scripts_cosim/other.py",))
    monkeypatch.setattr(G, "OS_R", r)
    with pytest.raises(SystemExit, match="not exactly"):
        G.os_chain(str(repo))


@pytest.mark.parametrize("kw, match", [({"seeds": "1,2"}, "--seeds 1,2,3,4,5,6"), ({"confirm-mp-twin": None}, "its own family"),
                                       ({"twin": G.TWIN}, "--twin ra_twin_eng_combfl_dgu_nosplit"), ({"w-pins": "abc"}, "pins are fixed"),
                                       ({"record": "x"}, "seal"), ({"topos-sha256": "0" * 64}, "20508-20531"), ({"pin": "deadbeef"}, "bar's pin"),
                                       ({"mlp": G.CONFIRM_MLP}, "only")])
def test_own_state_arguments_fail_loud(os_, kw, match):
    os_["fill"]()
    with pytest.raises(SystemExit, match=match):
        os_["read"](**kw)


def test_own_state_secondary_only_after_the_primary_is_filed(os_):
    os_["fill"]()
    os_["read"]()
    p = str(os_["tmp"] / "p.json")
    head = subprocess.run(["git", "-C", str(os_["repo"]), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with pytest.raises(SystemExit, match="the own-state primary read .* is not filed"):
        os_["read"]("secondary", "s.json", **{"primary-read": p, "record": head})
    res = os_["secondary"](p)
    d = res["descriptive"]
    assert set(d) == {"note", "gnn_vs_bar", "new_vs_old_gnn", "new_vs_old_twin", "old_gnn_vs_old_twin"}
    assert all(len(d[k]) == 4 and all("p_holm" not in e for e in d[k]) for k in d if k != "note")
    assert all(e["median_pct"] < 0 for e in d["gnn_vs_bar"]) and all(e["median_pct"] < 0 for e in d["new_vs_old_gnn"])
    assert all(e["a_seeds"] == [1, 2] for e in d["new_vs_old_twin"]) and len(res["cotenancy"]["descriptive"]["gnn_vs_bar"]) == 4
    assert res["confirmation"]["primary_read"]["md5"] == G.md5(p) and res["F2"]["label"] == "MP-ATTRIBUTED"


def test_own_state_written_once(os_):
    os_["fill"]()
    os_["read"]()
    with pytest.raises(SystemExit, match="written once"):
        os_["read"]()


def test_own_state_flags_are_refused_by_the_filed_modes():
    for flag in ("--old-gate", "--bar-gate", "--react-gate", "--ident-gate", "--ident-ref", "--ckpt-snapshot", "--inputs-group", "--cell-inputs"):
        with pytest.raises(SystemExit, match="belong to --confirm-own-state"):
            G.main(["--gate", "x", "--topos", "1", "--topos-sha256", "y", "--pin", "x", "--primary", "p", "--checkpoint-md5", "c", flag, "m"])


def test_own_state_constants_are_the_registered_ones():
    assert hashlib.sha256(",".join(map(str, range(20508, 20532))).encode()).hexdigest() == G.OS_TOPOS_SHA256 == \
        "d4b1fb268c14af65a7c201ef27a97fbfb3a3033a964218cfbf443f060032ac37"
    assert sum(G.OS_CELLS.values()) == 5760 and G.OS_CELLS == {"new": 4608, "old": 768, "bar": 192, "react": 192}
    assert (G.OS_BASE[:8], G.OS_G1[:8], G.OS_G2[:8], G.OS_R[:8], G.OS_SERVE[:8]) == ("a459dd98", "271b6665", "0ab430d4", "211241a7", "05f52abb")
    assert G.OS_CKPT_SNAPSHOT_MD5 == "80e3b6b45f9f28a5fdcd65865fcd94a3"
    assert G.OS_INPUTS_MD5SUMS == "72a7db697771082a14916ad82e540507" and G.OS_SEEDS == (1, 2, 3, 4, 5, 6)


def test_own_state_serving_pin(os_, monkeypatch):
    repo, r = os_["repo"], os_["r"]
    later = commit(repo, {"scripts_cosim/later.py": "1\n"}, "later code, same w")
    monkeypatch.setattr(G, "OS_SERVE", later)
    assert set(G.os_chain(str(repo))) == {os_["base"], r, later}
    w = json.loads(json.dumps(os_["wr_body"]))
    w["w_ms_per_plan"][GNN] = {"heavy": 9.9, "moderate": 9.9}
    monkeypatch.setattr(G, "OS_SERVE", commit(repo, {G.W_FILE: w}, "later w moved"))
    with pytest.raises(SystemExit, match="serving pin .* differs"):
        G.os_chain(str(repo))
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "side", os_["g2"]], check=True)
    monkeypatch.setattr(G, "OS_SERVE", commit(repo, {"scripts_cosim/side.py": "1\n"}, "not on R"))
    with pytest.raises(SystemExit, match="does not descend"):
        G.os_chain(str(repo))


def test_own_state_identity_allows_the_new_telemetry_only(os_):
    os_["fill"]()
    f = os_["dirs"]["ident"] / f"cc40s30149__g0heavy__{BAR}_k100_s0.summary.json"
    x = json.loads(f.read_text())
    x["peerExchangeByAccessClass"] = {"wired+wired": 1}
    x["peerExchangeOverlap"] = {"pairs": 2}
    x.setdefault("schedulerCounters", {}).update({"pg_fsaware_decisions": 0, "pg_fsaware_decisions_nonzero": 0})
    f.write_text(json.dumps(x))
    assert os_["read"]()["identity"].startswith("PASS")
    x["schedulerCounters"]["pg_fsaware_decisions"] = 3
    f.write_text(json.dumps(x))
    with pytest.raises(SystemExit, match="misses its filed fl3 cell"):
        os_["read"](out="p2.json")


def _move_to(os_, sha):
    for d in ("new", "old", "ident"):
        for f in os_["dirs"][d].glob("*.summary.json"):
            x = json.loads(f.read_text())
            old = x["code"]["commit"]
            x["code"]["commit"] = sha
            f.write_text(json.dumps(x))
            lg = Path(str(f).replace(".summary.json", ".log"))
            if lg.exists():
                lg.write_text(lg.read_text().replace(old, sha).replace(old[:8], sha[:8]))


def test_own_state_reads_end_to_end_at_a_later_serving_pin(os_, monkeypatch):
    """The served cells (learned and identity) at a descendant of the retest pin with code changes and the same w: the whole read passes."""
    later = commit(os_["repo"], {"scripts_cosim/later.py": "1\n"}, "later code, same w")
    monkeypatch.setattr(G, "OS_SERVE", later)
    os_["fill"]()
    _move_to(os_, later)
    res = os_["read"]()
    assert res["F2"]["label"] == "MP-ATTRIBUTED" and res["serving_pin"] == later and res["identity"].startswith("PASS")


def test_own_state_refuses_cells_at_the_retest_pin_when_serving_is_later(os_, monkeypatch):
    later = commit(os_["repo"], {"scripts_cosim/later.py": "1\n"}, "later code, same w")
    monkeypatch.setattr(G, "OS_SERVE", later)
    os_["fill"]()
    with pytest.raises(SystemExit, match="not the serving pin"):
        os_["read"]()
