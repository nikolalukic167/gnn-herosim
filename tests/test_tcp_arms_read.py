"""Tests for scripts_cosim/tcp_arms_read.py on a synthetic tree (three topologies, 24 servers): the grid, the asserts and the statistics."""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import tcp_arms_read as T  # noqa: E402

SIZE, TOPOS = 24, [1, 2, 3]
FACT = {"off": {"bar": 1.0, "gnn": 0.90, "mlp": 0.92, "twin": 0.93}, "on": {"bar": 1.0, "gnn": 0.95, "mlp": 0.96, "twin": 0.97}}


def elapsed(mode, arm, t, w, r, sd):
    base = 10.0 + t + int(w[1:]) + (5.0 if r == "heavy" else 0.0)
    if w == "g3" and t == 1 and r == "moderate":
        base *= 2.0 if arm == "bar" else 1.0
    return base * FACT[mode][arm]


def summary(cfg, key, stem, md5s, models):
    mode, size, arm, sd, t, w, r = key
    e = elapsed(mode, arm, t, w, r, sd)
    pin = cfg["pins"][os.path.dirname(stem)]
    s = {"num_tasks": 100, "code": {"commit": pin + "0" * (40 - len(pin)), "dirty": False}, "servers": size, "topology": t,
         "scaleOut": {"target": 0.7}, "env": {}, "peerExchangeByAccessClass": {"a>b": {"transfers": 40, "seconds": 80.0}},
         "averageElapsedTime": e, "steady_state": {"after_360s": {"mean": e * 0.8}}, "totalPeerExchangeTime": 150.0 if mode == "off" else 200.0,
         "lock_wait": {"mean": 0.3}, "replica_count_series": {"time_mean": 40.0 if mode == "off" else 44.0}, "averageColdStartTime": 0.1,
         "endTime": 1000.0, "arrival_end": {"last_arrival_s": 990.0}}
    if mode == "on":
        s["peerExchangeTcp"] = {**T.TCP_BLOCK, "transfers": 40, "extra_seconds": 50.0}
    if arm != "bar":
        s["env"]["GNN_MODEL_PATH"] = models[(arm, sd)]
    return s


@pytest.fixture
def tree(tmp_path):
    U = str(tmp_path)
    cfg = T.make_cfg(U)
    cfg["topos"] = {SIZE: TOPOS}
    models, md5s = {}, {}
    os.makedirs(f"{U}/models")
    for arm in ("gnn", "mlp", "twin"):
        for sd in (1, 2):
            p = f"{U}/models/{T.CKPT_FILE[arm].format(sd)}"
            open(p, "wb").write(f"{arm}{sd}".encode())
            models[(arm, sd)] = p
            md5s[f"{T.CKPT_KEY[arm]}|s{sd}"] = hashlib.md5(f"{arm}{sd}".encode()).hexdigest()
    for r in T.RUNGS:
        for w in T.WINDOWS:
            os.makedirs(f"{cfg['wl'][SIZE]}/wf1_{r}/wl", exist_ok=True)
            json.dump({"peer_exchange": list(range(60))}, open(f"{cfg['wl'][SIZE]}/wf1_{r}/wl/grounded_{w}_n50000.json", "w"))
    g = T.grid(cfg)
    for key, stem in g.items():
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        json.dump(summary(cfg, key, stem, md5s, models), open(stem + ".summary.json", "w"))
    lines = []
    for key, stem in g.items():
        d = os.path.dirname(stem)
        if d.startswith(f"{cfg['A']}/tcp_arms/"):
            lines.append(f"{d[len(cfg['A']) + 1:]}\t{os.path.basename(stem)}\t{'TCP=1' if key[0] == 'on' else 'TCP=0'}")
    lines.append("tcp_arms/ident24\tcc40s1__g0heavy__x_s1\tTCP=0\tvs comb/cells_comb")
    cf = f"{U}/cells.txt"
    open(cf, "w").write("\n".join(lines) + "\n")
    return cfg, g, md5s, cf


def load(cfg, md5s):
    return T.load_all(cfg, T.Ckpt(cfg, md5s))


def test_grid_is_exact_and_list_matches(tree):
    cfg, g, md5s, cf = tree
    n, ident = T.check_list(cfg, cf, None)
    assert ident == 1 and n == 96 + 48  # on: MLP and twin; off: twin
    cells, rows, nc = load(cfg, md5s)
    assert nc == len(g)


def test_list_and_grid_must_agree(tree):
    cfg, g, md5s, cf = tree
    open(cf, "a").write("tcp_arms/on24\tcc40s9__g0moderate__ra_twin_eng_combfl_nosplit_k100_s1\tTCP=1\n")
    with pytest.raises(SystemExit, match="cell list and grid differ"):
        T.check_list(cfg, cf, None)


def test_primary_is_the_frozen_statistic_and_window_mean_differs(tree):
    cfg, g, md5s, cf = tree
    cells, rows, _ = load(cfg, md5s)
    es = T.primary(cells["on"][SIZE], "gnn", "bar", TOPOS)
    assert [round(e["median_pct"], 6) for e in es] == [-5.0] * 4 and all(e["faster"] == 3 for e in es)
    wm = T.window_mean_vals(rows["on"][SIZE], "gnn", "bar", "moderate", TOPOS, "elapsed")
    assert wm[2] == pytest.approx(-5.0) and abs(wm[1] + 5.0) > 0.5


def test_shift_and_families(tree):
    cfg, g, md5s, cf = tree
    cells, rows, _ = load(cfg, md5s)
    res = T.build(cfg, cells, rows)
    f = res[SIZE]["contrasts"]["F-bar"]
    assert all(s["shift_median"] == pytest.approx(5.0) and s["toward_cmp"] == 3 for s in f["shift"])
    assert all("p_holm" in e for e in f["per"]["on"]) and "p_holm" not in res[SIZE]["desc"]["mlp-vs-bar"]["per"]["on"][0]
    txt = T.report(cfg, res, T.health(cfg, cells, rows), T.arm_table(cfg, rows), (0, 1), 0)
    assert "primary (Table I)" in txt and "window-mean" in txt and "MP rule at 24" in txt


def test_missing_and_failed_cells_fail_loud(tree):
    cfg, g, md5s, cf = tree
    key = next(k for k in g if k[0] == "on" and k[2] == "twin")
    os.remove(g[key] + ".summary.json")
    with pytest.raises(SystemExit, match="missing cell"):
        load(cfg, md5s)
    open(g[key] + ".failed.json", "w").write("{}")
    with pytest.raises(SystemExit, match="failed cell"):
        load(cfg, md5s)


@pytest.mark.parametrize("mut,msg", [
    (lambda s: s["code"].update(commit="deadbeef" + "0" * 32), "pin"),
    (lambda s: s["code"].update(dirty=True), "pin"),
    (lambda s: s.update(servers=64), "servers"),
    (lambda s: s["scaleOut"].update(target=0.45), "scaleOut"),
    (lambda s: s["env"].update(HEROSIM_KPA_KNATIVE_PANIC="1"), "env"),
])
def test_cell_assert_failures(tree, mut, msg):
    cfg, g, md5s, cf = tree
    key = next(k for k in g if k[0] == "off" and k[2] == "bar")
    p = g[key] + ".summary.json"
    s = json.load(open(p))
    mut(s)
    json.dump(s, open(p, "w"))
    with pytest.raises(SystemExit, match=msg):
        load(cfg, md5s)


def test_tcp_block_must_match_the_mode(tree):
    cfg, g, md5s, cf = tree
    on = next(k for k in g if k[0] == "on" and k[2] == "bar")
    p = g[on] + ".summary.json"
    s = json.load(open(p))
    del s["peerExchangeTcp"]
    json.dump(s, open(p, "w"))
    with pytest.raises(SystemExit, match="TCP block"):
        load(cfg, md5s)
    s["peerExchangeTcp"] = {**T.TCP_BLOCK, "transfers": 40, "extra_seconds": 1.0}
    json.dump(s, open(p, "w"))
    off = next(k for k in g if k[0] == "off" and k[2] == "bar")
    q = g[off] + ".summary.json"
    t = json.load(open(q))
    t["peerExchangeTcp"] = s["peerExchangeTcp"]
    json.dump(t, open(q, "w"))
    with pytest.raises(SystemExit, match="switch-off"):
        load(cfg, md5s)


def test_checkpoint_md5_and_name_are_asserted(tree):
    cfg, g, md5s, cf = tree
    bad = dict(md5s)
    bad[f"{T.CKPT_KEY['twin']}|s1"] = "0" * 32
    with pytest.raises(SystemExit, match="checkpoint md5"):
        load(cfg, bad)
    key = next(k for k in g if k[2] == "mlp" and k[3] == 2)
    p = g[key] + ".summary.json"
    s = json.load(open(p))
    s["env"]["GNN_MODEL_PATH"] = s["env"]["GNN_MODEL_PATH"].replace("seed2", "seed1")
    json.dump(s, open(p, "w"))
    with pytest.raises(SystemExit, match="model path"):
        load(cfg, md5s)


def test_frozen_modules_are_blob_checked():
    here = os.path.join(os.path.dirname(__file__), "..", "scripts_cosim")
    assert T.G.__frozen_blob__ == T.FROZEN_BLOBS["uncut_arm_gate_read.py"] and T.G.__name__ == "uncut_arm_gate_read@04e368af"
    for f, want in T.FROZEN_BLOBS.items():
        assert T.git_blob(os.path.join(here, f)) == want or f == "uncut_arm_gate_read.py"  # the reader has since gained additions; loaded at its blob


def test_ident_cells_never_in_grid(tree):
    cfg, g, md5s, cf = tree
    assert not any("tcp_arms/ident" in s for s in g.values())


# ---------------------------------------------------------------- identity cells
import numpy as np  # noqa: E402
import tcp_arms_ident as I  # noqa: E402


def _ident_pair(tmp_path, arm="ra_gnn_eng_combfl_nosplit", extra_new=None, rerun_edit=None, array_edit=False):
    U = tmp_path
    name = f"cc40s1__g0heavy__{arm}_k100_s1"
    old, new = U / "size64" / "cells_learned" / name, U / "anomaly" / "tcp_arms" / "ident64" / name
    model = U / "models" / I.CKPT[arm].format(1)
    model.parent.mkdir(exist_ok=True)
    model.write_bytes(b"w")
    base = {"averageElapsedTime": 2.0, "scaleOut": {"target": 0.7}, "peerExchangeByAccessClass": {"a>b": {"transfers": 3, "seconds": 4.0}},
            "env": {"GNN_MODEL_PATH": str(model)}, "code": {"commit": "74ddf70e" + "0" * 32, "dirty": False}, "wallclock_s": 10.0}
    rerun = json.loads(json.dumps(base))
    rerun["code"] = {"commit": I.PIN_ON, "dirty": False}
    rerun["wallclock_s"] = 99.0
    rerun["peerExchangeOverlap"] = {"x": 1}
    rerun["peerExchangeByAccessClass"]["a>b"]["bytes"] = 5
    if extra_new:
        rerun[extra_new] = 1
    if rerun_edit:
        rerun_edit(rerun)
    for d, s in ((old, base), (new, rerun)):
        d.parent.mkdir(parents=True, exist_ok=True)
        json.dump(s, open(str(d) + ".summary.json", "w"))
        for e in I.ARRAYS:
            np.save(str(d) + e[:-4], np.arange(5) + (1 if array_edit and d is new and e == ".tasks.npy" else 0))
    return str(new), str(old), {f"{arm}|s1": hashlib.md5(b"w").hexdigest()}


def test_ident_passes_with_named_additions(tmp_path):
    new, old, m = _ident_pair(tmp_path)
    why, news, _ = I.judge(new, old, m, {})
    assert not why and news == ["peerExchangeByAccessClass.a>b.bytes", "peerExchangeOverlap"]


@pytest.mark.parametrize("kw,msg", [
    ({"extra_new": "surprise"}, "fields"),
    ({"rerun_edit": lambda r: r.update(averageElapsedTime=2.1)}, "fields"),
    ({"rerun_edit": lambda r: r.update(peerExchangeTcp={})}, "TCP"),
    ({"rerun_edit": lambda r: r["code"].update(commit="0" * 40)}, "pin"),
    ({"array_edit": True}, "arrays"),
])
def test_ident_failures(tmp_path, kw, msg):
    new, old, m = _ident_pair(tmp_path, **kw)
    why, _, _ = I.judge(new, old, m, {})
    assert why and msg in " ".join(why)


def test_ident_checkpoint_md5(tmp_path):
    new, old, m = _ident_pair(tmp_path)
    why, _, _ = I.judge(new, old, {k: "0" * 32 for k in m}, {})
    assert "checkpoint md5" in " ".join(why)


# ---------------------------------------------------------------- per-size runs
def test_list_restricted_to_the_size_but_whole_list_md5_checked(tree):
    cfg, g, md5s, cf = tree
    open(cf, "a").write("tcp_arms/on64\tcc40s30129__g0moderate__ra_twin_eng_combfl_nosplit_k100_s1\tTCP=1\n"
                        "tcp_arms/ident64\tcc40s30129__g0heavy__x_s1\tTCP=0\tvs size64/cells_bar\n")
    n, ident = T.check_list(cfg, cf, None)
    assert (n, ident) == (144, 1)
    with pytest.raises(SystemExit, match="cell list md5"):
        T.check_list(cfg, cf, "0" * 32)


def test_main_reads_one_size_once_and_refuses_a_partial_size(tree, monkeypatch, capsys):
    cfg, g, md5s, cf = tree
    U = cfg["U"]
    os.makedirs(f"{U}/comb", exist_ok=True)
    os.makedirs(f"{U}/fl3", exist_ok=True)
    json.dump({k: v for k, v in md5s.items() if "twin" not in k}, open(f"{U}/comb/md5_comball.json", "w"))
    json.dump({k: v for k, v in md5s.items() if "twin" in k}, open(f"{U}/fl3/md5_twin.json", "w"))
    monkeypatch.setenv("TCP_ARMS_U", U)
    monkeypatch.setattr(T, "CELLS_MD5", hashlib.md5(open(cf, "rb").read()).hexdigest())
    monkeypatch.setitem(T.TOPOS, 24, TOPOS)
    out = f"{U}/read24.txt"
    T.main(["x", cf, "--size", "24", "--out", out])
    assert os.path.exists(out) and "24 servers" in open(out).read() and "64 servers" not in open(out).read()
    with pytest.raises(SystemExit, match="read once"):
        T.main(["x", cf, "--size", "24", "--out", out])
    key = next(k for k in g if k[0] == "on" and k[2] == "mlp")
    os.remove(g[key] + ".summary.json")
    with pytest.raises(SystemExit, match="missing cell"):
        T.main(["x", cf, "--size", "24", "--out", f"{U}/read24b.txt"])
    assert not os.path.exists(f"{U}/read24b.txt")
    with pytest.raises(SystemExit):
        T.main(["x", cf])


def test_ident_main_is_per_size_and_counts_entries(tmp_path, monkeypatch):
    new, old, m = _ident_pair(tmp_path)
    cf = tmp_path / "cells.txt"
    cf.write_text("tcp_arms/ident64\tcc40s1__g0heavy__ra_gnn_eng_combfl_nosplit_k100_s1\tTCP=0\tvs size64/cells_learned\n")
    monkeypatch.setenv("TCP_ARMS_U", str(tmp_path))
    monkeypatch.setattr(I, "CELLS_MD5", hashlib.md5(cf.read_bytes()).hexdigest())
    os.makedirs(tmp_path / "comb")
    os.makedirs(tmp_path / "fl3")
    json.dump(m, open(tmp_path / "comb" / "md5_comball.json", "w"))
    json.dump({}, open(tmp_path / "fl3" / "md5_twin.json", "w"))
    with pytest.raises(SystemExit, match="1 identity entries at 64, want 4"):
        I.main(["x", str(cf), "--size", "64"])
    with pytest.raises(SystemExit):
        I.main(["x", str(cf)])
