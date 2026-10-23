"""Tests for scripts_cosim/target_arms_read.py on a synthetic tree (three topologies)."""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import target_arms_read as T  # noqa: E402

TOPOS = [1, 2, 3]
# gnn / bar ratio per target: the gain shrinks as the target falls
RATIO = {0.7: 0.90, 0.45: 0.96, 0.294: 1.02}
BAR_BASE = {0.7: 3.0, 0.45: 2.5, 0.294: 2.0}


def elapsed(arm, t, tp, w, sd):
    b = BAR_BASE[t] + tp * 0.1 + int(w[1:]) * 0.05
    return b if arm == "bar" else b * RATIO[t]


def summary(cfg, key, md5s, models):
    arm, t, sd, tp, w = key
    e = elapsed(arm, t, tp, w, sd)
    s = {"num_tasks": 100, "code": {"commit": T.PIN + "0" * 32, "dirty": False}, "servers": 24, "topology": tp,
         "scaleOut": {"mode": "kpa", "target": t}, "env": {}, "peerExchangeByAccessClass": {"a>b": {"transfers": 40, "seconds": 80.0}},
         "averageElapsedTime": e, "steady_state": {"after_360s": {"mean": e * 0.8}}, "totalPeerExchangeTime": 150.0,
         "lock_wait": {"mean": 0.3}, "averageColdStartTime": 0.1, "endTime": 1000.0, "arrival_end": {"last_arrival_s": 990.0},
         "replica_count_series": {"t": [0.0, 500.0, 1000.0], "total": [0, 100, 120], "time_mean": 50.0 + 100 * (0.7 - t)}}
    if arm == "gnn":
        s["env"]["GNN_MODEL_PATH"] = models[sd]
    return s


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "TOPOS", TOPOS)
    U = str(tmp_path)
    cfg = T.make_cfg(U)
    models, md5s = {}, {}
    os.makedirs(f"{U}/models")
    os.makedirs(f"{U}/comb")
    for sd in (1, 2):
        p = f"{U}/models/{T.CKPT_FILE.format(sd)}"
        open(p, "wb").write(f"g{sd}".encode())
        models[sd] = p
        md5s[f"{T.CKPT_KEY}|s{sd}"] = hashlib.md5(f"g{sd}".encode()).hexdigest()
    json.dump(md5s, open(cfg["ckpt"], "w"))
    for w in T.WINDOWS:
        os.makedirs(f"{cfg['wl']}/wf1_moderate/wl", exist_ok=True)
        json.dump({"peer_exchange": list(range(60))}, open(f"{cfg['wl']}/wf1_moderate/wl/grounded_{w}_n50000.json", "w"))
    g = T.grid(cfg)
    for key, stem in g.items():
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        json.dump(summary(cfg, key, md5s, models), open(stem + ".summary.json", "w"))
    lines = [f"{os.path.dirname(stem)[len(U) + len('/anomaly/'):]}\t{os.path.basename(stem)}\tR1A_QUEUE_LENGTH={key[1]}\tTCP=0"
             for key, stem in g.items() if (key[0], key[1]) in T.NEW]
    cf = f"{U}/cells.txt"
    open(cf, "w").write("\n".join(lines) + "\n")
    return cfg, g, md5s, cf


def load(cfg, md5s):
    return T.load_all(cfg, T.Ckpt(cfg, md5s))


def test_list_equals_the_new_cells_only(tree):
    cfg, g, md5s, cf = tree
    assert T.check_list(cfg, cf, None) == 2 * 3 * 4 * 2
    open(cf, "a").write("dose_7cb483c5/cells_gnn_mod045\tcc40s9__g0moderate__x_s1\tR1A_QUEUE_LENGTH=0.45\tTCP=0\n")
    with pytest.raises(SystemExit, match="cell list and grid differ"):
        T.check_list(cfg, cf, None)
    with pytest.raises(SystemExit, match="cell list md5"):
        T.check_list(cfg, cf, "0" * 32)


def test_primary_holm_and_descriptive_rows(tree):
    cfg, g, md5s, cf = tree
    cells, rows, n = load(cfg, md5s)
    assert n == len(g)
    res = T.build(cells, rows)
    ft = res["ft"]
    assert [(e["a"], e["stat"]) for e in ft] == [("gnn@0.45", "whole"), ("gnn@0.45", "after360"), ("gnn@0.294", "whole"), ("gnn@0.294", "after360")]
    assert [round(e["median_pct"], 6) for e in ft] == [-4.0, -4.0, 2.0, 2.0] and all("p_holm" in e for e in ft)
    d7 = res["desc"][("gnn", 0.7, "bar", 0.7)]
    assert round(d7[0]["median_pct"], 6) == -10.0 and "p_holm" not in d7[0]
    x = res["desc"][("gnn", 0.7, "bar", 0.294)][0]
    assert x["faster"] == 0 and x["median_pct"] > 0
    m, up, p = res["change"][("whole", 0.7, 0.45)]
    assert m == pytest.approx(6.0) and up == 3
    txt = T.report(res, rows, T.health(cells, rows), 192, n)
    assert "primary (Table I)" in txt and "replica-s/task" in txt and "F-target (Holm 4)" in txt
    assert "slower than the bar @0.294 on 12/12" in txt
    assert "the item holds only if both: True" in txt and "whole True" in txt and "after 360 s True" in txt


def test_replica_seconds_per_task(tree):
    cfg, g, md5s, cf = tree
    cells, rows, _ = load(cfg, md5s)
    r = rows[("bar@0.7", 0, 1, "g0")]
    assert r["replica_s_per_task"] == pytest.approx(50.0 * 1000.0 / 100)
    assert r["at_cap_share"] == pytest.approx(0.5)


def test_missing_failed_and_asserts(tree):
    cfg, g, md5s, cf = tree
    key = next(k for k in g if k[0] == "gnn" and k[1] == 0.45)
    p = g[key] + ".summary.json"
    s = json.load(open(p))
    for mut, msg in ((lambda x: x["scaleOut"].update(target=0.7), "scaleOut"), (lambda x: x["code"].update(dirty=True), "pin"),
                     (lambda x: x.update(peerExchangeTcp={}), "TCP"), (lambda x: x["scaleOut"].update(knative_panic=True), "scaleOut"),
                     (lambda x: x.update(servers=64), "servers")):
        t = json.loads(json.dumps(s))
        mut(t)
        json.dump(t, open(p, "w"))
        with pytest.raises(SystemExit, match=msg):
            load(cfg, md5s)
    json.dump(s, open(p, "w"))
    open(g[key] + ".failed.json", "w").write("{}")
    with pytest.raises(SystemExit, match="failed cell"):
        load(cfg, md5s)
    os.remove(g[key] + ".failed.json")
    os.remove(p)
    with pytest.raises(SystemExit, match="missing cell"):
        load(cfg, md5s)


def test_checkpoint_asserts(tree):
    cfg, g, md5s, cf = tree
    with pytest.raises(SystemExit, match="checkpoint md5"):
        load(cfg, {k: "0" * 32 for k in md5s})
    key = next(k for k in g if k[0] == "gnn" and k[2] == 2)
    p = g[key] + ".summary.json"
    s = json.load(open(p))
    s["env"]["GNN_MODEL_PATH"] = s["env"]["GNN_MODEL_PATH"].replace("seed2", "seed1")
    json.dump(s, open(p, "w"))
    with pytest.raises(SystemExit, match="model path"):
        load(cfg, md5s)


def test_main_writes_once(tree, monkeypatch):
    cfg, g, md5s, cf = tree
    monkeypatch.setenv("TARGET_ARMS_U", cfg["U"])
    monkeypatch.setattr(T, "CELLS_MD5", hashlib.md5(open(cf, "rb").read()).hexdigest())
    out = f"{cfg['U']}/read.txt"
    T.main(["x", cf, "--out", out])
    assert os.path.exists(out)
    with pytest.raises(SystemExit, match="written once"):
        T.main(["x", cf, "--out", out])


def test_frozen_modules_are_blob_checked():
    here = os.path.join(os.path.dirname(__file__), "..", "scripts_cosim")
    assert T.G.__frozen_blob__ == T.FROZEN_BLOBS["uncut_arm_gate_read.py"] and T.G.__name__ == "uncut_arm_gate_read@04e368af"
    for f, want in T.FROZEN_BLOBS.items():
        assert T.git_blob(os.path.join(here, f)) == want or f == "uncut_arm_gate_read.py"  # the reader has since gained additions; loaded at its blob
