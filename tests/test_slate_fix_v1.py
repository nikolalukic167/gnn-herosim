"""slate_fix_v1: serving slate from the sidecar, sidecheck rule, cache check, configs and the gate reader (docs/lineages/archive/slate_fix_v1.md)."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_cosim"))

UNC, CAP = "declared_pruning_v1_uncapped", "declared_pruning_v1"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.delenv("R1A_SLATE", raising=False)
    return importlib.import_module("fresh_topo_burst_v1_gate")


def _side(tmp_path, rule):
    p = tmp_path / "ck.contract.json"
    p.write_text(json.dumps({"candidate_slate": rule}))
    return str(p)


def test_serving_slate_default_is_the_capped_rule(gate, tmp_path):
    assert gate.r1a_slate() == CAP
    assert gate.ra_serving_slate(_side(tmp_path, CAP)) == CAP
    with pytest.raises(SystemExit):
        gate.ra_serving_slate(_side(tmp_path, UNC))


def test_serving_slate_uncapped_when_pinned(gate, tmp_path, monkeypatch):
    monkeypatch.setenv("R1A_SLATE", UNC)
    assert gate.ra_serving_slate(_side(tmp_path, UNC)) == UNC
    with pytest.raises(SystemExit):
        gate.ra_serving_slate(_side(tmp_path, CAP))
    monkeypatch.setenv("R1A_SLATE", "top7")
    with pytest.raises(SystemExit):
        gate.r1a_slate()


def test_twin_nosplit_is_a_gate_arm(gate):
    assert "ra_twin_eng_nosplit" in gate.R1A_ARMS


def _sidecheck(tmp_path, rule, env_rule):
    split = tmp_path / "split.json"
    split.write_text("{}")
    import hashlib
    sc = {"arm_kind": "gnn", "disable_message_passing": False, "partial_state_contract": "partial_state_v5", "load_seconds": True,
          "exchange_seconds": True, "full_context_ce_weight": 1.0, "dag_alpha_key": "inf", "mp_residual": False, "plan_raw": False,
          "plan_raw_sum": False, "plan_raw_local": False, "mp_bipartite_edge_conv": True, "mp_bipartite_edge_attr_zero": True,
          "mp_peer_edges": True, "partial_state_feature_dim": 27, "candidate_slate": rule, "physics_env": {"x": 1},
          "split_artifact": {"sha256": hashlib.sha256(split.read_bytes()).hexdigest()}}
    side = tmp_path / "c.contract.json"
    side.write_text(json.dumps(sc))
    env = {"PATH": "/usr/bin:/bin", **({"R1A_SLATE": env_rule} if env_rule else {})}
    return subprocess.run([sys.executable, str(ROOT / "scripts_cosim/joint_burst_v2_sidecheck.py"), str(side), "ra_gnn_eng", str(split), "inf"],
                          env=env, capture_output=True, text=True).returncode


@pytest.mark.parametrize("rule,env_rule,ok", [(CAP, None, True), (UNC, None, False), (UNC, UNC, True), (CAP, UNC, False)])
def test_sidecheck_slate_rule(tmp_path, rule, env_rule, ok):
    assert (_sidecheck(tmp_path, rule, env_rule) == 0) is ok


def test_configs_are_the_g4_recipe():
    g4 = yaml.safe_load((ROOT / "experiments/accel_replica_v1_gnn_eng_g4.yaml").read_text())
    for arm in ("gnn_eng", "twin_eng"):
        c = yaml.safe_load((ROOT / f"experiments/slate_fix_v1_{arm}.yaml").read_text())
        assert c["args"] == g4["args"] and c["unset_env"] == g4["unset_env"] and c["lineage"] == "slate_fix_v1"
        assert c["cache_dir"] == "/share/nikola.lukic/accel_uncap/4f41b771/cache_run/cache"  # S7's uncapped recache
        env = dict(g4["env"])
        if arm == "twin_eng":
            for k in ("NEAR_RTT_MP_BIPARTITE_EDGE_CONV", "NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO"):
                env.pop(k)
            env["GNN_DISABLE_MESSAGE_PASSING"] = "1"
        assert c["env"] == env


def test_cache_check():
    cc = importlib.import_module("slate_fix_v1_cache_check")
    base = {k: f"v_{k}" for k in cc.SAME_KEYS}
    old = {**base, "candidate_slate": CAP, "parent_dataset_ids": ["a", "b"]}
    new = {**base, "candidate_slate": UNC, "candidate_slate_rules": [CAP, UNC], "parent_dataset_ids": ["b", "a"]}
    split = {"train": ["a"], "val": ["b"], "test": []}
    assert cc.check(new, old, split, cc.SPLIT_SHA) == []
    assert cc.check({**new, "candidate_slate": CAP}, old, split, cc.SPLIT_SHA)
    assert cc.check({**new, "partial_state_contract": "x"}, old, split, cc.SPLIT_SHA)
    assert cc.check({**new, "parent_dataset_ids": ["a"]}, old, split, cc.SPLIT_SHA)
    assert cc.check(new, old, split, "0" * 64)
    assert cc.check(new, old, {"train": ["a"], "val": [], "test": []}, cc.SPLIT_SHA)


def _cell(d, topo, win, rung, kind, seed, lat):
    q = {"quarter_mean_queue": [5.0 * lat, lat / 10, lat / 10, lat / 10]}  # a cold first quarter, then flat
    (d / f"cc40s{topo}__{win}{rung}__{kind}_s{seed}.summary.json").write_text(json.dumps({"averageElapsedTime": lat, "wallclock_s": 1.0, "num_tasks": 50000, "queue_drift": q}))


def _gate_dirs(tmp_path, new_pct, old_pct, twin_pct=0.0):
    new, old = tmp_path / "new", tmp_path / "old"
    new.mkdir(), old.mkdir()
    for i, t in enumerate(range(19420, 19432)):
        for w in ("g0", "g1"):
            for r in ("moderate", "heavy"):
                e = 10.0 + 0.1 * i
                _cell(old, t, w, r, "cd_exactS", 0, e)
                _cell(old, t, w, r, "cd", 0, e * 1.2)
                _cell(old, t, w, r, "cd_exactS_ib0", 0, e * 0.99)
                for s in (1, 2):
                    jit = 0.01 * (i + 1)
                    for kind in ("ra_gnn_eng_selfsearch", "ra_gnn_eng_nosplit"):
                        _cell(new, t, w, r, kind, s, e * (1 + (new_pct - jit) / 100))
                        _cell(old, t, w, r, kind, s, e * (1 + (old_pct + jit) / 100))
                    for kind in ("ra_twin_eng_selfsearch", "ra_twin_eng_nosplit"):
                        _cell(new, t, w, r, kind, s, e * (1 + (new_pct - jit - twin_pct) / 100))
                    _cell(new, t, w, r, "ra_gnn_eng_cdxapply", s, e)
    return str(new), str(old)


def test_reader_win_and_gap(tmp_path):
    rd = importlib.import_module("slate_fix_v1_gate_read")
    r = rd.read(*_gate_dirs(tmp_path, -3.0, 3.6))
    m = r["main"]
    assert m["comparable"] and r["n_summaries"] == 12 * 2 * 2 * (3 + 2 * 7)
    assert m["primary"]["holm_over"] == 8 and set(m["primary"]["labels"].values()) == {"WIN"}
    assert "new self-search vs cd_exactS_ib0|heavy" in m["primary"]["tests"]
    qp = m["queue_q234_proxy"]
    assert qp["label"] == "queue component only, Q1 dropped, outside every family" and len(qp["tests"]) == 8
    for k, c in qp["tests"].items():  # queue is proportional to latency in the synthetic cells: the proxy equals the primary contrast
        assert abs(c["median_pct"] - m["primary"]["tests"][k]["median_pct"]) < 1e-9 and "holm_p" not in c
    assert all(c["holm_p"] < 0.05 for c in m["primary"]["tests"].values())
    g = m["gap_change_pp"]["ra_gnn_eng_selfsearch|heavy"]
    assert g["n_topologies"] == 12 and -6.9 < g["median_pp"] < -6.4 and g["shrank"] == 12
    assert set(m["mp"]["rung_labels"].values()) == {"NOT-SEPARATED"}  # gnn and twin cells are identical here: no difference, no p


def test_reader_reports_the_served_slate_per_arm(tmp_path, capsys):
    rd = importlib.import_module("slate_fix_v1_gate_read")
    new, old = _gate_dirs(tmp_path, -3.0, 3.6)
    f = Path(new) / "cc40s19420__g0heavy__ra_gnn_eng_selfsearch_s1.summary.json"
    d = json.loads(f.read_text())
    d["schedulerCounters"] = {"slate_declared_batches": 100, "slate_declared_sub_batched": 10, "slate_declared_unsplit": 10,
                              "slate_declared_tasks": 400, "slate_declared_sub_batched_tasks": 80, "ss_batches": 100,
                              "ss_exact_batches": 70, "ss_ascent_batches": 30}
    f.write_text(json.dumps(d))
    r = rd.read(new, old)
    c = r["slate"]["new:ra_gnn_eng_selfsearch|heavy"]
    assert c["slate_declared_sub_batched"] == 10 and c["sub_batched_task_share"] == 0.2 and c["ss_ascent_share"] == 0.3
    rd.print_report(r)
    out = capsys.readouterr().out
    assert "served slate per arm" in out and "-- code: " in out
    assert r["code_commits"][new] == ["unrecorded"] and len(r["code_commits"]["reader"]) == 1


def test_reader_loss_and_refusals(tmp_path):
    rd = importlib.import_module("slate_fix_v1_gate_read")
    new, old = _gate_dirs(tmp_path, 2.5, 3.6)
    assert set(rd.read(new, old)["main"]["primary"]["labels"].values()) == {"E-FASTER"}
    _cell(Path(new), 19500, "g0", "heavy", "cd", 0, 1.0)
    with pytest.raises(SystemExit):
        rd.read(new, old)


def test_reader_refuses_a_hand_cell_in_both_dirs(tmp_path):
    rd = importlib.import_module("slate_fix_v1_gate_read")
    new, old = _gate_dirs(tmp_path, -3.0, 3.6)
    _cell(Path(new), 19420, "g0", "heavy", "cd_exactS", 0, 10.0)
    with pytest.raises(SystemExit):
        rd.read(new, old)


def test_reader_refuses_unregistered_windows(tmp_path):
    rd = importlib.import_module("slate_fix_v1_gate_read")
    new, old = _gate_dirs(tmp_path, -3.0, 3.6)
    _cell(Path(old), 19420, "g2", "heavy", "cd", 0, 10.0)
    with pytest.raises(SystemExit):
        rd.read(new, old)
