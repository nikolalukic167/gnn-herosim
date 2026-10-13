import hashlib
import json

import pytest

import scripts_cosim.uncut_groups_hand_read as U

TOPOS = [1, 2]


def cell(mean, a360=None, tail=10.0, qs=0.3, busy=0.0, wait=0.0, n=100, whole=95, split=5):
    return {"averageElapsedTime": mean, "steady_state": {"after_360s": {"mean": a360 if a360 is not None else mean}}, "endTime": 100.0 + tail,
            "arrival_end": {"last_arrival_s": 100.0}, "effective_queue_share": qs, "num_tasks": n,
            "schedulerCounters": {"sched_busy_seconds": busy, "sched_queue_wait_seconds": wait, "peer_group_incomplete_batches": 0,
                                **{f"pgb_whole_{b}": whole for b in ("le10", "11_50", "51_100", "gt100")},
                                **{f"pgb_split_{b}": split for b in ("le10", "11_50", "51_100", "gt100")}}}


def write(d, arm, topo, win, rung, s, kind="summary"):
    (d / f"cc40s{topo}__{win}{rung}__{arm}_s0.{kind}.json").write_text(json.dumps(s))


def fill(d, arms, collapse=()):
    for rung in U.RUNGS:
        for t in TOPOS:
            for w in U.WINDOWS:
                for arm, base in arms.items():
                    m = base * (5.0 if (arm, t, w, rung) in collapse else 1.0)
                    write(d, arm, t, w, rung, cell(m, busy=0.5, wait=0.5))


def test_collapse_and_tail_counts(tmp_path):
    fill(tmp_path, {"cd": 1.0, "cd_expand": 0.9, "cd_exactS": 0.95}, collapse={("cd_exactS", 1, "g1", "heavy")})
    write(tmp_path, "cd_expand", 1, "g0", "heavy", cell(0.9, tail=130.0))
    cells, failed = U.load_gate(str(tmp_path))
    ht = U.health_table(cells, ["cd", "cd_expand", "cd_exactS"], TOPOS)
    assert ht["cd_exactS|heavy"]["collapse_any"] == 1 and ht["cd_exactS|heavy"]["collapsed_cells"] == [(1, "g1")]
    assert ht["cd|heavy"]["collapse_any"] == 0 and ht["cd_expand|heavy"]["tail_flag"] == 1 and ht["cd_expand|heavy"]["collapse_any"] == 0


def test_k0_health_needs_ten_of_twelve(tmp_path):
    fill(tmp_path, {"cd": 1.0, "cd_exactS": 1.0})
    cells, _ = U.load_gate(str(tmp_path))
    out, verdict = U.k0_health(cells, TOPOS)
    assert out["heavy"]["healthy_g0"] == {"cd": 2, "cd_exactS": 2} and not out["heavy"]["pass"] and verdict == "NO-SEAT"
    topos12 = list(range(1, 13))
    d = tmp_path / "t"
    d.mkdir()
    for rung in U.RUNGS:
        for t in topos12:
            for arm in ("cd", "cd_exactS"):
                write(d, arm, t, "g0", rung, cell(1.0, qs=0.3 if t <= 10 else 0.9))
    cells, _ = U.load_gate(str(d))
    assert U.k0_health(cells, topos12)[1] == "K0_OK_BOTH_RUNGS"


def test_uncut_share_pooled_and_loud(tmp_path):
    fill(tmp_path, {"cd": 1.0, "cd_expand": 1.0})
    cells, _ = U.load_gate(str(tmp_path))
    r = U.whole_group_share(cells, "cd_expand", TOPOS)
    assert r["heavy"]["pooled"] == pytest.approx(0.95) and r["heavy"]["pass"]
    s = cell(1.0)
    del s["schedulerCounters"]["pgb_split_gt100"]
    write(tmp_path, "cd_expand", 1, "g0", "heavy", s)
    cells, _ = U.load_gate(str(tmp_path))
    with pytest.raises(SystemExit):
        U.whole_group_share(cells, "cd_expand", TOPOS)


def test_hand_arm_skips_a_collapsing_best_and_headroom_from_counters(tmp_path):
    fill(tmp_path, {"cd_k100": 1.0, "cd_expand_k100": 1.2, "cd_exactS_k100": 0.8}, collapse={("cd_exactS_k100", 2, "g3", "heavy")})
    cells, _ = U.load_gate(str(tmp_path))
    pick, _ = U.pick_hand_arm(cells, ["cd", "cd_expand", "cd_exactS"], TOPOS, "_k100")
    assert pick["heavy"]["median_best"] == "cd_exactS" and pick["heavy"]["median_best_collapses"] and pick["heavy"]["decides"] == "cd"
    assert pick["moderate"]["decides"] == "cd_exactS"
    h = U.headroom(cells, "cd_expand", "_k100", TOPOS, "heavy")
    assert h["whole"]["median"] == pytest.approx((1.0 / 100) / 1.2)


def test_replay_regret_seal_and_mean(tmp_path):
    ids = [f"s{i}" for i in range(4)]
    p = tmp_path / "ids.json"
    p.write_text(json.dumps(ids))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    rows = [{"state_id": i, "rung": "heavy", "arm": "cd_expand", "hand_rtt": 100 + d, "ils_rtt": 100.0, "bucket": "11-50"} for i, d in zip(ids, (0, 0, 0, 3))]
    r = U.replay_regret(rows, ids, sha, str(p), {"heavy": "cd_expand", "moderate": "cd_expand"})
    assert r["mean"] == pytest.approx(0.0075) and r["pass"] and r["share_within_1pct"] == 0.75
    with pytest.raises(SystemExit):
        U.replay_regret(rows, ids, "0" * 64, str(p), {"heavy": "cd_expand"})
    with pytest.raises(SystemExit):
        U.replay_regret(rows[:3], ids, sha, str(p), {"heavy": "cd_expand"})
    with pytest.raises(SystemExit):
        U.replay_regret(rows, ids, sha, str(p), {"heavy": "cd"})


def rr(m, p):
    return {r: {"median": m, "p_holm": p} for r in U.RUNGS}


def test_f1_labels():
    win, no = rr(-3.0, 0.01), rr(-3.0, 0.2)
    clean, brittle = {"heavy": 0, "moderate": 0}, {"heavy": 3, "moderate": 0}
    assert U.f1_label(win, win, clean, clean, True) == "WIN"
    assert U.f1_label(win, win, clean, brittle, True) == "WIN-ROBUST"
    assert U.f1_label(win, no, clean, brittle, True) == "WIN-OVER-BRITTLE-BAR"
    assert U.f1_label(win, win, brittle, clean, True) == "TIES"
    assert U.f1_label(win, win, clean, clean, False) == "NO-GO"
    assert U.f1_label(no, no, clean, clean, True) == "TIES"
    assert U.f1_label(rr(4.0, 0.01), no, clean, clean, True) == "E-FASTER"
    assert U.f1_label({"heavy": {"median": -3.0, "p_holm": 0.01}, "moderate": {"median": -1.0, "p_holm": 0.01}}, win, clean, clean, True) == "TIES"


def test_wn_table_flags_a_bucket_beyond_1_5x(tmp_path):
    import scripts_cosim.uncut_groups_wn_table as T
    (tmp_path / "w.json").write_text(json.dumps({"w_ms_per_plan": {"cd_expand": {"heavy": 0.01}}}))
    s = cell(1.0)
    s["schedulerCounters"]["sched_plan_evals"] = 300
    s["decisionTiming"] = {"by_bucket": {"le10": {"calls": 5, "total_s": 0.001, "plan_evals": 100}, "11_50": {"calls": 3, "total_s": 0.0014, "plan_evals": 100},
                                         "gt50": {"calls": 1, "total_s": 0.0035, "plan_evals": 100}}}
    write(tmp_path, "cd_expand", 1, "g0", "heavy", s)
    o = T.table(str(tmp_path), str(tmp_path / "w.json"))["cd_expand|heavy"]
    assert o["ms_per_plan"]["le10"] == pytest.approx(0.01) and o["exceeds_1_5x"] == ["gt50"]
    s["schedulerCounters"]["sched_plan_evals"] = 999
    write(tmp_path, "cd_expand", 1, "g0", "heavy", s)
    with pytest.raises(SystemExit):
        T.table(str(tmp_path), str(tmp_path / "w.json"))


def test_wall_infeasible_is_its_own_count(tmp_path):
    write(tmp_path, "cd_exactS", 1, "g0", "heavy", {"why": "watchdog: projected", "watchdog": {"detail": "projected: 287/3933"}}, "failed")
    write(tmp_path, "cd_exactS", 1, "g1", "heavy", {"why": "oom"}, "failed")
    write(tmp_path, "cd_exactS", 2, "g0", "heavy", cell(10.0))
    cells, failed = U.load_gate(str(tmp_path))
    w = U.wall_infeasible(failed, ["cd_exactS"], TOPOS)["cd_exactS|heavy"]
    assert (w["wall_infeasible"], w["wall_infeasible_g0"], w["other_failed"]) == (1, 1, 1)
    assert w["cells"][0]["detail"] == "projected: 287/3933"
    assert U.counts(cells, failed, ["cd_exactS"], TOPOS)["cd_exactS|heavy"]["finished"] == 1


def test_sibling_gate_union_asserts_identical_duplicates(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    write(a, "cd", 1, "g0", "heavy", cell(10.0)); write(b, "cd", 1, "g0", "heavy", cell(10.0)); write(b, "cd", 2, "g0", "heavy", cell(11.0))
    cells, _ = U.load_gate(f"{a},{b}")
    assert len(cells) == 2
    write(b, "cd", 1, "g0", "heavy", cell(99.0))
    with pytest.raises(SystemExit):
        U.load_gate(f"{a},{b}")
