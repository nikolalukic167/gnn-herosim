import hashlib
import json

import pytest

import scripts_cosim.uncut_groups_hand_read as U

TOPOS = [1, 2]


def cell(mean, a360=None, tail=10.0, qs=0.3, busy=0.0, wait=0.0, n=100, big=0, whole=0):
    return {"averageElapsedTime": mean, "steady_state": {"after_360s": {"mean": a360 if a360 is not None else mean}}, "endTime": 100.0 + tail,
            "arrival_end": {"last_arrival_s": 100.0}, "effective_queue_share": qs, "num_tasks": n,
            "schedulerCounters": {"sched_busy_seconds": busy, "sched_queue_wait_seconds": wait, "peer_group_incomplete_batches": 0, "big": big, "whole": whole}}


def write(d, arm, topo, win, rung, s, kind="summary"):
    (d / f"cc40s{topo}__{win}{rung}__{arm}_s0.{kind}.json").write_text(json.dumps(s))


def fill(d, arms, collapse=()):
    for rung in U.RUNGS:
        for t in TOPOS:
            for w in U.WINDOWS:
                for arm, base in arms.items():
                    m = base * (5.0 if (arm, t, w, rung) in collapse else 1.0)
                    write(d, arm, t, w, rung, cell(m, busy=0.5, wait=0.5, big=100, whole=95))


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
    r = U.whole_group_share(cells, "cd_expand", TOPOS, "big", "whole")
    assert r["heavy"]["pooled"] == pytest.approx(0.95) and r["heavy"]["pass"]
    with pytest.raises(SystemExit):
        U.whole_group_share(cells, "cd_expand", TOPOS, "nope", "whole")


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
