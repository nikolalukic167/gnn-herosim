import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import plan_oracle_read as P  # noqa: E402

TOPOS = list(range(19300, 19306))


def cell(lat, oracle=False, cold=0.5, exch=1.0, lock=0.4):
    c = {"averageElapsedTime": lat, "num_tasks": 100, "averageColdStartTime": cold, "totalPeerExchangeTime": exch * 100, "lock_wait": {"mean": lock},
         "schedulerCounters": {"pg_batches": 100, "pg_exact_batches": 100}}
    if oracle:
        c["schedulerCounters"].update(pg_oracle_batches=80, pg_oracle_seen_batches=100, pg_oracle_over_cap_batches=20, pg_oracle_seen_tasks=400,
                                      pg_oracle_over_cap_tasks=160, pg_oracle_plans=8000, pg_oracle_changed=20, pg_oracle_gain_seconds=40.0,
                                      pg_oracle_seconds=160.0)
    return c


def synth(o_pct, topo_pct=None):
    cells = {}
    for t in TOPOS:
        for r in P.RUNGS:
            for w in P.WINDOWS:
                p = (topo_pct or {}).get(t, o_pct)
                cells[(P.C, 0, t, w, r)] = cell(3.3)
                cells[(P.E, 0, t, w, r)] = cell(3.0)
                cells[(P.O, 0, t, w, r)] = cell(3.0 * (1 + p / 100), oracle=True, cold=0.4)
    return cells


def run(monkeypatch, cells, failed=None):
    monkeypatch.setattr(P.R, "load", lambda gates: (cells, failed or {}))
    return P.read(["g"], TOPOS)


def test_go_needs_four_percent_and_five_of_six_at_both_rungs(monkeypatch):
    r = run(monkeypatch, synth(-5.0))
    assert r["GO"] and r["GO_per_rung"] == {"moderate": True, "heavy": True}
    assert not run(monkeypatch, synth(-3.0))["GO"]


def test_one_odd_topology_cannot_decide_but_two_can_block(monkeypatch):
    one = {19300: +2.0}
    assert run(monkeypatch, synth(-6.0, one))["GO"]            # 5 of 6 faster, median still <= -4
    two = {19300: +2.0, 19301: +1.0}
    r = run(monkeypatch, synth(-6.0, two))
    assert not r["GO"] and r["contrasts"][f"{P.O} vs {P.E}|heavy"]["wins"] == 4


def test_the_bar_is_per_rung(monkeypatch):
    cells = synth(-6.0)
    for k in list(cells):
        if k[0] == P.O and k[4] == "heavy":
            cells[k] = cell(3.0 * 0.99, oracle=True)
    r = run(monkeypatch, cells)
    assert r["GO_per_rung"] == {"moderate": True, "heavy": False} and not r["GO"]


def test_counts_books_and_buckets(monkeypatch):
    r = run(monkeypatch, synth(-5.0))
    assert r["counts"][f"{P.O}|heavy"] == {"finished": 12, "failed": 0, "expected": 12, "complete": True}
    b = r["oracle_books"]["heavy"]
    assert b["books_close"] and b["over_cap_share_of_batches"] == pytest.approx(0.2) and b["over_cap_share_of_tasks"] == pytest.approx(0.4)
    assert b["plans_per_served_batch"] == pytest.approx(100.0) and b["changed_share"] == pytest.approx(0.25)
    assert b["decision_wall_s_per_served_batch"] == pytest.approx(2.0)
    k = r["buckets_O_minus_E_pp"]["heavy"]
    assert k["elapsed"] == pytest.approx(-5.0) and k["cold_start"] == pytest.approx(100 * (-0.1) / 3.0)
    assert k["exchange"] == pytest.approx(0.0) and k["n_cells"] == 12


def test_failed_cells_are_counted_not_dropped(monkeypatch):
    cells = synth(-5.0)
    gone = (P.O, 0, 19300, "g0", "heavy")
    del cells[gone]
    r = run(monkeypatch, cells, {gone: {"why": "watchdog: stalled"}})
    assert r["counts"][f"{P.O}|heavy"] == {"finished": 11, "failed": 1, "expected": 12, "complete": False} and r["n_failed"] == 1


def test_foreign_topology_window_or_arm_is_refused(monkeypatch):
    for bad in ((P.O, 0, 19999, "g0", "heavy"), (P.O, 0, 19300, "g2", "heavy"), ("cd_expand", 0, 19300, "g0", "heavy")):
        cells = synth(-5.0)
        cells[bad] = cell(3.0)
        with pytest.raises(SystemExit, match="FAIL LOUD"):
            run(monkeypatch, cells)


def test_stacking_compares_the_two_plans_on_the_same_state():
    rows = [{"n": 4, "s_plan": [[1, 1], [1, 1], [2, 2], [3, 3]], "best": [[1, 1], [1, 1], [1, 1], [1, 1]]}]
    s = P.stacking(rows)
    assert s["s_plan"]["tasks_per_occupied_platform"] == pytest.approx(4 / 3) and s["s_plan"]["share_tasks_sharing_a_platform"] == pytest.approx(0.5)
    assert s["best"]["tasks_per_occupied_platform"] == pytest.approx(4.0) and s["best"]["share_tasks_sharing_a_platform"] == pytest.approx(1.0)


def test_first_quarter_share_of_the_gap(tmp_path):
    def raw(name, el):
        trs = [{"taskId": i, "scheduledTime": 0.0, "doneTime": float(x)} for i, x in enumerate(el)]
        (tmp_path / name).write_text(json.dumps({"stats": {"taskResults": trs}}))
        return str(tmp_path / name)
    a = raw("a.json", [5, 5, 1, 1, 1, 1, 1, 1])
    b = raw("b.json", [1, 1, 1, 1, 1, 1, 1, 1])
    g = P.quarter_gap(a, b)
    assert g["gap_s"] == 8.0 and g["first_quarter_gap_s"] == 8.0 and g["first_quarter_share"] == 1.0
