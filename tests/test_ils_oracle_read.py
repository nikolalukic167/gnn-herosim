import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import ils_oracle_read as O  # noqa: E402

TOPOS = [1, 2, 3, 4, 5, 6]


def summ(whole, after, exch=0.8, queue=0.5, caps=(0, 0, 0), batches=(10, 10, 2)):
    s = {"cell": "x", "averageElapsedTime": whole, "steady_state": {"after_360s": {"mean": after}}, "num_tasks": 1000,
         "totalPeerExchangeTime": exch * 1000, "averageQueueTime": queue, "endTime": 100.0, "arrival_end": {"last_arrival_s": 99.0},
         "code": {"commit": "abc123ff", "dirty": False}}
    c = {"pg_ils_batches": sum(batches), "pg_ils_tasks_moved": 5, "pg_ils_gain_seconds": 1.0, "pg_ils_budget_stops": 0}
    for b, n, h in zip(O.BUCKETS, batches, caps):
        c[f"pg_ils_batches_{b}"], c[f"pg_ils_cap_hits_{b}"] = n, h
    s["schedulerCounters"] = c
    return s


def write(d, arm, t, rung, s):
    with open(os.path.join(d, f"cc40s{t}__g0{rung}__{arm}_s0.summary.json"), "w") as fh:
        json.dump(s, fh)


@pytest.fixture
def dirs(tmp_path):
    o, r = tmp_path / "o", tmp_path / "r"
    o.mkdir()
    r.mkdir()
    for t in TOPOS:
        for rung in O.RUNGS:
            for arm in (O.BAR, O.EXS, O.CD):
                write(r, arm, t, rung, summ(2.0, 1.0))
    return str(o), str(r)


def run(o, r, *extra):
    return O.main(["--oracle", o, "--oracle-pin", "abc123", "--ref", r, "--topos", ",".join(map(str, TOPOS)), *extra])


def test_transfers_b_not_run(dirs):
    o, r = dirs
    for t in TOPOS:
        for rung in O.RUNGS:
            write(o, O.ORACLE, t, rung, summ(1.8, 0.9, exch=0.6, queue=0.52))
    res = run(o, r)
    assert res["A"]["heavy"]["after360"]["median_pct"] == pytest.approx(-10.0)
    assert res["label"].startswith("TRANSFERS + B-NOT-RUN")


def test_queue_rise_beats_exchange_fall_is_myopic(dirs):
    o, r = dirs
    for t in TOPOS:
        for rung in O.RUNGS:
            write(o, O.ORACLE, t, rung, summ(1.8, 0.9, exch=0.7, queue=0.7))
    assert run(o, r)["label"].startswith("MYOPIC-LIVE")


def test_small_gain_is_partial_and_cap_flag(dirs):
    o, r = dirs
    for t in TOPOS:
        for rung in O.RUNGS:
            write(o, O.ORACLE, t, rung, summ(1.9, 0.96, exch=0.7, queue=0.5, caps=(0, 3, 1)))
    res = run(o, r)
    assert res["label"].startswith("PARTIAL + CEILING-TRUNCATED + B-NOT-RUN")


def test_incomplete_gives_no_label(dirs):
    o, r = dirs
    write(o, O.ORACLE, 1, "heavy", summ(1.8, 0.9))
    res = run(o, r)
    assert res["label"].startswith("DESCRIPTIVE (interim, 1 cells of 12)")


def test_b_runs_only_with_identity(dirs, tmp_path):
    o, r = dirs
    e = tmp_path / "e"
    e.mkdir()
    for t in TOPOS:
        for rung in O.RUNGS:
            write(o, O.ORACLE, t, rung, summ(1.8, 0.9, exch=0.6, queue=0.52))
            write(e, O.EXACT10, t, rung, summ(2.0, 1.0, exch=0.6, queue=0.52))
    assert run(o, r, "--exact10", str(e))["B_status"] == "B-NOT-RUN"
    res = run(o, r, "--exact10", str(e), "--exact10-identity-pass")
    assert res["B_status"] == "RUN" and res["label"] == "PARTIAL"
    for t in TOPOS:
        for rung in O.RUNGS:
            write(e, O.EXACT10, t, rung, summ(2.1, 1.05, exch=0.6, queue=0.52))
    assert run(o, r, "--exact10", str(e), "--exact10-identity-pass")["label"] == "MYOPIC-LIVE"


def test_missing_after360_fails_loud(dirs):
    o, r = dirs
    s = summ(1.8, 0.9)
    s["steady_state"] = {}
    write(o, O.ORACLE, 1, "heavy", s)
    with pytest.raises(SystemExit):
        run(o, r)


def test_wrong_pin_fails_loud(dirs):
    o, r = dirs
    s = summ(1.8, 0.9)
    s["code"]["commit"] = "deadbeef"
    write(o, O.ORACLE, 1, "heavy", s)
    with pytest.raises(SystemExit):
        run(o, r)


def test_failed_cell_is_wall_infeasible_not_imputed(dirs):
    o, r = dirs
    for t in TOPOS:
        for rung in O.RUNGS:
            if (t, rung) == (6, "heavy"):
                with open(os.path.join(o, f"cc40s{t}__g0{rung}__{O.ORACLE}_s0.failed.json"), "w") as fh:
                    json.dump({"why": "watchdog: projected 1/2"}, fh)
                continue
            write(o, O.ORACLE, t, rung, summ(1.8, 0.9, exch=0.6, queue=0.52))
    res = run(o, r)
    h = res["A"]["heavy"]
    assert h["n"] == 5 and h["wall_infeasible"] == 1 and h["after360"]["faster"] == 5
    assert res["label"].startswith("TRANSFERS")
