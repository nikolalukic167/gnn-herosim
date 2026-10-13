import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import ib0_confirm_oracle_read as P  # noqa: E402


def cell(lat, books=None):
    return {"averageElapsedTime": lat, "schedulerCounters": books or {}}


def books(seen=100, served=70, searched=27, unplaced=3, over=0):
    return {"pg_oracle_seen_tasks": seen, "pg_oracle_tasks": served, "pg_oracle_search_tasks": searched, "pg_oracle_unplaced_tasks": unplaced,
            "pg_oracle_over_cap_tasks": over, "pg_oracle_seen_batches": 10, "pg_oracle_batches": 6, "pg_oracle_search_batches": 3,
            "pg_oracle_unplaced_batches": 1, "pg_oracle_over_cap_batches": 0}


def synth(mod_pct, heavy_pct, bk=None):
    cells = {}
    for t in P.TOPOS:
        for r, p in (("moderate", mod_pct), ("heavy", heavy_pct)):
            cells[(P.IB0, 0, t, "g0", r)] = cell(3.0)
            cells[(P.E, 0, t, "g0", r)] = cell(3.1)
            cells[(P.ORC, 1, t, "g0", r)] = cell(3.0 * (1 + p / 100), bk or books())
    return cells


def run(monkeypatch, cells, failed=None):
    monkeypatch.setattr(P.R, "load", lambda gates: (cells, failed or {}))
    return P.read(["g"])


def test_family_closes_when_oracle_under_three_percent(monkeypatch):
    v = run(monkeypatch, synth(-1.5, -4.0))["verdict"]
    assert v["ORACLE"] == "FAMILY-CLOSES"


def test_family_stays_open_at_three_percent(monkeypatch):
    assert run(monkeypatch, synth(-3.5, -1.0))["verdict"]["ORACLE"] == "FAMILY-OPEN"


def test_heavy_cannot_rescue_moderate(monkeypatch):
    assert run(monkeypatch, synth(-0.5, -9.0))["verdict"]["ORACLE"] == "FAMILY-CLOSES"


def test_low_coverage_is_inconclusive(monkeypatch):
    r = run(monkeypatch, synth(-5.0, -5.0, books(served=60, searched=20, unplaced=20)))
    assert r["verdict"]["ORACLE"] == "INCONCLUSIVE"
    assert r["coverage"]["moderate"]["labelled_share"] == pytest.approx(0.80)


def test_missing_cell_is_incomplete(monkeypatch):
    cells = synth(-5.0, -5.0)
    del cells[(P.ORC, 1, 19410, "g0", "heavy")]
    r = run(monkeypatch, cells)
    assert r["verdict"]["ORACLE"] == "INCOMPLETE"
    assert r["counts"][f"{P.ORC}|heavy"]["finished"] == 5


def test_books_close_flag_and_coverage_share(monkeypatch):
    c = run(monkeypatch, synth(-2.0, -2.0))["coverage"]["heavy"]
    assert c["books_close"] is True and c["labelled_share"] == pytest.approx(0.97)
