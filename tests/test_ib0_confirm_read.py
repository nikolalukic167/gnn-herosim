import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import ib0_confirm_read as P  # noqa: E402

TOPOS = list(range(19405, 19417))


def cell(lat):
    return {"averageElapsedTime": lat, "num_tasks": 100, "averageColdStartTime": 0.5, "totalPeerExchangeTime": 100.0, "lock_wait": {"mean": 0.4},
            "schedulerCounters": {"pg_batches": 100, "cdr_batches": 100, "pg_cd_passes": 200, "pg_expand_batches": 100}}


def synth(ib0_pct, seed_pct=0.0, per_topo=None):
    """E = 3.0 s everywhere; IB0 = E*(1+pct); the hand-seed control = IB0; the GNN seed = hand control*(1+seed_pct)."""
    cells = {}
    for t in TOPOS:
        for r in P.RUNGS:
            for w in P.WINDOWS4:
                p = (per_topo or {}).get(t, ib0_pct)
                e = 3.0
                ib0 = e * (1 + p / 100)
                cells[(P.E, 0, t, w, r)] = cell(e)
                cells[(P.C, 0, t, w, r)] = cell(3.4)
                cells[(P.IB0, 0, t, w, r)] = cell(ib0)
                cells[(P.CDXH, 0, t, w, r)] = cell(ib0)
                cells[(P.CDXI, 1, t, w, r)] = cell(ib0 * (1 + seed_pct / 100))
                if w in P.WINDOWS2:
                    for arm in (P.A05, P.A0):
                        cells[(arm, 0, t, w, r)] = cell(ib0)
                    cells[(P.TWIN, 1, t, w, r)] = cell(ib0)
    return cells


def run(monkeypatch, cells, failed=None):
    monkeypatch.setattr(P.R, "load", lambda gates: (cells, failed or {}))
    return P.read(["g"], TOPOS)


def test_complete_counts_and_a_win(monkeypatch):
    r = run(monkeypatch, synth(-4.0))
    assert all(c["complete"] for c in r["counts"].values()) and r["n_failed"] == 0
    assert r["verdict"]["PRIMARY"] == "WIN" and r["holm_primary"]["moderate"] < 0.05
    c = r["contrasts"][f"{P.IB0} vs {P.E}|moderate"]
    assert c["median_pct"] == pytest.approx(-4.0) and c["wins"] == 12 and c["n_topologies"] == 12


def test_a_two_percent_miss_is_not_a_win_and_a_slower_ib0_loses(monkeypatch):
    assert run(monkeypatch, synth(-1.0))["verdict"]["PRIMARY"] == "TIES"
    assert run(monkeypatch, synth(+3.0))["verdict"]["PRIMARY"] == "LOSES"


def test_win_needs_nine_of_twelve_topologies(monkeypatch):
    four_odd = {t: +1.0 for t in TOPOS[:4]}   # 8 of 12 faster, median still <= -2
    assert run(monkeypatch, synth(-4.0, per_topo=four_odd))["verdict"]["PRIMARY"] == "TIES"
    three_odd = {t: +1.0 for t in TOPOS[:3]}  # 9 of 12
    assert run(monkeypatch, synth(-4.0, per_topo=three_odd))["verdict"]["PRIMARY"] == "WIN"


def test_the_gnn_seed_is_credited_only_against_the_hand_seed_control(monkeypatch):
    better = run(monkeypatch, synth(-4.0, seed_pct=-3.0))
    assert better["verdict"]["GNN_SEED_WIN"] and better["contrasts"][f"{P.CDXI} vs {P.CDXH}|heavy"]["median_pct"] == pytest.approx(-3.0)
    assert not run(monkeypatch, synth(-4.0, seed_pct=0.0))["verdict"]["GNN_SEED_WIN"]
    # the seed beating ib0 alone is not the credit: the control equals ib0 here, so both contrasts move together; the verdict reads the control one
    assert set(better["holm_secondary"]) == {f"{a} vs {b}|{r}" for a, b in P.SECONDARY for r in P.RUNGS}


def test_failed_cells_are_counted_and_an_unexpected_window_fails_loud(monkeypatch):
    cells = synth(-4.0)
    key = (P.IB0, 0, TOPOS[0], "g3", "heavy")
    failed = {key: {"why": "watchdog: projected"}}
    del cells[key]
    r = run(monkeypatch, cells, failed)
    assert r["counts"][f"{P.IB0}|heavy"] == {"finished": 47, "failed": 1, "expected": 48, "complete": False}
    bad = synth(-4.0)
    bad[(P.A0, 0, TOPOS[0], "g2", "heavy")] = cell(3.0)
    with pytest.raises(SystemExit, match="FAIL LOUD"):
        run(monkeypatch, bad)


def test_search_books_put_the_equal_budget_on_the_page(monkeypatch):
    r = run(monkeypatch, synth(-4.0))
    a, b = r["search_books"]["cdx_ib0|moderate"], r["search_books"][f"{P.CDXI}|moderate"]
    assert a["pg_cd_passes"] == b["pg_cd_passes"] == 200 * 48 and a["pg_expand_batches"] == b["pg_expand_batches"]
