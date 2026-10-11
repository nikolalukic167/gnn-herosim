import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import gnn_overcap_gate_read as G  # noqa: E402

TOPOS = list(range(19206, 19218))
PCT = {G.C: 0.0, G.E: -8.0, G.H: -11.0, G.T: -10.0}


def cell(lat, arm):
    c = {"averageElapsedTime": lat, "num_tasks": 100,
         "schedulerCounters": {"pg_batches": 100, "cdr_batches": 100, "cdr_tasks": 800, "pg_exact_batches": 60, "pg_exact_fallbacks": 40, "pg_overcap_pred": 40}}
    if arm in G.OVERCAP_ARMS:
        c["schedulerCounters"].update(pg_exact_fallbacks=0, pg_overcap_gnn_batches=40, pg_overcap_gnn_tasks=40 * 8)
    return c


def synth(pct=PCT, topos=TOPOS, windows=G.WINDOWS):
    cells = {}
    for t in topos:
        for r in G.RUNGS:
            for w in windows:
                for arm, p in pct.items():
                    cells[(arm, 0 if arm in (G.C, G.E) else 1, t, w, r)] = cell(3.0 * (1 + p / 100), arm)
    return cells


def patch(mp, cells, failed=None):
    mp.setattr(G.R, "load", lambda gate: (cells, failed or {}))


def test_counts_books_contrasts_and_holm(monkeypatch):
    patch(monkeypatch, synth())
    r = G.read(["x"], TOPOS)
    assert all(c["complete"] for c in r["counts"].values()) and r["counts"][f"{G.H}|heavy"]["expected"] == 48
    b = r["books"][f"{G.H}|heavy"]
    assert b["over_cap_share_of_batches"] == pytest.approx(0.4) and b["served_equals_predicted"] and b["no_fallback_reached"]
    assert r["books"][f"{G.E}|heavy"]["over_cap_share_of_batches"] == pytest.approx(0.4) and r["books"][f"{G.E}|heavy"]["predicted_equals_fallbacks"]
    c = r["contrasts"][f"{G.H} vs {G.E}|moderate"]
    assert c["family"] == "primary" and c["wins"] == 12 and c["median_pct"] == pytest.approx(100 * (0.89 / 0.92 - 1))
    assert c["p_holm"] == pytest.approx(min(1.0, 2 * c["p"]))
    assert r["contrasts"][f"{G.H} vs {G.T}|heavy"]["family"] == "secondary" and "p_holm" in r["contrasts"][f"{G.H} vs {G.C}|heavy"]
    assert r["contrasts"][f"{G.E} vs {G.C}|heavy"]["family"] == "descriptive"
    assert r["WIN"] and not r["LOSS"] and r["E_FASTER_RUNGS"] == []


def test_a_median_short_of_two_percent_is_not_a_win(monkeypatch):
    patch(monkeypatch, synth({G.C: 0.0, G.E: -8.0, G.H: -9.0, G.T: -9.0}))   # H is 1.1 % faster than E: beats it, short of the bar
    r = G.read(["x"], TOPOS)
    assert not r["WIN"] and not r["LOSS"] and r["E_FASTER_RUNGS"] == []


def test_a_significant_loss_at_one_rung_is_a_loss_and_never_a_win(monkeypatch):
    cells = synth()
    for t in TOPOS:
        for w in G.WINDOWS:
            cells[(G.H, 1, t, w, "heavy")] = cell(3.0 * 0.94, G.H)   # slower than cd_exactS (0.92) at heavy
    patch(monkeypatch, cells)
    r = G.read(["x"], TOPOS)
    assert not r["WIN"] and r["LOSS"] and r["E_FASTER_RUNGS"] == ["heavy"]


def test_incomplete_cells_and_failed_are_counted_not_hidden(monkeypatch):
    cells = synth()
    del cells[(G.T, 1, 19206, "g0", "heavy")]
    patch(monkeypatch, cells, {(G.T, 1, 19206, "g0", "heavy"): {"why": "killed by the progress watchdog"}})
    r = G.read(["x"], TOPOS)
    assert not r["counts"][f"{G.T}|heavy"]["complete"] and r["counts"][f"{G.T}|heavy"]["failed"] == 1
    assert r["failed_why"] == ["killed by the progress watchdog"]


def test_refuses_other_topologies_and_windows(monkeypatch):
    patch(monkeypatch, synth(topos=TOPOS[:-1] + [19999]))
    with pytest.raises(SystemExit, match="outside --topos"):
        G.read(["x"], TOPOS)
    cells = synth()
    cells[(G.C, 0, 19206, "g9", "heavy")] = cell(3.0, G.C)
    patch(monkeypatch, cells)
    with pytest.raises(SystemExit, match="windows outside"):
        G.read(["x"], TOPOS)
    patch(monkeypatch, synth())
    with pytest.raises(SystemExit, match="outside the windows"):
        G.read(["x"], TOPOS, windows=("g0",))


def test_the_secondary_seat_reads_on_one_window_with_three_arms(monkeypatch):
    patch(monkeypatch, synth({G.C: 0.0, G.E: -8.0, G.H: -9.0}, windows=("g0",)))
    r = G.read(["x"], TOPOS, windows=("g0",))
    assert r["counts"][f"{G.H}|heavy"]["expected"] == 12 and G.T not in {k.split("|")[0] for k in r["counts"]}
    assert f"{G.H} vs {G.T}|heavy" not in r["contrasts"] and f"{G.H} vs {G.C}|heavy" in r["contrasts"]


def test_prints_counts_before_contrasts(monkeypatch, capsys):
    patch(monkeypatch, synth())
    G.print_report(G.read(["x"], TOPOS))
    out = capsys.readouterr().out
    assert out.index("cells per arm") < out.index("over-cap books") < out.index("paired %")
