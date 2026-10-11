import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import prune_g8_gate_read as G  # noqa: E402

TOPOS = list(range(18500, 18512))
PCT = {G.C: 0.0, G.E: -8.0, G.A: -9.0, G.B: -8.5, G.T: -8.8}


def cell(lat, arm):
    c = {"averageElapsedTime": lat, "num_tasks": 100, "schedulerCounters": {"pg_batches": 100, "pg_exact_batches": 60, "pg_exact_fallbacks": 40}}
    if arm in G.PRUNE_ARMS:
        c["schedulerCounters"].update(pg_exact_prune_batches=40, pg_exact_prune_differs=30, pg_exact_prune_better=20, pg_exact_prune_worse=5,
                                      pg_exact_prune_gain_fb_seconds=80.0, pg_exact_prune_gain_pass_seconds=400.0, pg_exact_prune_set_differs=36,
                                      pg_exact_prune_plans=2_000_000)
    return c


def synth(pct=PCT, topos=TOPOS):
    cells = {}
    for t in topos:
        for r in G.RUNGS:
            for w in G.WINDOWS:
                for arm, p in pct.items():
                    cells[(arm, 0 if arm in (G.C, G.E) else 1, t, w, r)] = cell(3.0 * (1 + p / 100), arm)
    return cells


def patch(mp, cells, failed=None):
    mp.setattr(G.R, "load", lambda gate: (cells, failed or {}))


def test_counts_books_contrasts_and_holm(monkeypatch):
    patch(monkeypatch, synth())
    r = G.read(["x"], TOPOS)
    assert all(c["complete"] for c in r["counts"].values()) and r["counts"][f"{G.A}|heavy"]["expected"] == 48
    b = r["books"][f"{G.A}|heavy"]
    assert b["over_cap_share_of_batches"] == pytest.approx(0.4) and b["prune_equals_fallbacks"]
    assert b["gain_vs_expansion_s_per_batch"] == pytest.approx(2.0) and b["topk_set_differs_share"] == pytest.approx(0.9)
    c = r["contrasts"][f"{G.A} vs {G.E}|moderate"]
    assert c["family"] == "primary" and c["wins"] == 12 and c["median_pct"] == pytest.approx(100 * (0.91 / 0.92 - 1))
    assert c["p_holm"] == pytest.approx(min(1.0, 2 * c["p"]))
    assert r["contrasts"][f"{G.A} vs {G.B}|heavy"]["family"] == "secondary" and "p_holm" in r["contrasts"][f"{G.A} vs {G.B}|heavy"]
    assert r["contrasts"][f"{G.E} vs {G.C}|heavy"]["family"] == "descriptive"
    assert r["PRIMARY_BOTH_RUNGS_FASTER"] and r["PRIMARY_HOLM_SIG"]


def test_flags_off_when_a_loses_at_one_rung(monkeypatch):
    cells = synth()
    for t in TOPOS:
        for w in G.WINDOWS:
            cells[(G.A, 1, t, w, "heavy")] = cell(3.0 * 0.93, G.A)   # slower than cd_exactS (0.92) at heavy
    patch(monkeypatch, cells)
    r = G.read(["x"], TOPOS)
    assert not r["PRIMARY_BOTH_RUNGS_FASTER"] and not r["PRIMARY_HOLM_SIG"]


def test_incomplete_cells_and_failed_are_counted_not_hidden(monkeypatch):
    cells = synth()
    del cells[(G.T, 1, 18500, "g0", "heavy")]
    patch(monkeypatch, cells, {(G.T, 1, 18500, "g0", "heavy"): {"why": "killed by the progress watchdog"}})
    r = G.read(["x"], TOPOS)
    assert not r["counts"][f"{G.T}|heavy"]["complete"] and r["counts"][f"{G.T}|heavy"]["failed"] == 1
    assert r["failed_why"] == ["killed by the progress watchdog"]


def test_refuses_other_topologies_and_windows(monkeypatch):
    patch(monkeypatch, synth(topos=TOPOS[:-1] + [18999]))
    with pytest.raises(SystemExit, match="outside --topos"):
        G.read(["x"], TOPOS)
    cells = synth()
    cells[(G.C, 0, 18500, "g9", "heavy")] = cell(3.0, G.C)
    patch(monkeypatch, cells)
    with pytest.raises(SystemExit, match="windows outside"):
        G.read(["x"], TOPOS)


def test_prints_counts_before_contrasts(monkeypatch, capsys):
    patch(monkeypatch, synth())
    G.print_report(G.read(["x"], TOPOS))
    out = capsys.readouterr().out
    assert out.index("cells per arm") < out.index("prune books") < out.index("paired %")
