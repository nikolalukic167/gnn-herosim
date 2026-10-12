import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import gnn_selfsearch_read as G  # noqa: E402

TOPOS = list(range(16301, 16313))
RUNGS = ("moderate", "heavy")


def cell(lat, ss=False):
    c = {"averageElapsedTime": lat, "num_tasks": 100, "wallclock_s": 10.0,
         "decisionTiming": {"per_task_median_s": 0.01, "per_task_mean_s": 0.02}, "schedulerCounters": {}}
    if ss:
        c["schedulerCounters"] = {"ss_batches": 10, "ss_exact_batches": 7, "ss_ascent_batches": 3, "ss_changed_batches": 5,
                                  "ss_changed_tasks": 20, "ss_tasks": 40, "ss_scored": 99, "slate_declared_batches": 10}
    return c


def synth(pct, topos=TOPOS):
    cells = {}
    for t in topos:
        for r in RUNGS:
            cells[("cd", 0, t, "g0", r)] = cell(3.0)
            cells[("cd_exactS", 0, t, "g0", r)] = cell(3.0)
            cells[("ra_gnn_eng_nosplit", 1, t, "g0", r)] = cell(3.0)
            cells[("ra_gnn_eng_cdxapply", 1, t, "g0", r)] = cell(3.0)
            cells[(G.ARM, 1, t, "g0", r)] = cell(3.0 * (1 + pct / 100), ss=True)
    return cells


def patch(mp, cells, failed=None):
    mp.setattr(G.R, "load", lambda gate: (cells, failed or {}))


def test_trigger_needs_both_rungs(monkeypatch):
    patch(monkeypatch, synth(-10.0))
    r = G.read(["x"])
    assert r["trigger"] and r["contrasts"][f"{G.ARM} vs cd|heavy"]["wins"] == 12
    patch(monkeypatch, synth(-3.0))
    assert not G.read(["x"])["trigger"]


def test_search_counts_and_shares(monkeypatch):
    patch(monkeypatch, synth(-10.0))
    c = G.read(["x"])["search"]["moderate"]
    assert c["batches"] == 120 and c["exact_batches"] == 84 and c["ascent_batches"] == 36
    assert abs(c["changed_batch_share"] - 0.5) < 1e-9 and abs(c["changed_task_share"] - 0.5) < 1e-9


def test_refuses_other_topologies(monkeypatch):
    patch(monkeypatch, synth(-10.0, topos=TOPOS[:-1] + [16313]))
    with pytest.raises(SystemExit, match="outside 16301-16312"):
        G.read(["x"])


def test_cd_identity_flags_a_difference(monkeypatch):
    cells = synth(-10.0)
    ref = dict(cells)
    ref[("cd", 0, 16301, "g0", "heavy")] = cell(3.1)
    calls = iter([(cells, {}), (ref, {})])
    monkeypatch.setattr(G.R, "load", lambda gate: next(calls))
    r = G.read(["x"], ["y"])
    assert r["cd_identity"]["compared"] == 24 and r["cd_identity"]["different"] == 1


def test_counts_print_first(monkeypatch, capsys):
    patch(monkeypatch, synth(-10.0))
    G.print_report(G.read(["x"]))
    assert capsys.readouterr().out.splitlines()[0].endswith("0 failed, 12 topologies")
