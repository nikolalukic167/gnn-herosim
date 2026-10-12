import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import slate_bar_v1_gate_read as G  # noqa: E402

TOPOS = list(range(16490, 16514))


def cell(lat, arm):
    c = {"averageElapsedTime": lat, "num_tasks": 100, "schedulerCounters": {"pg_batches": 100, "pg_exact_batches": 60, "pg_exact_fallbacks": 40}}
    if arm in (G.S, G.K):
        c["schedulerCounters"].update(pg_declared_batches=100, pg_declared_groups=100, pg_declared_out_of_slate=0)
    return c


def synth(pct, topos=TOPOS):
    cells = {}
    for t in topos:
        for r in G.RUNGS:
            for w in G.WINDOWS:
                for arm, p in pct.items():
                    for seed in ((1, 2) if arm == G.G else (0,)):
                        cells[(arm, seed, t, w, r)] = cell(3.0 * (1 + p / 100), arm)
    return cells


def patch(mp, cells, failed=None):
    mp.setattr(G.R, "load", lambda gate: (cells, failed or {}))


def vary(base, t_idx):
    return base + (t_idx % 5 - 2) * 0.1   # small spread so the Wilcoxon has no ties


def synth_var(pct):
    cells = synth({k: 0.0 for k in pct})
    for (arm, seed, t, w, r) in list(cells):
        cells[(arm, seed, t, w, r)] = cell(3.0 * (1 + vary(pct[arm], t - 16490) / 100), arm)
    return cells


def test_counts_instrument_and_handicap_label(monkeypatch):
    # E 0, S +4 (the slate costs 4 %), G +3.5 (the GNN ties the slate arm within noise, ahead of it): SLATE-HANDICAP
    patch(monkeypatch, synth_var({G.C: 9.0, G.E: 0.0, G.S: 4.0, G.K: 2.0, G.G: 3.0}))
    r = G.read(["x"], TOPOS)
    assert r["counts"][f"{G.G}|heavy"]["expected"] == 192 and all(c["complete"] for c in r["counts"].values())
    assert r["INSTRUMENT_OK"] and r["books"][f"{G.S}|heavy"]["out_of_slate_zero"]
    assert r["contrasts"][f"{G.G} vs {G.S}|heavy"]["family"] == "primary" and r["contrasts"][f"{G.S} vs {G.E}|heavy"]["family"] == "secondary"
    assert r["contrasts"][f"{G.K} vs {G.S}|heavy"]["family"] == "descriptive"
    assert r["LABEL"] == "SLATE-HANDICAP" and r["labels"]["moderate"] == "SLATE-HANDICAP"


def test_irrelevant_when_the_slate_costs_nothing(monkeypatch):
    patch(monkeypatch, synth_var({G.C: 9.0, G.E: 0.0, G.S: 0.0, G.K: 0.0, G.G: 3.0}))
    r = G.read(["x"], TOPOS)
    assert r["labels"]["heavy"] in ("SLATE-IRRELEVANT", "SLATE-IS-DISCLOSED-LIMIT")
    cells = synth_var({G.C: 9.0, G.E: 0.0, G.S: 3.0, G.K: 3.0, G.G: 3.0})
    for k, v in cells.items():
        if k[0] == G.S:
            cells[k] = cell(3.0 * (1 + vary(0.0, k[2] - 16490) / 100), G.S)   # S ties E; G ties S
        if k[0] == G.G:
            cells[k] = cell(3.0 * (1 + vary(0.0, k[2] - 16490) / 100), G.G)
    patch(monkeypatch, cells)
    assert G.read(["x"], TOPOS)["LABEL"] == "SLATE-IRRELEVANT"


def test_disclosed_limit_when_the_gnn_loses_to_the_slate_arm(monkeypatch):
    patch(monkeypatch, synth_var({G.C: 9.0, G.E: 0.0, G.S: 4.0, G.K: 2.0, G.G: 8.0}))
    assert G.read(["x"], TOPOS)["LABEL"] == "SLATE-IS-DISCLOSED-LIMIT"


def test_out_of_slate_pick_breaks_the_instrument(monkeypatch):
    cells = synth_var({G.C: 9.0, G.E: 0.0, G.S: 4.0, G.K: 2.0, G.G: 3.0})
    k = next(k for k in cells if k[0] == G.S)
    cells[k]["schedulerCounters"]["pg_declared_out_of_slate"] = 1
    patch(monkeypatch, cells)
    assert not G.read(["x"], TOPOS)["INSTRUMENT_OK"]


def test_refuses_a_foreign_topology_or_arm(monkeypatch):
    cells = synth_var({G.C: 9.0, G.E: 0.0, G.S: 4.0, G.K: 2.0, G.G: 3.0})
    cells[(G.S, 0, 99999, "g0", "heavy")] = cell(3.0, G.S)
    patch(monkeypatch, cells)
    with pytest.raises(SystemExit):
        G.read(["x"], TOPOS)
