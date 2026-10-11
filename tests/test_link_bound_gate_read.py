import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import link_bound_gate_read as G  # noqa: E402

TOPOS = list(range(17007, 17019))


def cell(lat, ss=False, ex=False):
    c = {"averageElapsedTime": lat, "num_tasks": 100, "schedulerCounters": {}, "queue_share": 0.2, "effective_queue_share": 0.3,
         "arrival_end": {"end_over_last_arrival": 1.01},
         "link_telemetry": {"link_wait": {"mean": 0.03 * lat}, "core_util_median": 0.04, "core_util_max": 0.08}}
    if ss:
        c["schedulerCounters"] = {"ss_batches": 10, "ss_exact_batches": 6, "ss_ascent_batches": 4, "ss_changed_batches": 5, "ss_scored": 50}
    if ex:
        c["schedulerCounters"] = {"pg_batches": 20, "pg_exact_batches": 18, "pg_exact_fallbacks": 2, "pg_exact_plans": 1000}
    return c


def synth(pct_ss, pct_ex, topos=TOPOS):
    cells = {}
    for t in topos:
        for r in G.RUNGS:
            for w in G.WINDOWS:
                cells[("cd", 0, t, w, r)] = cell(3.0)
                cells[("cd_exactS", 0, t, w, r)] = cell(3.0 * (1 + pct_ex / 100), ex=True)
                cells[(G.ARM, 1, t, w, r)] = cell(3.0 * (1 + pct_ss / 100), ss=True)
    return cells


def patch(mp, cells, failed=None):
    mp.setattr(G.R, "load", lambda gate: (cells, failed or {}))


def test_contrasts_counts_and_shares(monkeypatch):
    patch(monkeypatch, synth(-10.0, -8.0))
    r = G.read(["x"], TOPOS)
    assert r["counts"][f"{G.ARM}|heavy"]["complete"] and r["counts"][f"{G.ARM}|heavy"]["expected"] == 48
    c = r["contrasts"][f"{G.ARM} vs cd|moderate"]
    assert c["n_topologies"] == 12 and c["wins"] == 12 and c["median_pct"] == pytest.approx(-10.0)
    assert r["contrasts"][f"cd_exactS vs cd|heavy"]["median_pct"] == pytest.approx(-8.0)
    assert r["contrasts"][f"{G.ARM} vs cd_exactS|heavy"]["median_pct"] == pytest.approx(100 * (0.9 / 0.92 - 1))
    assert r["search"]["heavy"]["exact_share"] == pytest.approx(0.6) and r["search"]["heavy"]["ascent_share"] == pytest.approx(0.4)
    assert r["cd_exactS"]["heavy"]["fallback_share_of_batches"] == pytest.approx(0.1)
    assert r["trigger_vs_cd"]


def test_incomplete_cells_are_flagged(monkeypatch):
    cells = synth(-10.0, -8.0)
    del cells[("cd", 0, 17007, "g0", "heavy")]
    patch(monkeypatch, cells)
    assert not G.read(["x"], TOPOS)["counts"]["cd|heavy"]["complete"]


def test_refuses_other_topologies_and_windows(monkeypatch):
    patch(monkeypatch, synth(-10.0, -8.0, topos=TOPOS[:-1] + [17999]))
    with pytest.raises(SystemExit, match="outside --topos"):
        G.read(["x"], TOPOS)
    cells = synth(-10.0, -8.0)
    cells[("cd", 0, 17007, "g9", "heavy")] = cell(3.0)
    patch(monkeypatch, cells)
    with pytest.raises(SystemExit, match="windows outside"):
        G.read(["x"], TOPOS)


def test_prints(monkeypatch, capsys):
    patch(monkeypatch, synth(-3.0, -8.0))
    r = G.read(["x"], TOPOS)
    G.print_report(r)
    out = capsys.readouterr().out
    assert out.index("cells per arm") < out.index("paired %") and "False" in out.splitlines()[-1]


def test_cd_health_flags_unhealthy_cells_and_keeps_them_in_the_contrasts(monkeypatch):
    cells = synth(-10.0, -8.0)
    cells[("cd", 0, 17007, "g0", "heavy")]["arrival_end"] = {"end_over_last_arrival": 1.2}
    patch(monkeypatch, cells)
    r = G.read(["x"], TOPOS)
    bad = [h for h in r["cd_health"] if h["unhealthy"]]
    assert len(bad) == 1 and (bad[0]["topology"], bad[0]["window"], bad[0]["rung"]) == (17007, "g0", "heavy")
    assert len(r["cd_health"]) == 96 and r["contrasts"][f"{G.ARM} vs cd|heavy"]["n_topologies"] == 12
    assert r["link"][f"{G.ARM}|heavy"]["link_wait_share_median"] == pytest.approx(0.03)
