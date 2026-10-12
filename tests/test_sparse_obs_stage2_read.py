import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import sparse_obs_v2_gate_read as G  # noqa: E402

TOPOS = list(range(19440, 19452))


def cell(lat, arm):
    c = {n: 0 for n in G.BOOKS}
    if arm != G.A0:
        c["pg_cold_tail_calls"] = 10
    if arm in (G.WN, G.WE):
        c["obs_cold_degraded"] = 10
    return {"averageElapsedTime": lat, "num_tasks": 100, "schedulerCounters": c}


def synth(pct):
    cells = {(a, 0, t, "g0", r): cell(3.0 * (1 + (p + (t % 5 - 2) * 0.1) / 100), a) for t in TOPOS for r in G.RUNGS for a, p in pct.items()}
    for (a, _, t, _, r), c in cells.items():   # steady-state keys: a different effect size so the labelled lines are distinguishable
        base = c["averageElapsedTime"]
        c["steady_state"] = {"after_360s": {"mean": base * 0.5, "n": 10}, "after_p5_dispatch": {"mean": base * 0.6, "n": 10}}
    return cells


def patch(mp, cells):
    mp.setattr(G.R, "load", lambda gate: (cells, {}))


def test_pass_and_instrument(monkeypatch):
    patch(monkeypatch, synth({G.A0: 0.0, G.AP: -2.0, G.WN: 18.0, G.WE: 9.0}))
    r = G.read(["x"], TOPOS)
    assert r["INSTRUMENT_OK"] and r["W1_PASS"] and not r["LEAK"]
    assert r["flags"]["heavy"]["A0P_GE_1"] and r["contrasts"][f"{G.AP} vs {G.A0}|heavy"]["family"] == "primary"


def test_kill_when_cold_seconds_are_not_worth_it(monkeypatch):
    patch(monkeypatch, synth({G.A0: 0.0, G.AP: 0.0, G.WN: 3.0, G.WE: 1.0}))
    assert not G.read(["x"], TOPOS)["W1_PASS"]


def test_leak_flag_when_a_degraded_arm_beats_its_reference(monkeypatch):
    patch(monkeypatch, synth({G.A0: 0.0, G.AP: 0.0, G.WN: -3.0, G.WE: 0.5}))
    r = G.read(["x"], TOPOS)
    assert r["LEAK"] and r["flags"]["moderate"]["LEAK"][G.WN]


def test_instrument_fails_when_w1_reaches_the_exact_cold_term(monkeypatch):
    cells = synth({G.A0: 0.0, G.AP: 0.0, G.WN: 5.0, G.WE: 5.0})
    cells[(G.WE, 0, TOPOS[0], "g0", "heavy")]["schedulerCounters"]["obs_cold_exact"] = 1
    patch(monkeypatch, cells)
    assert not G.read(["x"], TOPOS)["INSTRUMENT_OK"]


def test_unregistered_arm_is_refused(monkeypatch):
    cells = synth({G.A0: 0.0, G.AP: 0.0, G.WN: 5.0, G.WE: 5.0})
    cells[("cd_exactS", 0, TOPOS[0], "g0", "heavy")] = cell(3.0, "cd_exactS")
    patch(monkeypatch, cells)
    with pytest.raises(SystemExit):
        G.read(["x"], TOPOS)


def test_steady_state_lines_are_labelled_descriptive_and_outside_the_flags(monkeypatch):
    patch(monkeypatch, synth({G.A0: 0.0, G.AP: -2.0, G.WN: 18.0, G.WE: 9.0}))
    r = G.read(["x"], TOPOS)
    ss = r["steady_state_descriptive"]
    assert ss[f"{G.WN} vs {G.AP}|heavy|after_360s"]["median_pct"] == pytest.approx(r["contrasts"][f"{G.WN} vs {G.AP}|heavy"]["median_pct"], abs=0.5)
    assert all(c["family"].startswith("steady:") for c in ss.values()) and len(ss) == 3 * 2 * 2
    assert "steady" not in str(r["flags"])
