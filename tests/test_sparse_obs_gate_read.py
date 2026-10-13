import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import sparse_obs_v1_gate_read as G  # noqa: E402

TOPOS = list(range(19440, 19452))


def cell(lat, arm):
    c = {"averageElapsedTime": lat, "num_tasks": 100, "schedulerCounters": {n: 0 for n in G.OBS_BOOKS}}
    if arm != G.A0:
        bundle = "_bundle_" in arm
        c["schedulerCounters"].update(obs_drain_degraded=10, obs_exec_degraded=10 if bundle else 0, obs_partner_hidden=10 if bundle else 0,
                                      obs_partner_predicted=5 if (bundle and arm.endswith("_ec")) else 0)
    return c


def synth(pct):
    cells = {}
    for t in TOPOS:
        for r in G.RUNGS:
            for arm, p in pct.items():
                cells[(arm, 0, t, "g0", r)] = cell(3.0 * (1 + (p + (t % 5 - 2) * 0.1) / 100), arm)
    return cells


def patch(mp, cells, failed=None):
    mp.setattr(G.R, "load", lambda gate: (cells, failed or {}))


def pct(naive, ec):
    d = {G.A0: 0.0}
    for k in G.DEGRADATIONS:
        d[G.arm(k, "naive")], d[G.arm(k, "ec")] = naive, ec
    return d


def test_counts_instrument_and_pass(monkeypatch):
    patch(monkeypatch, synth(pct(30.0, 8.0)))
    r = G.read(["x"], TOPOS)
    assert all(c["complete"] and c["expected"] == 12 for c in r["counts"].values())
    assert r["INSTRUMENT_OK"] and r["STAGE1_PASS"]
    assert r["contrasts"]["VoI bundle|heavy"]["median_pct"] == pytest.approx(30.0, abs=0.3)
    assert r["flags"]["queue|heavy"]["D_PASS"] and "p_holm" in r["contrasts"]["R queue|moderate"]


def test_kill_when_the_channel_is_worth_little_or_the_residual_is_small(monkeypatch):
    patch(monkeypatch, synth(pct(30.0, 2.0)))
    assert not G.read(["x"], TOPOS)["STAGE1_PASS"]
    d = pct(5.0, 1.0)
    d[G.arm("bundle", "naive")], d[G.arm("bundle", "ec")] = 100.0, 60.0   # a big bundle does not pass the drain channel
    patch(monkeypatch, synth(d))
    r = G.read(["x"], TOPOS)
    assert not r["STAGE1_PASS"] and r["BUNDLE_VOI_PCT"]["heavy"] == pytest.approx(100.0, abs=0.3)
    patch(monkeypatch, synth(pct(5.0, 1.0)))
    assert not G.read(["x"], TOPOS)["STAGE1_PASS"]


def test_instrument_fails_on_an_exact_queue_walk_or_a_degraded_reference(monkeypatch):
    cells = synth(pct(30.0, 8.0))
    k = (G.arm("queue", "ec"), 0, TOPOS[0], "g0", "heavy")
    cells[k]["schedulerCounters"]["obs_drain_exact"] = 1
    patch(monkeypatch, cells)
    assert not G.read(["x"], TOPOS)["INSTRUMENT_OK"]
    cells = synth(pct(30.0, 8.0))
    cells[(G.A0, 0, TOPOS[0], "g0", "heavy")]["schedulerCounters"]["obs_exec_degraded"] = 1
    patch(monkeypatch, cells)
    assert not G.read(["x"], TOPOS)["INSTRUMENT_OK"]


def test_unregistered_arm_or_topology_is_refused(monkeypatch):
    cells = synth(pct(30.0, 8.0))
    cells[("cd_exactS", 0, TOPOS[0], "g0", "heavy")] = cell(3.0, "cd_exactS")
    patch(monkeypatch, cells)
    with pytest.raises(SystemExit):
        G.read(["x"], TOPOS)
    patch(monkeypatch, synth(pct(30.0, 8.0)))
    with pytest.raises(SystemExit):
        G.read(["x"], TOPOS[:-1] + [1])
