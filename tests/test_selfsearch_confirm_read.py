import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import selfsearch_confirm_v1_read as C  # noqa: E402

TOPOS = list(range(16490, 16514))
RUNGS = ("moderate", "heavy")


def cell(lat, ss=False):
    c = {"averageElapsedTime": lat, "num_tasks": 100, "wallclock_s": 10.0, "schedulerCounters": {}, "decisionTiming": {"per_task_median_s": 0.01}}
    if ss:
        c["schedulerCounters"] = {"ss_batches": 10, "slate_declared_batches": 10, "ss_exact_batches": 8, "ss_ascent_batches": 2,
                                  "ss_changed_batches": 4, "ss_changed_tasks": 10, "ss_tasks": 40}
    return c


def synth(pct: dict, topos=TOPOS, windows=("g0", "g1"), twin=None):
    """cd 3.0 s (seed 0); no-split arms at 3.0 s; each self-search arm at pct[ckpt] (a number or per-rung dict) % off cd; seeds 1, 2."""
    cells = {}
    for t in topos:
        for w in windows:
            for r in RUNGS:
                for ref in ("cd", "cd_exactS"):
                    cells[(ref, 0, t, w, r)] = cell(3.0)
                for sd in (1, 2):
                    cells[("ra_gnn_eng_cdxapply", sd, t, w, r)] = cell(3.0)
                    for ck, p in pct.items():
                        v = p.get(r, 0) if isinstance(p, dict) else p
                        cells[(C.NOSPLIT[ck], sd, t, w, r)] = cell(3.0)
                        cells[(C.SS[ck], sd, t, w, r)] = cell(3.0 * (1 + v / 100), ss=True)
                    if twin is not None:
                        cells[(C.TWIN, sd, t, w, r)] = cell(3.0 * (1 + twin / 100), ss=True)
    return cells


def patch(mp, cells, failed=None):
    mp.setattr(C.R, "load", lambda gate: (cells, failed or {}))


def test_rung_labels_and_win_rules():
    c = lambda m, h: {"median_pct": m, "holm_p": h}
    assert C.rung_label(c(-6, .01)) == "CONFIRMED" and C.rung_label(c(-4, .01)) == "DIRECTION-ONLY"
    assert C.rung_label(c(3, .01)) == "CD-FASTER" and C.rung_label(c(1, .5)) == "NOT-SEPARATED"
    assert C.label({"moderate": c(-6, .01), "heavy": c(-6, .01)}) == "WIN"
    assert C.label({"moderate": c(-6, .01), "heavy": c(3, .01)}) == "CD-FASTER"
    assert C.label({"moderate": c(-6, .01), "heavy": c(-3, .01)}) == "DIRECTION-ONLY"
    assert C.label({"moderate": c(-6, .01), "heavy": c(None, None)}) == "NOT-SEPARATED"


def test_primary_is_holm_over_four_and_labelled(monkeypatch):
    patch(monkeypatch, synth({"ra_gnn_eng": -10.0, "ra_gnn_eng_physmp": 0.0}))
    r = C.read(["x"])["main"]["primary"]
    assert r["holm_over"] == 4 and r["labels"] == {"ra_gnn_eng": "WIN", "ra_gnn_eng_physmp": "NOT-SEPARATED"}


def test_a_cd_faster_rung_blocks_the_win(monkeypatch):
    patch(monkeypatch, synth({"ra_gnn_eng": {"moderate": -10.0, "heavy": 6.0}, "ra_gnn_eng_physmp": -10.0}))
    p = C.read(["x"])["main"]["primary"]
    assert p["labels"]["ra_gnn_eng"] == "CD-FASTER" and p["labels"]["ra_gnn_eng_physmp"] == "WIN"


def test_search_adds_family_pairs_each_checkpoint_with_its_own_no_split(monkeypatch):
    patch(monkeypatch, synth({"ra_gnn_eng": -10.0, "ra_gnn_eng_physmp": -10.0}))
    f = C.read(["x"])["main"]["search_adds"]
    assert f["holm_over"] == 4 and abs(f["tests"]["ra_gnn_eng self-search vs no-split|heavy"]["median_pct"] + 10.0) < 1e-9


def test_twin_family_runs_only_with_twin_cells(monkeypatch):
    patch(monkeypatch, synth({"ra_gnn_eng": -10.0, "ra_gnn_eng_physmp": -10.0}))
    r = C.read(["x"])
    assert r["main"]["mp"] == {"not_run": True} and not r["twin_run"]
    patch(monkeypatch, synth({"ra_gnn_eng": -10.0, "ra_gnn_eng_physmp": -10.0}, twin=-2.0))
    r = C.read(["x"])
    assert r["twin_run"] and r["main"]["mp"]["holm_over"] == 2
    assert r["main"]["mp"]["tests"]["gnn_eng self-search vs twin self-search|moderate"]["median_pct"] < 0


def test_descriptive_contrasts_stay_outside_the_families(monkeypatch):
    patch(monkeypatch, synth({"ra_gnn_eng": -10.0, "ra_gnn_eng_physmp": -10.0}))
    d = C.read(["x"])["main"]["descriptive"]
    assert "ra_gnn_eng self-search vs cd_exactS|heavy" in d and "ra_gnn_eng no-split vs cd|moderate" in d
    assert abs(d["ra_gnn_eng self-search vs cd_exactS|heavy"]["median_pct"] + 10.0) < 1e-9


def test_refuses_topologies_outside_the_range(monkeypatch):
    patch(monkeypatch, synth({"ra_gnn_eng": -10.0}, topos=TOPOS[:-1] + [16514]))
    with pytest.raises(SystemExit, match="outside 16490-16513"):
        C.read(["x"])


def test_failed_cell_counted_and_sensitivity_drops_its_topology(monkeypatch):
    cells = synth({"ra_gnn_eng": -10.0, "ra_gnn_eng_physmp": -10.0})
    k = ("ra_gnn_eng_selfsearch", 1, 16491, "g0", "heavy")
    del cells[k]
    patch(monkeypatch, cells, {k: {"why": "watchdog: projected"}})
    r = C.read(["x"])
    assert r["n_failed"] == 1 and r["sensitivity"]["excluded_topologies"] == [16491]
    assert r["sensitivity"]["families"]["primary"]["tests"]["ra_gnn_eng self-search vs cd|heavy"]["n_topologies"] == 23


def test_search_counts_and_counts_first(monkeypatch, capsys):
    patch(monkeypatch, synth({"ra_gnn_eng": -10.0}))
    r = C.read(["x"])
    s = r["search"]["ra_gnn_eng_selfsearch|moderate"]
    assert s["exact_batches"] + s["ascent_batches"] == s["batches"] and abs(s["changed_task_share"] - 0.25) < 1e-9
    C.print_report(r)
    assert capsys.readouterr().out.splitlines()[0].endswith("0 failed, 24 topologies")
