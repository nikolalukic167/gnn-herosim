import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import cost_to_go_v1_read as C  # noqa: E402

TOPOS = list(range(16369, 16393))
RUNGS = ("moderate", "heavy")


def synth(arm_pct: dict, topos=TOPOS, windows=("g0", "g1"), seeds=(1, 2)):
    """cd_exactS = 3.0 s everywhere (seed 0); each arm is `pct` % off it; descriptive refs 3.3 s."""
    cells = {}
    for t in topos:
        for w in windows:
            for r in RUNGS:
                for ref in ("cd_exactS", "cd", "cd_expand", "cdxapply"):
                    cells[(ref, 0, t, w, r)] = {"averageElapsedTime": 3.0 if ref == "cd_exactS" else 3.3}
                for a, pct in arm_pct.items():
                    for sd in seeds:
                        cells[(a, sd, t, w, r)] = {"averageElapsedTime": 3.0 * (1 + pct.get(r, 0) / 100) if isinstance(pct, dict) else 3.0 * (1 + pct / 100)}
    return cells


def patch(monkeypatch, cells, failed=None):
    monkeypatch.setattr(C.R, "load", lambda gate: (cells, failed or {}))


def test_rung_labels():
    c = lambda m, h: {"median_pct": m, "holm_p": h}
    assert C.rung_label(c(-6.0, 0.01)) == "CONFIRMED"
    assert C.rung_label(c(-4.0, 0.01)) == "DIRECTION-ONLY"
    assert C.rung_label(c(3.0, 0.01)) == "REF-FASTER"
    assert C.rung_label(c(1.0, 0.5)) == "NOT-SEPARATED"
    assert C.rung_label(c(None, None)) == "NOT-SEPARATED"


def test_win_needs_both_rungs_and_no_ref_faster():
    c = lambda m, h: {"median_pct": m, "holm_p": h}
    assert C.label({"moderate": c(-6, .01), "heavy": c(-6, .01)}) == "WIN"
    assert C.label({"moderate": c(-6, .01), "heavy": c(-3, .01)}) == "DIRECTION-ONLY"
    assert C.label({"moderate": c(-6, .01), "heavy": c(3, .01)}) == "REF-FASTER"
    assert C.label({"moderate": c(-6, .01), "heavy": c(None, None)}) == "NOT-SEPARATED"


def test_a_clear_win_is_labelled_win_with_holm_over_arms_times_rungs(monkeypatch):
    patch(monkeypatch, synth({"cdexs_handv": -10.0, "mlpv": 0.0}))
    r = C.read(["x"], ["cdexs_handv", "mlpv"])
    assert r["main"]["holm_over"] == 4 and r["main"]["ref"] == "cd_exactS"
    assert r["main"]["labels"]["cdexs_handv"] == "WIN"
    assert r["main"]["labels"]["mlpv"] == "NOT-SEPARATED"
    t = r["main"]["tests"]["cdexs_handv|moderate"]
    assert abs(t["median_pct"] + 10.0) < 1e-9 and t["wins"] == 24 and t["n_topologies"] == 24


def test_a_ref_faster_arm_is_not_a_win(monkeypatch):
    patch(monkeypatch, synth({"cdexs_handv": {"moderate": -10.0, "heavy": 8.0}}))
    r = C.read(["x"], ["cdexs_handv"])
    assert r["main"]["labels"]["cdexs_handv"] == "REF-FASTER"
    assert r["main"]["rung_labels"]["cdexs_handv|moderate"] == "CONFIRMED"
    assert r["main"]["rung_labels"]["cdexs_handv|heavy"] == "REF-FASTER"


def test_a_missing_arm_counts_as_p_one_and_keeps_the_family_size(monkeypatch):
    patch(monkeypatch, synth({"cdexs_handv": -10.0}))
    r = C.read(["x"], ["cdexs_handv", "mlpv"])
    assert r["main"]["holm_over"] == 4
    assert r["main"]["tests"]["mlpv|moderate"]["holm_p"] is None  # p absent: reported as not run
    assert r["main"]["labels"]["mlpv"] == "NOT-SEPARATED"
    assert r["main"]["labels"]["cdexs_handv"] == "WIN"


def test_descriptive_contrasts_are_outside_the_family_and_use_their_own_refs(monkeypatch):
    patch(monkeypatch, synth({"cdexs_handv": -10.0}))
    r = C.read(["x"], ["cdexs_handv"])
    assert r["main"]["holm_over"] == 2
    d = r["main"]["descriptive"]
    assert set(d) == {f"cdexs_handv vs {x}|{u}" for x in ("cd", "cd_expand", "cdxapply") for u in RUNGS}
    assert abs(d["cdexs_handv vs cd|moderate"]["median_pct"] - (100 * (2.7 - 3.3) / 3.3)) < 1e-9


def test_refuses_a_topology_outside_the_registered_range(monkeypatch):
    patch(monkeypatch, synth({"cdexs_handv": -10.0}, topos=TOPOS[:-1] + [16393]))
    with pytest.raises(SystemExit, match="outside 16369-16392"):
        C.read(["x"], ["cdexs_handv"])
    patch(monkeypatch, synth({"cdexs_handv": -10.0}, topos=[16368] + TOPOS[1:]))
    with pytest.raises(SystemExit, match="16368"):
        C.read(["x"], ["cdexs_handv"])


def test_a_failed_cell_is_counted_and_sensitivity_drops_its_topology(monkeypatch):
    cells = synth({"cdexs_handv": -10.0})
    del cells[("cdexs_handv", 1, 16370, "g0", "heavy")]
    failed = {("cdexs_handv", 1, 16370, "g0", "heavy"): {"why": "watchdog: projected"}}
    patch(monkeypatch, cells, failed)
    r = C.read(["x"], ["cdexs_handv"])
    assert r["n_failed"] == 1
    assert r["arms"]["cdexs_handv|heavy"]["failed"] == 1 and r["arms"]["cdexs_handv|heavy"]["failed_why"] == {"watchdog": 1}
    assert r["sensitivity"]["excluded_topologies"] == [16370]
    assert r["sensitivity"]["family"]["tests"]["cdexs_handv|heavy"]["n_topologies"] == 23


def test_report_prints_counts_first(monkeypatch, capsys):
    patch(monkeypatch, synth({"cdexs_handv": -10.0}))
    C.print_report(C.read(["x"], ["cdexs_handv"]))
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("0 failed, 24 topologies") and out[1].startswith("-- cells per arm")


def test_importing_the_reader_leaves_the_other_readers_alone():
    import scale_160_v1_read as S
    assert S.ARMS == ("ra_gnn_eng", "ra_gnn_eng_physmp") and S.RUNGS == ("moderate", "heavy")
