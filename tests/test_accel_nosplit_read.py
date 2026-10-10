import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import accel_nosplit_v1_read as A  # noqa: E402
import scale_160_v1_read as S  # noqa: E402


def test_rung_labels():
    c = lambda m, h: {"median_pct": m, "holm_p": h}
    assert A.rung_label(c(-6.0, 0.01)) == "CONFIRMED"
    assert A.rung_label(c(-4.0, 0.01)) == "DIRECTION-ONLY"
    assert A.rung_label(c(3.0, 0.01)) == "CD-FASTER"
    assert A.rung_label(c(1.0, 0.5)) == "NOT-SEPARATED"
    assert A.rung_label(c(None, None)) == "NOT-SEPARATED"


def test_win_needs_both_rungs_and_no_cd_faster():
    c = lambda m, h: {"median_pct": m, "holm_p": h}
    assert S.label({"moderate": c(-6, .01), "heavy": c(-6, .01)}) == "WIN"
    assert S.label({"moderate": c(-6, .01), "heavy": c(-3, .01)}) == "DIRECTION-ONLY"
    assert S.label({"moderate": c(-6, .01), "heavy": c(3, .01)}) == "CD-FASTER"


def test_family_is_holm_over_four_on_the_nosplit_arms_and_restores_globals(monkeypatch):
    monkeypatch.setattr(S, "ARMS", A.ARMS)
    monkeypatch.setattr(S, "DESCRIPTIVE", A.DESCRIPTIVE)
    cells = {}
    for t in range(16345, 16357):
        for w in ("g0", "g1"):
            for r in ("moderate", "heavy"):
                cells[("cd", 0, t, w, r)] = {"averageElapsedTime": 3.0}
                for a in A.ARMS + A.DESCRIPTIVE:
                    for sd in (1, 2):
                        cells[(a, sd, t, w, r)] = {"averageElapsedTime": 2.8}
    fam = S.family(cells, list(range(16345, 16357)))
    assert fam["holm_over"] == 4 and set(fam["labels"]) == set(A.ARMS)


def test_importing_the_reader_leaves_the_scale_160_arms_alone():
    assert S.ARMS == ("ra_gnn_eng", "ra_gnn_eng_physmp")


def test_vs_cd_expand_pairs_against_seed_zero(monkeypatch):
    monkeypatch.setattr(S, "RUNGS", ("moderate", "heavy"))
    cells = {}
    for t in range(16345, 16349):
        cells[("cd_expand", 0, t, "g0", "moderate")] = {"averageElapsedTime": 3.0}
        cells[("ra_gnn_eng_nosplit", 1, t, "g0", "moderate")] = {"averageElapsedTime": 2.7}
    monkeypatch.setattr(S.R, "load", lambda gate: (cells, {}))
    out = A.vs_cd_expand(["x"], list(range(16345, 16349)))
    c = out["ra_gnn_eng_nosplit|moderate"]
    assert c["n_topologies"] == 4 and abs(c["median_pct"] + 10.0) < 1e-9 and c["wins"] == 4
    assert out["ra_gnn_eng_physmp_nosplit|moderate"]["median_pct"] is None
