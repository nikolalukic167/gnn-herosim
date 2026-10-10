import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import accel_moderate_confirm_v1_read as A  # noqa: E402
import scale_160_v1_read as S  # noqa: E402


def test_one_rung_family_of_two_and_labels(monkeypatch):
    monkeypatch.setattr(S, "RUNGS", A.RUNGS)
    c = lambda m, h: {"median_pct": m, "holm_p": h}
    assert A.RENAME[S.label({"moderate": c(-6.0, 0.01)})] == "CONFIRMED"
    assert S.label({"moderate": c(-4.0, 0.01)}) == "DIRECTION-ONLY"
    assert S.label({"moderate": c(-6.0, 0.20)}) == "DIRECTION-ONLY"
    assert S.label({"moderate": c(3.0, 0.01)}) == "CD-FASTER"
    assert S.label({"moderate": c(1.0, 0.50)}) == "NOT-SEPARATED"


def test_family_holm_over_two(monkeypatch):
    monkeypatch.setattr(S, "RUNGS", A.RUNGS)
    cells = {}
    for t in range(16321, 16333):
        for w in ("g0", "g1"):
            cells[("cd", 0, t, w, "moderate")] = {"averageElapsedTime": 3.0}
            for a in S.ARMS:
                for sd in (1, 2):
                    cells[(a, sd, t, w, "moderate")] = {"averageElapsedTime": 2.7 + 0.001 * (t - 16321)}
    fam = S.family(cells, list(range(16321, 16333)))
    assert fam["holm_over"] == 2 and set(fam["labels"]) == set(S.ARMS)


def test_importing_the_reader_leaves_the_scale_160_rungs_alone():
    assert S.RUNGS == ("moderate", "heavy")
