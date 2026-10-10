import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import accel_nosplit_classical_read as C  # noqa: E402

TOPOS = list(range(16345, 16369))
RUNGS = ("moderate", "heavy")
GNN = ("ra_gnn_eng_nosplit", "ra_gnn_eng_physmp_nosplit")


def cell(lat, end=1.0, share=0.3):
    return {"averageElapsedTime": lat, "num_tasks": 100, "wallclock_s": 10.0, "effective_queue_share": share,
            "arrival_end": {"end_over_last_arrival": end}, "latency_percentiles": {"p95": 2 * lat}}


def synth(pct: dict, topos=TOPOS, windows=("g0", "g1")):
    """cd and the no-split GNN arms are 3.0 s (CD seed 0, GNN seeds 1 and 2); each classical arm at seed 0 is pct[arm] % off."""
    cells = {}
    for t in topos:
        for w in windows:
            for r in RUNGS:
                cells[("cd", 0, t, w, r)] = cell(3.0)
                for g in GNN:
                    for sd in (1, 2):
                        cells[(g, sd, t, w, r)] = cell(3.0)
                for a, p in pct.items():
                    cells[(a, 0, t, w, r)] = cell(3.0 * (1 + p / 100))
    return cells


def patch(monkeypatch, cells, failed=None):
    monkeypatch.setattr(C.R, "load", lambda gate: (cells, failed or {}))


def test_a_seed_zero_arm_pairs_with_a_seeded_reference(monkeypatch):
    patch(monkeypatch, synth({"reactive": 20.0}))
    c = C.contrast(synth({"reactive": 20.0}), "reactive", "ra_gnn_eng_nosplit", "heavy", TOPOS)
    assert c["n_topologies"] == 24 and abs(c["median_pct"] - 20.0) < 1e-9 and c["wins"] == 0


def test_three_families_each_holm_over_eight(monkeypatch):
    patch(monkeypatch, synth({"reactive": 20.0, "random": 40.0, "batched": 5.0, "locality": -3.0}))
    r = C.read(["x"])
    assert set(r["families"]) == set(C.REFS) and all(f["holm_over"] == 8 for f in r["families"].values())
    lab = r["families"]["cd"]["labels"]
    assert lab["reactive|heavy"] == "SLOWER" and lab["locality|heavy"] == "FASTER"


def test_a_missing_arm_is_not_separated_and_keeps_the_family_size(monkeypatch):
    patch(monkeypatch, synth({"reactive": 20.0}))
    r = C.read(["x"])
    f = r["families"]["cd"]
    assert f["holm_over"] == 8 and f["tests"]["random|heavy"]["p"] is None and f["labels"]["random|heavy"] == "NOT-SEPARATED"


def test_collapse_is_recorded_not_dropped(monkeypatch):
    cells = synth({"reactive": 20.0})
    cells[("reactive", 0, 16345, "g0", "heavy")] = cell(30.0, end=3.0, share=0.95)
    patch(monkeypatch, cells)
    r = C.read(["x"])
    c = r["collapse"]["reactive|heavy"]
    assert c["finished"] == 48 and c["end_over_last_arrival_gt_1.5"] == 1 and c["effective_share_gt_0.80"] == 1
    assert r["families"]["cd"]["tests"]["reactive|heavy"]["n_topologies"] == 24


def test_refuses_a_topology_outside_the_range(monkeypatch):
    patch(monkeypatch, synth({"reactive": 1.0}, topos=TOPOS[:-1] + [16369]))
    with pytest.raises(SystemExit, match="outside 16345-16368"):
        C.read(["x"])


def test_a_failed_cell_is_counted_and_sensitivity_drops_its_topology(monkeypatch):
    cells = synth({"reactive": 20.0})
    del cells[("reactive", 0, 16346, "g0", "heavy")]
    patch(monkeypatch, cells, {("reactive", 0, 16346, "g0", "heavy"): {"why": "watchdog: projected"}})
    r = C.read(["x"])
    assert r["n_failed"] == 1 and r["collapse"]["reactive|heavy"]["failed"] == 1
    assert r["sensitivity"]["excluded_topologies"] == [16346]
    assert r["sensitivity"]["families"]["cd"]["tests"]["reactive|heavy"]["n_topologies"] == 23


def test_counts_print_first(monkeypatch, capsys):
    patch(monkeypatch, synth({"reactive": 20.0}))
    C.print_report(C.read(["x"]))
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("0 failed, 24 topologies") and out[1].startswith("-- cells per arm")
