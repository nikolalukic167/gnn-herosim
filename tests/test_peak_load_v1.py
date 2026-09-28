import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import peak_load_v1_read as R  # noqa: E402

TOPOS = list(range(9001, 9013))


def _write(d, topo, window, kind, seed, elapsed, queue=1.0):
    name = f"cc40s{topo}__{window}__{kind}_s{seed}"
    with open(os.path.join(d, name + ".summary.json"), "w") as fh:
        json.dump({"arm": name, "topology": topo, "window": window, "checkpoint_seed": seed,
                   "averageElapsedTime": elapsed, "averageQueueTime": queue}, fh)


def test_part_b_primary_and_twin(tmp_path):
    d = str(tmp_path)
    for t in TOPOS:
        for w in R.B_WINDOWS:
            for k in R.B_RULES:
                _write(d, t, w, k, 0, 10.0, 9.0)
            for s in R.SEEDS:
                _write(d, t, w, "xs1load_selfref", s, 7.0)
                _write(d, t, w, "xs1load", s, 8.0)
                _write(d, t, w, "gnnedge0", s, 9.0)
                _write(d, t, w, "mpoff", s, 9.5)
    res = R.read_b(d, TOPOS)
    assert res["primary"]["label"] == "CONFIRMED" and res["primary"]["median_pct"] == pytest.approx(-30.0)
    assert res["mp_twin"]["median_pct"] == pytest.approx(100 * (9.0 - 9.5) / 9.5)
    assert all(v == pytest.approx(0.9) for v in res["reactive_queue_share"].values())


def test_part_a_witness_and_fill(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for t in TOPOS:
        for w in R.A_WINDOWS:
            for k in ("cd", "cd_inflight", "selfpredict", "reactive"):
                _write(str(old), t, w, k, 0, 10.0)
            for s in R.SEEDS:
                _write(str(old), t, w, R.GNN, s, 8.0)
            for k in ("random", "batched", "decima"):
                _write(str(new), t, w, k, 0, 20.0)
    for t in (9119, 9420):
        for k, s, e in (("cd", 0, 10.0), (R.GNN, 1, 8.0)):
            _write(str(old), t, "w0x15d1", k, s, e)
            _write(str(new), t, "w0x15d1", k, s, e)
    res = R.read_a(str(old), str(new), TOPOS)
    assert res["W"]["pass"]
    assert res["vs"]["random"]["median_pct"] == pytest.approx(-60.0)
    assert res["primary"]["median_pct"] == pytest.approx(-20.0)
