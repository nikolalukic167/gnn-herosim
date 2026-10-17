import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import uncut_te3_read as T  # noqa: E402


def _cell(d, topo, w, rung, arm, seed, lat):
    p = os.path.join(d, f"cc40s{topo}__{w}{rung}__{arm}_s{seed}.summary.json")
    json.dump({"averageElapsedTime": lat, "steady_state": {"after_360s": {"mean": lat}}}, open(p, "w"))


def _mk(tmp, factor):
    bar, cells = str(tmp / "bar"), {}
    os.makedirs(bar)
    for n, kind in T.ARMS.items():
        cells[n] = str(tmp / n)
        os.makedirs(cells[n])
    for t in range(100, 136):
        for w in ("g0", "g1", "g2", "g3"):
            for rung in ("moderate", "heavy"):
                _cell(bar, t, w, rung, "cd_ib0_k100", 0, 100.0)
                for n, kind in T.ARMS.items():
                    for sd in (1, 2):
                        for ch in T.CHARGES:
                            _cell(cells[n], t, w, rung, f"{kind}_{ch}", sd, 100.0 * factor + (t % 5) * 0.01)
    return bar, cells


def test_sign_interval_index_matches_the_filed_coverages():
    assert T.sign_interval_index(12)[0] == 3
    assert T.sign_interval_index(36)[0] == 12 and abs(T.sign_interval_index(36)[1] - 0.9712) < 1e-3
    assert T.sign_interval_index(48)[0] == 17


def test_labels_and_the_conservative_combination():
    assert T.label(0.01, -3.0, [-3.0] * 36) == "CLAIM"
    assert T.label(0.01, 3.0, [3.0] * 36) == "REVERSED"
    assert T.label(0.5, 0.2, [0.1 * i for i in range(36)]) == "CLEAN-NEGATIVE"
    assert T.label(0.5, -2.0, [-3.0 + 0.1 * i for i in range(36)]) == "UNRESOLVED"
    assert T.combine(["CLAIM", "CLAIM"]) == "CLAIM"
    assert T.combine(["CLAIM", "UNRESOLVED"]) == "UNRESOLVED"
    assert T.combine(["CLEAN-NEGATIVE", "CLAIM"]) == "UNRESOLVED"


def test_an_arm_uniformly_5_percent_faster_is_a_claim_on_all_six_tests(tmp_path):
    bar, cells = _mk(tmp_path, 0.95)
    res = T.read(bar, cells)
    assert res["charges"]["k100"]["n_new_topologies"] == 36
    for ch in T.CHARGES:
        assert all(t["label"] == "CLAIM" for t in res["charges"][ch]["tests"].values())
    assert all(row["heavy_rung"] == "CLAIM" for row in res["combined"].values())


def test_a_topology_missing_a_cell_is_dropped_and_listed(tmp_path):
    bar, cells = _mk(tmp_path, 0.95)
    os.remove(os.path.join(cells["arm3"], f"cc40s100__g0heavy__{T.ARMS['arm3']}_k100_s2.summary.json"))
    res = T.read(bar, cells)
    assert res["charges"]["k100"]["dropped"]["arm3/heavy"] == [100]
    assert res["charges"]["k100"]["tests"]["arm3/heavy/whole"]["n_topologies"] == 35
    assert res["charges"]["kr"]["dropped"]["arm3/heavy"] == []
