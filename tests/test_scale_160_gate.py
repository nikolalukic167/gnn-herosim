import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import fresh_topo_burst_v1_gate as G  # noqa: E402
import scale_160_v1_read as S  # noqa: E402

SEL = os.path.join(os.path.dirname(__file__), "..", "scripts_cosim", "scale_160_v1_selected.json")


def test_selection_is_12_fresh_ids_disjoint_from_the_record():
    sel = json.load(open(SEL))
    assert sel["verdict"] == "DESIGN-READY" and len(sel["topologies"]) == 12 and len(set(sel["topologies"])) == 12
    assert all(t >= 16001 for t in sel["topologies"] + sel["spares"])
    assert not set(sel["topologies"]) & set(range(16101, 16201))  # S6's corpus pool
    assert not set(sel["topologies"]) & (set(range(9901, 9909)) | {9101})


@pytest.fixture
def s160_env(monkeypatch):
    for k, v in {"HEROSIM_TRANSFER_MODEL": "pipelined", "HEROSIM_REPLICA_RELEASE": "1", "HEROSIM_SCALEOUT": "kpa",
                 "GATE_FIXED_POLICY_TIME_SCALE": "1.0", "HEROSIM_SHARED_AUTOSCALER": "0"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(G, "WF1_LADDER", {"g0moderate": ("a", "wf1_moderate"), "g0heavy": ("a", "wf1_heavy")})
    monkeypatch.setattr(G, "WF1_TAGS", ["moderate", "heavy"])
    monkeypatch.setenv("R1A_ARMS", "cd,ra_gnn_eng,ra_gnn_eng_physmp,selfpredict,ra_gnn_eng_cdapply")


def test_cell_grid(s160_env):
    cells = G.r1a_tasks(json.load(open(SEL)))
    # 12 topologies x 2 rungs x 4 windows; CD and self-predict once, the three learned/seeded arms at seeds 1 and 2
    assert len(cells) == 12 * 2 * 4 * (2 + 3 * 2)
    assert {c["seed"] for c in cells if c["kind"] in ("cd", "selfpredict")} == {0}
    assert {c["seed"] for c in cells if c["kind"] == "ra_gnn_eng_physmp"} == {1, 2}


def _cell(cells, kind, seed, topo, win, rung, lat, defer=100):
    cells[(kind, seed, topo, win, rung)] = {"averageElapsedTime": lat, "averageWaitTime": 0.01,
                                           "schedulerCounters": {"prefix_tasks_deferred": defer},
                                           "decisionTiming": {"per_task_median_s": 0.001}}


def _grid(delta):
    cells = {}
    for topo in range(16001, 16013):
        for rung in S.RUNGS:
            for win in ("g0", "g1"):
                _cell(cells, "cd", 0, topo, win, rung, 100.0)
                for arm in S.ARMS:
                    for seed in (1, 2):
                        _cell(cells, arm, seed, topo, win, rung, 100.0 * (1 + delta[arm][rung] / 100.0))
    return cells


def _labels(delta):
    cells = _grid(delta)
    return S.family(cells, sorted({k[2] for k in cells}))["labels"]


def test_labels():
    win = {"moderate": -8.0, "heavy": -6.0}
    mid = {"moderate": -2.0, "heavy": -3.0}
    slower = {"moderate": 4.0, "heavy": 6.0}
    mixed = {"moderate": -6.0, "heavy": 1.0}
    out = _labels({"ra_gnn_eng": win, "ra_gnn_eng_physmp": slower})
    assert out == {"ra_gnn_eng": "WIN", "ra_gnn_eng_physmp": "CD-FASTER"}
    out = _labels({"ra_gnn_eng": mid, "ra_gnn_eng_physmp": mixed})
    assert out == {"ra_gnn_eng": "DIRECTION-ONLY", "ra_gnn_eng_physmp": "CD-FASTER"}


def test_a_win_needs_both_rungs_and_five_percent():
    out = _labels({"ra_gnn_eng": {"moderate": -8.0, "heavy": -3.0}, "ra_gnn_eng_physmp": {"moderate": -8.0, "heavy": -8.0}})
    assert out == {"ra_gnn_eng": "DIRECTION-ONLY", "ra_gnn_eng_physmp": "WIN"}


def test_side_by_side_is_unpaired_and_reports_median_and_iqr():
    import scale_vs_accel_side_by_side as SB

    def read(vals):
        return {"main": {"tests": {f"{a}|{r}": {"per_topology": {str(i): v for i, v in enumerate(vals)}} for a in SB.ARMS for r in SB.RUNGS}}}

    out = SB.table(read([1.0, 2.0, 3.0, 4.0, 5.0]), read([-5.0, -4.0, -3.0, -2.0, -1.0]))
    assert "unpaired, descriptive" in out
    assert "+3.00 % [+2.00, +4.00] n=5" in out and "-3.00 % [-4.00, -2.00] n=5" in out
