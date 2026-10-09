"""r1_attribution_v1 live gate: cell list, progress watchdog, pause line."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import fresh_topo_burst_v1_gate as G  # noqa: E402
import progress_watchdog as pw  # noqa: E402


def test_decide_waits_for_grace_then_projects():
    assert pw.decide(300, 100.0, 1000.0, 8100, 0) is None  # inside the grace period nothing is killed
    assert pw.decide(700, 500.0, 1000.0, 8100, 0) is None  # projected 1,400 s
    v = pw.decide(700, 5.0, 1000.0, 8100, 0)  # projected 140,000 s
    assert v["reason"] == "projected" and v["projected_s"] == pytest.approx(140000.0)


def test_decide_stall_and_no_progress():
    assert pw.decide(900, 400.0, 1000.0, 8100, 700)["reason"] == "stall"
    assert pw.decide(900, None, 1000.0, 8100, 100)["reason"] == "no-progress"


def test_watchdog_kills_a_slow_process_group(tmp_path):
    prog = tmp_path / "p.progress"
    prog.write_text(json.dumps({"sim_now": 1.0, "wall": time.time()}))
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    dog = pw.Watchdog(str(prog), last_arrival_s=1000.0, limit_s=100.0, grace_s=0.2, stall_s=1000.0, poll_s=0.05)
    dog.attach(proc)
    assert proc.wait(timeout=10) != 0
    assert dog.verdict["reason"] == "projected"


def test_pause_line(tmp_path):
    for i in range(40):  # 39 done, none hung: not paused
        (tmp_path / f"cc40s{9000 + i}__g0heavy__cd_s0.summary.json").write_text("{}")
    assert not pw.paused(pw.rung_state(str(tmp_path), "heavy"))
    for i in range(3):  # 3 of 43 hung = 7 %
        (tmp_path / f"cc40s{9100 + i}__g1heavy__selfpredict_s0.failed.json").write_text(json.dumps({"why": "watchdog: stall"}))
    st = pw.rung_state(str(tmp_path), "heavy")
    assert st == {"decided": 43, "hung": 3} and pw.paused(st)
    assert not pw.paused(pw.rung_state(str(tmp_path), "light"))


def test_pause_ignores_knative_and_non_hang_failures(tmp_path):
    for i in range(45):
        (tmp_path / f"cc40s{9000 + i}__g0moderate__cd_s0.summary.json").write_text("{}")
    for i in range(10):
        (tmp_path / f"cc40s{9200 + i}__g0moderate__reactive_s0.failed.json").write_text(json.dumps({"why": "timeout"}))
    (tmp_path / "cc40s9300__g0moderate__locality_s0.failed.json").write_text(json.dumps({"why": "num_tasks=1"}))
    st = pw.rung_state(str(tmp_path), "moderate")
    assert st == {"decided": 46, "hung": 0}


@pytest.fixture
def r1a_env(monkeypatch):
    for k, v in {"HEROSIM_TRANSFER_MODEL": "pipelined", "HEROSIM_REPLICA_RELEASE": "1", "HEROSIM_SCALEOUT": "kpa",
                 "GATE_FIXED_POLICY_TIME_SCALE": "1.0", "HEROSIM_SHARED_AUTOSCALER": "0"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(G, "WF1_LADDER", {"g0light": ("a", "wf1_light")})
    monkeypatch.setattr(G, "WF1_TAGS", ["light", "moderate", "heavy"])


def test_r1a_cell_list(r1a_env, monkeypatch):
    sel = {"topologies": list(range(19))}
    cells = G.r1a_tasks(sel)
    # 19 topologies x 3 rungs x 4 windows; 6 classical arms once, 7 learned + 2 seeded-CD arms at 2 seeds
    assert len(cells) == 19 * 3 * 4 * (6 + 9 * 2)
    assert {c["seed"] for c in cells if c["kind"] in G.R1A_CLASSICAL} == {0}
    assert {c["seed"] for c in cells if c["kind"] == "ra_gnn_eng"} == {1, 2}
    monkeypatch.setenv("R1A_SHARD", "1/4")
    shard = G.r1a_tasks(sel)
    assert len(shard) == len(cells) // 4 and shard[0] == cells[1]


def test_r1a_refuses_other_physics(r1a_env, monkeypatch):
    monkeypatch.setenv("HEROSIM_SCALEOUT", "legacy")
    with pytest.raises(SystemExit):
        G.r1a_tasks({"topologies": [1]})
