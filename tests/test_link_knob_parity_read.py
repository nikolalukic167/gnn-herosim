import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import link_knob_parity_read as P  # noqa: E402


def fixture(lat_shift=0.0, node="n1", core=62.5):
    raw = {"stats": {"taskResults": [{"taskId": 0, "executionNode": "n1", "executionPlatform": 0, "linkWaitTime": 0.5}]}}
    tab = {"scheduled": [10.0], "done": [20.0]}
    snaps = [json.dumps({"snapshot_id": 3, "fidelity": {"batch": [{"gid": 0}]}})]
    row = {"tag": "x|3", "batch_lat": [10.0 + lat_shift], "core_bw": core,
           "batch_detail": [{"executionNode": node, "executionPlatform": 0, "linkWaitTime": 0.5}]}
    return [row], tab, raw, snaps


def test_pass_within_the_poll_offset_and_not_vacuous():
    r = P.read(*fixture(0.0009), 62.5, 1000.0, 62.5)
    assert r["verdict"] == "PASS" and r["link_wait_replay_nonzero"] == 1


def test_fail_on_plan_latency_or_vacuous_bandwidth():
    assert P.read(*fixture(0.0), 62.5, 1000.0, 62.5)["verdict"] == "PASS"
    assert P.read(*fixture(0.01), 62.5, 1000.0, 62.5)["verdict"] == "FAIL"
    assert P.read(*fixture(0.0, node="n2"), 62.5, 1000.0, 62.5)["verdict"] == "FAIL"
    assert P.read(*fixture(0.0), 1000.0, 1000.0, 1000.0)["verdict"] == "FAIL"
    assert P.read(*fixture(0.0, core=1000.0), 62.5, 1000.0, 62.5)["verdict"] == "FAIL"
    assert P.read(*fixture(0.0), 62.5, 1000.0, 1000.0)["verdict"] == "FAIL"
