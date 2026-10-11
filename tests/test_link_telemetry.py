import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))
import fresh_topo_burst_v1_gate as G  # noqa: E402


def test_link_telemetry_splits_core_and_access_and_reads_per_task_wait():
    st = {"endTime": 100.0,
          "fabricLinkBusy": {"core0|core1": 50.0, "core1|core2": 10.0, "n0|core0": 5.0},
          "fabricLinkBandwidth": {"core0|core1": 250.0, "core1|core2": 250.0, "n0|core0": 250.0, "n1|core1": 250.0},
          "taskResults": [{"taskId": i, "linkWaitTime": w} for i, w in enumerate([0.0, 1.0, 3.0])] + [{"taskId": -1, "linkWaitTime": 99.0}]}
    t = G.link_telemetry(st)
    assert t["core_links"] == 2 and t["access_links"] == 2
    assert t["core_util_max"] == 0.5 and t["core_util_median"] == 0.3
    assert t["access_util_median"] == 0.025
    assert t["link_wait"]["n"] == 3 and t["link_wait"]["median"] == 1.0 and t["link_wait"]["max"] == 3.0
    assert t["core_bandwidth_mbps"] == [250.0]


def test_no_fabric_gives_none():
    assert G.link_telemetry({"endTime": 10.0, "taskResults": []}) is None
