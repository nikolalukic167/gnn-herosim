"""workload_fix_v1 W4 live-path check: reachability, platform types, memory and distinct-platform fit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import workload_fix_v1_reachability_live as live  # noqa: E402

TYPES = {
    "a": {"platforms": ["cpu"], "memoryRequirements": {"cpu": 1.0}},
    "b": {"platforms": ["cpu", "gpu"], "memoryRequirements": {"cpu": 1.0, "gpu": 4.0}},
}


def infra(server_platforms, mem=8, net=("client_node0",), fabric=None):
    return {
        "nodes": [{"node_name": "client_node0", "platforms": ["cpu"], "memory": 8, "network_map": {"node0": 0.1}},
                  {"node_name": "node0", "platforms": server_platforms, "memory": mem,
                   "network_map": {c: 0.1 for c in net}}],
        "link_topology": fabric,
    }


def test_reachable_and_unreachable():
    assert live.reachable_platforms(infra(["cpu"]), TYPES, "client_node0", "a", True, None)[0] == [("node0", 0)]
    out, why = live.reachable_platforms(infra(["cpu"], net=()), TYPES, "client_node0", "a", True, None)
    assert (out, why) == ([], "no-server-reaches-client")
    assert live.reachable_platforms(infra(["gpu"]), TYPES, "client_node0", "a", True, None)[1] == "no-compatible-platform"
    assert live.reachable_platforms(infra(["cpu"], mem=0.5), TYPES, "client_node0", "a", True, None)[1] == "memory"
    assert live.reachable_platforms(infra(["cpu"], fabric={"routes": {}}), TYPES, "client_node0", "a", True, None)[1] == "no-route"
    assert live.reachable_platforms(infra(["cpu"], fabric={"routes": {"client_node0": {"node0": []}}}),
                                    TYPES, "client_node0", "a", True, None)[0] == [("node0", 0)]


def test_client_hosts_only_when_not_server_only():
    assert live.reachable_platforms(infra([], net=()), TYPES, "client_node0", "a", True, None)[0] == []
    assert live.reachable_platforms(infra([], net=()), TYPES, "client_node0", "a", False, None)[0] == [("client_node0", 0)]


def test_allowed_platform_types_filter():
    assert live.reachable_platforms(infra(["cpu"]), TYPES, "client_node0", "a", True, {"gpu"})[1] == "no-compatible-platform"


def test_distinct_platforms_are_needed():
    one = infra(["cpu"])
    used = {"client_node0": {"a", "b"}}
    res = live.check_window(one, TYPES, used, True, None)
    assert not res["ok"] and res["bad_clients"]["client_node0"] == {"*": "no-distinct-platforms"}
    assert live.check_window(infra(["cpu", "cpu"]), TYPES, used, True, None)["ok"]
    assert live.check_window(infra(["cpu", "cpu"], mem=1.5), TYPES, used, True, None)["ok"] is False  # memory is shared
    assert live.check_window(infra(["cpu", "cpu"]), TYPES, {"client_node0": {"a"}}, True, None)["min_reachable_platforms"] == 2
