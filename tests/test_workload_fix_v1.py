"""workload_fix_v1: W2 payload sampler, W3 access-link classes, W4 reachability repair.

Every flag is off by default, and the default path must replay byte for byte: the golden hashes below were taken
from origin/reference-physics before any of this code existed.
"""
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import pytest
import simpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_cosim"))
import grounded_workload_v1_mint as M  # noqa: E402
import workload_fix_v1_reachability_check as RC  # noqa: E402

from src.generate_infrastructure import (  # noqa: E402
    ReachabilityRepairError,
    generate_deterministic_infrastructure,
    resolve_reachability_repair_types,
)
from src.placement import workload_payloads as WP  # noqa: E402
from src.placement.dag_workload import route_hops_and_bottleneck  # noqa: E402
from src.placement.network_fabric import (  # noqa: E402
    DEFAULT_ACCESS_MIX,
    NetworkFabric,
    directed_bandwidth,
    draw_access_classes,
)

CFG = ROOT / "tests" / "fixtures" / "workload_fix_v1" / "cc40s9473.json"
SIM_INPUT = ROOT / "data" / "nofs-ids"
SEED = 9473

GOLDEN_MINT = {1: "af1407e12aa0436a", 3: "1ba9a42aaafb0ddc"}
GOLDEN_INFRA = "82cddbe372de5b7799d5ee4e6e7ed2f3337ce3cd368a7bcc3897abff1ee259c5"


def _lib_base():
    lib = {"groups": [{"t0_ms": 1000 * i, "offsets_ms": [0, 1, 5, 9][: 1 + i % 4]} for i in range(80)]}
    base = {"rps": 1, "duration": 10, "events": [
        {"timestamp": 0.5 * i, "application": {"name": "nofs-dnn1", "dag": {"dnn1": []}}, "qos": {"name": "m"},
         "node_name": "c0"} for i in range(120)]}
    return lib, base


def _gen(cfg: dict, tmp_path: Path, name: str = "x"):
    space = tmp_path / f"{name}.cfg.json"
    space.write_text(json.dumps(cfg))
    out = tmp_path / f"{name}.infra.json"
    with contextlib.redirect_stdout(io.StringIO()):
        infra = generate_deterministic_infrastructure(str(space), SIM_INPUT, str(out), SEED)
    return json.loads(out.read_text()) if infra is not None else None


def _norm_hash(infra: dict) -> str:
    d = copy.deepcopy(infra)
    d["metadata"].pop("generation_time")
    d["metadata"].pop("config_file")
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def _cfg() -> dict:
    return json.loads(CFG.read_text())


# ----------------------------------------------------------------------------- default path replays byte for byte

@pytest.mark.parametrize("merge_k", [1, 3])
def test_default_mint_replays_pre_change_bytes(merge_k):
    lib, base = _lib_base()
    doc, meta = M.mint(lib, base, seed=1, n_tasks=100, merge_k=merge_k)
    assert hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16] == GOLDEN_MINT[merge_k]
    assert "payload_sampler" not in meta
    explicit, _ = M.mint(lib, base, seed=1, n_tasks=100, merge_k=merge_k, payload_sampler="legacy")
    assert explicit == doc


def test_default_infrastructure_replays_pre_change_bytes(tmp_path):
    infra = _gen(_cfg(), tmp_path)
    assert _norm_hash(infra) == GOLDEN_INFRA
    assert "access_classes" not in infra["link_topology"]
    assert not any("access_node" in a for a in infra["link_topology"]["links"].values())


# ----------------------------------------------------------------------------- W2

def test_sampler_is_seeded_and_deterministic():
    a = [WP.sample_payload_bytes(WP.payload_rng(7)) for _ in range(1)]
    b = [WP.sample_payload_bytes(WP.payload_rng(7)) for _ in range(1)]
    c = [WP.sample_payload_bytes(WP.payload_rng(8)) for _ in range(1)]
    assert a == b and a != c
    r1, r2 = WP.payload_rng(3), WP.payload_rng(3)
    assert [WP.sample_payload_bytes(r1) for _ in range(50)] == [WP.sample_payload_bytes(r2) for _ in range(50)]


def test_sampler_matches_the_registered_distribution():
    rng = WP.payload_rng(0)
    xs = [WP.sample_payload_bytes(rng) for _ in range(400_000)]
    under_10mb = sum(x < 10e6 for x in xs) / len(xs)
    assert under_10mb == pytest.approx(0.9 * 0.7774, abs=0.006)  # 0.9 * Phi(ln 2.5 / 1.2) ~ 0.70
    body_median = statistics.median(x for x in xs if x < 50e6)
    assert body_median == pytest.approx(4e6, rel=0.05)
    assert statistics.fmean(xs) / 1e6 == pytest.approx(26.9, rel=0.05)  # the node's "about 27 MB"
    tiers = WP.payload_rng(0)  # same stream: draw 1 of each pair is the tier
    is_heavy = []
    for _ in xs:
        is_heavy.append(tiers.random() < WP.HEAVY_SHARE)
        tiers.random()
    assert sum(is_heavy) / len(xs) == pytest.approx(0.10, abs=0.002)
    heavy_bytes = sum(x for x, h in zip(xs, is_heavy) if h) / sum(xs)
    assert heavy_bytes == pytest.approx(0.72, abs=0.03)  # the node's "about 72 % of the new bytes"


class _Scripted:
    def __init__(self, values):
        self.values = list(values)

    def random(self):
        return self.values.pop(0)


def test_heavy_tier_is_log_uniform_on_50_to_500_mb():
    lo = WP.sample_payload_bytes(_Scripted([0.05, 0.0]))
    hi = WP.sample_payload_bytes(_Scripted([0.05, 1.0]))
    mid = WP.sample_payload_bytes(_Scripted([0.05, 0.5]))
    assert lo == pytest.approx(50e6) and hi == pytest.approx(500e6)
    assert mid == pytest.approx(math.sqrt(50e6 * 500e6))
    assert WP.sample_payload_bytes(_Scripted([0.10, 0.5])) == pytest.approx(4e6)  # tier edge: 0.10 is the body


def test_sampler_always_consumes_two_draws():
    rng = random.Random(1)
    WP.sample_payload_bytes(rng)
    ref = random.Random(1)
    ref.random(), ref.random()
    assert rng.random() == ref.random()


def test_unknown_sampler_fails_loudly():
    lib, base = _lib_base()
    with pytest.raises(ValueError, match="payload sampler"):
        M.mint(lib, base, seed=1, n_tasks=100, payload_sampler="wf2")


@pytest.mark.parametrize("merge_k", [1, 3])
def test_wf1_mint_changes_payloads_only(merge_k):
    lib, base = _lib_base()
    old, _ = M.mint(lib, base, seed=1, n_tasks=100, merge_k=merge_k)
    new, meta = M.mint(lib, base, seed=1, n_tasks=100, merge_k=merge_k, payload_sampler="wf1_v1")
    assert old["events"] == new["events"]
    assert [p[:2] for p in old["peer_exchange"]] == [p[:2] for p in new["peer_exchange"]]
    assert [p[2] for p in old["peer_exchange"]] != [p[2] for p in new["peer_exchange"]]
    assert meta["payload_sampler"]["median_bytes"] == 4e6
    again, _ = M.mint(lib, base, seed=1, n_tasks=100, merge_k=merge_k, payload_sampler="wf1_v1")
    assert again == new


@pytest.mark.parametrize("merge_k", [1, 3])
def test_minting_wf1_equals_resampling_the_legacy_file(merge_k):
    lib, base = _lib_base()
    old, _ = M.mint(lib, base, seed=5, n_tasks=100, merge_k=merge_k)
    new, _ = M.mint(lib, base, seed=5, n_tasks=100, merge_k=merge_k, payload_sampler="wf1_v1")
    assert new["peer_exchange"] == WP.resample_peer_exchange(old["peer_exchange"], seed=5)


def test_resample_peer_exchange_keeps_structure():
    pairs = [[0, 1, 5.0], [1, 2, 6.0], [0, 2, 7.0]]
    out = WP.resample_peer_exchange(pairs, seed=4)
    assert [p[:2] for p in out] == [p[:2] for p in pairs]
    assert out == WP.resample_peer_exchange(pairs, seed=4)
    assert all(p[2] > 0 for p in out)


# ----------------------------------------------------------------------------- W3

def test_access_class_draw_is_deterministic_and_matches_the_mix():
    names = [f"n{i}" for i in range(6000)]
    a = draw_access_classes(names, seed=5)
    assert a == draw_access_classes(names, seed=5)
    assert a != draw_access_classes(names, seed=6)
    share = {c: sum(v["class"] == c for v in a.values()) / len(a) for c in DEFAULT_ACCESS_MIX}
    for cls, want in DEFAULT_ACCESS_MIX.items():
        assert share[cls] == pytest.approx(want, abs=0.02)
    for v in a.values():
        if v["class"] == "wired":
            assert (v["out_mbps"], v["in_mbps"]) == (117.0, 117.0)
        elif v["class"] == "cellular":
            assert (v["out_mbps"], v["in_mbps"]) == (4.0, 75.0)
        else:
            assert 7.0 <= v["out_mbps"] == v["in_mbps"] <= 14.0


def test_access_class_mix_validation():
    with pytest.raises(ValueError, match="sum to 1"):
        draw_access_classes(["a"], 1, {"wired": 0.5, "wifi": 0.4})
    with pytest.raises(ValueError, match="unknown class"):
        draw_access_classes(["a"], 1, {"wired": 0.5, "satellite": 0.5})
    only = draw_access_classes([f"n{i}" for i in range(20)], 1, {"cellular": 1.0})
    assert {v["class"] for v in only.values()} == {"cellular"}


def _with_classes(extra=None):
    cfg = _cfg()
    cfg["network"]["backbone"]["access_classes"] = extra if extra is not None else {}
    return cfg


@pytest.fixture(scope="module")
def default_infra(tmp_path_factory):
    return _gen(_cfg(), tmp_path_factory.mktemp("default"))


@pytest.fixture(scope="module")
def classed_infra(tmp_path_factory):
    return _gen(_with_classes(), tmp_path_factory.mktemp("classed"))


def test_classes_leave_everything_but_access_bandwidth_alone(default_infra, classed_infra):
    for key in ("network_maps", "replica_placements", "queue_distributions"):
        assert default_infra[key] == classed_infra[key]
    d, c = default_infra["link_topology"], classed_infra["link_topology"]
    assert d["routes"] == c["routes"]
    assert set(d["links"]) == set(c["links"])
    for key, attrs in d["links"].items():
        assert c["links"][key]["latency"] == attrs["latency"]
        left, _, right = key.partition("|")
        if left.startswith("core") and right.startswith("core"):
            assert c["links"][key] == attrs, "core links must be unchanged"


def test_access_links_carry_their_nodes_class(classed_infra):
    lt = classed_infra["link_topology"]
    classes = lt["access_classes"]
    assert set(classes) == set(classed_infra["network_maps"])
    seen = set()
    for key, attrs in lt["links"].items():
        left, _, right = key.partition("|")
        if left.startswith("core") and right.startswith("core"):
            continue
        node = right if left.startswith("core") else left
        cls = classes[node]
        seen.add(cls["class"])
        if cls["class"] == "cellular":
            assert attrs["access_node"] == node
            assert (attrs["bandwidth_out_mbps"], attrs["bandwidth_in_mbps"]) == (4.0, 75.0)
            assert attrs["bandwidth_mbps"] == 4.0
        else:
            assert "access_node" not in attrs
            assert attrs["bandwidth_mbps"] == cls["out_mbps"]
    assert seen == {"wired", "wifi", "cellular"}  # 46 nodes at 40/40/20


def test_classes_apply_to_servers_as_well_as_clients(classed_infra):
    classes = classed_infra["link_topology"]["access_classes"]
    server_classes = {v["class"] for n, v in classes.items() if not n.startswith("client_node")}
    assert len(server_classes) >= 2


def test_class_draw_does_not_depend_on_the_global_stream(tmp_path):
    random.seed(1)
    a = _gen(_with_classes(), tmp_path, "a")["link_topology"]["access_classes"]
    random.seed(2)
    b = _gen(_with_classes(), tmp_path, "b")["link_topology"]["access_classes"]
    assert a == b


def test_cellular_only_mix_is_directional_end_to_end(tmp_path):
    infra = _gen(_with_classes({"mix": {"cellular": 1.0}}), tmp_path)
    lt = infra["link_topology"]
    fabric = NetworkFabric(simpy.Environment(), lt)
    client = "client_node0"
    server = next(s for s in infra["network_maps"][client] if not s.startswith("client_node"))
    up = fabric.hops(client, server)
    down = fabric.hops(server, client)
    assert up[0][1] == 4.0 and up[-1][1] == 75.0
    assert down[0][1] == 4.0 and down[-1][1] == 75.0
    assert [k for k, _ in up] == [k for k, _ in reversed(down)]
    assert route_hops_and_bottleneck(lt["routes"], lt["links"], client, server) == (len(up), 4.0)
    # a server-to-server exchange sends at the sender's out rate and lands at the receiver's in rate
    other = next(s for s in infra["network_maps"][server] if not s.startswith("client_node"))
    ex = fabric.hops(server, other)
    assert ex[0][1] == 4.0 and ex[-1][1] == 75.0


def test_uniform_topology_hops_are_unchanged(default_infra):
    lt = default_infra["link_topology"]
    fabric = NetworkFabric(simpy.Environment(), lt)
    client = "client_node0"
    server = next(s for s in default_infra["network_maps"][client] if not s.startswith("client_node"))
    for key, bw in fabric.hops(client, server) + fabric.hops(server, client):
        assert bw == fabric.bandwidth(key) == lt["links"][key]["bandwidth_mbps"]


def test_directed_bandwidth_unit():
    link = {"bandwidth_mbps": 4.0, "access_node": "n", "bandwidth_out_mbps": 4.0, "bandwidth_in_mbps": 75.0}
    assert directed_bandwidth(link, "n", "core0") == 4.0
    assert directed_bandwidth(link, "core0", "n") == 75.0
    assert directed_bandwidth({"bandwidth_mbps": 9.0}, "a", "b") == 9.0
    with pytest.raises(KeyError):
        directed_bandwidth(link, "x", "y")


def test_directional_link_without_both_rates_fails_loudly():
    lt = {"links": {"core0|n": {"latency": 0.01, "bandwidth_mbps": 4.0, "access_node": "n", "bandwidth_out_mbps": 4.0}},
          "routes": {"n": {"core0": ["n", "core0"]}}}
    with pytest.raises(ValueError, match="bandwidth_in_mbps"):
        NetworkFabric(simpy.Environment(), lt)


def test_classes_require_a_dict_and_known_keys(tmp_path):
    cfg = _cfg()
    cfg["network"]["backbone"]["access_classes"] = {"mixx": {}}
    with pytest.raises(ValueError, match="unknown key"):
        _gen(cfg, tmp_path)


# ----------------------------------------------------------------------------- W4

def test_repair_block_validation():
    types = {"dnn1": {}, "dnn2": {}, "rf": {}, "cnn": {}}
    assert resolve_reachability_repair_types({"network": {}}, types) is None
    assert resolve_reachability_repair_types({"network": {"reachability_repair": {"task_types": "all"}}}, types) == list(types)
    assert resolve_reachability_repair_types(
        {"network": {"reachability_repair": {"task_types": ["rf"]}}}, types) == ["rf"]
    for bad in ({"task_types": []}, {"task_types": "some"}, {"types": "all"}, "all"):
        with pytest.raises(ValueError):
            resolve_reachability_repair_types({"network": {"reachability_repair": bad}}, types)


def _starve(cfg: dict, task_type: str) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["replicas"][task_type] = {"per_client": 0, "per_server": 0}
    return cfg


def test_task_type_without_server_replica_is_skipped_silently_by_default(tmp_path):
    infra = _gen(_starve(_cfg(), "cnn"), tmp_path)
    assert not infra["replica_placements"]["cnn"]
    report = RC.check_infrastructure(infra, ["dnn1", "dnn2", "rf", "cnn"])
    assert not report["ok"] and report["no_server_replica"] == ["cnn"]


def test_repair_fails_loudly_for_a_used_type_with_no_server_replica(tmp_path):
    cfg = _starve(_cfg(), "cnn")
    cfg["network"]["reachability_repair"] = {"task_types": "all"}
    with pytest.raises(ReachabilityRepairError, match="cnn"):
        _gen(cfg, tmp_path)
    cfg["network"]["reachability_repair"] = {"task_types": ["dnn1", "dnn2"]}
    _gen(cfg, tmp_path, "scoped")  # cnn is not used here, so it is not required


def test_repair_fails_loudly_for_a_type_missing_from_the_replicas_config(tmp_path):
    cfg = _cfg()
    del cfg["replicas"]["rf"]
    cfg["network"]["reachability_repair"] = {"task_types": "all"}
    with pytest.raises(ReachabilityRepairError, match="rf"):
        _gen(cfg, tmp_path)


def test_repair_on_a_healthy_topology_changes_nothing(tmp_path, default_infra):
    cfg = _cfg()
    cfg["network"]["reachability_repair"] = {"task_types": "all"}
    repaired = _gen(cfg, tmp_path)
    assert _norm_hash(repaired) == _norm_hash(default_infra)


def test_check_finds_an_unreachable_client():
    infra = {
        "network_maps": {"client_node0": {"node0": 0.1}, "client_node1": {}, "node0": {"client_node0": 0.1}},
        "replica_placements": {"rf": [{"node_name": "node0", "platform_id": 0, "platform_type": "rpiCpu"}]},
        "link_topology": None,
    }
    report = RC.check_infrastructure(infra, ["rf"])
    assert not report["ok"] and report["unreachable_clients"] == {"rf": ["client_node1"]}


def test_check_requires_a_route_when_a_fabric_exists():
    infra = {
        "network_maps": {"client_node0": {"node0": 0.1}, "node0": {"client_node0": 0.1}},
        "replica_placements": {"rf": [{"node_name": "node0", "platform_id": 0, "platform_type": "rpiCpu"}]},
        "link_topology": {"links": {"x": {}}, "routes": {}},
    }
    assert RC.check_infrastructure(infra, ["rf"])["unreachable_clients"] == {"rf": ["client_node0"]}
    infra["link_topology"]["routes"] = {"client_node0": {"node0": ["client_node0", "core0", "node0"]}}
    assert RC.check_infrastructure(infra, ["rf"])["ok"]


def test_check_passes_on_the_fixture_topology_with_and_without_repair():
    for repair in (False, True):
        row = RC.run_topology(CFG, SIM_INPUT, repair, None)
        assert row["ok"], row


def test_check_reports_a_generator_refusal_as_a_failure(tmp_path):
    cfg = _starve(_cfg(), "cnn")
    p = tmp_path / "cc40s1.json"
    p.write_text(json.dumps(cfg))
    assert not RC.run_topology(p, SIM_INPUT, False, None)["ok"]
    row = RC.run_topology(p, SIM_INPUT, True, None)
    assert not row["ok"] and "ReachabilityRepairError" in row["generator_error"]


def test_platform_charges_the_direction_of_travel():
    from types import SimpleNamespace

    from src.placement.infrastructure import Platform

    lt = {
        "links": {
            "cell|core0": {"latency": 0.01, "bandwidth_mbps": 4.0, "access_node": "cell",
                           "bandwidth_out_mbps": 4.0, "bandwidth_in_mbps": 75.0},
            "core0|wire": {"latency": 0.01, "bandwidth_mbps": 117.0},
        },
        "routes": {"cell": {"wire": ["cell", "core0", "wire"]}},
    }
    fabric = NetworkFabric(simpy.Environment(), lt)
    mb = 1024.0 * 1024.0
    payload = 75.0 * mb

    def seconds(parent, child):
        fake = SimpleNamespace(node=SimpleNamespace(fabric=fabric, node_name=child))
        return Platform._payload_transfer_time(fake, parent, payload)

    # cell -> wire leaves the cellular node at 4 MB/s; wire -> cell enters it at 75 MB/s, so 117 does not bind
    assert seconds("cell", "wire") == pytest.approx(2 * payload / (4.0 * mb))
    assert seconds("wire", "cell") == pytest.approx(2 * payload / (75.0 * mb))
