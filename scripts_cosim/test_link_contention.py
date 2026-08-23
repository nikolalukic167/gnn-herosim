#!/usr/bin/env python3
"""Physics tests for link_contention_v1 — per-link capacity over a multi-hop backbone.

The 2026-08-17 separability measurement showed the co-sim target is additive to 98.8-100%,
and four attempts to break that all collapsed under the same one-integer control: a single
*node-occupancy excess* column ("how many tasks landed on host X") repaired most of the
pointwise fit's regret.

The node ingress pipe (`network_contention_v1`) is indexed by the destination node, so its
cost is a function of destination occupancy *by construction* — one integer was always going
to repair it. A link is crossed by paths to many different destinations, so two tasks that
share no node at all can still queue behind each other.

That single distinction is the entire reason this mechanism might survive where the others
did not, so it gets its own test: `test_shared_core_link_serializes_different_destinations`.
If that property ever breaks, the lineage is pointless.

The other guards:

1. **Default-off is bit-identical.** No `link_topology` ⇒ no fabric, no pipes, no time.
2. **Failures are loud.** A missing route or a non-positive bandwidth raises; it must never
   silently charge 0.0, which is how `network_latency_between` under-charges today.
3. **The hold is long enough to overlap.** `node_contention_v3` failed because its resource
   was held for ~0.024 s and tasks never overlapped, leaving `nodeContentionTime` exactly
   0.0 everywhere.
4. **The ECT mirror agrees with the simulation**, so Knative/ECT baselines stay fair.

Run: pipenv run python3 -m pytest scripts_cosim/test_link_contention.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import simpy
from simpy.resources.store import FilterStore

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.placement.infrastructure import Node  # noqa: E402
from src.placement.network_fabric import (  # noqa: E402
    NetworkFabric,
    build_fabric,
    is_core_link,
    link_key,
    route_links,
)
from src.placement.scheduling_cost import (  # noqa: E402
    link_transfer_cost,
    link_wait,
    transfer_time,
)

INPUT_BYTES = 153_600
BANDWIDTH_MBPS = 1.5
MIB = 1024.0 * 1024.0
EXPECTED_TRANSFER = INPUT_BYTES / (BANDWIDTH_MBPS * MIB)  # ~0.0977 s
EXEC_TIME = 0.0239175

APP_NAME = "nofs-dnn2"
TASK_TYPE = {
    "name": "dnn2",
    "executionTime": {"xavierCpu": EXEC_TIME},
    "stateSize": {APP_NAME: {"input": INPUT_BYTES, "output": 8000}},
}


class _FakeApplication:
    def __init__(self, name: str = APP_NAME) -> None:
        self.type = {"name": name}


class _FakeTask:
    def __init__(self, application=None, node_name: str = "client_node0") -> None:
        self.type = TASK_TYPE
        self.application = _FakeApplication() if application is None else application
        self.node_name = node_name


# A deliberately small backbone with a genuinely shared middle segment:
#
#   client_node0 -- core0 --\
#                            core1 -- core2 -- node0
#   client_node1 ------------/          \----- node1
#   client_node2 ------------/
#
# core0|core1 carries client_node0's traffic only; core1|core2 is crossed by every
# client-to-server path. client_node1 and client_node2 are equidistant from that shared
# segment, so their transfers arrive together and genuinely collide — which is what the
# discriminating test needs. client_node0 sits one hop further out, giving a staggered
# case as well.
def _topology(bandwidth: float = BANDWIDTH_MBPS) -> dict:
    links = {
        link_key("client_node0", "core0"): {"latency": 0.02, "bandwidth_mbps": bandwidth},
        link_key("client_node1", "core1"): {"latency": 0.02, "bandwidth_mbps": bandwidth},
        link_key("client_node2", "core1"): {"latency": 0.02, "bandwidth_mbps": bandwidth},
        link_key("core0", "core1"): {"latency": 0.004, "bandwidth_mbps": bandwidth},
        link_key("core1", "core2"): {"latency": 0.004, "bandwidth_mbps": bandwidth},
        link_key("core2", "node0"): {"latency": 0.02, "bandwidth_mbps": bandwidth},
        link_key("core2", "node1"): {"latency": 0.02, "bandwidth_mbps": bandwidth},
    }
    routes = {
        "client_node0": {
            "node0": ["client_node0", "core0", "core1", "core2", "node0"],
            "node1": ["client_node0", "core0", "core1", "core2", "node1"],
        },
        "client_node1": {
            "node0": ["client_node1", "core1", "core2", "node0"],
            "node1": ["client_node1", "core1", "core2", "node1"],
        },
        "client_node2": {
            "node0": ["client_node2", "core1", "core2", "node0"],
            "node1": ["client_node2", "core1", "core2", "node1"],
        },
    }
    return {"links": links, "routes": routes}


def _node(env, fabric, node_name: str = "node0"):
    return Node(
        env=env,
        node_id=0,
        memory=8.0,
        platforms=FilterStore(env),
        storage=FilterStore(env),
        network_map={},
        network={"bandwidth": 100.0},
        policy=None,
        data=None,
        node_type="xavier",
        node_name=node_name,
        fabric=fabric,
    )


# --- 1. default-off is node_disk_v2 -----------------------------------------------


def test_no_fabric_is_node_disk_v2():
    env = simpy.Environment()
    assert build_fabric(env, None) is None
    node = _node(env, None)
    assert node.fabric is None
    assert link_transfer_cost(None, _FakeTask(), "client_node0", node) == 0.0
    assert link_wait(None, _FakeTask(), "client_node0", node, added_on_links={"a|b": 3}) == 0.0


def test_generator_emits_no_backbone_without_config():
    """The generator contract: no `network.backbone` block ⇒ no routes, and the one-hop
    latencies every pre-existing corpus was built with are left untouched."""
    import random

    from src.generate_infrastructure import build_core_backbone

    nodes = [{"node_name": f"client_node{i}", "type": "rpi"} for i in range(2)] + [
        {"node_name": f"node{i}", "type": "rpi"} for i in range(2)
    ]
    network_maps = {
        "client_node0": {"node0": 0.1},
        "node0": {"client_node0": 0.1},
        "client_node1": {"node1": 0.2},
        "node1": {"client_node1": 0.2},
    }
    before = {name: dict(edges) for name, edges in network_maps.items()}

    assert build_core_backbone(network_maps, nodes, {}, random.Random(0)) is None
    assert build_core_backbone(network_maps, nodes, {"network": {}}, random.Random(0)) is None
    assert network_maps == before


def test_generator_rejects_a_backbone_without_bandwidth():
    """A backbone with no capacity is a config mistake — it would silently produce routes
    the simulator then cannot charge."""
    import random

    from src.generate_infrastructure import build_core_backbone

    nodes = [{"node_name": "client_node0", "type": "rpi"}, {"node_name": "node0", "type": "rpi"}]
    network_maps = {"client_node0": {"node0": 0.1}, "node0": {"client_node0": 0.1}}
    with pytest.raises(ValueError, match="bandwidth_mbps must be > 0"):
        build_core_backbone(
            network_maps, nodes, {"network": {"backbone": {"n_core": 4}}}, random.Random(0)
        )


def test_independent_rng_stream_is_immune_to_prior_consumption():
    """rng_stream=independent_v1: backbone links must be identical no matter how many
    draws the shared stream consumed beforehand (the corpus-side reachability repair vs
    the live path's clean stream — the parity divergence recorded in LINEAGES.md,
    link_contention_v1 2026-08-21). legacy_v0 must stay position-dependent so every
    pre-2026-08-22 corpus regenerates byte-identically from its config."""
    import random

    from src.generate_infrastructure import build_core_backbone

    nodes = [{"node_name": f"client_node{i}", "type": "rpi"} for i in range(3)] + [
        {"node_name": f"node{i}", "type": "rpi"} for i in range(3)
    ]

    def fresh_maps():
        return {
            "client_node0": {"node0": 0.1},
            "node0": {"client_node0": 0.1},
            "client_node1": {"node1": 0.2},
            "node1": {"client_node1": 0.2},
            "client_node2": {"node2": 0.3},
            "node2": {"client_node2": 0.3},
        }

    def build(stream, consume_first):
        rng = random.Random(7)
        if consume_first:
            rng.shuffle(list(range(50)))  # stand-in for the reachability repair's draws
            rng.random()
        config = {"network": {"backbone": {
            "n_core": 4, "attach_degree": 1, "bandwidth_mbps": 1.5, "rng_stream": stream,
        }}}
        return build_core_backbone(fresh_maps(), nodes, config, rng, seed=7)

    clean = build("independent_v1", consume_first=False)
    offset = build("independent_v1", consume_first=True)
    assert clean["links"] == offset["links"]
    assert clean["routes"] == offset["routes"]
    assert clean["params"]["rng_stream"] == "independent_v1"

    legacy_clean = build("legacy_v0", consume_first=False)
    legacy_offset = build("legacy_v0", consume_first=True)
    assert legacy_clean["links"] != legacy_offset["links"], (
        "legacy_v0 stopped depending on stream position — old corpora would no longer "
        "regenerate from their configs"
    )
    assert legacy_clean["params"]["rng_stream"] == "legacy_v0"


def test_independent_rng_stream_requires_a_seed():
    import random

    from src.generate_infrastructure import build_core_backbone

    nodes = [{"node_name": "client_node0", "type": "rpi"}, {"node_name": "node0", "type": "rpi"}]
    network_maps = {"client_node0": {"node0": 0.1}, "node0": {"client_node0": 0.1}}
    config = {"network": {"backbone": {
        "n_core": 2, "attach_degree": 1, "bandwidth_mbps": 1.5, "rng_stream": "independent_v1",
    }}}
    with pytest.raises(ValueError, match="requires the topology seed"):
        build_core_backbone(network_maps, nodes, config, random.Random(0))


def test_unknown_rng_stream_fails_loudly():
    import random

    from src.generate_infrastructure import build_core_backbone

    nodes = [{"node_name": "client_node0", "type": "rpi"}, {"node_name": "node0", "type": "rpi"}]
    network_maps = {"client_node0": {"node0": 0.1}, "node0": {"client_node0": 0.1}}
    config = {"network": {"backbone": {
        "n_core": 2, "attach_degree": 1, "bandwidth_mbps": 1.5, "rng_stream": "typo_v9",
    }}}
    with pytest.raises(ValueError, match="rng_stream must be"):
        build_core_backbone(network_maps, nodes, config, random.Random(0), seed=0)


def test_empty_fabric_fails_loudly():
    """An empty links dict is a config mistake, not a quiet no-op."""
    env = simpy.Environment()
    with pytest.raises(ValueError, match="link_topology.links is empty"):
        NetworkFabric(env, {"links": {}, "routes": {}})


def test_invalid_bandwidth_fails_loudly():
    env = simpy.Environment()
    topology = _topology()
    topology["links"][link_key("core0", "core1")]["bandwidth_mbps"] = 0.0
    with pytest.raises(ValueError, match="bandwidth_mbps must be > 0"):
        NetworkFabric(env, topology)


def test_missing_route_fails_loudly():
    """Silently charging 0.0 is how an additive term gets under-counted unnoticed."""
    env = simpy.Environment()
    fabric = build_fabric(env, _topology())
    with pytest.raises(KeyError, match="no route between"):
        fabric.hops("client_node9", "node0")


# --- 2. routing and the transfer term ----------------------------------------------


def test_route_links_are_multi_hop():
    routes = _topology()["routes"]
    hops = route_links(routes, "client_node0", "node0")
    assert len(hops) == 4
    assert link_key("core1", "core2") in hops


def test_route_lookup_is_symmetric():
    """Routes are stored client-side; the reverse direction must resolve to the same links."""
    routes = _topology()["routes"]
    assert route_links(routes, "node0", "client_node0") == list(
        reversed(route_links(routes, "client_node0", "node0"))
    )


def test_core_links_are_distinguished_from_access_links():
    assert is_core_link(link_key("core1", "core2"))
    assert not is_core_link(link_key("client_node0", "core0"))
    assert not is_core_link(link_key("core2", "node0"))


def test_transfer_time_matches_size_over_bandwidth():
    got = transfer_time(_FakeTask(), BANDWIDTH_MBPS)
    assert got == pytest.approx(EXPECTED_TRANSFER)
    assert got == pytest.approx(0.09766, abs=1e-4)


def test_transfer_scales_inversely_with_bandwidth():
    slow = transfer_time(_FakeTask(), 0.5)
    mid = transfer_time(_FakeTask(), 1.5)
    fast = transfer_time(_FakeTask(), 5.0)
    assert slow > mid > fast
    assert slow == pytest.approx(3 * mid)


def test_transfer_is_long_enough_to_overlap():
    """The guard against node_contention_v3's `nodeContentionTime == 0.0` everywhere.

    A hold of ~0.024 s never overlaps a batch-mate's. One hop must dominate execution.
    """
    hold = transfer_time(_FakeTask(), BANDWIDTH_MBPS)
    assert hold > 4 * EXEC_TIME


def test_route_cost_grows_with_hop_count():
    """Store-and-forward: each hop pays a full transmission, so a longer path costs more."""
    env = simpy.Environment()
    fabric = build_fabric(env, _topology())
    far = link_transfer_cost(
        fabric, _FakeTask(node_name="client_node0"), "client_node0", _node(env, fabric)
    )
    near = link_transfer_cost(
        fabric, _FakeTask(node_name="client_node1"), "client_node1", _node(env, fabric)
    )
    assert far == pytest.approx(4 * EXPECTED_TRANSFER)
    assert near == pytest.approx(3 * EXPECTED_TRANSFER)
    assert far > near


def test_local_execution_pays_nothing():
    env = simpy.Environment()
    fabric = build_fabric(env, _topology())
    node = _node(env, fabric, node_name="client_node0")
    assert link_transfer_cost(fabric, _FakeTask(), "client_node0", node) == 0.0


def test_task_without_matching_state_size_pays_nothing():
    task = _FakeTask(application=_FakeApplication("some-other-app"))
    assert transfer_time(task, BANDWIDTH_MBPS) == 0.0


# --- 3. the discriminating property ------------------------------------------------


def _drain(env, fabric, src: str, dst: str, finished: list, label: str, start_at: float = 0.0):
    """Walk a route store-and-forward exactly as Platform.platform_process does."""

    def process():
        if start_at:
            yield env.timeout(start_at)
        for key, bandwidth in fabric.hops(src, dst):
            hold = transfer_time(_FakeTask(node_name=src), bandwidth)
            wait_start = env.now
            with fabric.pipe(key).request() as hop:
                yield hop
                fabric.link_wait_total += env.now - wait_start
                yield env.timeout(hold)
        finished.append((label, env.now))

    return process()


def test_shared_core_link_serializes_different_destinations():
    """THE test. Two tasks from different clients to DIFFERENT servers still contend.

    Their destination nodes are disjoint, so node-occupancy excess — the repair column that
    killed the four previous mechanisms — is identically zero for this pair. Yet they share
    core1|core2 and must serialize. If this ever passes trivially (equal finish times), the
    mechanism has degenerated into co-location and the lineage is over.
    """
    env = simpy.Environment()
    fabric = build_fabric(env, _topology())
    finished: list = []
    # Equidistant from the shared segment and bound for DIFFERENT servers, so the only
    # thing they have in common is core1|core2.
    env.process(_drain(env, fabric, "client_node1", "node0", finished, "to_node0"))
    env.process(_drain(env, fabric, "client_node2", "node1", finished, "to_node1"))
    env.run()

    times = sorted(t for _label, t in finished)
    assert len(times) == 2
    # Solo, each route is 3 hops = 3 transfers. One of them must queue a full transfer on
    # the shared middle segment, so the pair cannot both finish at the solo time.
    assert times[0] == pytest.approx(3 * EXPECTED_TRANSFER)
    assert times[1] == pytest.approx(4 * EXPECTED_TRANSFER)


def test_disjoint_routes_do_not_serialize():
    """The anti-case: no shared link means no coupling, so the effect is not global."""
    env = simpy.Environment()
    topology = {
        "links": {
            link_key("client_node0", "coreA"): {"latency": 0.02, "bandwidth_mbps": BANDWIDTH_MBPS},
            link_key("coreA", "node0"): {"latency": 0.02, "bandwidth_mbps": BANDWIDTH_MBPS},
            link_key("client_node1", "coreB"): {"latency": 0.02, "bandwidth_mbps": BANDWIDTH_MBPS},
            link_key("coreB", "node1"): {"latency": 0.02, "bandwidth_mbps": BANDWIDTH_MBPS},
        },
        "routes": {
            "client_node0": {"node0": ["client_node0", "coreA", "node0"]},
            "client_node1": {"node1": ["client_node1", "coreB", "node1"]},
        },
    }
    fabric = build_fabric(env, topology)
    finished: list = []
    env.process(_drain(env, fabric, "client_node0", "node0", finished, "a"))
    env.process(_drain(env, fabric, "client_node1", "node1", finished, "b"))
    env.run()

    times = [t for _label, t in finished]
    assert times[0] == pytest.approx(times[1])
    assert times[0] == pytest.approx(2 * EXPECTED_TRANSFER)


def test_staggered_transfers_still_overlap():
    """Real batches do not arrive simultaneously; the hold must outlast the spread."""
    env = simpy.Environment()
    fabric = build_fabric(env, _topology())
    finished: list = []
    stagger = EXPECTED_TRANSFER / 3
    env.process(_drain(env, fabric, "client_node0", "node0", finished, "first"))
    env.process(_drain(env, fabric, "client_node1", "node1", finished, "second", start_at=stagger))
    env.run()

    times = sorted(t for _label, t in finished)
    assert times[1] > times[0]


def test_fabric_accumulates_total_link_wait():
    """Telemetry mirrors Node.ingress_wait_total, so a sweep can see whether the physics
    fired at all — node_contention_v3 shipped looking correct while its counter was 0.0
    on every task in every dataset."""
    env = simpy.Environment()
    fabric = build_fabric(env, _topology())
    finished: list = []
    env.process(_drain(env, fabric, "client_node1", "node0", finished, "a"))
    env.process(_drain(env, fabric, "client_node2", "node1", finished, "b"))
    env.run()
    assert fabric.link_wait_total == pytest.approx(EXPECTED_TRANSFER)


def test_no_contention_leaves_the_counter_at_zero():
    env = simpy.Environment()
    fabric = build_fabric(env, _topology())
    finished: list = []
    env.process(_drain(env, fabric, "client_node1", "node0", finished, "solo"))
    env.run()
    assert fabric.link_wait_total == pytest.approx(0.0)


# --- 4. the ECT mirror --------------------------------------------------------------


def test_ect_mirror_charges_each_crossed_link():
    """The scheduler's anticipated wait must name the same links the simulator will."""
    env = simpy.Environment()
    fabric = build_fabric(env, _topology())
    node = _node(env, fabric, node_name="node0")
    task = _FakeTask(node_name="client_node0")

    # One batch-mate already committed to cross the shared middle segment.
    added = {link_key("core1", "core2"): 1}
    assert link_wait(fabric, task, "client_node0", node, added_on_links=added) == pytest.approx(
        EXPECTED_TRANSFER
    )

    # A link the route does not cross must cost nothing.
    unrelated = {link_key("core2", "node1"): 4}
    assert link_wait(fabric, task, "client_node0", node, added_on_links=unrelated) == 0.0


def test_ect_mirror_scales_with_crossing_count():
    env = simpy.Environment()
    fabric = build_fabric(env, _topology())
    node = _node(env, fabric, node_name="node0")
    task = _FakeTask(node_name="client_node0")
    key = link_key("core1", "core2")
    one = link_wait(fabric, task, "client_node0", node, added_on_links={key: 1})
    three = link_wait(fabric, task, "client_node0", node, added_on_links={key: 3})
    assert three == pytest.approx(3 * one)


def test_ect_mirror_matches_simulated_wait():
    """Two tasks to different destinations sharing core1|core2: the ECT term must equal
    the wait the simulator actually imposes on the second one."""
    env = simpy.Environment()
    fabric = build_fabric(env, _topology())
    node = _node(env, fabric, node_name="node1")
    task = _FakeTask(node_name="client_node1")

    waits: list = []

    def process(src, dst, label):
        for key, bandwidth in fabric.hops(src, dst):
            hold = transfer_time(_FakeTask(node_name=src), bandwidth)
            start = env.now
            with fabric.pipe(key).request() as hop:
                yield hop
                if label == "second":
                    waits.append(env.now - start)
                yield env.timeout(hold)

    env.process(process("client_node2", "node0", "first"))
    env.process(process("client_node1", "node1", "second"))
    env.run()

    observed = sum(waits)
    predicted = link_wait(
        fabric, task, "client_node1", node,
        added_on_links={link_key("core1", "core2"): 1},
    )
    # The second task queues behind the first on the one segment they share, and the ECT
    # model must predict that wait from the batch-mate's link crossing alone.
    assert observed > 0
    assert observed == pytest.approx(predicted)
