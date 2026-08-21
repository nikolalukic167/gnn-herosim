"""link_contention_v1 — per-link capacity over a multi-hop core backbone.

Why this exists
---------------
Every coupling mechanism tried before this one collapsed under the one-integer control
in ``scripts_cosim/separability_diagnostic.py``: a single *node-occupancy excess* column
("how many tasks landed on host X") repaired most of the pointwise fit's regret, so the
"coupling" was a count a pointwise MLP learns from one extra input.

The node ingress pipe (``network_contention_v1``) is the clearest case — it is indexed by
the destination node, so its cost is a function of the destination-occupancy count by
construction. A per-*link* model is the first candidate whose contended object has more
identities than the destination-node count: two tasks routed to **different** destination
nodes can still queue behind each other on a shared core segment, and no node-occupancy
column can express that at any value.

That only holds if the routes actually overlap, which is why ``link_topology`` describes a
core tier rather than the bare client<->server adjacency. The 4 tasks in a co-sim batch
always originate at 4 *distinct* clients (``generate_workload_templates`` draws them from
``random.Random(42)``; all 10 templates give distinct sources), so an access link is used
by at most one task and is perfectly additive. Only the core segments are shared.

Model
-----
Propagation latency stays the un-serialized ``env.timeout`` it has always been — additive,
unchanged. The *transmission* of a task's input walks its route store-and-forward,
holding one ``simpy.Resource(capacity=1)`` per hop. The wait for those pipes is the
non-additive term.

Absent ``link_topology`` there is no fabric and nothing is charged, so every existing
corpus replays bit-identically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from simpy import Resource

LINK_SEP = "|"

# Core routers are the only shared segment; access links are per-client/per-server and
# stay additive. The prefix is how every consumer tells the two apart without needing the
# topology config -- the diagnostic and the pre-check both rely on it.
CORE_PREFIX = "core"


def link_key(a: str, b: str) -> str:
    """Canonical undirected key for the link between two nodes.

    Sorted so ``link_key(x, y) == link_key(y, x)`` -- the pipe must be the same object in
    both directions or traffic would not contend.
    """
    return f"{a}{LINK_SEP}{b}" if a <= b else f"{b}{LINK_SEP}{a}"


def is_core_link(key: str) -> bool:
    """True when both endpoints are core routers, i.e. the link is genuinely shared."""
    left, _, right = key.partition(LINK_SEP)
    return left.startswith(CORE_PREFIX) and right.startswith(CORE_PREFIX)


def route_links(
    routes: Mapping[str, Mapping[str, Sequence[str]]],
    src: str,
    dst: str,
) -> List[str]:
    """Link keys along the route from ``src`` to ``dst``.

    Pure, env-free, and shared with ``separability_diagnostic.py`` and the overlap
    pre-check so the offline analysis maps a plan onto exactly the links the simulator
    would charge. Routes are symmetric; the reverse direction is accepted so callers do
    not have to care which end stored it.
    """
    path = _lookup_path(routes, src, dst)
    return [link_key(path[i], path[i + 1]) for i in range(len(path) - 1)]


def _lookup_path(
    routes: Mapping[str, Mapping[str, Sequence[str]]],
    src: str,
    dst: str,
) -> Sequence[str]:
    forward = routes.get(src, {}).get(dst)
    if forward:
        return forward
    reverse = routes.get(dst, {}).get(src)
    if reverse:
        return list(reversed(reverse))
    raise KeyError(
        f"link_contention_v1: no route between {src!r} and {dst!r}. Every logical "
        f"network_map edge must have a route -- a silent 0.0 here is how the additive "
        f"term gets under-charged without anyone noticing."
    )


class NetworkFabric:
    """Per-link pipes plus the frozen route table, shared by every Node in a run.

    Built once in ``src/placement/simulation.py`` because it is the only place that owns
    cross-node state; ``Node`` objects are constructed per-node in a loop and none of them
    can own an edge.
    """

    def __init__(self, env: Any, link_topology: Mapping[str, Any]) -> None:
        links: Mapping[str, Mapping[str, Any]] = link_topology.get("links") or {}
        routes = link_topology.get("routes") or {}
        if not links:
            raise ValueError(
                "link_contention_v1: link_topology.links is empty. Pass None instead of "
                "an empty fabric so the default-off path stays bit-identical."
            )

        self._bandwidth: Dict[str, float] = {}
        for key, attrs in links.items():
            bandwidth = attrs.get("bandwidth_mbps")
            if bandwidth is None or float(bandwidth) <= 0:
                raise ValueError(
                    f"link_contention_v1: link {key!r} bandwidth_mbps must be > 0 when "
                    f"set, got {bandwidth}"
                )
            self._bandwidth[key] = float(bandwidth)

        self._pipes: Dict[str, Resource] = {
            key: Resource(env, capacity=1) for key in links
        }
        self._routes = routes
        # Kept verbatim so the feature builders can put the fabric *into the graph*
        # (`src/placement/network_graph.py`) without re-deriving latencies the simulator
        # already holds. Read-only: nothing here mutates it.
        self._link_topology = link_topology
        # Telemetry, mirroring Node.ingress_wait_total.
        self.link_wait_total: float = 0.0

    @property
    def link_topology(self) -> Mapping[str, Any]:
        """The `links` + `routes` block this fabric was built from, unmodified."""
        return self._link_topology

    def hops(self, src: str, dst: str) -> List[Tuple[str, float]]:
        """``[(link_key, bandwidth_mbps)]`` along the route, in traversal order."""
        hops: List[Tuple[str, float]] = []
        for key in route_links(self._routes, src, dst):
            if key not in self._bandwidth:
                raise KeyError(
                    f"link_contention_v1: route {src!r}->{dst!r} traverses link {key!r} "
                    f"which has no capacity entry in link_topology.links"
                )
            hops.append((key, self._bandwidth[key]))
        return hops

    def pipe(self, key: str) -> Resource:
        return self._pipes[key]

    def bandwidth(self, key: str) -> float:
        return self._bandwidth[key]

    @property
    def link_keys(self) -> List[str]:
        return list(self._bandwidth)


def build_fabric(env: Any, link_topology: Optional[Mapping[str, Any]]) -> Optional[NetworkFabric]:
    """``None`` in, ``None`` out -- the default-off path builds no pipes at all."""
    if not link_topology:
        return None
    return NetworkFabric(env, link_topology)
