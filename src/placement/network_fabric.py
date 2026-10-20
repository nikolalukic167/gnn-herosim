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

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from simpy import Resource

LINK_SEP = "|"

# Core routers are the only shared segment; access links are per-client/per-server and
# stay additive. The prefix is how every consumer tells the two apart without needing the
# topology config -- the diagnostic and the pre-check both rely on it.
CORE_PREFIX = "core"


TRANSFER_MODEL_ENV = "HEROSIM_TRANSFER_MODEL"
TRANSFER_MODELS = ("store_forward", "pipelined")


def transfer_model() -> str:
    """How a payload crosses a multi-hop route. ``store_forward`` (default, every run before 2026-10-07): each hop
    carries the whole payload in turn, n_hops x size / bottleneck. ``pipelined``: the payload streams through every
    hop at once, size / bottleneck, as a packet network moves a large transfer. Propagation latency is charged
    separately either way."""
    model = os.environ.get(TRANSFER_MODEL_ENV, "store_forward")
    if model not in TRANSFER_MODELS:
        raise ValueError(f"{TRANSFER_MODEL_ENV}={model!r}; expected one of {TRANSFER_MODELS}")
    return model


def transmission_hops(n_hops: float) -> float:
    """Multiplier on size / bottleneck for a route of ``n_hops`` links under the active transfer model."""
    if transfer_model() == "pipelined":
        return 1.0 if n_hops > 0 else 0.0
    return n_hops


def link_key(a: str, b: str) -> str:
    """Canonical undirected key for the link between two nodes.

    Sorted so ``link_key(x, y) == link_key(y, x)`` -- the pipe must be the same object in
    both directions or traffic would not contend.
    """
    return f"{a}{LINK_SEP}{b}" if a <= b else f"{b}{LINK_SEP}{a}"


# workload_fix_v1 W3: per-node access-link classes, MB/s (the unit every charge divides by, 1024**2 per MB).
# ``out`` leaves the node, ``in`` enters it. Wi-Fi draws one symmetric rate per node from ``range``.
ACCESS_CLASSES: Dict[str, Dict[str, Any]] = {
    "wired": {"out": 117.0, "in": 117.0},
    "wifi": {"range": (7.0, 14.0)},
    "cellular": {"out": 4.0, "in": 75.0},
}
DEFAULT_ACCESS_MIX: Dict[str, float] = {"wired": 0.4, "wifi": 0.4, "cellular": 0.2}


# HEROSIM_EXCHANGE_TCP=1 (default off): a disclosed sensitivity on the peer-exchange charge (uncut_groups_s0_v1 Record,
# 2026-10-20 "Simulator component check" and the TCP-transfer sensitivity line). Real TCP over the simulator's own rates and
# latencies costs MTU framing, a handshake and slow start; the plain charge size / bottleneck has none of them. Under the switch each
# non-co-located peer transfer is `tcp_transfer(size, bottleneck, lat) + lat` instead of `size / bottleneck + lat`: one handshake RTT,
# slow start from 10 segments doubling per RTT up to the bandwidth-delay product, the rest at goodput. No fitted value. Every remote
# partner is its own connection (its own handshake) and the terms are summed serially, as the exchange loop already does.
EXCHANGE_TCP_ENV = "HEROSIM_EXCHANGE_TCP"
TCP_MSS = 1448
TCP_FRAME = 1514
TCP_INITCWND_SEGMENTS = 10


def exchange_tcp_enabled() -> bool:
    """True under HEROSIM_EXCHANGE_TCP=1; unset or '0' is off; any other value raises."""
    raw = os.environ.get(EXCHANGE_TCP_ENV)
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"{EXCHANGE_TCP_ENV}={raw!r}; expected '0' or '1'")


def tcp_transfer(size_bytes: float, bottleneck_mbs: float, lat: float) -> float:
    """Seconds to move ``size_bytes`` over a route of bottleneck ``bottleneck_mbs`` (MB/s, 1024**2 bytes) and one-way latency
    ``lat`` (s), excluding the one-way ``lat`` the caller still adds once."""
    if not bottleneck_mbs > 0:
        raise ValueError(f"{EXCHANGE_TCP_ENV}: bottleneck {bottleneck_mbs} MB/s must be > 0")
    rtt = 2 * lat
    g = bottleneck_mbs * 1024 ** 2 / (TCP_FRAME / TCP_MSS)
    bdp = g * rtt
    t = rtt
    sent = 0.0
    cw = TCP_INITCWND_SEGMENTS * TCP_MSS
    while sent < size_bytes and cw < bdp:
        sent += cw
        t += rtt
        cw *= 2
    if sent < size_bytes:
        t += (size_bytes - sent) / g
    else:
        t -= rtt
    return t


def tcp_startup_and_rest(size_bytes: float, bottleneck_mbs: float, lat: float) -> Tuple[float, float]:
    """`tcp_transfer` split into its start-up and its rate-limited part: ``(b, rest_bytes)`` with
    ``b + rest_bytes / goodput == tcp_transfer(size, bottleneck, lat)``. ``b`` is the handshake and the slow-start rounds that
    carry the first ``size - rest`` bytes at a window-limited rate; ``rest_bytes`` is what is left to move at the link rate."""
    if not bottleneck_mbs > 0:
        raise ValueError(f"{EXCHANGE_TCP_ENV}: bottleneck {bottleneck_mbs} MB/s must be > 0")
    rtt = 2 * lat
    g = bottleneck_mbs * 1024 ** 2 / (TCP_FRAME / TCP_MSS)
    bdp = g * rtt
    t = rtt
    sent = 0.0
    cw = TCP_INITCWND_SEGMENTS * TCP_MSS
    while sent < size_bytes and cw < bdp:
        sent += cw
        t += rtt
        cw *= 2
    if sent < size_bytes:
        return t, size_bytes - sent
    return t - rtt, 0.0


# HEROSIM_EXCHANGE_FAIRSHARE=1 (default off): peer-exchange flows that overlap in time share each link, processor sharing
# (uncut_groups_s0_v1 Record, 2026-10-20). A flow's rate is the minimum over the links of its route of (link capacity / flows
# active on that link), recomputed whenever a flow starts or ends; links are directional (full duplex: A>B and B>A are separate
# capacities). A flow alone gets the whole bottleneck, so a single exchange costs what it cost before. The route latency is not a
# shared resource: it is added once after the bytes have gone, as in the plain charge. Composes with HEROSIM_EXCHANGE_TCP: the
# per-transfer start-up (handshake, slow-start rounds) is the TCP model's and is paid per transfer without holding any link; only
# the remaining bytes (at goodput, i.e. capacity x MSS / frame) share the links. A task's exchanges with several partners start
# one after another (HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS=serial, the default and SCIENTIST's ruling A1: the switch changes one thing, the
# contention between different tasks' flows); `concurrent` starts them together and is not used by any registered study. Neither
# scheduler prices the shared rate. The pipelined transfer model only.
EXCHANGE_FAIRSHARE_ENV = "HEROSIM_EXCHANGE_FAIRSHARE"
EXCHANGE_FAIRSHARE_PARTNERS_ENV = "HEROSIM_EXCHANGE_FAIRSHARE_PARTNERS"
FAIRSHARE_PARTNER_MODES = ("concurrent", "serial")
FAIRSHARE_MODEL = "processor_sharing_v1"


def exchange_fairshare_enabled() -> bool:
    """True under HEROSIM_EXCHANGE_FAIRSHARE=1; unset or '0' is off; any other value raises."""
    raw = os.environ.get(EXCHANGE_FAIRSHARE_ENV)
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"{EXCHANGE_FAIRSHARE_ENV}={raw!r}; expected '0' or '1'")


def exchange_fairshare_partners() -> str:
    """How one task's exchanges with several partners start: ``serial`` (default) or ``concurrent``; any other value raises."""
    raw = os.environ.get(EXCHANGE_FAIRSHARE_PARTNERS_ENV)
    if raw is None or raw == "":
        return "serial"
    if raw in FAIRSHARE_PARTNER_MODES:
        return raw
    raise ValueError(f"{EXCHANGE_FAIRSHARE_PARTNERS_ENV}={raw!r}; expected one of {FAIRSHARE_PARTNER_MODES}")


class _Flow:
    __slots__ = ("hops", "nbytes", "rem", "scale", "rate", "event")

    def __init__(self, hops: Sequence[Tuple[str, float]], nbytes: float, scale: float, event: Any) -> None:
        self.hops, self.nbytes, self.rem, self.scale, self.rate, self.event = list(hops), float(nbytes), float(nbytes), float(scale), 0.0, event


class FairShare:
    """Processor sharing over named directional links (capacities in MB/s, 1024**2 bytes). `start` returns an event that
    succeeds when the flow's bytes have all been moved; the rates of every active flow are recomputed whenever a flow starts or
    ends. Work-conserving: n equal flows on one link each take n times as long as one alone."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.flows: List[_Flow] = []
        self._last = env.now
        self._gen = 0
        self.peak_flows_on_a_link = 0

    def start(self, hops: Sequence[Tuple[str, float]], nbytes: float, scale: float = 1.0) -> Any:
        if not nbytes > 0 or not hops:
            raise ValueError("FairShare.start needs bytes > 0 and a route")
        ev = self.env.event()
        self._advance()
        self.flows.append(_Flow(hops, nbytes, scale, ev))
        self._reschedule()
        return ev

    def _advance(self) -> None:
        dt = self.env.now - self._last
        if dt > 0:
            for f in self.flows:
                f.rem -= f.rate * dt
        self._last = self.env.now

    def _reschedule(self) -> None:
        counts: Dict[str, int] = {}
        for f in self.flows:
            for key, _bw in f.hops:
                counts[key] = counts.get(key, 0) + 1
        if counts:
            self.peak_flows_on_a_link = max(self.peak_flows_on_a_link, max(counts.values()))
        for f in self.flows:
            f.rate = f.scale * min(bw * 1024 ** 2 / counts[key] for key, bw in f.hops)
        self._gen += 1
        if self.flows:
            self.env.process(self._wake(self._gen, max(0.0, min(f.rem / f.rate for f in self.flows))))

    def _wake(self, gen: int, dt: float):
        yield self.env.timeout(dt)
        if gen != self._gen:
            return
        self._advance()
        done = [f for f in self.flows if f.rem <= 1e-9 * max(f.nbytes, 1.0)]
        for f in done:
            self.flows.remove(f)
            f.event.succeed()
        self._reschedule()


def draw_access_classes(
    node_names: Sequence[str], seed: int, mix: Optional[Mapping[str, float]] = None
) -> Dict[str, Dict[str, Any]]:
    """One class per node, drawn once per topology from a stream derived from the topology seed alone.

    Returns ``{node: {"class", "out_mbps", "in_mbps"}}``. The stream is dedicated, so enabling classes never
    shifts any other draw of the generator.
    """
    import random

    mix = dict(DEFAULT_ACCESS_MIX if mix is None else mix)
    unknown = sorted(set(mix) - set(ACCESS_CLASSES))
    if unknown:
        raise ValueError(f"access_classes.mix names unknown class(es) {unknown}; known: {sorted(ACCESS_CLASSES)}")
    total = sum(float(v) for v in mix.values())
    if any(float(v) < 0 for v in mix.values()) or abs(total - 1.0) > 1e-9:
        raise ValueError(f"access_classes.mix must be non-negative and sum to 1, got {mix} (sum {total})")
    names = [name for name in ACCESS_CLASSES if mix.get(name, 0.0) > 0]
    rng = random.Random(f"{seed}:access_classes_v1")
    drawn: Dict[str, Dict[str, Any]] = {}
    for node_name in node_names:
        u = rng.random()
        acc = 0.0
        chosen = names[-1]
        for name in names:
            acc += float(mix[name])
            if u < acc:
                chosen = name
                break
        spec = ACCESS_CLASSES[chosen]
        if "range" in spec:
            rate = rng.uniform(*spec["range"])
            out_mbps = in_mbps = rate
        else:
            out_mbps, in_mbps = spec["out"], spec["in"]
        drawn[node_name] = {"class": chosen, "out_mbps": out_mbps, "in_mbps": in_mbps}
    return drawn


def directed_bandwidth(attrs: Mapping[str, Any], frm: str, to: str) -> float:
    """Bandwidth of one link traversed ``frm`` -> ``to``. A link with no ``access_node`` is symmetric and returns
    ``bandwidth_mbps`` exactly as before; a directional access link returns the access node's out rate when
    traversed away from it and its in rate when traversed toward it."""
    node = attrs.get("access_node")
    if node is None:
        return float(attrs["bandwidth_mbps"])
    if frm == node:
        return float(attrs["bandwidth_out_mbps"])
    if to == node:
        return float(attrs["bandwidth_in_mbps"])
    raise KeyError(f"access link of {node!r} traversed {frm!r}->{to!r}, which touches neither end")


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
            if attrs.get("access_node") is not None:
                for field in ("bandwidth_out_mbps", "bandwidth_in_mbps"):
                    if attrs.get(field) is None or float(attrs[field]) <= 0:
                        raise ValueError(
                            f"link_contention_v1: directional link {key!r} {field} must be > 0, "
                            f"got {attrs.get(field)}"
                        )

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
        # workload_fix_v1 W3: per-node access class, and peer-exchange seconds by the classes of the two servers.
        # Empty on every topology without `access_classes`, which then records nothing.
        self.access_classes: Dict[str, str] = {
            name: str(spec["class"]) for name, spec in (link_topology.get("access_classes") or {}).items()
        }
        self.exchange_by_class: Dict[str, Dict[str, float]] = {}
        # Per sending node (its access link): (start, end, class pair) of every peer exchange it sends. Telemetry
        # only; exchanges are charged size/bottleneck + latency with no shared pipe, and this is what exposes it.
        self.exchange_intervals: Dict[str, List[Tuple[float, float, str]]] = {}
        # HEROSIM_EXCHANGE_TCP: seconds the switch added to the booked exchange seconds. None (and no summary key) when off.
        self.exchange_tcp: Optional[Dict[str, Any]] = (
            {"model": "slow_start_v1", "mss": TCP_MSS, "frame": TCP_FRAME, "initcwnd_segments": TCP_INITCWND_SEGMENTS,
             "per_partner": True, "transfers": 0, "extra_seconds": 0.0}
            if exchange_tcp_enabled() else None
        )
        # HEROSIM_EXCHANGE_FAIRSHARE: the processor-sharing engine of this run and its booked totals. None (and no summary key) when off.
        self.fairshare: Optional[FairShare] = FairShare(env) if exchange_fairshare_enabled() else None
        self.exchange_fairshare: Optional[Dict[str, Any]] = (
            {"model": FAIRSHARE_MODEL, "partners": exchange_fairshare_partners(), "directional_links": True,
             "transfers": 0, "shared_transfers": 0, "flow_seconds": 0.0, "extra_seconds": 0.0, "peak_flows_on_a_link": 0}
            if self.fairshare is not None else None
        )

    @property
    def link_topology(self) -> Mapping[str, Any]:
        """The `links` + `routes` block this fabric was built from, unmodified."""
        return self._link_topology

    def hops(self, src: str, dst: str) -> List[Tuple[str, float]]:
        """``[(link_key, bandwidth_mbps)]`` along the route, in traversal order. The bandwidth is the one for the
        direction of travel when the link is a directional access link."""
        hops: List[Tuple[str, float]] = []
        links = self._link_topology.get("links") or {}
        path = _lookup_path(self._routes, src, dst)
        for a, b in zip(path, path[1:]):
            key = link_key(a, b)
            if key not in self._bandwidth:
                raise KeyError(
                    f"link_contention_v1: route {src!r}->{dst!r} traverses link {key!r} "
                    f"which has no capacity entry in link_topology.links"
                )
            attrs = links[key]
            bandwidth = (
                directed_bandwidth(attrs, a, b) if attrs.get("access_node") is not None else self._bandwidth[key]
            )
            hops.append((key, bandwidth))
        return hops

    def record_exchange(self, node_a: str, node_b: str, seconds: float, bytes_: Optional[float] = None,
                        sender: Optional[str] = None, start: Optional[float] = None) -> None:
        """Add one peer exchange to its unordered class pair, e.g. ``cellular+wired``. ``bytes_`` is the payload the
        caller charged; ``sender`` and ``start`` (the sending node and the exchange's start time) feed
        `exchange_overlap`. Each is optional and adds a key or an interval only when given."""
        pair = "+".join(sorted((self.access_classes[node_a], self.access_classes[node_b])))
        slot = self.exchange_by_class.setdefault(pair, {"seconds": 0.0, "transfers": 0})
        slot["seconds"] += float(seconds)
        slot["transfers"] += 1
        if bytes_ is not None:
            slot["bytes"] = slot.get("bytes", 0.0) + float(bytes_)
        if sender is not None and start is not None:
            self.exchange_intervals.setdefault(sender, []).append((float(start), float(start) + float(seconds), pair))

    def record_exchange_tcp(self, extra_seconds: float) -> None:
        """Book what HEROSIM_EXCHANGE_TCP added to one exchange (its TCP time minus the plain size / bottleneck term)."""
        slot = self.exchange_tcp
        if slot is None:
            raise RuntimeError("record_exchange_tcp called with HEROSIM_EXCHANGE_TCP off")
        slot["transfers"] += 1
        slot["extra_seconds"] += float(extra_seconds)

    def directed_hops(self, src: str, dst: str) -> List[Tuple[str, float]]:
        """``[(directed_link_id, bandwidth_mbps)]`` along the route: the id carries the direction of travel (``a>b``), so the two
        directions of a link are separate capacities. Same bandwidths and order as `hops`."""
        links = self._link_topology.get("links") or {}
        path = _lookup_path(self._routes, src, dst)
        out: List[Tuple[str, float]] = []
        for a, b in zip(path, path[1:]):
            key = link_key(a, b)
            if key not in self._bandwidth:
                raise KeyError(f"route {src!r}->{dst!r} traverses link {key!r} which has no capacity entry in link_topology.links")
            attrs = links[key]
            bw = directed_bandwidth(attrs, a, b) if attrs.get("access_node") is not None else self._bandwidth[key]
            out.append((f"{a}>{b}", bw))
        return out

    def record_exchange_fairshare(self, actual_seconds: float, solo_seconds: float) -> None:
        """Book one finished fair-share exchange: its seconds as run and what it would have taken alone on its route."""
        slot = self.exchange_fairshare
        if slot is None or self.fairshare is None:
            raise RuntimeError("record_exchange_fairshare called with HEROSIM_EXCHANGE_FAIRSHARE off")
        slot["transfers"] += 1
        slot["flow_seconds"] += float(actual_seconds)
        slot["extra_seconds"] += float(actual_seconds) - float(solo_seconds)
        if float(actual_seconds) > float(solo_seconds) * (1.0 + 1e-9) + 1e-9:
            slot["shared_transfers"] += 1
        slot["peak_flows_on_a_link"] = self.fairshare.peak_flows_on_a_link

    def exchange_overlap(self) -> Dict[str, Any]:
        """Share of exchange seconds that run concurrently with another exchange leaving the same source access
        link, and the share of transfers with any such overlap (open intervals: touching ends do not overlap).
        ``union_over_sum`` is sum over sources of the union length over the sum of lengths."""
        total = overlapped = union = 0.0
        n = n_over = 0
        by: Dict[str, Dict[str, float]] = {}
        for items in self.exchange_intervals.values():
            events = []
            for i, (a, b, pair) in enumerate(items):
                if b > a:
                    events.append((b, 0, i))
                    events.append((a, 1, i))
            events.sort()
            active: Dict[int, None] = {}
            flagged = set()
            prev = None
            for t, kind, i in events:
                if prev is not None and active and t > prev:
                    seg = t - prev
                    union += seg
                    if len(active) >= 2:
                        for j in active:
                            slot = by.setdefault(items[j][2], {"seconds": 0.0, "overlap_seconds": 0.0, "transfers": 0,
                                                               "transfers_with_overlap": 0})
                            slot["overlap_seconds"] += seg
                        overlapped += seg * len(active)
                if kind == 0:
                    active.pop(i, None)
                else:
                    if active:
                        flagged.add(i)
                        flagged.update(active)
                    active[i] = None
                prev = t
            for i, (a, b, pair) in enumerate(items):
                slot = by.setdefault(pair, {"seconds": 0.0, "overlap_seconds": 0.0, "transfers": 0,
                                            "transfers_with_overlap": 0})
                slot["seconds"] += b - a
                slot["transfers"] += 1
                slot["transfers_with_overlap"] += i in flagged
                total += b - a
                n += 1
                n_over += i in flagged
        return {"overlap_seconds_share": overlapped / total if total else 0.0,
                "transfers_with_overlap_share": n_over / n if n else 0.0,
                "union_over_sum": union / total if total else 1.0,
                "exchange_seconds": total, "transfers": n,
                "by_class": {k: {**v, "overlap_seconds_share": v["overlap_seconds"] / v["seconds"] if v["seconds"] else 0.0}
                             for k, v in sorted(by.items())}}

    def has_route(self, src: str, dst: str) -> bool:
        try:
            _lookup_path(self._routes, src, dst)
        except KeyError:
            return False
        return True

    def route_latency(self, src: str, dst: str) -> float:
        """Propagation latency (s) along the stored route: the sum of its links' latencies."""
        links = self._link_topology.get("links") or {}
        return sum(float(links[key]["latency"]) for key in route_links(self._routes, src, dst))

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
