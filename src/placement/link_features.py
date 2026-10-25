"""partial_state_v7: the live link-sharing state as platform and edge columns (option B, 2026-10-23).

v7 is v6 plus what the fair-share link fabric (HEROSIM_EXCHANGE_FAIRSHARE=1) knows and the scorer could not see: which links are already carrying
peer-exchange flows. The columns are computed by the functions below and nowhere else; two front ends only build their input, `ActiveFlows`:

    live      `flows_live(share, now)`      from `fabric.fairshare.flows` (the transfers holding a link right now)
    stored    `flows_stored(block)`         from `fidelity["flows"]` of a recaptured state (src/placement/flow_snapshot.py, schema v1)

`ActiveFlows` is the active transfers in FairShare.flows order, each as (remaining bytes at the instant, ((directed link id, MB/s), ...)). Only
transfers holding a link count: startup, latency_tail and pending transfers are not in it, as they are not in FairShare.flows. The live front end
never calls `FairShare._advance` (it would split the float subtraction of `rem` and change later bits of the run); it computes
`rem - rate * (now - last)`, the very expression `_advance` evaluates, which is what the stored `rem` is (flow_snapshot: "after `_advance`").

Platform columns (appended after the v6 in-flight columns), per platform, from the flows crossing its NODE's access links (hop ids "node>x" and "x>node"):

    P1  log1p(number of active flows leaving the node)     P2  log1p(number entering it)
    P3  log1p(max of the two seconds to drain the link at full capacity = sum over its flows of rem / (MB/s * 1024**2))

Edge columns (appended after the v5 push columns of the partial-state block), per (task, candidate), over the task's COMMITTED partners (already placed, so
their node is known; the pull direction, partner node -> candidate node, is the direction the platform charges):

    E1  log1p(shared exchange seconds)  = sum over the remote committed partners of the price of moving the payload at the rate the flow would get on the
        live links, bytes / min over hops of (MB/s * 1024**2 / (active flows on the hop + 1)), plus the route latency (fairshare_aware.payload_seconds,
        batch's own transfers not counted); a route with no active flow is priced solo, b * seconds-per-byte, as `_xchg` prices it
    E2  log(E1 seconds / the same sum priced solo); 0 on an empty fabric (the ratio is >= 1)
    E3  log1p(largest number of active flows on any hop of any of those routes, before adding this one)

The edge columns REFRESH with the committed prefix (live_link_features_v1, the `_commit` fix of the fair-share-aware rule): the flows the already-placed
batch-mates will run (`pending_counts`) are added to the active counts before any of E1-E3 is priced, at every decode step and every refine pass, for every
arm. P1-P3 are the at-capture state: the platform block is encoded once per batch and never re-encoded per commit (that is the Arm 2 committed-inputs
path), so they do not move with the prefix.

The cache builder and the live batch both pass `partial_state_columns` the same two tables (`link_counts`: link id -> active flows, `link_routes`:
(node, node) -> directed hops), so one definition serves training and serving (tests/test_link_features.py).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LINK_PLATFORM_DIM = 3
LINK_EDGE_DIM = 3
LINK_STATIC_DIM = 3
LINK_OUT_DIM = 4
MIB = 1024 ** 2

Hop = Tuple[str, float]
ActiveFlows = Tuple[Tuple[float, Tuple[Hop, ...]], ...]

# wall seconds spent in the live link capture (flows_live + node_table + the per-platform lookups) and how many captures; the GNN scheduler folds the seconds of each decision's
# capture into that decision's timed span (decision_timing), for the v7 kinds only
CAPTURE: Dict[str, float] = {"seconds": 0.0, "calls": 0}
NODE_KEY = "link_node"  # temporal-state key of a platform: the raw (up, down, drain seconds) triple


def link_enabled(contract: Optional[str] = None) -> bool:
    from src.policy.tabular.reduced_features import LINK_CONTRACTS, resolve_partial_state_contract

    return resolve_partial_state_contract(contract) in LINK_CONTRACTS


def flows_live(share: Any, now: float) -> ActiveFlows:
    """The active transfers of a live `FairShare`, with `rem` brought to `now` without touching the object."""
    dt = float(now) - float(share._last)
    if dt < 0:
        raise RuntimeError(f"FAIL LOUD: FairShare._last {share._last!r} is after now {now!r}")
    out = []
    for f in share.flows:
        rem = f.rem - f.rate * dt if dt > 0 else f.rem
        out.append((float(rem), tuple((str(k), float(bw)) for k, bw in f.hops)))
    return tuple(out)


def flows_stored(block: Mapping[str, Any]) -> ActiveFlows:
    """The active transfers of a stored `fidelity["flows"]` block (flow_snapshot schema v1), in `active` (= FairShare.flows) order."""
    from src.placement import flow_snapshot

    flow_snapshot.validate(dict(block))
    by_owner = {ex["owner_tid"]: ex for ex in block["exchanges"]}
    out = []
    for owner, idx in block["active"]:
        t = by_owner[owner]["transfers"][idx]
        out.append((float(t["rem"]), tuple((str(k), float(bw)) for k, bw in t["hops"])))
    return tuple(out)


def counts_of(flows: ActiveFlows) -> Dict[str, int]:
    """link id -> number of active flows crossing it."""
    counts: Dict[str, int] = {}
    for _rem, hops in flows:
        for key, _bw in hops:
            counts[key] = counts.get(key, 0) + 1
    return counts


def node_table(flows: ActiveFlows) -> Dict[str, Tuple[int, int, float]]:
    """node name -> (flows leaving it, flows entering it, seconds to drain its busier access link), for every name a hop id starts or ends with, in one pass over the
    flows. The per-node sums add in flow order and hop order, as `node_raw` always did, so the floats are the same."""
    n_up: Dict[str, int] = {}
    n_down: Dict[str, int] = {}
    s_up: Dict[str, float] = {}
    s_down: Dict[str, float] = {}
    for rem, hops in flows:
        for key, bw in hops:
            tail, _, _ = key.partition(">")
            head = key.rpartition(">")[2]
            secs = rem / (bw * MIB)
            n_up[tail] = n_up.get(tail, 0) + 1
            s_up[tail] = s_up.get(tail, 0.0) + secs
            n_down[head] = n_down.get(head, 0) + 1
            s_down[head] = s_down.get(head, 0.0) + secs
    return {n: (n_up.get(n, 0), n_down.get(n, 0), max(s_up.get(n, 0.0), s_down.get(n, 0.0))) for n in set(n_up) | set(n_down)}


def node_raw(flows: ActiveFlows, node_name: str, table: Optional[Mapping[str, Tuple[int, int, float]]] = None) -> Tuple[int, int, float]:
    """(flows leaving the node, flows entering it, seconds to drain the busier access link) at the instant; pass `table` (`node_table(flows)`) to avoid a pass per node."""
    return (table if table is not None else node_table(flows)).get(node_name, (0, 0, 0.0))


def platform_columns(raw: Tuple[int, int, float]) -> np.ndarray:
    """P1-P3 from `node_raw`, shared by the live and the cache builders."""
    n_up, n_down, drain = raw
    return np.asarray([math.log1p(float(n_up)), math.log1p(float(n_down)), math.log1p(max(0.0, float(drain)))], dtype=np.float64)


def edge_columns(
    counts: Mapping[str, int],
    routes: Mapping[Tuple[Any, Any], Sequence[Hop]],
    node_exchange: Mapping[Tuple[Any, Any], Tuple[float, float]],
    partners: Iterable[Tuple[Any, float]],
    node: Any,
) -> Tuple[float, float, float]:
    """E1-E3 of one candidate `node` over the committed `partners` = (partner node, payload bytes). Remote partners only (a co-located partner moves nothing)."""
    shared = solo = 0.0
    busiest = 0
    for node_j, b in partners:
        if node_j == node:
            continue
        pb, lat = node_exchange[(node_j, node)]
        lat = lat if pb > 0.0 else 0.0
        solo_s = b * pb + lat
        hops = routes[(node_j, node)]
        active = [counts.get(key, 0) for key, _bw in hops]
        if any(active):
            rate = min(bw * MIB / (n + 1) for (_k, bw), n in zip(hops, active))
            shared += b / rate + lat
            busiest = max(busiest, max(active))
        else:
            shared += solo_s
        solo += solo_s
    e2 = math.log(shared / solo) if solo > 0.0 and shared > solo else 0.0
    return math.log1p(shared), e2, math.log1p(float(busiest))


def out_edge_columns(
    counts: Mapping[str, int],
    routes: Mapping[Tuple[Any, Any], Sequence[Hop]],
    node_exchange: Mapping[Tuple[Any, Any], Tuple[float, float]],
    partners: Iterable[Tuple[Any, float]],
    node: Any,
) -> Tuple[float, float, float, float]:
    """partial_state_v9 R1-R3 and U1 of one candidate `node` over the committed `partners` = (partner node, payload bytes): the OUTBOUND direction of the exchange.

    The pair table is symmetric, so every remote committed partner also pulls from the candidate: the flow runs node -> partner over the candidate's own access
    uplink first. E1-E3 price only the partner -> node direction. `counts` are the active flows plus the pending flows of the committed prefix, as for E1-E3.

        R1  log1p(sum_j shared_j)   shared_j = b_j / min over the hops of node -> node_j of (MB/s * MiB / (active + 1)) plus the route latency; the solo price,
                                    b * seconds-per-byte of the pair, when no hop of the route is loaded
        R2  log(sum shared / sum solo), 0 when the solo sum is 0 or the shared sum does not exceed it
        R3  log1p(largest active count on any hop of any of those routes, before adding this flow)
        U1  log1p(B * (c + k) / (bw * MiB)), the seconds the candidate's uplink (the first hop of its routes: bandwidth bw MB/s, c flows on it) needs for this
            task's k remote outbound flows of B bytes in all when they share it with the c flows already there; 0 for k = 0. R1 prices each flow as the one new
            flow on its route, U1 prices the k flows as sharing the uplink with one another.

    Remote partners only (a co-located partner moves nothing)."""
    shared = solo = 0.0
    busiest = 0
    k = 0
    total = 0.0
    c = 0
    bw_up = 0.0
    up_key = None
    for node_j, b in partners:
        if node_j == node:
            continue
        pb, lat = node_exchange[(node, node_j)]
        lat = lat if pb > 0.0 else 0.0
        solo_s = b * pb + lat
        hops = routes[(node, node_j)]
        active = [counts.get(key, 0) for key, _bw in hops]
        if any(active):
            rate = min(bw * MIB / (n + 1) for (_k, bw), n in zip(hops, active))
            shared += b / rate + lat
            busiest = max(busiest, max(active))
        else:
            shared += solo_s
        solo += solo_s
        if k and hops[0][0] != up_key:
            raise ValueError(f"FAIL LOUD: out_edge_columns: the routes of node {node!r} do not share one first hop ({up_key!r} vs {hops[0][0]!r}); U1 is defined for an access-link topology")
        up_key = hops[0][0]
        k += 1
        total += b
        c = active[0]
        bw_up = hops[0][1]
    r2 = math.log(shared / solo) if solo > 0.0 and shared > solo else 0.0
    u1 = math.log1p(total * (c + k) / (bw_up * MIB)) if k else 0.0
    return math.log1p(shared), r2, math.log1p(float(busiest)), u1


def static_edge_columns(
    counts: Mapping[str, int],
    routes: Mapping[Tuple[Any, Any], Sequence[Hop]],
    node_exchange: Mapping[Tuple[Any, Any], Tuple[float, float]],
    partners: Iterable[Tuple[Sequence[Any], float]],
    node: Any,
) -> Tuple[float, float, float]:
    """partial_state_v8 S1-S3 of one candidate `node`: the lookahead twin of E1-E3. `partners` = (candidate nodes of the in-batch peer, payload bytes) for EVERY peer
    of the task, placed or not (prefix-free: the same for every committed set), each candidate list with duplicates kept (the weight is the candidate multiplicity, as the
    peer-mass lookahead has it). Per peer, the price of its payload moving from each of its candidate nodes to `node` is averaged over its candidate list; a candidate on
    `node` itself moves nothing and contributes 0 (it still counts in the mean). Only the CAPTURED active counts price the links: no pending flows. Peers with an empty
    candidate list are skipped, sums run in the order of `partners`, and a task with no peers gives zeros.

        S1  log1p(sum_j shared_j)           shared_j = mean over j's candidates of the E1 price (bytes / min over hops of MB/s*MiB/(active+1), plus the route latency;
                                            the solo price when no hop of the route is loaded)
        S2  log(sum_j shared_j / sum_j solo_j), 0 when the solo sum is 0 or the shared sum does not exceed it
        S3  log1p(max_j busiest_j)          busiest_j = mean over j's candidates of the largest active count on any hop of the route
    """
    shared = solo = 0.0
    busiest_max = 0.0
    for nodes_j, b in partners:
        k = len(nodes_j)
        if not k:
            continue
        sh = so = 0.0
        bz = 0
        for nj in nodes_j:
            if nj == node:
                continue
            pb, lat = node_exchange[(nj, node)]
            lat = lat if pb > 0.0 else 0.0
            solo_s = b * pb + lat
            hops = routes[(nj, node)]
            active = [counts.get(key, 0) for key, _bw in hops]
            if any(active):
                rate = min(bw * MIB / (n + 1) for (_k, bw), n in zip(hops, active))
                sh += b / rate + lat
                bz += max(active)
            else:
                sh += solo_s
            so += solo_s
        shared += sh / k
        solo += so / k
        busiest_max = max(busiest_max, bz / k)
    s2 = math.log(shared / solo) if solo > 0.0 and shared > solo else 0.0
    return math.log1p(shared), s2, math.log1p(busiest_max)


def pending_counts(
    routes: Mapping[Tuple[Any, Any], Sequence[Hop]],
    peers_of: Mapping[Any, Iterable[Tuple[Any, float]]],
    node_of_committed: Mapping[Any, Any],
) -> Dict[str, int]:
    """link id -> flows the already-placed batch-mates will run: for every placed task u and every placed peer v of u on another node, one flow v -> u
    (a task's input stage pulls from each of its partners; both directions of a pair are counted, as `_pg_remote_edges` does). Peers outside the batch are
    invisible to the context by contract and not counted. Integer counts, so the order of the walk does not matter."""
    counts: Dict[str, int] = {}
    for u, nu in node_of_committed.items():
        for v, _b in peers_of.get(u, ()):
            nv = node_of_committed.get(v)
            if nv is None or nv == nu:
                continue
            for key, _bw in routes[(nv, nu)]:
                counts[key] = counts.get(key, 0) + 1
    return counts


def with_pending(counts: Mapping[str, int], pending: Mapping[str, int]) -> Dict[str, int]:
    out = dict(counts)
    for key, n in pending.items():
        out[key] = out.get(key, 0) + n
    return out


def routes_between(link_topology: Mapping[str, Any], names: Mapping[Any, str]) -> Dict[Tuple[Any, Any], Tuple[Hop, ...]]:
    """(a, b) -> directed hops for every ordered pair of distinct candidate nodes `names` = {node id: node name}; one definition, cache and live."""
    from src.placement.network_fabric import directed_hops_from_topology

    out: Dict[Tuple[Any, Any], Tuple[Hop, ...]] = {}
    for a, na in names.items():
        for b, nb in names.items():
            if a != b:
                out[(a, b)] = tuple((str(k), float(bw)) for k, bw in directed_hops_from_topology(link_topology, na, nb))
    return out


def widen_graph(graph: Any) -> None:
    """Cache side: append the held-apart link columns to platform_features, in place, and drop them."""
    import torch

    pe = getattr(graph, "link_platform_extra", None)
    if pe is None:
        raise ValueError("partial_state_v7: graph carries no link columns (built under another contract)")
    graph.platform_features = torch.cat([graph.platform_features, pe.to(graph.platform_features.dtype)], dim=1)
    del graph.link_platform_extra
