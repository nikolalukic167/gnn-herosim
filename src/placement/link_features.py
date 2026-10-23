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

The cache builder and the live batch both pass `partial_state_columns` the same two tables (`link_counts`: link id -> active flows, `link_routes`:
(node, node) -> directed hops), so one definition serves training and serving (tests/test_link_features.py).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LINK_PLATFORM_DIM = 3
LINK_EDGE_DIM = 3
MIB = 1024 ** 2

Hop = Tuple[str, float]
ActiveFlows = Tuple[Tuple[float, Tuple[Hop, ...]], ...]

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


def node_raw(flows: ActiveFlows, node_name: str) -> Tuple[int, int, float]:
    """(flows leaving the node, flows entering it, seconds to drain the busier access link) at the instant."""
    up_pre, down_suf = f"{node_name}>", f">{node_name}"
    n_up = n_down = 0
    s_up = s_down = 0.0
    for rem, hops in flows:
        for key, bw in hops:
            if key.startswith(up_pre):
                n_up += 1
                s_up += rem / (bw * MIB)
            elif key.endswith(down_suf):
                n_down += 1
                s_down += rem / (bw * MIB)
    return n_up, n_down, max(s_up, s_down)


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
