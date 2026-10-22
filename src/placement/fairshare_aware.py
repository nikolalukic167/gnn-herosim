"""HEROSIM_PG_FAIRSHARE_AWARE=1: the hand rule prices a peer exchange at the rate the flow would get on the live fair-share links.

The bar prices an exchange at the solo bottleneck rate (`Platform._payload_transfer_time`). Under HEROSIM_EXCHANGE_FAIRSHARE=1 the run charges
it at the shared rate, so the bar mis-prices every link that already carries flows. With this switch, the price of a transfer of B bytes over
the directed route (the one `Platform._fairshare_plan` uses) is B / min over hops of (bandwidth / (active FairShare flows on that directional
link + 1)). The counts are read from the live `FairShare` object; nothing is simulated and no physics changes. With no active flow on any hop
of the route the solo price is returned unchanged (bit-identical to the bar); switch off, every call is `_payload_transfer_time`.
"""
from __future__ import annotations

import os
from typing import Any, Dict

AWARE_ENV = "HEROSIM_PG_FAIRSHARE_AWARE"
DRAIN_ENV = "HEROSIM_PG_FAIRSHARE_AWARE_DRAIN"    # diagnostic variant (a): 0 = the drain walk keeps the solo price (default 1)
COMMIT_ENV = "HEROSIM_PG_FAIRSHARE_AWARE_COMMIT"  # diagnostic variant (b): 1 = the batch's own other transfers are counted as flows (default 0)
HERD_ENV = "HEROSIM_PG_HERD"                      # diagnostic: per batch, the share of its remote transfers crossing its busiest route link
MODEL = "live_rate_v1"
STATS: Dict[str, int] = {"priced": 0, "nonzero": 0}
HERD: Dict[str, float] = {"batches": 0, "flows": 0, "top_share_sum": 0.0, "top_count_sum": 0, "top_share_ge_half": 0}
PENDING: Dict[str, int] = {}  # link -> flows the rest of the batch will create (variant b), set around one task's choice
_ENABLED = None
_FLAGS: Dict[str, bool] = {}
_CACHE: Dict[str, Any] = {"share": None, "gen": None, "active": {}}  # link -> active flows; FairShare bumps `_gen` whenever a flow starts or ends


def forget_switch() -> None:
    """Re-read the environment at the next call (a scheduler is built once per run)."""
    global _ENABLED
    _ENABLED = None
    _FLAGS.clear()


def reset() -> None:
    """Forget the cached switch and zero the books (tests; one run per process)."""
    forget_switch()
    STATS["priced"] = 0
    STATS["nonzero"] = 0
    _CACHE.update(share=None, gen=None, active={})
    PENDING.clear()
    HERD.update(batches=0, flows=0, top_share_sum=0.0, top_count_sum=0, top_share_ge_half=0)


def fairshare_aware_enabled() -> bool:
    """True under HEROSIM_PG_FAIRSHARE_AWARE=1; unset or '0' is off; any other value raises; on without HEROSIM_EXCHANGE_FAIRSHARE=1 raises."""
    global _ENABLED
    if _ENABLED is not None:
        return _ENABLED
    raw = os.environ.get(AWARE_ENV)
    if raw is None or raw == "" or raw == "0":
        on = False
    elif raw == "1":
        on = True
    else:
        raise ValueError(f"FAIL LOUD: {AWARE_ENV}={raw!r}; expected '0' or '1'")
    if on and os.environ.get("HEROSIM_EXCHANGE_FAIRSHARE") != "1":
        raise RuntimeError(f"FAIL LOUD: {AWARE_ENV}=1 prices exchanges at the live fair-share rate but HEROSIM_EXCHANGE_FAIRSHARE is not 1")
    flags = {}
    for env, default in ((DRAIN_ENV, True), (COMMIT_ENV, False)):
        v = os.environ.get(env)
        if v not in (None, "", "0", "1"):
            raise ValueError(f"FAIL LOUD: {env}={v!r}; expected '0' or '1'")
        if v not in (None, "") and not on:
            raise RuntimeError(f"FAIL LOUD: {env} is set but {AWARE_ENV} is not 1")
        flags[env] = default if v in (None, "") else v == "1"
    _FLAGS.update(flags)
    _ENABLED = on
    return on


def drain_enabled() -> bool:
    fairshare_aware_enabled()
    return _FLAGS.get(DRAIN_ENV, True)


def commit_enabled() -> bool:
    fairshare_aware_enabled()
    return _FLAGS.get(COMMIT_ENV, False)


def herd_enabled() -> bool:
    raw = os.environ.get(HERD_ENV)
    if raw not in (None, "", "0", "1"):
        raise ValueError(f"FAIL LOUD: {HERD_ENV}={raw!r}; expected '0' or '1'")
    return raw == "1"


def payload_seconds(platform: Any, parent_node_name: str, payload_bytes: float, drain: bool = False) -> float:
    """Seconds to move `payload_bytes` from `parent_node_name` to `platform`'s node, as the rule prices it (route latency is not included)."""
    solo = platform._payload_transfer_time(parent_node_name, payload_bytes)
    if not fairshare_aware_enabled() or (drain and not drain_enabled()):
        return solo
    fabric = getattr(platform.node, "fabric", None)
    share = getattr(fabric, "fairshare", None)
    if share is None:
        raise RuntimeError(f"FAIL LOUD: {AWARE_ENV}=1 needs the link fabric of the run built with HEROSIM_EXCHANGE_FAIRSHARE=1")
    hops = fabric.directed_hops(parent_node_name, platform.node.node_name)
    if not hops:
        return solo
    STATS["priced"] += 1
    if _CACHE["share"] is not share or _CACHE["gen"] != share._gen:
        active: Dict[str, int] = {}
        for flow in share.flows:
            for key, _bw in flow.hops:
                active[key] = active.get(key, 0) + 1
        _CACHE.update(share=share, gen=share._gen, active=active)
    active = _CACHE["active"]
    pending = PENDING if (PENDING and not drain) else {}
    if not any(active.get(key, 0) + pending.get(key, 0) for key, _bw in hops):
        return solo
    STATS["nonzero"] += 1
    rate = min(bw * 1024 ** 2 / (active.get(key, 0) + pending.get(key, 0) + 1) for key, bw in hops)
    return payload_bytes / rate


def pending_flows(fabric: Any, edges: Any) -> Dict[str, int]:
    """link -> number of flows from `edges`, an iterable of (source node, destination node) remote transfers, over their directed routes."""
    counts: Dict[str, int] = {}
    for src, dst in edges:
        if src == dst:
            continue
        for key, _bw in fabric.directed_hops(src, dst):
            counts[key] = counts.get(key, 0) + 1
    return counts


def record_herd(fabric: Any, edges: Any) -> None:
    """One batch's remote transfers: the largest number crossing one directed link, as a share of all of them."""
    edges = [(a, b) for a, b in edges if a != b]
    if len(edges) < 2:
        return
    counts = pending_flows(fabric, edges)
    top = max(counts.values())
    HERD["batches"] += 1
    HERD["flows"] += len(edges)
    HERD["top_count_sum"] += top
    HERD["top_share_sum"] += top / len(edges)
    HERD["top_share_ge_half"] += int(top / len(edges) >= 0.5)


def summary_block() -> Dict[str, Any]:
    return {"model": MODEL, "priced_transfers": STATS["priced"], "priced_transfers_nonzero_flows": STATS["nonzero"],
            "drain_walk_shared": drain_enabled(), "batch_commit_counted": commit_enabled()}


def herd_block() -> Dict[str, Any]:
    n = HERD["batches"]
    return {"batches_with_2plus_remote_transfers": n, "remote_transfers": HERD["flows"],
            "mean_share_on_busiest_link": HERD["top_share_sum"] / n if n else None, "mean_flows_on_busiest_link": HERD["top_count_sum"] / n if n else None,
            "batches_with_half_or_more_on_one_link": HERD["top_share_ge_half"]}
