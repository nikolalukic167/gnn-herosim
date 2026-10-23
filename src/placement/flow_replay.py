"""Replay of the stored peer-exchange flows (option B, 2026-10-23): the consumer of `flow_snapshot` v1.

A fidelity snapshot taken under HEROSIM_EXCHANGE_FAIRSHARE=1 stores, for every task whose input stage is reading or exchanging,
where it is in its serial exchange (src/placement/flow_snapshot.py). Without this module the replay turns such a ghost into
`env.timeout(io_remaining)` (snapshot_fidelity._serve), so a candidate batch is priced against links that carry only its own flows.
With HEROSIM_REPLAY_STORED_FLOWS=1 (default off; unset or '0' leaves the replay byte-identical), `snapshot_fidelity.apply_platforms`:

  1. `start_active` re-creates the active `FairShare` flows, in the stored `active` (start) order, with their remaining bytes, scale and
     hops, in one reschedule; the recomputed rate of each is checked against the stored one (a mismatch is a capture or link error and raises);
  2. each ghost with an ExchangeRecord runs `exchange_process` instead of the single timeout: the rest of the pre-exchange read, then its
     transfers serially as live's `Platform._fairshare_exchange.one` does (TCP start-up that holds no link, all bytes through the shared links,
     then the route latency), starting from each transfer's stored status.

Only the physics is replayed: the fabric's booked exchange tallies (`record_exchange*`) are telemetry that nothing reads back and are not re-booked.
Ghosts whose record is `planned` (admitted, input stage not begun) run the plan as evaluated at the snapshot instant (all their partners were placed
then, so live's plan at input start is the same).

Ghosts in the capture's `unplanned` list (rendezvous, or admitted with an unplaced partner) carry `flows["unplanned_detail"][tid]` = {dst_node, stage,
read_duration, partners [[peer_id, node or null, payload]]}: `snapshot_fidelity.apply_orchestrator` registers the partner table as the ghost's
`peer_exchange` row (partners in the snapshot's batch/queue map to their replay tasks, others become `PeerStub`s), and `unplanned_process` then does what
`Platform.platform_process` does for a real task: wait for the unplaced partners (`_peer_rendezvous_events`), plan the serial exchange at that instant from where the
partners are NOW, so the candidate plan's batch partners are priced where the candidate put them (`_fairshare_plan`), run the read, then the exchange
(`_fairshare_exchange`). A partner that is not placed and is outside the snapshot stays unplaced for ever (the ghost never leaves its wait), as live's
continuation is beyond the replay's horizon. Without `unplanned_detail` (a capture before schema v1.1) these ghosts keep the old path, and `require` says so.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Generator, Optional, Tuple

from src.placement import flow_snapshot
from src.placement.network_fabric import FairShare, _Flow, exchange_fairshare_enabled

ENV = "HEROSIM_REPLAY_STORED_FLOWS"
UNPLANNED_OLD_ENV = "HEROSIM_REPLAY_UNPLANNED_OLD"
RATE_RTOL = 1e-9

ActiveEvents = Dict[Tuple[int, int], Any]


def enabled() -> bool:
    """True under HEROSIM_REPLAY_STORED_FLOWS=1; unset or '0' is off; any other value raises."""
    raw = os.environ.get(ENV)
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"{ENV}={raw!r}; expected '0' or '1'")


def require(fidelity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The validated `flows` block when the switch is on, else None. Fails loudly when the switch is on and the snapshot has no block (it was not
    captured with fair share and snapshot fidelity on) or fair share is off in this run."""
    if not enabled():
        return None
    if not exchange_fairshare_enabled():
        raise RuntimeError(f"{ENV}=1 needs HEROSIM_EXCHANGE_FAIRSHARE=1 in the replay")
    block = fidelity.get(flow_snapshot.KEY)
    if block is None:
        raise RuntimeError(f"{ENV}=1 but the snapshot has no fidelity['{flow_snapshot.KEY}'] block; capture it with "
                           "HEROSIM_SNAPSHOT_FIDELITY=1 and HEROSIM_EXCHANGE_FAIRSHARE=1")
    flow_snapshot.validate(block)
    if block.get("unplanned") and "unplanned_detail" not in block and os.environ.get(UNPLANNED_OLD_ENV) != "1":
        raise RuntimeError(f"{ENV}=1: the snapshot lists unplanned ghosts {block['unplanned'][:5]} but carries no flows['unplanned_detail'] (a capture before schema v1.1); "
                           f"recapture it, or set {UNPLANNED_OLD_ENV}=1 to replay those ghosts the old way (no exchange)")
    return block


def unplanned_detail(block: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """`flows["unplanned_detail"]` by owner task id; {} when the capture predates it."""
    return {int(k): v for k, v in (block.get("unplanned_detail") or {}).items()}


def ghost_key(tid: int) -> int:
    """The id `snapshot_fidelity.GhostTask` gives the ghost of live task `tid`: the key of its row in the orchestrator's `peer_exchange` table."""
    return -1 - int(tid)


def unplanned_process(platform: Any, ghost: Any, detail: Dict[str, Any]) -> Generator:
    """The input stage of an unplanned ghost, as `Platform.platform_process` runs it for a task: rendezvous with unplaced partners, plan from their placements
    at that instant, the read, then the serial exchange."""
    env = platform.env
    for ready in platform._peer_rendezvous_events(ghost):
        yield ready
    flows = platform._fairshare_plan(ghost, record=False)
    live_id = (getattr(platform.node.orchestrator_ref, "fid_live_peer", None) or {}).get(ghost.id)
    if live_id:                                    # the live plan runs the partners in LIVE task-id order; the replay's own ids (batch tasks first, stubs after) are not that order
        flows.sort(key=lambda f: live_id[f["peer_id"]])
    yield env.timeout(float(detail["read_duration"]))
    if flows:
        yield from platform._fairshare_exchange(ghost, flows)


def _records(block: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(ex["owner_tid"]): ex for ex in block["exchanges"]}


def start_active(fairshare: FairShare, block: Dict[str, Any]) -> ActiveEvents:
    """Re-create the stored active flows in `active` order and return their completion events by (owner_tid, partner_index). One reschedule, after
    all are in: rates depend on the set of flows only. A flow stored with rem == 0 stays in the list and ends at the instant, as live's wake does."""
    recs = _records(block)
    fairshare._advance()
    events: ActiveEvents = {}
    made = []
    for tid, idx in block["active"]:
        t = recs[int(tid)]["transfers"][int(idx)]
        ev = fairshare.env.event()
        fl = _Flow([(k, bw) for k, bw in t["hops"]], t["rest"], t["scale"], ev, (int(tid), int(idx)))
        fl.rem = float(t["rem"])
        fairshare.flows.append(fl)
        events[(int(tid), int(idx))] = ev
        made.append((fl, t, tid, idx))
    fairshare._reschedule()
    for fl, t, tid, idx in made:
        if abs(fl.rate - t["rate"]) > RATE_RTOL * max(fl.rate, t["rate"]):
            raise RuntimeError(f"flow replay: task {tid} partner {t['partner_id']}: recomputed rate {fl.rate!r} B/s differs from the stored "
                               f"{t['rate']!r}; the links or the active set of the replay do not match the capture")
    return events


def exchange_process(env: Any, fairshare: FairShare, ex: Dict[str, Any], events: ActiveEvents) -> Generator:
    """The rest of one task's input stage after the snapshot: what `Platform.platform_process` and `_fairshare_exchange` do from the stored state."""
    tid = int(ex["owner_tid"])
    if ex["read_remaining"] > 0:
        yield env.timeout(ex["read_remaining"])
    for i, t in enumerate(ex["transfers"]):
        status = t["status"]
        if status == "done":
            continue
        if status == "latency_tail":
            yield env.timeout(t["latency_remaining"])
            continue
        if status == "active":
            yield events[(tid, i)]
        else:
            if status == "startup":
                yield env.timeout(t["startup_remaining"])
            elif t["b"] > 0:
                yield env.timeout(t["b"])
            if t["rest"] > 0:
                yield fairshare.start([(k, bw) for k, bw in t["hops"]], t["rest"], t["scale"], owner=(tid, i))
        if t["latency"] > 0:
            yield env.timeout(t["latency"])
