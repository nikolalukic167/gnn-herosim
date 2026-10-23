"""Stored-flow schema: what a fidelity snapshot records about peer-exchange transfers in flight or still to run (option B, 2026-10-23). Version 1.

`HEROSIM_SNAPSHOT_FIDELITY=1` + `HEROSIM_EXCHANGE_FAIRSHARE=1` add one more key, `flows`, to the `fidelity` block of a live audit snapshot
(snapshot_fidelity.capture). Off, no key is added and the snapshot is byte-identical to before. All times are seconds relative to the snapshot
instant (>= 0, the future), as in snapshot_fidelity; bytes are bytes; rates are bytes/s; link bandwidths are MB/s (1024**2 bytes), exactly as
`FairShare` holds them; link ids are the DIRECTED ids `fabric.directed_hops` returns ("a>b").

    fidelity["flows"] = {
        "version": 1,
        "model": "processor_sharing_v1",     # network_fabric.FAIRSHARE_MODEL
        "partners": "serial",                # network_fabric.exchange_fairshare_partners(); only serial is defined
        "tcp": bool,                         # HEROSIM_EXCHANGE_TCP was on
        "fairshare_last": float,             # FairShare._last relative to the instant (<= 0). The capture does NOT advance FairShare (that would split a
                                             # flow's float arithmetic and move the run); `rem` below is projected: flow.rem - flow.rate * (now - _last)
        "active": [[owner_tid, partner_index], ...],   # FairShare.flows in list (start) order: owner task and index into that owner's `transfers`
        "exchanges": [ExchangeRecord, ...],  # one per task that has, or will run, an exchange with at least one remote partner
        "unplanned": [owner_tid, ...],       # ghosts that will exchange but whose plan could not be made at the instant (rendezvous: partner unplaced;
                                             # admitted with an unplaced partner): they have NO ExchangeRecord, a replay plans them at input start
    }

ExchangeRecord (key: owner_tid = the ghost's `tid`):
    owner_tid        int    live id of the task whose input stage runs the exchange (the receiver)
    dst_node         str    its node
    stage            str    the ghost's own stage: "input_io" (read running or exchange running) or "cold" / "ingress" (admitted, input stage not begun).
                            Rendezvous ghosts and ghosts past the exchange (lock_wait, compute) have no record
    planned          bool   False when `transfers` is live's `_fairshare_plan` of this task as the input stage built it (stage input_io);
                            True when the task has not begun its input stage and `transfers` is the plan evaluated at the snapshot instant (the real plan is
                            made at input start from the partners' placements then)
    read_remaining   float  seconds of the pre-exchange `env.timeout(input_duration)` still to run (live yields it BEFORE the serial exchange); 0.0 once over.
                            While > 0 no transfer of this task is started: every status is "pending". For planned=True it is the full read duration,
                            which only starts when the input stage begins (after the task's ingress and cold start)
    transfers        [Transfer, ...]   the remote partners in the order `_fairshare_plan` runs them (sorted peer id; co-located partners absent)
    `io_remaining` of the ghost is left as it is (io_end - now, io_end = start + SOLO estimate); under fair share it is wrong, so a replay treats
    `read_remaining` + `transfers` as authoritative for any ghost that has an ExchangeRecord.

Transfer (a `_fairshare_plan` entry + its progress; `one(fl)` in `Platform._fairshare_exchange` runs: startup b, FairShare flow, latency):
    partner_id       int
    src_node         str    the partner's node (bytes leave here)
    payload          float  bytes of the partner's payload
    latency          float  route latency (added once, after the bytes)
    hops             [[directed_link_id, bandwidth_MBps], ...]   flow.hops
    b                float  TCP start-up delay of the plan (0.0 unless HEROSIM_EXCHANGE_TCP)
    rest             float  bytes handed to FairShare.start (== payload)
    scale            float  FairShare rate scale (1.0)
    status           str    "done" | "latency_tail" | "active" | "startup" | "pending"
    startup_remaining float seconds of b still to run (status startup; b when pending; else 0.0)
    rem              float  bytes still to move at the instant: flow.rem projected as above (status active); rest when startup/pending; 0.0 when latency_tail/done
    rate             float  bytes/s the flow has at the instant (status active), else 0.0
    latency_remaining float seconds of the post-bytes latency timeout still to run (status latency_tail; latency when active/startup/pending; 0.0 done)

Invariants (`validate` checks them): serial mode, so in an ExchangeRecord the statuses read done* [current] pending*, with at most one
transfer in {latency_tail, active, startup}; read_remaining > 0 => all pending; `active` lists exactly the transfers with status "active", once each,
in FairShare.flows order; 0 <= rem <= rest; an active transfer has hops and rate > 0.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict

KEY = "flows"
VERSION = 1
STATUSES = ("done", "latency_tail", "active", "startup", "pending")
_CURRENT = ("latency_tail", "active", "startup")
_STAGES = ("input_io", "rendezvous", "cold", "ingress")


class Transfer(TypedDict):
    partner_id: int
    src_node: str
    payload: float
    latency: float
    hops: List[Tuple[str, float]]
    b: float
    rest: float
    scale: float
    status: str
    startup_remaining: float
    rem: float
    rate: float
    latency_remaining: float


class ExchangeRecord(TypedDict):
    owner_tid: int
    dst_node: str
    stage: str
    planned: bool
    read_remaining: float
    transfers: List[Transfer]


class FlowBlock(TypedDict):
    version: int
    model: str
    partners: str
    tcp: bool
    fairshare_last: float
    active: List[Tuple[int, int]]
    exchanges: List[ExchangeRecord]
    unplanned: List[int]


def validate(block: Dict[str, Any]) -> None:
    """Fail loudly when a stored `fidelity["flows"]` block breaks the schema or its invariants (module docstring)."""
    if block.get("version") != VERSION:
        raise ValueError(f"flow_snapshot: version {block.get('version')!r}, expected {VERSION}")
    if block.get("partners") != "serial":
        raise ValueError(f"flow_snapshot: partners {block.get('partners')!r}; only serial is defined")
    if not block.get("fairshare_last", 1.0) <= 0.0:
        raise ValueError(f"flow_snapshot: fairshare_last {block.get('fairshare_last')!r} is after the snapshot instant")
    by_owner: Dict[int, ExchangeRecord] = {}
    for ex in block["exchanges"]:
        tid = ex["owner_tid"]
        if tid in by_owner:
            raise ValueError(f"flow_snapshot: two exchange records for task {tid}")
        by_owner[tid] = ex
        if ex["stage"] not in _STAGES:
            raise ValueError(f"flow_snapshot: task {tid}: unknown stage {ex['stage']!r}")
        if ex["read_remaining"] < 0:
            raise ValueError(f"flow_snapshot: task {tid}: negative read_remaining")
        statuses = [t["status"] for t in ex["transfers"]]
        bad = [s for s in statuses if s not in STATUSES]
        if bad:
            raise ValueError(f"flow_snapshot: task {tid}: unknown status {bad[0]!r}")
        order = [0 if s == "done" else 1 if s in _CURRENT else 2 for s in statuses]
        if order != sorted(order) or order.count(1) > 1:
            raise ValueError(f"flow_snapshot: task {tid}: statuses {statuses} are not done* [current] pending* (serial)")
        if ex["read_remaining"] > 0 and any(s != "pending" for s in statuses):
            raise ValueError(f"flow_snapshot: task {tid}: a transfer has started while the pre-exchange read has {ex['read_remaining']} s left")
        for t in ex["transfers"]:
            if not (0.0 <= t["rem"] <= t["rest"] * (1 + 1e-12)):
                raise ValueError(f"flow_snapshot: task {tid} partner {t['partner_id']}: rem {t['rem']!r} outside [0, rest {t['rest']!r}]")
            if t["status"] == "active" and (not t["hops"] or not t["rate"] > 0):
                raise ValueError(f"flow_snapshot: task {tid} partner {t['partner_id']}: active without hops or a positive rate")
    listed = [tuple(a) for a in block["active"]]
    if len(set(listed)) != len(listed):
        raise ValueError("flow_snapshot: a flow is listed twice in `active`")
    want = {(ex["owner_tid"], i) for ex in block["exchanges"] for i, t in enumerate(ex["transfers"]) if t["status"] == "active"}
    if set(listed) != want:
        raise ValueError(f"flow_snapshot: `active` {sorted(listed)} is not the set of active transfers {sorted(want)}")
