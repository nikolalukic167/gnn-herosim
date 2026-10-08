"""physics_audit_v1 I11: what a live snapshot has to carry for co-simulation to reproduce live latency under R1.

`HEROSIM_SNAPSHOT_FIDELITY=1` (set for BOTH the live run that captures and the replay that applies) adds a
`fidelity` block to every live audit snapshot and makes `apply_live_snapshot_seed` / `start_simulation` honour it.
Off (the default), nothing here runs: no key is added to a snapshot and a snapshot without the key replays exactly
as before. The block carries the five things the original snapshot left out (found by `i11_replay.py`):

  1. KPA window history, panic state, replica birth causes and the phase of the next reconcile tick
  2. (replay side) the live autoscaler's settings and autoscaler class -- `live_run_params`, `swap_autoscaler`
  3. tasks a released replica holds between pop and release: ingress, cold start, rendezvous, input I/O, waiting for
     the compute lock, computing -- each resumed from the stage it was caught in (`GhostTask`)
  4. replicas still pulling their image (they stay uninitialised for the remaining pull, not ready at once)
  5. per-platform sandbox warmth (`previous_task`) and node image caches

All times in the block are relative to the snapshot instant (<= 0 for the past, >= 0 for the future); the replay's
clock starts at 0, so a relative time is its absolute replay time.
"""
from __future__ import annotations

import copy
import math
import os
from types import SimpleNamespace

from typing import Any, Dict, Generator, List, Optional, Set, Tuple

ENV = "HEROSIM_SNAPSHOT_FIDELITY"
VERSION = 1


def enabled() -> bool:
    raw = os.environ.get(ENV, "0")
    if raw not in ("0", "1"):
        raise ValueError(f"{ENV}={raw!r}; expected 0 or 1")
    return raw == "1"


def _rel(x: Any, now: float) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x - now if math.isfinite(x) else None


# ---------------------------------------------------------------------------------------------------------------
# replay parameters (2)
# ---------------------------------------------------------------------------------------------------------------
def live_run_params() -> Dict[str, Any]:
    """The autoscaler settings `executesimulation.main` passes for the live run in this environment: target
    concurrency (0.7 under kpa), policy-time-scaled keep-alive and reconcile tick. The oracle's hard-coded
    `KEEP_ALIVE` / `QUEUE_LENGTH` (30 / 100) are not these."""
    from src.executesimulation import _resolve_keep_alive, _resolve_queue_length
    from src.placement.constants import RECONCILE_INTERVAL
    from src.placement.scaleout import policy_time_scale

    scale = policy_time_scale()
    return {
        "keep_alive": _resolve_keep_alive(scale),
        "queue_length": _resolve_queue_length(None),
        "reconcile_interval": RECONCILE_INTERVAL if scale == 1.0 else RECONCILE_INTERVAL * scale,
    }


# ---------------------------------------------------------------------------------------------------------------
# capture (live side)
# ---------------------------------------------------------------------------------------------------------------
def _ghost_record(platform: Any, task: Any, stage_of_pool: str, now: float, fn: Optional[str],
                  rendezvous: bool) -> Dict[str, Any]:
    fid = getattr(task, "_fid", None)
    if fid is None:
        raise RuntimeError(f"fidelity capture: task {task.id} on {platform} has no stage markers; was the live run "
                           f"started with {ENV}=1?")
    if fid.get("unsupported"):
        raise RuntimeError(
            f"fidelity capture: task {task.id} is in an ingress path the resume does not model (ingress pipe or "
            "store-and-forward fabric); R1 uses the pipelined fabric only")
    ptype = platform.type["shortName"]
    nominal = float(task.type["executionTime"][ptype])
    from src.placement.exec_physics import realized_factor

    exec_seconds = nominal * realized_factor(platform, int(task.id))
    from src.placement.infrastructure import _audit_input_bytes

    rec: Dict[str, Any] = {
        "q": f"{platform.node.node_name}:{platform.id}", "tid": int(task.id), "fn": task.type["name"],
        "src": task.node_name, "exec": exec_seconds, "output": _output_seconds(task),
        "input_bytes": _audit_input_bytes(task), "pop": float(fid["pop"]) - now,
    }
    link_stage = fid.get("link_stage", "done")
    if stage_of_pool == "admitted":
        rec["io_remaining"] = _expected_input_seconds(platform, task)
        if link_stage == "net":
            rec.update(stage="ingress", link_stage="net", net_remaining=max(0.0, fid["net_end"] - now),
                       route=None, hold=0.0)
        elif link_stage == "wait":
            rec.update(stage="ingress", link_stage="wait", net_remaining=0.0, route=fid["route"], hold=fid["hold"])
        elif link_stage == "hold":
            rec.update(stage="ingress", link_stage="hold", net_remaining=0.0, route=fid["route"], hold=fid["hold"],
                       hold_remaining=max(0.0, fid["hold_end"] - now))
        else:
            rec.update(stage="cold", cold_remaining=max(0.0, fid.get("cold_end", now) - now))
        return rec
    # in the replica's inflight list: past ingress and cold start
    if rendezvous:
        rec.update(stage="rendezvous", io_remaining=0.0)
    elif fid.get("compute_end") is not None:
        rec.update(stage="compute", compute_remaining=max(0.0, fid["compute_end"] - now))
    elif fid.get("io_end") is not None and now >= fid["io_end"]:
        rec.update(stage="lock_wait", order=fid["io_end"] - now)
    elif fid.get("io_end") is not None:
        rec.update(stage="input_io", io_remaining=fid["io_end"] - now, order=fid["io_end"] - now)
    else:
        rec.update(stage="input_io", io_remaining=0.0, order=0.0)
    return rec


def _expected_input_seconds(platform: Any, task: Any) -> float:
    """The input stage a task popped but not yet started will run (`Platform._serve_task`): read the input from the
    node's remote storage, plus the peer exchange to partners already placed. The same terms, evaluated now. A
    partner not placed yet is a rendezvous whose length no snapshot holds; it contributes 0 here."""
    if task.dependencies:
        raise RuntimeError(f"fidelity capture: task {task.id} has dependencies; the resumed input stage models "
                           "single-task applications")
    node = platform.node
    storage = next((st for st in node._fid_storages if st.type.get("remote")), None)
    if storage is None:
        raise RuntimeError(f"fidelity capture: {node.node_name} has no remote storage to read input from")
    state = task.type["stateSize"][task.application.type["name"]]
    speed = min(storage.type["throughput"]["read"], node.network["bandwidth"])
    seconds = float(state["input"]) / (speed * 1024 * 1024) + float(storage.type["latency"]["read"])
    try:
        seconds += float(platform._peer_exchange_time(task))
    except RuntimeError:
        pass  # a partner with no placement yet: the live task rendezvouses, which is not capturable
    return seconds


def _output_seconds(task: Any) -> float:
    from src.placement.infrastructure import _output_estimate_seconds

    return float(_output_estimate_seconds(task))


def capture(scheduler: Any, system_state: Any, batch_tasks: List[Any]) -> Dict[str, Any]:
    """The `fidelity` block for a snapshot taken now, inside the scheduler's decision."""
    from src.placement.infrastructure import REPLICA_RELEASE

    if not REPLICA_RELEASE:
        raise RuntimeError("fidelity capture models released replicas (HEROSIM_REPLICA_RELEASE=1); with held "
                           "replicas the running task is current_task and the legacy backlog already covers it")
    env = scheduler.env
    now = float(env.now)
    replica_fn: Dict[Any, str] = {}
    for fn, reps in system_state.replicas.items():
        for _node, platform in reps:
            replica_fn[platform] = fn

    platforms: Dict[str, Any] = {}
    ghosts: List[Dict[str, Any]] = []
    for node in scheduler.nodes.items:
        for p in node.platforms.items:
            key = f"{node.node_name}:{p.id}"
            prev = p.previous_task.type.get("name") if p.previous_task is not None else None
            rec: Dict[str, Any] = {
                "initialized": bool(p.initialized.triggered), "prev": prev, "idle_since": _rel(p.idle_since, now), "last_started": _rel(p.last_started, now),
                "last_allocated": _rel(p.last_allocated, now), "last_removed": _rel(p.last_removed, now),
            }
            platforms[key] = rec
            rv = getattr(p, "rendezvous_procs", None) or {}
            if p.admitted is not None:
                ghosts.append(_ghost_record(p, p.admitted, "admitted", now, replica_fn.get(p), False))
            for t in list(p.inflight):
                ghosts.append(_ghost_record(p, t, "inflight", now, replica_fn.get(p), t in rv))

    pulls = _capture_pulls(scheduler, platforms, now)
    for p, fn in replica_fn.items():
        if not p.initialized.triggered and platforms[f"{p.node.node_name}:{p.id}"].get("pull_remaining") is None:
            raise RuntimeError(f"fidelity capture: replica {p} of {fn} is uninitialised but no image pull is "
                               "recorded for it; it would replay as ready")

    autoscaler = getattr(scheduler, "autoscaler", None)
    kpa_block = None
    kpa = getattr(autoscaler, "kpa", None)
    if kpa is not None:
        batch_ids = {int(t.id) for t in batch_tasks}
        functions = {}
        last_tick = None
        for fn, st in kpa.functions.items():
            functions[fn] = {
                "samples": [[float(t) - now, float(v)] for t, v in st.samples],
                "first_sample": _rel(st.first_sample, now), "last_traffic": _rel(st.last_traffic, now),
                "panic_since": _rel(st.panic_since, now), "panic_last_over": _rel(st.panic_last_over, now),
                "max_panic_pods": int(st.max_panic_pods),
            }
            if st.samples:
                last_tick = max(last_tick if last_tick is not None else -math.inf, float(st.samples[-1][0]))
        pending_extra = {
            fn: sum(1 for t in lst if not t.scheduled.triggered and int(t.id) not in batch_ids)
            for fn, lst in (getattr(autoscaler, "_kpa_pending", None) or {}).items()
        }
        next_wake, wake_kind = _autoscaler_wake(env, autoscaler, now)
        kpa_block = {
            "next_wake": next_wake, "wake_kind": wake_kind,
            "functions": functions,
            "born": [[fn, int(nid), int(pid), float(b) - now, autoscaler._replica_cause.get((fn, nid, pid))]
                     for (fn, nid, pid), b in autoscaler._replica_born.items()],
            "tick_interval": float(autoscaler.reconcile_interval),
            "last_tick": None if last_tick is None else last_tick - now,
            "pending_outside_batch": pending_extra,
        }

    from src.placement.live_audit import orchestrator_of

    def event_of(t: Any, q: Optional[str] = None) -> Dict[str, Any]:
        if getattr(t, "is_internal", False):
            raise RuntimeError(f"fidelity capture: internal task {t.id} in the system")
        if len(t.application.tasks) != 1:
            raise RuntimeError(f"fidelity capture: task {t.id} belongs to a {len(t.application.tasks)}-task "
                               "application; replay events are single-task")
        rec = {"gid": int(t.id), "fn": t.type["name"], "src": t.node_name,
               "app": copy.deepcopy(t.application.type), "qos": copy.deepcopy(t.application.qos)}
        if q is not None:
            rec["q"] = q
        return rec

    queued: List[Dict[str, Any]] = []
    for node in scheduler.nodes.items:
        for p in node.platforms.items:
            for t in list(p.queue.items):
                rec = event_of(t, f"{node.node_name}:{p.id}")
                rec["node_id"], rec["platform_id"] = int(node.id), int(p.id)
                queued.append(rec)
    batch = [event_of(t) for t in batch_tasks]

    orch = orchestrator_of(scheduler)
    table = (getattr(orch, "peer_exchange", None) or {}) if orch is not None else {}
    by_id = (getattr(orch, "task_by_id", None) or {}) if orch is not None else {}
    included = {r["gid"] for r in queued} | {r["gid"] for r in batch}
    pairs: List[List[Any]] = []
    peers: Dict[str, List[List[Any]]] = {}
    for gid in sorted(included):
        outside = []
        for other, payload in sorted((table.get(gid) or {}).items()):
            if other in included:
                if gid < other:
                    pairs.append([gid, int(other), float(payload)])
                continue
            peer = by_id.get(other)
            node_name = None
            if peer is not None:
                node_name = (peer.platform.node.node_name if getattr(peer, "platform", None) is not None
                             else getattr(peer, "planned_node_name", None))
            outside.append([int(other), node_name, float(payload)])
        if outside:
            peers[str(gid)] = outside

    disk: Dict[str, List[List[Any]]] = {}
    for node in scheduler.nodes.items:
        locals_ = [s for s in node._fid_storages if not s.type.get("remote")]
        entries = [[idx, short, tt["name"]] for idx, s in enumerate(locals_) for short, tt in s.functions_cache]
        if entries:
            disk[node.node_name] = entries

    return {"version": VERSION, "platforms": platforms, "ghosts": ghosts, "kpa": kpa_block, "disk": disk, "pulls": pulls,
            "batch": batch, "queued": queued, "pairs": pairs, "peers": peers}


def _autoscaler_wake(env: Any, autoscaler: Any, now: float) -> Tuple[Optional[float], str]:
    """When the autoscaler process wakes next, read from the event it is suspended on. Its loop calls
    `env.step()` itself (it processes the next queued event, whatever it is, and can advance the clock), so ticks
    are not `last_tick + interval`; the only reliable phase is the pending Timeout. A process suspended on the
    mutex (queued behind the decision in progress) has no wake time: its tick follows that decision."""
    run = getattr(autoscaler, "run", None)
    target = getattr(run, "target", None)
    if target is None:
        return None, "none"
    kind = type(target).__name__
    if kind != "Timeout":
        return None, kind
    queue = getattr(env, "_queue", None)
    if queue is None:
        raise RuntimeError("fidelity capture: this SimPy has no Environment._queue; cannot read the autoscaler's wake")
    for t, _prio, _eid, event in queue:
        if event is target:
            return float(t) - now, kind
    raise RuntimeError("fidelity capture: the autoscaler's Timeout is not in the event queue")


def _capture_pulls(scheduler: Any, platforms: Dict[str, Any], now: float) -> List[Dict[str, Any]]:
    """Every unfinished `initialize_replica` call, in the order the node's storage will serve them.

    A pull holds its node's local storage for its whole duration and the next call on that node waits for it (a
    FilterStore get). A call that has not started pulling is priced by playing the node's queue forward: the holder's
    remaining time, then each waiter in the order it asked, a waiter finding the image already cached (by an earlier
    call in the queue) paying only the wait. Returns one record per call: its own hold time and its rank. Sets
    `pull_remaining` on a platform not yet initialised to when its first call finishes (that call initialises it;
    a duplicate call's `succeed()` is swallowed by the live code)."""
    from src.placement.warmth import PLATFORM_REUSE_V1, needs_image_pull

    physics = getattr(scheduler.env, "warmth_physics", PLATFORM_REUSE_V1)
    out: List[Dict[str, Any]] = []
    for node in scheduler.nodes.items:
        calls = [c for c in node.__dict__.get("_fid_pull_calls", []) if not c["done"]]
        if not calls:
            continue
        locals_ = [s for s in node._fid_storages if not s.type.get("remote")]
        if len(locals_) != 1:
            raise RuntimeError(f"fidelity capture: {node.node_name} has {len(locals_)} local storages; the pull "
                               "queue is modelled for exactly one")
        storage = locals_[0]
        holders = sorted((c for c in calls if c["end"] is not None), key=lambda c: (c["end"], c["init_start"]))
        waiters = sorted((c for c in calls if c["end"] is None),
                         key=lambda c: (c["init_start"], c["platform"].id))
        cached = {(short, tt["name"]) for short, tt in storage.functions_cache}
        cursor = now
        rank = 0
        finish: Dict[int, float] = {}

        def emit(call: Dict[str, Any], own: float, estimated: bool) -> None:
            nonlocal rank
            p = call["platform"]
            out.append({"q": f"{node.node_name}:{p.id}", "own": own, "rank": rank, "estimated": estimated,
                        "node": node.node_name, "short": p.type["shortName"], "fn": call["fn"]})
            rank += 1
            if not p.initialized.triggered:
                done_at = cursor - now
                finish[p.id] = min(finish.get(p.id, math.inf), done_at)

        for c in holders:
            own = max(0.0, float(c["end"]) - now)
            cursor = max(cursor, float(c["end"]))
            emit(c, own, False)
        for c in waiters:
            p, fn = c["platform"], c["fn"]
            short = p.type["shortName"]
            task_type = scheduler.data.task_types[fn]
            own = 0.0
            if (short, fn) not in cached and needs_image_pull(physics, p, node, task_type):
                size = task_type["imageSize"][short]
                speed = min(storage.type["throughput"]["write"], node.network["bandwidth"])
                own = float(size / (speed / 1024) + storage.type["latency"]["write"])
                cached.add((short, fn))
            cursor += own
            emit(c, own, True)
        for pid, remaining in finish.items():
            platforms[f"{node.node_name}:{pid}"]["pull_remaining"] = max(0.0, remaining)
    return out


# ---------------------------------------------------------------------------------------------------------------
# apply (replay side)
# ---------------------------------------------------------------------------------------------------------------
def uninitialized_keys(fidelity: Optional[Dict[str, Any]]) -> Set[Tuple[str, int]]:
    """Platforms the seed must leave uninitialised: a replica whose image pull has not finished, and every free
    platform (releasing a replica gives it a fresh, untriggered event). The base seed triggers every platform it is
    told about, which would make a replica the replay's autoscaler creates on a free platform ready at once."""
    out: Set[Tuple[str, int]] = set()
    for key, rec in ((fidelity or {}).get("platforms") or {}).items():
        if not rec["initialized"]:
            node, plat = key.rsplit(":", 1)
            out.add((node, int(plat)))
    return out


class GhostTask:
    """A live task the replay resumes. Shaped like what the platform bookkeeping reads from a Task."""

    def __init__(self, rec: Dict[str, Any], task_type: Dict[str, Any]):
        self.id = -1 - int(rec["tid"])
        self.type = task_type
        self.node_name = rec["src"]
        self.rec = rec
        self.held: List[Tuple[str, Any]] = []  # (link key, request) already taken at apply time
        # what state_capture / scheduling_cost read off a platform's current_task
        self.cold_started = False
        self.started_time = None
        self.arrived_time = None
        self.application = None


def _pull(env: Any, node: Any, platform: Any, request: Any, own: float, image: Tuple[str, Dict[str, Any]]) -> Generator:
    """An image pull in progress: it holds the node's local storage (checked out of the FilterStore) for its whole
    duration, so every task on the node that needs storage waits for it, then the replica becomes ready. The
    request was issued at apply time, in the order the live pulls queued."""
    storage = yield request
    if own > 0:
        # a pull caches its image when it starts, so a call behind it in the queue finds it there
        storage.store_function(image[0], image[1])
        yield env.timeout(own)
    yield node.storage.put(storage)
    if not platform.initialized.triggered:
        platform.initialized.succeed()
    platform._fid_pull_pending = not platform.initialized.triggered


def _ingress(platform: Any, ghost: GhostTask) -> Generator:
    """The rest of a popped task's pipelined ingress, as platform_process runs it: propagation, then every link on
    the route taken in sorted key order and held together for one transmission at the bottleneck. Hold-stage and
    wait-stage ghosts took their first pipe requests at apply time (so queue order does not depend on which
    platform process starts first); this finishes the rest."""
    env, rec = platform.env, ghost.rec
    fabric = platform.node.fabric
    stage = rec["link_stage"]
    if stage == "net":
        if rec["net_remaining"] > 0:
            yield env.timeout(rec["net_remaining"])
        route = fabric.hops(rec["src"], platform.node.node_name) if fabric is not None else []
        keys = sorted({k for k, _bw in route})
        hold = rec["input_bytes"] / (min(bw for _k, bw in route) * 1024.0 * 1024.0) if route and rec["input_bytes"] else 0.0
    else:
        keys = sorted(set(rec["route"]))
        hold = rec["hold_remaining"] if stage == "hold" else rec["hold"]
    try:
        if hold > 0:
            held = dict(ghost.held)
            for key in keys:
                req = held.get(key)
                if req is None:
                    req = fabric.pipe(key).request()
                    ghost.held.append((key, req))
                yield req
            yield env.timeout(hold)
    finally:
        for key, req in ghost.held:
            fabric.pipe(key).release(req)
        ghost.held = []


def _resume_admitted(ghost: GhostTask) -> Any:
    def run(platform: Any) -> Generator:
        from src.placement.warmth import sandbox_is_warm

        env, rec = platform.env, ghost.rec
        if rec["stage"] == "ingress":
            yield from _ingress(platform, ghost)
            cold = 0.0 if sandbox_is_warm(platform, ghost.type) else float(
                ghost.type["coldStartDuration"][platform.type["shortName"]])
        else:
            cold = rec["cold_remaining"]
        if cold > 0:
            yield env.timeout(cold)
        platform.previous_task = SimpleNamespace(type={"name": rec["fn"]})
        platform.inflight.append(ghost)
        platform.admitted = None
        ghost.rec = dict(rec, stage="input_io", io_remaining=rec.get("io_remaining", 0.0))
        env.process(_serve(platform, ghost))

    return run


def _serve(platform: Any, ghost: GhostTask) -> Generator:
    """The rest of _serve_task for a resumed task: remaining I/O, then the compute lock for execution and output,
    then release."""
    env, rec = platform.env, ghost.rec
    stage = rec["stage"]
    if stage in ("input_io", "rendezvous") and rec.get("io_remaining", 0.0) > 0:
        yield env.timeout(rec["io_remaining"])
    compute = platform.compute_lock.request()
    yield compute
    platform.current_task = ghost
    execute = rec["compute_remaining"] if stage == "compute" else rec["exec"]
    if execute > 0:
        yield env.timeout(execute)
    # output: the node's local storage is fetched (it waits while a pull holds it), then written
    storage = yield platform.node.storage.get(lambda st: not st.type.get("remote"))
    yield platform.node.storage.put(storage)
    if rec["output"] > 0:
        yield env.timeout(rec["output"])
    platform.previous_task = SimpleNamespace(type={"name": rec["fn"]})
    platform.current_task = None
    platform.compute_lock.release(compute)
    platform.inflight.remove(ghost)
    if not platform.inflight:
        platform.idle_since = env.now


def apply_platforms(plat_map: Dict[Tuple[str, int], Tuple[Any, Any]], simulation_data: Any, env: Any,
                    fidelity: Dict[str, Any]) -> None:
    """After the base seed: warmth and timestamps per platform, pulls in progress, node image caches, and the
    tasks held by released replicas."""
    if int(fidelity.get("version", 0)) != VERSION:
        raise RuntimeError(f"fidelity block version {fidelity.get('version')!r}; this build reads {VERSION}")
    for key, rec in fidelity["platforms"].items():
        node_name, plat_id = key.rsplit(":", 1)
        found = plat_map.get((node_name, int(plat_id)))
        if found is None:
            continue
        node, plat = found
        plat.previous_task = None if rec["prev"] is None else SimpleNamespace(type={"name": rec["prev"]})
        for attr in ("idle_since", "last_started", "last_allocated", "last_removed"):
            if rec.get(attr) is not None:
                setattr(plat, attr, float(rec[attr]))

    for node_name, entries in (fidelity.get("disk") or {}).items():
        node = next(n for (nn, _pid), (n, _p) in plat_map.items() if nn == node_name)
        locals_ = [s for s in node._fid_storages if not s.type.get("remote")]
        for idx, short, fn in entries:
            if not locals_[idx].store_function(short, simulation_data.task_types[fn]):
                raise RuntimeError(f"fidelity apply: node {node_name} cannot hold its captured image {short}/{fn}")

    for pull in sorted(fidelity.get("pulls") or [], key=lambda x: (x["node"], x["rank"])):
        node, plat = plat_map[(pull["q"].rsplit(":", 1)[0], int(pull["q"].rsplit(":", 1)[1]))]
        request = node.storage.get(lambda st: not st.type.get("remote"))
        plat._fid_pull_pending = True  # the determined scheduler may place on it before the pull ends
        env.process(_pull(env, node, plat, request, float(pull["own"]),
                          (pull["short"], simulation_data.task_types[pull["fn"]])))

    def order(g: Dict[str, Any]) -> Tuple:
        ingress = g["stage"] == "ingress"
        # hold-stage pipes first, then wait-stage in pop order: the order the live pipe queues had
        return (0 if ingress and g["link_stage"] == "hold" else 1 if ingress and g["link_stage"] == "wait" else 2,
                g.get("pop", 0.0), g["q"], 0 if g["stage"] == "compute" else 1, g.get("order", 0.0), g["tid"])

    for rec in sorted(fidelity.get("ghosts") or [], key=order):
        node_name, plat_id = rec["q"].rsplit(":", 1)
        node, plat = plat_map[(node_name, int(plat_id))]
        ghost = GhostTask(rec, simulation_data.task_types[rec["fn"]])
        if rec["stage"] in ("ingress", "cold"):
            if plat.admitted is not None:
                raise RuntimeError(f"fidelity apply: two admitted tasks on {rec['q']}")
            plat.admitted = ghost
            plat._fid_resume = _resume_admitted(ghost)
            if rec["stage"] == "ingress" and rec["link_stage"] in ("hold", "wait") and rec["hold"] > 0:
                keys = sorted(set(rec["route"]))
                for key in (keys if rec["link_stage"] == "hold" else keys[:1]):
                    ghost.held.append((key, node.fabric.pipe(key).request()))
        else:
            plat.inflight.append(ghost)
            env.process(_serve(plat, ghost))


class PeerStub:
    """A replayed task's partner outside the snapshot's task set. The orchestrator's peer exchange charge reads
    only where it runs; a partner the live run had not placed yet stays unplaced (its `scheduled` event pending)
    until `_place_later` fires it."""

    def __init__(self, env: Any, gid: int, node_name: Optional[str]):
        self.id = STUB_BASE + gid
        self.platform = None
        self.planned_node_name = node_name
        self.scheduled = env.event()
        if node_name is not None:
            self.scheduled.succeed()


def _place_later(env: Any, stub: PeerStub, delay: float, node_name: str) -> Generator:
    if delay > 0:
        yield env.timeout(delay)
    stub.planned_node_name = node_name
    stub.scheduled.succeed()


STUB_BASE = 10_000_000


def local_ids(fidelity: Dict[str, Any]) -> Dict[int, int]:
    """Live task id -> replay task id: queued tasks first (they were placed before the batch), then the batch."""
    order = [r["gid"] for r in fidelity["queued"]] + [r["gid"] for r in fidelity["batch"]]
    return {gid: i for i, gid in enumerate(order)}


def replay_workload(fidelity: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[int, Tuple[int, int]], Dict[int, int]]:
    """The co-sim workload for a snapshot: one event per queued task and per batch task, all at t = 0, with the
    peer pairs among them. Returns (workload, forced placements of the queued tasks, live id -> replay id). The
    caller adds the batch's own placements. Queued tasks go through the real pipeline from their live platform's
    queue, instead of being compressed into a virtual backlog."""
    ids = local_ids(fidelity)
    events = []
    forced: Dict[int, Tuple[int, int]] = {}
    for rec in fidelity["queued"] + fidelity["batch"]:
        events.append({"timestamp": 0.0, "application": copy.deepcopy(rec["app"]), "qos": copy.deepcopy(rec["qos"]),
                       "node_name": rec["src"]})
    for rec in fidelity["queued"]:
        forced[ids[rec["gid"]]] = (int(rec["node_id"]), int(rec["platform_id"]))
    workload: Dict[str, Any] = {"rps": len(events), "duration": 1, "events": events}
    if fidelity["pairs"]:
        workload["peer_exchange"] = [[ids[a], ids[b], payload] for a, b, payload in fidelity["pairs"]]
    return workload, forced, ids


def apply_orchestrator(orchestrator: Any, fidelity: Dict[str, Any]) -> int:
    """Give each replayed task its partners outside the snapshot's task set, as stubs that only know where they
    run. A partner the live run had not placed yet is a live rendezvous: the task waits until it is. A snapshot
    alone cannot say when (that is a later scheduling decision), so such a partner is placed only if the block
    carries `future` = {partner id: [seconds from the snapshot, node name]}, which a test harness fills from the
    live run's own continuation. Returns the number of partners left unplaced forever (the replay would wait on
    them for ever; the caller must treat that replay as failed)."""
    env = orchestrator.env
    ids = local_ids(fidelity)
    future = fidelity.get("future") or {}
    stubs: Dict[int, PeerStub] = {}
    never = 0
    for gid, row in (fidelity.get("peers") or {}).items():
        local = ids[int(gid)]
        for other, node_name, payload in row:
            stub = stubs.get(int(other))
            if stub is None:
                stub = stubs[int(other)] = PeerStub(env, int(other), node_name)
                orchestrator.task_by_id[stub.id] = stub
                if node_name is None:
                    if str(other) in future:
                        delay, where = future[str(other)]
                        env.process(_place_later(env, stub, float(delay), where))
                    else:
                        never += 1
            orchestrator.peer_exchange.setdefault(local, {})[stub.id] = float(payload)
    return never


def apply_autoscaler(autoscaler: Any, fidelity: Dict[str, Any]) -> None:
    """KPA window history, panic state, replica birth causes and the tick phase."""
    block = fidelity.get("kpa")
    kpa = getattr(autoscaler, "kpa", None)
    if block is None and kpa is None:
        return
    if (block is None) != (kpa is None):
        raise RuntimeError("fidelity apply: snapshot and replay disagree on HEROSIM_SCALEOUT (kpa on one side only)")
    from collections import deque

    from src.placement.scaleout import KpaFunctionState

    for fn, rec in block["functions"].items():
        st = kpa.functions.setdefault(fn, KpaFunctionState())
        st.samples = deque((float(t), float(v)) for t, v in rec["samples"])
        st.first_sample = rec["first_sample"]
        st.last_traffic = -math.inf if rec["last_traffic"] is None else rec["last_traffic"]
        st.panic_since = rec["panic_since"]
        st.panic_last_over = rec["panic_last_over"]
        st.max_panic_pods = int(rec["max_panic_pods"])
    for fn, nid, pid, born, cause in block["born"]:
        autoscaler._replica_born[(fn, nid, pid)] = born
        if cause is not None:
            autoscaler._replica_cause[(fn, nid, pid)] = cause
    # the first replayed tick follows the scheduler's first decision (a live tick suspended on the mutex waits for
    # the decision in progress) and not before the captured wake time
    autoscaler._kpa_start_gate = autoscaler.env.event()
    if block["next_wake"] is not None:
        autoscaler._kpa_start_delay = block["next_wake"]


def swap_autoscaler(autoscaler_type: Any) -> Any:
    """Under the fidelity replay the co-sim `determined` arm runs the autoscaler the live run used (kpa_scaleout_v1
    A4 gave every live arm the GNN-family one)."""
    from src.placement.scaleout import shared_autoscaler

    if not (enabled() and shared_autoscaler()):
        return autoscaler_type
    from src.policy.determined.autoscaler import DeterminedAutoscaler
    from src.policy.gnn.autoscaler import KnativeAutoscaler as GNNAutoscaler

    return GNNAutoscaler if autoscaler_type is DeterminedAutoscaler else autoscaler_type
