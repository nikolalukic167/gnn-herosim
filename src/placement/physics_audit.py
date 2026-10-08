"""physics_audit_v1: an opt-in trace of what the simulation did, for the invariant checks I1-I12.

`HEROSIM_AUDIT_TRACE=<path>` turns it on and writes one JSON object per line to `<path>`. Unset (the default),
`AUDIT` is None, every hook is a single `is not None` test, and nothing is recorded, scheduled or changed.

The recorder only reads simulation state and schedules nothing: it adds no SimPy event, process or timeout. That
is load-bearing. The autoscaler loops call `env.step()` themselves (autoscaler.py, "Next event"), so one extra
scheduled event is consumed in place of a real one and changes the result: a 0.1 s occupancy sampler tried first
moved CD's total RTT on 9483/g0/x20 from 1288.5 s to 1077.7 s. All that is added to the simulation are attributes
on Task and Platform objects (`_audit_*`) that carry stage times to the row written when the task finishes, and
one callback on `Platform.initialized`. `replay_identity.py` checks trace on vs off.

Row kinds (field `k`):
  header      run configuration: physics flags, policy time constants, KPA config        (I12)
  arrive      a task handed to the scheduler                                               (I1, I7)
  enqueue     a task put on a platform's queue                                             (I1)
  svc         one served task with every stage time, warmth and the replica it ran on      (I1, I4, I9)
  xfer        one transfer: ingress over the fabric, or a peer / parent->child payload     (I2, I3)
  rep_up      replica created (cause, memory)          rep_init  replica initialised      (I4-I6, I9)
  rep_down    replica released (already_removed = drained)
  kpa         one KPA decision per function per tick                                       (I5)
  i11         ground-truth state at a live decision next to what a live snapshot carries  (I11)
  decision    sim time before/after one batch decision and its wall-clock duration        (I10)
  end         task conservation counts at the end of the run                               (I7)
"""
from __future__ import annotations

import atexit
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional

AUDIT_ENV = "HEROSIM_AUDIT_TRACE"
_FLUSH_EVERY = 20000

# environment flags whose value is physics (recorded in the header so a checker never infers them)
PHYSICS_ENV = (
    "HEROSIM_TRANSFER_MODEL", "HEROSIM_REPLICA_RELEASE", "HEROSIM_SCALEOUT", "HEROSIM_POLICY_TIME_SCALE",
    "HEROSIM_KEEP_ALIVE", "HEROSIM_QUEUE_LENGTH", "HEROSIM_PEER_EXCHANGE", "HEROSIM_SERVER_ONLY_REPLICAS",
    "HEROSIM_WARMTH_PHYSICS", "HEROSIM_DATA_LOCALITY", "HEROSIM_INFLIGHT_CAPTURE", "HEROSIM_MAX_EVENTS",
    "COSIM_AUTOSCALER_RECONCILE_INTERVAL", "GNN_CAPTURE_DATASET_STATE",
)


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


class AuditRecorder:
    def __init__(self, path: str):
        self.path = path
        self._rows: List[str] = []
        self._fh = open(path, "w")
        self._closed = False
        atexit.register(self.close)

    # ---- plumbing -------------------------------------------------------------------------------------------
    def emit(self, _row_kind: str, _t: Optional[float], **fields: Any) -> None:
        row = {"k": _row_kind, "t": _f(_t)}
        row.update(fields)
        self._rows.append(json.dumps(row, separators=(",", ":"), default=str))
        if len(self._rows) >= _FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if self._rows and not self._closed:
            self._fh.write("\n".join(self._rows) + "\n")
            self._fh.flush()
            self._rows = []

    def close(self) -> None:
        if not self._closed:
            self.flush()
            self._fh.close()
            self._closed = True

    # ---- run configuration (I12) -----------------------------------------------------------------------------
    def header(self, orchestrator: Any) -> None:
        autoscaler = getattr(orchestrator, "autoscaler", None)
        policy = getattr(orchestrator, "policy", None)
        kpa = getattr(autoscaler, "kpa", None)
        nodes = []
        for node in orchestrator.nodes.items:
            nodes.append({"node": node.node_name, "id": int(node.id), "memory": _f(node.memory),
                          "available_memory": _f(node.available_memory),
                          "platforms": len(node.platforms.items)})
        self.emit(
            "header", orchestrator.env.now,
            env={k: os.environ.get(k) for k in PHYSICS_ENV},
            policy=type(orchestrator.scheduler).__name__,
            autoscaler=type(autoscaler).__name__ if autoscaler is not None else None,
            keep_alive=_f(getattr(policy, "keep_alive", None)),
            reconcile_interval=_f(getattr(policy, "reconcile_interval", None)),
            queue_length=getattr(policy, "queue_length", None),
            kpa=kpa.config.describe() if kpa is not None else None,
            warmth_physics=getattr(orchestrator.env, "warmth_physics", None),
            fabric=any(getattr(n, "fabric", None) is not None for n in orchestrator.nodes.items),
            ingress_pipe=any(getattr(n, "ingress_pipe", None) is not None for n in orchestrator.nodes.items),
            nodes=nodes,
        )

    # ---- initial replica set ---------------------------------------------------------------------------------
    def initial_replicas(self, env: Any, system_state: Any, data: Any) -> None:
        """The replica set the run starts with, which no rep_up row covers."""
        for fn, reps in system_state.replicas.items():
            for node, platform in sorted(reps, key=lambda np: (np[0].node_name, np[1].id)):
                self.replica_up(env, fn, node, platform, "initial",
                                data.task_types[fn]["memoryRequirements"][platform.type["shortName"]])

    # ---- tasks -----------------------------------------------------------------------------------------------
    def arrive(self, env: Any, task: Any) -> None:
        self.emit("arrive", env.now, task=int(task.id), type=task.type["name"], src=task.node_name)

    def enqueue(self, env: Any, task: Any, platform: Any) -> None:
        self.emit("enqueue", env.now, task=int(task.id), q=f"{platform.node.node_name}:{platform.id}")

    def served(self, env: Any, task: Any, platform: Any, release: bool) -> None:
        g = lambda name: _f(getattr(task, name, None))  # noqa: E731
        self.emit(
            "svc", env.now, task=int(task.id), type=task.type["name"], q=f"{platform.node.node_name}:{platform.id}",
            ptype=platform.type["shortName"], release=bool(release),
            pop=g("_audit_pop"), arrived=g("arrived_time"), cold_end=g("_audit_cold_end"),
            io_start=g("_audit_io_start"), storage_got=g("_audit_storage_got"),
            rendezvous_end=g("_audit_rendezvous_end"), io_end=g("_audit_io_end"), started=g("started_time"), compute_start=g("_audit_compute_start"),
            exec_end=g("_audit_exec_end"), done=_f(env.now), scheduled=g("scheduled_time"),
            dispatched=g("dispatched_time"), cold=_f(task.cold_start_time), cold_started=bool(task.cold_started),
            warm=bool(getattr(task, "_audit_warm", False)), prev_type=getattr(task, "_audit_prev_type", None),
            inflight_at_pop=getattr(task, "_audit_inflight_at_pop", None),
            rendezvous=_f(task.peer_rendezvous_wait), exchange=_f(task.peer_exchange_time),
            exec=_f(task.execution_time), incarnation=getattr(task, "_audit_incarnation", None),
        )

    # ---- transfers (I2, I3) ----------------------------------------------------------------------------------
    def failed(self, env: Any, task: Any, platform: Any, reason: str) -> None:
        self.emit("fail", env.now, task=int(task.id), type=task.type["name"],
                  q=f"{platform.node.node_name}:{platform.id}", reason=reason,
                  scheduled=_f(getattr(task, "scheduled_time", None)), dispatched=_f(getattr(task, "dispatched_time", None)))

    def transfer(self, env: Any, kind: str, task: Any, src: str, dst: str, size_bytes: float,
                 route: Iterable, model: str, charged: float, latency: Optional[float] = None,
                 wait: Optional[float] = None, store_forward: bool = False,
                 route_latency: Optional[float] = None) -> None:
        self.emit("xfer", env.now, kind=kind, task=int(task.id), src=src, dst=dst, bytes=_f(size_bytes),
                  route=[[k, _f(bw)] for k, bw in route], model=model, charged=_f(charged), latency=_f(latency),
                  wait=_f(wait), sf=bool(store_forward), route_latency=_f(route_latency))

    # ---- replicas (I4-I6, I9) --------------------------------------------------------------------------------
    def replica_up(self, env: Any, function_name: str, node: Any, platform: Any, cause: str,
                   memory_required: float) -> None:
        platform._audit_incarnation = getattr(platform, "_audit_incarnation", 0) + 1
        self.emit("rep_up", env.now, fn=function_name, q=f"{node.node_name}:{platform.id}", cause=cause,
                  mem=_f(memory_required), node_mem=_f(node.memory), node_avail=_f(node.available_memory),
                  inc=platform._audit_incarnation)
        event = platform.initialized
        q = f"{node.node_name}:{platform.id}"
        inc = platform._audit_incarnation
        if event.triggered:
            self.emit("rep_init", env.now, fn=function_name, q=q, inc=inc, immediate=True)
        else:
            event.callbacks.append(
                lambda _ev: self.emit("rep_init", env.now, fn=function_name, q=q, inc=inc, immediate=False))

    def replica_down(self, env: Any, function_name: str, node: Any, platform: Any, memory_released: float,
                     already_removed: bool, inflight: int) -> None:
        self.emit("rep_down", env.now, fn=function_name, q=f"{node.node_name}:{platform.id}",
                  mem=_f(memory_released), node_avail=_f(node.available_memory), drained=bool(already_removed),
                  inflight=int(inflight), inc=getattr(platform, "_audit_incarnation", 0))

    def memory_refusal(self, env: Any, function_name: str, count: int) -> None:
        self.emit("mem_refuse", env.now, fn=function_name, n=int(count))

    # ---- KPA (I5) --------------------------------------------------------------------------------------------
    def kpa_tick(self, env: Any, function_name: str, observed: float, current: int, ready: int, decision: Any,
                 load_live: int, occupancy: Dict[str, int]) -> None:
        """`occupancy`: tasks each of the function's replicas holds right now (queued, admitted, in flight, running);
        the I1 check compares its time average with lambda x W from the task rows."""
        self.emit("kpa", env.now, fn=function_name, obs=_f(observed), cur=int(current), ready=int(ready),
                  desired=int(decision.desired), panic=bool(decision.panicking), stable=_f(decision.stable_avg),
                  load_live=int(load_live), occ=dict(occupancy))

    # ---- pool conservation (I13) -----------------------------------------------------------------------------
    def pool(self, env: Any, system_state: Any, draining: Any) -> None:
        """Per node at a KPA tick: [free platforms, platforms owned by a replica, platforms being drained, the node's own
        `available_platforms` counter]. With the node's platform count (header) the three must add up."""
        names = {}
        def slot(node):
            return names.setdefault(node.node_name, [0, 0, 0, None])
        for node, platforms in system_state.available_resources.items():
            row = slot(node)
            row[0] = len(platforms)
            row[3] = int(node.available_platforms)
        for replicas in system_state.replicas.values():
            for node, _platform in replicas:
                slot(node)[1] += 1
        for node_name, _pid in draining:
            names.setdefault(node_name, [0, 0, 0, None])[2] += 1
        self.emit("pool", env.now, nodes=names)

    # ---- decisions (I10) -------------------------------------------------------------------------------------
    def decision(self, env: Any, sim_before: float, sim_after: float, wall_s: float, n_tasks: int,
                 policy: str) -> None:
        self.emit("decision", env.now, before=_f(sim_before), after=_f(sim_after), wall=_f(wall_s),
                  n=int(n_tasks), policy=policy)

    # ---- co-simulation fidelity (I11) ------------------------------------------------------------------------
    def i11_probe(self, scheduler: Any, system_state: Any, batch_tasks: List[Any]) -> None:
        """State at a live decision that a co-sim replay would need, split into what a live audit snapshot
        (live_audit.maybe_capture_batch_live_audit_snapshot -> live_snapshot_seed) carries and what it does not.
        Counts only; a snapshot's own schema is the reference for "carried"."""
        env = scheduler.env
        now = env.now
        replica_of: Dict[Any, List[str]] = {}
        for fn, reps in system_state.replicas.items():
            for _node, platform in reps:
                replica_of.setdefault(platform, []).append(fn)

        stage_counts = {"rendezvous": 0, "input_io": 0, "lock_wait": 0, "compute_output": 0}
        inflight_tasks = admitted = draining_platforms = draining_tasks = 0
        uninit_replicas = uninit_pull_age = 0.0
        queued_on_replicas = queued_on_draining = 0
        service_end_overwritten = 0
        held_current_in_io = 0
        per_platform: List[Dict[str, Any]] = []
        for node in scheduler.nodes.items:
            for platform in node.platforms.items:
                inflight = list(getattr(platform, "inflight", None) or ())
                rendezvous = getattr(platform, "rendezvous_procs", None) or {}
                adm = getattr(platform, "admitted", None)
                current = getattr(platform, "current_task", None)
                q = len(platform.queue.items)
                is_replica = platform in replica_of
                if is_replica and not platform.initialized.triggered:
                    uninit_replicas += 1
                    uninit_pull_age += max(0.0, now - float(getattr(platform, "last_allocated", now)))
                holds = q + len(inflight) + (adm is not None) + (current is not None and current not in inflight)
                if not holds:
                    continue
                stages = {}
                for t in inflight:
                    if t in rendezvous:
                        st = "rendezvous"
                    elif getattr(t, "_audit_compute_start", None) is not None:
                        st = "compute_output"
                    elif getattr(t, "_audit_io_end", None) is not None:
                        st = "lock_wait"
                    else:
                        st = "input_io"
                    stage_counts[st] += 1
                    stages[st] = stages.get(st, 0) + 1
                inflight_tasks += len(inflight)
                admitted += adm is not None
                if len(inflight) > 1 and getattr(platform, "inflight_service_end", None) is not None:
                    service_end_overwritten += 1
                if not inflight and current is not None and getattr(current, "_audit_compute_start", None) is None:
                    held_current_in_io += 1
                if is_replica:
                    queued_on_replicas += q
                else:
                    draining_platforms += 1
                    draining_tasks += q + len(inflight) + (adm is not None) + (current is not None)
                    queued_on_draining += q
                per_platform.append({"q": f"{node.node_name}:{platform.id}", "fn": replica_of.get(platform),
                                     "queue": q, "inflight": len(inflight), "stages": stages,
                                     "admitted": adm is not None, "current": current is not None,
                                     "init": bool(platform.initialized.triggered)})

        fabric = next((n.fabric for n in scheduler.nodes.items if getattr(n, "fabric", None) is not None), None)
        links_busy = links_queued = 0
        if fabric is not None:
            for key in fabric.link_keys:
                pipe = fabric.pipe(key)
                links_busy += len(pipe.users) > 0
                links_queued += len(pipe.queue)

        autoscaler = getattr(scheduler, "autoscaler", None)
        kpa = getattr(autoscaler, "kpa", None)
        kpa_state = None
        if kpa is not None:
            window = kpa.config.stable_window
            kpa_state = {
                "functions": {fn: {"samples": len(st.samples), "panicking": st.panic_since is not None,
                                   "max_panic_pods": st.max_panic_pods,
                                   "window_covered": st.first_sample is not None
                                   and now - st.first_sample >= window,
                                   "since_traffic": _f(now - st.last_traffic)}
                              for fn, st in kpa.functions.items()},
                "pending": {fn: sum(1 for t in lst if not t.scheduled.triggered)
                            for fn, lst in (getattr(autoscaler, "_kpa_pending", None) or {}).items()},
                "born_within_window": {
                    cause: sum(1 for key, c in autoscaler._replica_cause.items()
                               if c == cause and now - autoscaler._replica_born.get(key, -math.inf) < window)
                    for cause in ("load", "reachability")},
            }
        # arrived, not in this batch, not yet on any platform (the scheduler's own task store)
        buffered = len(getattr(getattr(scheduler, "tasks", None), "items", ()) or ())
        self.emit(
            "i11", now, policy=type(scheduler).__name__, batch=[int(t.id) for t in batch_tasks],
            carried={"replicas": sum(len(r) for r in system_state.replicas.values()),
                     "queued_on_replicas": queued_on_replicas,
                     "uninit_flag_only": int(uninit_replicas)},
            missing={"inflight_tasks": inflight_tasks, "inflight_by_stage": stage_counts,
                     "admitted_in_ingress": admitted, "draining_platforms": draining_platforms,
                     "draining_tasks": draining_tasks, "queued_on_draining": queued_on_draining,
                     "uninit_replicas": int(uninit_replicas), "uninit_pull_age_s": _f(uninit_pull_age),
                     "links_busy": links_busy, "links_queued": links_queued,
                     "service_end_overwritten": service_end_overwritten,
                     "held_current_in_io": held_current_in_io,
                     "scheduler_buffered": buffered, "kpa": kpa_state},
            platforms=per_platform,
        )

    # ---- end (I7) --------------------------------------------------------------------------------------------
    def end(self, env: Any, task_archive: List[Any]) -> None:
        real = [t for t in task_archive if not getattr(t, "is_internal", False)]
        dispatched = [t for t in real if t.dispatched_time is not None]
        done = [t for t in dispatched if t.done.triggered]
        failed = [t for t in dispatched if getattr(t, "failed", False)]
        self.emit("end", env.now, created=len(real), dispatched=len(dispatched), done=len(done),
                  failed=len(failed), failed_ids=[int(t.id) for t in failed][:1000],
                  not_done_ids=[int(t.id) for t in dispatched if not t.done.triggered][:1000],
                  undispatched_ids=[int(t.id) for t in real if t.dispatched_time is None][:1000])
        self.flush()


def _open_from_env() -> Optional[AuditRecorder]:
    path = (os.environ.get(AUDIT_ENV) or "").strip()
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return AuditRecorder(path)


AUDIT: Optional[AuditRecorder] = _open_from_env()
