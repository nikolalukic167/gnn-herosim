"""r1_attribution_v1: a co-sim plan replayed the way physics_audit/i11_replay.py replays a live decision.

A warm-snapshot dataset made with `HEROSIM_SNAPSHOT_FIDELITY=1` carries the snapshot's `fidelity` block. This module is
the corpus' per-plan simulation under it, and it is deliberately the same code path as `i11_replay --params live`:

    base infra  = prepare_infrastructure_for_real_simulation(cell cfg)   # i11_replay.py:245 (the live driver's infrastructure)
    kw          = snapshot_fidelity.live_run_params()                    # i11_replay.py:267 (target 0.7, 1 s ticks, keep-alive)
    wl, forced, ids = snapshot_fidelity.replay_workload(fid)             # i11_replay.py:317
    infra[live_snapshot_seed | forced_placements | fast_forward_* | scheduler]   # i11_replay.py:321-325
    execute_simulation({"infrastructure": infra, "workload": wl}, sim_inputs,
        scheduling_strategy="determined_determined", cache_policy="fifo", task_priority="fifo", **kw)   # i11_replay.py:358-360
    with execute_simulation = src.executesimulation.execute_simulation (i11_replay.py:39-40), not the co-sim wrapper in
    executecosimulation.py (which differs only in model_locations, the trace name and the co-sim keep-alive / queue length).

Queued tasks are STATE: they enter the workload and are forced onto the platforms they were live-placed on, and they are
never decision variables. The decision is the batch's placement. The label is I11's quantity, the sum over the batch's
tasks of (done - scheduled). Everything downstream of the sweep (placements.jsonl, best.json, optimal_result.json, the
cache) sees a dataset whose tasks are the batch only: the replay's result is cut down to the batch and renumbered to the
dataset's own task ids.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

LABEL = "sum over the batch's tasks of (doneTime - scheduledTime), the quantity physics_audit/i11_replay.py scores"
SPEC_KEY = "fidelity_replay"


def spec_of(seed: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    return (seed or {}).get(SPEC_KEY)


def build_spec(snapshot: Mapping[str, Any], cell_config: Path, sim_input: Path) -> Dict[str, Any]:
    """What a dataset must carry to be replayed under fidelity: the RAW snapshot (every replica a candidate, the
    fidelity block attached), the cell it was captured on, and the parameters the replay will run with."""
    from src.placement import snapshot_fidelity

    if snapshot.get("fidelity") is None:
        raise ValueError("fidelity replay needs a snapshot captured with HEROSIM_SNAPSHOT_FIDELITY=1")
    if not snapshot_fidelity.enabled():
        raise ValueError(f"export {snapshot_fidelity.ENV}=1 for the corpus build as well as for the capture")
    return {
        "snapshot": snapshot,
        "cell_config": str(cell_config),
        "sim_input": str(sim_input),
        "label": LABEL,
        "live_run_params": snapshot_fidelity.live_run_params(),
        "queued_tasks": len(snapshot["fidelity"]["queued"]),
        "batch_tasks": len(snapshot["fidelity"]["batch"]),
    }


def exposure(snapshot: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Which datasets a replay-side net-stage fix can have moved: does the snapshot hold an ingress ghost still in its net stage
    (link_stage 'net', not yet on a pipe) on a node that is a candidate of some batch task? `candidates`: the offered candidate payload
    dicts (node_name per entry)."""
    nodes = {str(c["node_name"]) for c in candidates}
    net = [g for g in (snapshot.get("fidelity") or {}).get("ghosts") or []
           if g.get("stage") == "ingress" and g.get("link_stage") == "net"]
    on = sorted({str(g["q"]).rsplit(":", 1)[0] for g in net} & nodes)
    return {"net_ingress_ghost_on_candidate_node": bool(on), "net_ingress_ghost_nodes_on_candidates": on,
            "net_ingress_ghosts_total": len(net), "candidate_nodes": len(nodes)}


def restrict_to_gids(snapshot: Mapping[str, Any], gids: Sequence[int]) -> Dict[str, Any]:
    """A sub-batch as a snapshot of its own: only `gids` are batch tasks; their siblings in the original batch are absent from the
    replay (their load is not modelled) and partners outside the chunk are invisible (pairs and peer rows naming a sibling are
    dropped, as a peer outside the decoded batch is invisible to the decoder by contract). Queued tasks and ghosts are untouched."""
    keep = {int(g) for g in gids}
    out = deepcopy(dict(snapshot))
    fid = out["fidelity"]
    siblings = {int(r["gid"]) for r in fid["batch"]} - keep
    fid["batch"] = [r for r in fid["batch"] if int(r["gid"]) in keep]
    fid["pairs"] = [p for p in fid["pairs"] if int(p[0]) not in siblings and int(p[1]) not in siblings]
    fid["peers"] = {g: [row for row in rows if int(row[0]) not in siblings]
                    for g, rows in fid["peers"].items() if int(g) not in siblings}
    out["tasks"] = [t for t in out["tasks"] if int(t["task_id"]) in keep]
    return out


class FidelityReplay:
    """One per worker process. Holds the snapshot-derived pieces that do not depend on the plan."""

    def __init__(self, spec: Mapping[str, Any], dataset_types: Sequence[str], offered: Mapping[str, Sequence[Sequence[Any]]]):
        from src.executesimulation import load_simulation_inputs, prepare_infrastructure_for_real_simulation
        from src.placement import snapshot_fidelity
        from src.placement.live_snapshot_seed import build_live_snapshot_seed

        os.environ["SIM_FORCE_FULL_STATS"] = "1"  # taskResults are dropped above 20 tasks otherwise (orchestrator.py:736)
        self.snap = spec["snapshot"]
        fid = self.snap["fidelity"]
        self.wl, forced, ids = snapshot_fidelity.replay_workload(fid)
        self.queued_forced = dict(forced)
        by_type: Dict[str, List[int]] = {}
        for r in sorted(fid["batch"], key=lambda r: int(r["gid"])):
            by_type.setdefault(r["fn"], []).append(int(r["gid"]))
        # the dataset's task i is not the snapshot's i-th batch task: make_warm_corpus orders events by application
        # (stable within a type), so each dataset event takes the next unused batch record of its type
        self.batch_local = [ids[by_type[t].pop(0)] for t in dataset_types]
        if any(by_type.values()):
            raise ValueError(f"dataset types {list(dataset_types)} do not account for the snapshot's batch {fid['batch']}")
        space = json.loads(Path(spec["cell_config"]).read_text())
        self.base_infra = prepare_infrastructure_for_real_simulation(space, seed=None, sim_input_path=Path(spec["sim_input"]))
        self.seed = build_live_snapshot_seed(self.snap)  # fidelity attached: backlog fields zeroed, queued tasks replayed
        self.kw = snapshot_fidelity.live_run_params()
        self.sim_inputs = load_simulation_inputs(Path(spec["sim_input"]))
        self.full_queue = {str(k): int(v or 0) for k, v in (self.snap.get("full_queue_snapshot") or {}).items()}
        # what the live feature builder saw at the decision, per platform, from the snapshot's own capture (the replayed ghosts
        # are not exposed as platform.current_task, so the replay's temporal capture reads zero for them)
        plain = {k: v for k, v in self.snap.items() if k != "fidelity"}
        self.live_temporal = {
            f"{sp['node_name']}:{sp['platform_id']}": {
                "current_task_remaining": float(sp.get("current_task_remaining", 0.0) or 0.0),
                "cold_start_remaining": float(sp.get("cold_start_remaining", 0.0) or 0.0),
                "comm_remaining": float(sp.get("comm_remaining", 0.0) or 0.0),
            }
            for sp in build_live_snapshot_seed(plain)["platforms"]
        }
        self.offered = {t: [[str(k[0]), int(k[1])] for k in keys] for t, keys in offered.items()}
        self.spec_params = spec["live_run_params"]
        if self.kw != self.spec_params:
            raise ValueError(f"live_run_params at replay {self.kw} differ from the dataset's {self.spec_params}")

    def run(self, plan: Mapping[int, Tuple[int, int]]) -> Dict[str, Any]:
        from src.executesimulation import execute_simulation

        forced = dict(self.queued_forced)
        forced.update({self.batch_local[int(i)]: (int(p[0]), int(p[1])) for i, p in plan.items()})
        infra = deepcopy(self.base_infra)
        infra["live_snapshot_seed"] = self.seed
        infra["forced_placements"] = forced
        infra["fast_forward_warmup"] = True
        infra["fast_forward_threshold"] = 1
        infra["scheduler"] = {"batch_size": max(len(self.wl["events"]), 1), "batch_timeout": 0.02,
                           "exact_batch": os.environ.get("HEROSIM_REPLAY_EXACT_BATCH", "1") == "1"}  # 0: the pre-fix 1 ms poll, for A/B
        return execute_simulation({"infrastructure": infra, "workload": self.wl}, self.sim_inputs,
                                  scheduling_strategy="determined_determined", cache_policy="fifo",
                                  task_priority="fifo", **self.kw)

    def cut_to_batch(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """The replay's result as a dataset of the batch alone: taskResults renumbered to the dataset's ids, the label as
        total_rtt, the queue snapshot the live decision saw (the replay captures it before the queued tasks enter their
        queues), the scheduling-time replicas restricted to the offered slate. Everything else is the engine's."""
        stats = result["stats"]
        trs = {tr["taskId"]: tr for tr in stats["taskResults"] if tr.get("taskId", -1) >= 0}
        missing = [b for b in self.batch_local if b not in trs]
        if missing:
            raise RuntimeError(f"fidelity replay finished without results for batch tasks {missing}")
        rows: List[Dict[str, Any]] = []
        label = 0.0
        exchange = 0.0
        for i, b in enumerate(self.batch_local):
            tr = dict(trs[b])
            label += float(tr["doneTime"]) - float(tr["scheduledTime"])
            exchange += float(tr.get("peerExchangeTime") or 0.0)
            tr["taskId"] = i
            if tr.get("fullQueueSnapshot") is not None:
                tr["fullQueueSnapshot"] = dict(self.full_queue)
                at = tr.get("queueSnapshotAtScheduling") or {}
                tr["queueSnapshotAtScheduling"] = {k: self.full_queue.get(k, v) for k, v in at.items()}
            for field in ("fullTemporalStateAtScheduling", "temporalStateAtScheduling"):
                ts = tr.get(field)
                if ts:
                    tr[field] = {k: ({**v, **self.live_temporal[k]} if k in self.live_temporal else v) for k, v in ts.items()}
            rows.append(tr)
        every = {
            "n_tasks_replayed": len(self.wl["events"]),
            "queued_tasks": len(self.queued_forced),
            "total_rtt_all_tasks": stats.get("total_rtt"),
            "totalPeerExchangeTime_all_tasks": stats.get("totalPeerExchangeTime"),
            "endTime": stats.get("endTime"),
            "live_run_params": self.kw,
            "label": LABEL,
        }
        stats["taskResults"] = rows
        stats["num_tasks"] = len(rows)
        stats["total_rtt"] = label
        stats["totalPeerExchangeTime"] = exchange
        stats["fidelity_replay"] = every
        cap = stats.get("schedulingStateCapture")
        if cap:
            cap["replicas"] = self.scheduling_replicas()
        return result

    def scheduling_replicas(self) -> Dict[str, List[List[Any]]]:
        """The replica table the cache reads (SSC `replicas`): every live replica of every type. The candidate sets are the
        declared pruning's, restricted per task by `task_candidates` (src/placement/declared_slate.py), so the replica table is
        what the live builder sees and the platform replica flags (has_dnn1 / has_dnn2 and the four-type flags) mean the same in
        training and in serving."""
        return {t: [[str(sp["node_name"]), int(sp["platform_id"])] for sp in specs]
                for t, specs in (self.snap.get("replicas_by_type") or {}).items()}
