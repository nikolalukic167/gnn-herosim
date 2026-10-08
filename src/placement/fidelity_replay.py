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
        infra["scheduler"] = {"batch_size": max(len(self.wl["events"]), 1), "batch_timeout": 0.02}
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
            cap["replicas"] = deepcopy(self.offered)
        return result
