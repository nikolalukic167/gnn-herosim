"""Co-simulation RTT oracle for live audit snapshots (training-aligned physics)."""

from __future__ import annotations

import itertools
import json
import math
import os
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.executecosimulation import rtt_from_stats
from src.executesimulation import (
    execute_simulation,
    load_simulation_inputs,
    prepare_infrastructure_for_real_simulation,
)
from src.placement.constants import KEEP_ALIVE, QUEUE_LENGTH
from src.placement.live_snapshot_seed import build_live_snapshot_seed

Choice = Dict[str, Any]
Snapshot = Dict[str, Any]
PlacementPlan = Dict[int, Tuple[int, int]]


def placement_key(choice: Choice) -> str:
    return f"{choice['node_name']}:{choice['platform_id']}"


def snapshot_tasks(snapshot: Snapshot, horizon: Optional[int]) -> List[Dict[str, Any]]:
    tasks = list(snapshot.get("tasks", []))
    if horizon is not None:
        tasks = tasks[:horizon]
    return tasks


# Horizon arrivals are shifted to start this far after the batch (batch events sit at
# idx * 0.0001). A batch large enough to reach this offset would interleave with the
# horizon and silently corrupt the task-id -> forced-placement mapping, so the builder
# fails loud on it instead.
HORIZON_ARRIVAL_OFFSET = 0.01

# The determined scheduler's auto-resolve marker: least-loaded valid replica at
# schedule time (src/policy/determined/scheduler.py). Horizon arrivals carry it so the
# follow-on policy is the same fixed rule in every combo, reacting to the state the
# batch placement created — the P3 in-horizon dynamics being measured.
AUTO_RESOLVE = (-1, -1)


def slice_horizon_events(
    trace_events: Sequence[Mapping[str, Any]],
    snapshot_time: float,
    horizon_seconds: float,
) -> List[Dict[str, Any]]:
    """Trace arrivals in (snapshot_time, snapshot_time + horizon_seconds], shifted so the
    first possible arrival lands HORIZON_ARRIVAL_OFFSET after the batch at t=0."""
    if horizon_seconds <= 0:
        raise ValueError(f"horizon_seconds must be > 0, got {horizon_seconds}")
    out: List[Dict[str, Any]] = []
    for ev in trace_events:
        ts = float(ev["timestamp"])
        if snapshot_time < ts <= snapshot_time + horizon_seconds:
            shifted = deepcopy(dict(ev))
            shifted["timestamp"] = HORIZON_ARRIVAL_OFFSET + (ts - snapshot_time)
            out.append(shifted)
    out.sort(key=lambda e: float(e["timestamp"]))
    return out


def preflight_horizon_reachability(
    base_infrastructure: Mapping[str, Any],
    snapshot: Snapshot,
    horizon_events: Sequence[Mapping[str, Any]],
) -> List[Tuple[str, str]]:
    """Unplaceable (task_type, source_node) pairs among the horizon arrivals.

    Mirrors DeterminedScheduler._get_valid_replicas: a replica is valid for a task iff
    it sits on the task's own source node, or on a non-client node whose network_map
    contains the source. An unplaceable horizon task does not fail the mini co-sim —
    it retries forever and HANGS it — so the sweep must refuse the snapshot up front.
    Requires the snapshot's captured replicas_by_type (post-P3 captures).
    """
    captured = snapshot.get("replicas_by_type")
    if not captured:
        raise ValueError(
            "FAIL LOUD: snapshot has no replicas_by_type — it predates the P3 capture "
            "extension and cannot support horizon continuation (the batch candidates "
            "alone do not cover arrivals from other client nodes). Re-capture the cell."
        )
    node_maps = {
        str(n.get("node_name", "")): dict(n.get("network_map") or {})
        for n in base_infrastructure.get("nodes", [])
    }
    unplaceable: List[Tuple[str, str]] = []
    seen: set = set()
    for ev in horizon_events:
        source = str(ev.get("node_name", ""))
        for task_type in ev.get("application", {}).get("dag", {}):
            key = (str(task_type), source)
            if key in seen:
                continue
            seen.add(key)
            ok = False
            for spec in captured.get(str(task_type), []):
                rnode = str(spec.get("node_name", ""))
                if rnode == source:
                    ok = True
                    break
                if not rnode.startswith("client_node") and source in node_maps.get(
                    rnode, {}
                ):
                    ok = True
                    break
            if not ok:
                unplaceable.append(key)
    return unplaceable


def horizon_forced_placements(
    n_batch_tasks: int,
    horizon_events: Sequence[Mapping[str, Any]],
) -> Dict[int, Tuple[int, int]]:
    """Auto-resolve markers for every horizon task id.

    Task ids are assigned sequentially in event order (orchestrator.py), one per DAG
    entry. Batch events are single-task by construction (ids 0..n_batch-1); the
    determined scheduler hard-exits on any task with no forced placement, so every
    horizon task must carry an explicit marker.
    """
    forced: Dict[int, Tuple[int, int]] = {}
    next_id = n_batch_tasks
    for ev in horizon_events:
        dag = ev.get("application", {}).get("dag", {})
        n_tasks = max(len(dag), 1)
        for _ in range(n_tasks):
            forced[next_id] = AUTO_RESOLVE
            next_id += 1
    return forced


def build_workload_from_snapshot(
    tasks: Sequence[Mapping[str, Any]],
    horizon_events: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        task_type = str(task.get("task_type", "dnn1"))
        qos = task.get("qos") or {"name": "medium", "maxDurationDeviation": 15}
        events.append(
            {
                "timestamp": idx * 0.0001,
                "application": {
                    "name": f"nofs-{task_type}",
                    "dag": {task_type: []},
                },
                "qos": qos,
                "node_name": str(task.get("source_node", "client_node0")),
            }
        )
    if horizon_events:
        max_batch_ts = events[-1]["timestamp"] if events else 0.0
        if max_batch_ts >= HORIZON_ARRIVAL_OFFSET:
            raise ValueError(
                f"FAIL LOUD: batch of {len(events)} tasks reaches t={max_batch_ts}, at or "
                f"past HORIZON_ARRIVAL_OFFSET={HORIZON_ARRIVAL_OFFSET}; batch and horizon "
                "events would interleave and break the task-id mapping"
            )
        first_horizon_ts = float(horizon_events[0]["timestamp"])
        if first_horizon_ts < HORIZON_ARRIVAL_OFFSET:
            raise ValueError(
                f"FAIL LOUD: horizon events must be pre-shifted by slice_horizon_events "
                f"(first timestamp {first_horizon_ts} < offset {HORIZON_ARRIVAL_OFFSET})"
            )
        events = events + [dict(ev) for ev in horizon_events]
        duration = max(1, int(math.ceil(events[-1]["timestamp"])) + 1)
    else:
        duration = 1  # pre-P3 t=0 behaviour, bit-identical
    return {"rps": max(len(events), 1), "duration": duration, "events": events}


def candidate_lists_from_snapshot(
    tasks: Sequence[Mapping[str, Any]],
    *,
    initialized_only: bool = False,
) -> List[List[Choice]]:
    lists: List[List[Choice]] = []
    for task in tasks:
        candidates = list(task.get("candidates", []))
        if initialized_only:
            initialized = [c for c in candidates if c.get("initialized", True)]
            candidates = initialized or candidates
        if not candidates:
            raise ValueError(f"task {task.get('task_id')} has no candidates")
        lists.append(candidates)
    return lists


def combo_to_placement_plan(combo: Sequence[Choice]) -> PlacementPlan:
    return {
        idx: (int(choice["node_id"]), int(choice["platform_id"]))
        for idx, choice in enumerate(combo)
    }


def enumerate_combos(
    tasks: Sequence[Mapping[str, Any]],
    max_combos: int,
) -> Tuple[List[Tuple[Choice, ...]], int]:
    lists = candidate_lists_from_snapshot(tasks, initialized_only=False)
    total = math.prod(len(cands) for cands in lists)
    if total > max_combos:
        raise ValueError(f"snapshot has {total} combos, above max_combos={max_combos}")
    return list(itertools.product(*lists)), total


@dataclass
class CosimOracleContext:
    config_path: Path
    sim_input_path: Path
    seed: int = 101
    _space_config: Dict[str, Any] = field(init=False, repr=False)
    _sim_inputs: Dict[str, Any] = field(init=False, repr=False)
    _base_infrastructure: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        with open(self.config_path, "r") as f:
            self._space_config = json.load(f)
        self._sim_inputs = load_simulation_inputs(self.sim_input_path)
        self._base_infrastructure = prepare_infrastructure_for_real_simulation(
            self._space_config,
            seed=self.seed,
            sim_input_path=self.sim_input_path,
        )

    def validate_placement_plan(
        self,
        tasks: Sequence[Mapping[str, Any]],
        placement_plan: PlacementPlan,
    ) -> bool:
        for task_idx, task in enumerate(tasks):
            if task_idx not in placement_plan:
                return False
            node_id, plat_id = placement_plan[task_idx]
            candidates = task.get("candidates", [])
            if not any(
                int(c.get("node_id", -1)) == int(node_id)
                and int(c.get("platform_id", -1)) == int(plat_id)
                for c in candidates
            ):
                return False
        return True

    def run_placement_plan(
        self,
        snapshot: Snapshot,
        tasks: Sequence[Mapping[str, Any]],
        placement_plan: PlacementPlan,
        horizon_events: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> float:
        """Run one mini co-sim and return the RTT label.

        Without horizon_events: batch RTT (sum elapsedTime for batch tasks), the t=0
        behaviour, unchanged. With horizon_events (P3): the same trace's arrivals
        continue for the horizon window, placed by the auto-resolve rule
        (least-loaded valid replica) reacting to the state the batch placement
        created, and the label is the horizon return — total RTT over batch AND
        horizon tasks, so a placement that ignites queue runaway pays for it.
        """
        if not self.validate_placement_plan(tasks, placement_plan):
            return float("inf")

        infrastructure = deepcopy(self._base_infrastructure)
        n_tasks = len(tasks)
        seed_data = build_live_snapshot_seed({**snapshot, "tasks": list(tasks)})
        infrastructure["live_snapshot_seed"] = seed_data
        forced = {
            int(task_idx): (int(node_id), int(platform_id))
            for task_idx, (node_id, platform_id) in placement_plan.items()
        }
        if horizon_events:
            forced.update(horizon_forced_placements(n_tasks, horizon_events))
        infrastructure["forced_placements"] = forced
        infrastructure["fast_forward_warmup"] = True
        infrastructure["fast_forward_threshold"] = 1
        infrastructure["scheduler"] = {
            "batch_size": max(n_tasks, 1),
            "batch_timeout": 0.02,
        }

        workload = build_workload_from_snapshot(tasks, horizon_events=horizon_events)
        config = {"infrastructure": infrastructure, "workload": workload}

        prev_capture = os.environ.get("GNN_CAPTURE_DATASET_STATE")
        os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                result = execute_simulation(
                    config,
                    self._sim_inputs,
                    scheduling_strategy="determined_determined",
                    cache_policy="fifo",
                    task_priority="fifo",
                    keep_alive=KEEP_ALIVE,
                    queue_length=QUEUE_LENGTH,
                )
        except RuntimeError:
            return float("inf")
        finally:
            if prev_capture is None:
                os.environ.pop("GNN_CAPTURE_DATASET_STATE", None)
            else:
                os.environ["GNN_CAPTURE_DATASET_STATE"] = prev_capture

        stats = result.get("stats", {})
        rtt = rtt_from_stats(stats)
        return rtt if math.isfinite(rtt) else float("inf")


@dataclass
class CosimOracleResult:
    combo: List[Choice]
    rtt: float
    combo_count: int


def oracle_choice_cosim(
    ctx: CosimOracleContext,
    snapshot: Snapshot,
    tasks: Sequence[Mapping[str, Any]],
    max_combos: int,
    horizon_events: Optional[Sequence[Mapping[str, Any]]] = None,
) -> CosimOracleResult:
    combos, combo_count = enumerate_combos(tasks, max_combos)
    best_combo: Optional[Tuple[Choice, ...]] = None
    best_rtt = float("inf")
    for combo in combos:
        plan = combo_to_placement_plan(combo)
        rtt = ctx.run_placement_plan(snapshot, tasks, plan, horizon_events=horizon_events)
        if rtt < best_rtt and math.isfinite(rtt):
            best_rtt = rtt
            best_combo = combo
    if best_combo is None or not math.isfinite(best_rtt):
        raise ValueError("co-sim oracle found no valid placement combinations")
    return CosimOracleResult(list(best_combo), best_rtt, combo_count)


def policy_rtt_cosim(
    ctx: CosimOracleContext,
    snapshot: Snapshot,
    tasks: Sequence[Mapping[str, Any]],
    combo: Sequence[Choice],
    horizon_events: Optional[Sequence[Mapping[str, Any]]] = None,
) -> float:
    plan = combo_to_placement_plan(combo)
    return ctx.run_placement_plan(snapshot, tasks, plan, horizon_events=horizon_events)
