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
from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH
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


def build_workload_from_snapshot(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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
    return {"rps": max(len(events), 1), "duration": 1, "events": events}


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
    ) -> float:
        """Run one mini co-sim and return batch RTT (sum elapsedTime for batch tasks)."""
        if not self.validate_placement_plan(tasks, placement_plan):
            return float("inf")

        infrastructure = deepcopy(self._base_infrastructure)
        n_tasks = len(tasks)
        seed_data = build_live_snapshot_seed({**snapshot, "tasks": list(tasks)})
        infrastructure["live_snapshot_seed"] = seed_data
        infrastructure["forced_placements"] = {
            int(task_idx): (int(node_id), int(platform_id))
            for task_idx, (node_id, platform_id) in placement_plan.items()
        }
        infrastructure["fast_forward_warmup"] = True
        infrastructure["fast_forward_threshold"] = 1
        infrastructure["scheduler"] = {
            "batch_size": max(n_tasks, 1),
            "batch_timeout": 0.02,
        }

        workload = build_workload_from_snapshot(tasks)
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
) -> CosimOracleResult:
    combos, combo_count = enumerate_combos(tasks, max_combos)
    best_combo: Optional[Tuple[Choice, ...]] = None
    best_rtt = float("inf")
    for combo in combos:
        plan = combo_to_placement_plan(combo)
        rtt = ctx.run_placement_plan(snapshot, tasks, plan)
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
) -> float:
    plan = combo_to_placement_plan(combo)
    return ctx.run_placement_plan(snapshot, tasks, plan)
