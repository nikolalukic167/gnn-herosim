#!/usr/bin/env python3
"""Label captured live groups by the best plan in a deterministic sampled slate."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts_cosim.live_snapshot_cosim_oracle as oracle_module
from scripts_cosim.label_live_snapshots_for_training import (
    _inject_live_snapshots_into_task_results,
    _live_task_placements,
)
from scripts_cosim.live_snapshot_cosim_oracle import (
    CosimOracleContext,
    build_workload_from_snapshot,
    candidate_lists_from_snapshot,
    snapshot_tasks,
)
from src.executecosimulation import rtt_from_stats
from src.executesimulation import execute_simulation
from src.placement.constants import KEEP_ALIVE, QUEUE_LENGTH
from src.placement.live_snapshot_seed import build_live_snapshot_seed


def group_peer_exchange(snapshot: Mapping[str, Any], workload: Mapping[str, Any]) -> list[list[float]]:
    tasks = snapshot_tasks(snapshot, None)
    ids = [int(t["task_id"]) for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("snapshot has duplicate task IDs")
    local = {task_id: i for i, task_id in enumerate(ids)}
    pairs = []
    for triple in workload.get("peer_exchange") or []:
        if len(triple) != 3:
            raise ValueError(f"invalid peer_exchange triple: {triple!r}")
        a, b, payload = int(triple[0]), int(triple[1]), float(triple[2])
        if (a in local) != (b in local):
            raise ValueError(f"peer pair ({a}, {b}) crosses snapshot boundary")
        if a in local:
            if a == b or not math.isfinite(payload) or payload <= 0:
                raise ValueError(f"invalid peer pair ({a}, {b}, {payload})")
            pairs.append([local[a], local[b], payload])
    if not pairs:
        raise ValueError("snapshot has no peer-exchange pairs in supplied workload")
    return pairs


def group_events(snapshot: Mapping[str, Any], workload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = workload.get("events") or []
    matched = []
    for task in snapshot_tasks(snapshot, None):
        task_id = int(task["task_id"])
        if task_id < 0 or task_id >= len(events):
            raise ValueError(f"task ID {task_id} absent from source workload")
        event = events[task_id]
        dag = event.get("application", {}).get("dag") or {}
        if list(dag) != [str(task["task_type"])] or event.get("node_name") != task.get("source_node"):
            raise ValueError(f"source workload event {task_id} disagrees with snapshot task")
        scale = event.get("application", {}).get("demand_scale")
        if not isinstance(scale, dict) or set(scale) != set(dag):
            raise ValueError(f"source workload event {task_id} has no matching demand_scale")
        matched.append(event)
    return matched


def snapshot_replicas(snapshot: Mapping[str, Any]) -> dict[str, list[list[Any]]]:
    captured = snapshot.get("replicas_by_type")
    if not isinstance(captured, dict) or not captured:
        raise ValueError("snapshot has no scheduling-time replicas_by_type")
    replicas: dict[str, list[list[Any]]] = {}
    for task_type, entries in captured.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"snapshot has no replicas for {task_type}")
        rows = []
        for entry in entries:
            name = entry.get("node_name")
            platform_id = entry.get("platform_id")
            if not isinstance(name, str) or not name or platform_id is None:
                raise ValueError(f"malformed replica for {task_type}: {entry!r}")
            rows.append([name, int(platform_id)])
        replicas[str(task_type)] = rows
    return replicas


def _choice_key(candidate: Mapping[str, Any]) -> tuple[int, int]:
    return int(candidate["node_id"]), int(candidate["platform_id"])


def sampled_plans(
    tasks: Sequence[Mapping[str, Any]], count: int, seed: int,
    proposals: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
) -> list[dict[int, tuple[int, int]]]:
    if count < 2:
        raise ValueError("plans per snapshot must be at least two")
    lists = candidate_lists_from_snapshot(tasks)
    choices = [sorted(set(map(_choice_key, candidates))) for candidates in lists]
    if any(not row for row in choices):
        raise ValueError("snapshot task has no feasible candidate")
    if math.prod(map(len, choices)) < count:
        raise ValueError("requested slate exceeds distinct feasible placement count")
    rng = random.Random(seed)
    local_costs = [
        {
            _choice_key(c): sum(
                float(c.get(field, 0.0) or 0.0)
                for field in (
                    "execution_time", "communications_time", "network_latency",
                    "queue_drain_seconds", "cold_start_time",
                )
            )
            for c in row
        }
        for row in lists
    ]
    greedy = tuple(min(row, key=lambda key: (costs[key], key)) for row, costs in zip(choices, local_costs))
    seen: set[tuple[tuple[int, int], ...]] = set()
    keys: list[tuple[tuple[int, int], ...]] = []

    def add(key: tuple[tuple[int, int], ...]) -> None:
        if key not in seen:
            seen.add(key)
            keys.append(key)

    add(greedy)
    add(tuple(row[idx % len(row)] for idx, row in enumerate(choices)))
    proposal_keys = []
    if proposals is not None:
        for name in ("gnn", "mpoff", "hand"):
            entries = proposals.get(name)
            if not isinstance(entries, dict) or set(entries) != {str(i) for i in range(len(tasks))}:
                raise ValueError(f"proposal {name} must have all local task indices")
            key = tuple((int(entries[str(i)][0]), int(entries[str(i)][1])) for i in range(len(tasks)))
            for i, candidate in enumerate(key):
                if candidate not in choices[i]:
                    raise ValueError(f"proposal {name} has infeasible placement at task {i}")
            proposal_keys.append(key)
            add(key)
    for rank in range(max(map(len, choices))):
        add(tuple(row[rank % len(row)] for row in choices))
    if len(keys) > count:
        raise ValueError(
            f"slate size {count} too small for {len(keys)} mandatory greedy, diversified, "
            "proposal, and candidate-covering plans"
        )
    covered = [{key[i] for key in keys} for i in range(len(tasks))]
    if any(set(row) != covered[i] for i, row in enumerate(choices)):
        raise RuntimeError("mandatory slate rows failed to cover every task candidate")
    if proposal_keys:
        attempts = 0
        while len(keys) < count and attempts < count * 300:
            attempts += 1
            base = list(proposal_keys[(attempts - 1) % len(proposal_keys)])
            count_changes = 1 + ((attempts - 1) // len(proposal_keys)) % 4
            for idx in rng.sample(range(len(base)), k=min(count_changes, len(base))):
                alternatives = [choice for choice in choices[idx] if choice != base[idx]]
                if alternatives:
                    base[idx] = rng.choice(alternatives)
            add(tuple(base))
        if len(keys) != count:
            raise RuntimeError(f"could generate only {len(keys)} distinct proposal mutations of {count}")
        return [{idx: choice for idx, choice in enumerate(key)} for key in keys]
    attempts = 0
    while len(keys) < count and attempts < count * 200:
        attempts += 1
        if attempts % 2:
            key = tuple(rng.choice(row) for row in choices)
        else:
            base = list(keys[rng.randrange(len(keys))])
            for idx in rng.sample(range(len(base)), k=min(1 + attempts % 5, len(base))):
                base[idx] = rng.choice(choices[idx])
            key = tuple(base)
        add(key)
    if len(keys) != count:
        raise RuntimeError(f"could generate only {len(keys)} distinct plans of {count}")
    return [{idx: choice for idx, choice in enumerate(key)} for key in keys]


def validate_snapshot_infrastructure(ctx: CosimOracleContext, snapshot: Mapping[str, Any]) -> None:
    nodes = ctx._base_infrastructure.get("nodes") or []
    offsets = []
    count = 0
    for node in nodes:
        offsets.append(count)
        count += len(node.get("platforms") or [])
    for task in snapshot_tasks(snapshot, None):
        for c in task.get("candidates") or []:
            node_id = int(c["node_id"])
            if node_id < 0 or node_id >= len(nodes):
                raise ValueError(f"candidate node_id {node_id} absent from seeded infrastructure")
            node = nodes[node_id]
            if node.get("node_name") != c.get("node_name"):
                raise ValueError(f"candidate node name mismatch at ID {node_id}")
            platform_id = int(c["platform_id"])
            slot = platform_id - offsets[node_id]
            platforms = node.get("platforms") or []
            if slot < 0 or slot >= len(platforms):
                raise ValueError(f"candidate platform {platform_id} absent from node {node_id}")
            if platforms[slot] != c.get("platform_type"):
                raise ValueError(f"candidate platform type mismatch at ID {platform_id}")


def _peer_workload(
    tasks: Sequence[Mapping[str, Any]],
    pairs: Sequence[Sequence[float]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    built = build_workload_from_snapshot(tasks)
    if len(events) != len(built["events"]):
        raise ValueError("source event count differs from snapshot task count")
    for mini_event, source_event in zip(built["events"], events):
        mini_event["application"]["demand_scale"] = deepcopy(source_event["application"]["demand_scale"])
    built["peer_exchange"] = [list(row) for row in pairs]
    return built


def score_plan(
    ctx: CosimOracleContext,
    snapshot: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    plan: dict[int, tuple[int, int]],
    pairs: Sequence[Sequence[float]],
    events: Sequence[Mapping[str, Any]],
) -> float:
    def workload_with_peers(call_tasks: Sequence[Mapping[str, Any]], horizon_events=None):
        if horizon_events:
            raise ValueError("sampled-slate labels do not support horizon events")
        return _peer_workload(call_tasks, pairs, events)

    with patch.object(oracle_module, "build_workload_from_snapshot", workload_with_peers):
        return ctx.run_placement_plan(snapshot, tasks, plan)


def full_result(
    ctx: CosimOracleContext,
    snapshot: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    plan: dict[int, tuple[int, int]],
    pairs: Sequence[Sequence[float]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    infrastructure = deepcopy(ctx._base_infrastructure)
    infrastructure["live_snapshot_seed"] = build_live_snapshot_seed({**snapshot, "tasks": list(tasks)})
    infrastructure["forced_placements"] = plan
    infrastructure["fast_forward_warmup"] = True
    infrastructure["fast_forward_threshold"] = 1
    infrastructure["scheduler"] = {"batch_size": len(tasks), "batch_timeout": 0.02}
    config = {"infrastructure": infrastructure, "workload": _peer_workload(tasks, pairs, events)}
    previous = os.environ.get("GNN_CAPTURE_DATASET_STATE")
    previous_full_stats = os.environ.get("SIM_FORCE_FULL_STATS")
    os.environ["GNN_CAPTURE_DATASET_STATE"] = "0"
    os.environ["SIM_FORCE_FULL_STATS"] = "1"
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = execute_simulation(
                config, ctx._sim_inputs, scheduling_strategy="determined_determined",
                cache_policy="fifo", task_priority="fifo", keep_alive=KEEP_ALIVE,
                queue_length=QUEUE_LENGTH,
            )
    finally:
        if previous is None:
            os.environ.pop("GNN_CAPTURE_DATASET_STATE", None)
        else:
            os.environ["GNN_CAPTURE_DATASET_STATE"] = previous
        if previous_full_stats is None:
            os.environ.pop("SIM_FORCE_FULL_STATS", None)
        else:
            os.environ["SIM_FORCE_FULL_STATS"] = previous_full_stats
    _inject_live_snapshots_into_task_results(result, snapshot)
    result["sample"] = {
        "placement_plan": {str(k): list(v) for k, v in plan.items()},
        "label_provenance": "best_of_sampled_slate",
    }
    return result


def export_snapshot(
    ctx: CosimOracleContext,
    snapshot: Mapping[str, Any],
    workload: Mapping[str, Any],
    out_dir: Path,
    count: int,
    seed: int,
    proposal_trace: Mapping[tuple[int, ...], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if os.environ.get("HEROSIM_PEER_EXCHANGE") != "1":
        raise RuntimeError("HEROSIM_PEER_EXCHANGE=1 is required")
    tasks = snapshot_tasks(snapshot, None)
    if len(tasks) != 32:
        raise ValueError(f"expected 32 tasks, got {len(tasks)}")
    validate_snapshot_infrastructure(ctx, snapshot)
    pairs = group_peer_exchange(snapshot, workload)
    events = group_events(snapshot, workload)
    sid = int(snapshot["snapshot_id"])
    proposals = None
    if proposal_trace is not None:
        ids = tuple(int(t["task_id"]) for t in tasks)
        trace = proposal_trace.get(ids)
        if trace is None or tuple(trace["ids"]) != ids:
            raise ValueError(f"proposal trace has no exact task-ID match for snapshot {sid}")
        proposals = trace["proposals"]
    plans = sampled_plans(tasks, count, seed + sid * 1000003, proposals)
    rows = []
    for plan in plans:
        if not ctx.validate_placement_plan(tasks, plan):
            raise RuntimeError("generated invalid placement")
        rtt = score_plan(ctx, snapshot, tasks, plan, pairs, events)
        if not math.isfinite(rtt):
            raise RuntimeError(f"non-finite oracle RTT for snapshot {sid}")
        rows.append({
            "placement_plan": {str(k): list(v) for k, v in plan.items()},
            "rtt": float(rtt),
        })
    best_index = min(range(len(rows)), key=lambda i: rows[i]["rtt"])
    result = full_result(ctx, snapshot, tasks, plans[best_index], pairs, events)
    best_rtt = rows[best_index]["rtt"]
    if not math.isclose(rtt_from_stats(result.get("stats")), best_rtt, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError("best-plan full rerun disagrees with oracle label")
    peer_time = float(result.get("stats", {}).get("totalPeerExchangeTime") or 0.0)
    split_index = next((
        i for i, plan in enumerate(plans)
        if any(plan[int(a)][0] != plan[int(b)][0] for a, b, _ in pairs)
    ), None)
    if split_index is None:
        raise RuntimeError("sampled slate has no plan that splits a peer pair across nodes")
    split_result = full_result(ctx, snapshot, tasks, plans[split_index], pairs, events)
    split_peer_time = float(split_result.get("stats", {}).get("totalPeerExchangeTime") or 0.0)
    if split_peer_time <= 0:
        raise RuntimeError("split-peer full rerun has no peer-exchange charge")
    if not math.isclose(rtt_from_stats(split_result.get("stats")), rows[split_index]["rtt"], rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError("split-peer full rerun disagrees with oracle label")

    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "placements").mkdir()
    with (out_dir / "placements" / "placements.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    (out_dir / "optimal_result.json").write_text(json.dumps(result))
    (out_dir / "workload.json").write_text(json.dumps(_peer_workload(tasks, pairs, events)))
    dataset_infrastructure = deepcopy(ctx._base_infrastructure)
    dataset_infrastructure["network_maps"] = {
        str(node["node_name"]): dict(node.get("network_map") or {})
        for node in dataset_infrastructure.get("nodes") or []
    }
    (out_dir / "infrastructure.json").write_text(json.dumps(dataset_infrastructure))
    (out_dir / "space_with_network.json").write_text(json.dumps(ctx._space_config))
    (out_dir / "best.json").write_text(json.dumps({
        "rtt": best_rtt, "label_provenance": "best_of_sampled_slate",
        "best_slate_index": best_index,
    }))
    (out_dir / "system_state_captured_unique.json").write_text(json.dumps({
        "task_placements": _live_task_placements(snapshot),
        "replicas": snapshot_replicas(snapshot),
    }))
    (out_dir / "live_snapshot.json").write_text(json.dumps(snapshot))
    metadata = {
        "snapshot_id": sid,
        "topology_seed": ctx.seed,
        "label_provenance": "best_of_sampled_slate",
        "slate_size": len(rows),
        "best_slate_index": best_index,
        "best_slate_rtt": best_rtt,
        "peer_pairs": len(pairs),
        "best_plan_peer_exchange_time": peer_time,
        "split_plan_peer_exchange_time": split_peer_time,
        "sampling_seed": seed + sid * 1000003,
        "proposal_sources": list(proposals) if proposals is not None else [],
    }
    (out_dir / "placement_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--proposal-trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sim-input", type=Path, default=ROOT / "data" / "nofs-ids")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, default=101)
    parser.add_argument("--dataset-id-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--plans", type=int, default=16)
    args = parser.parse_args()
    if args.limit < 1 or not 2 <= args.plans <= 32:
        parser.error("limit must be positive and plans must be 2..32")
    if os.environ.get("HEROSIM_PEER_EXCHANGE") != "1":
        parser.error("HEROSIM_PEER_EXCHANGE=1 is required")
    workload = json.loads(args.workload.read_text())
    proposal_trace = None
    if args.proposal_trace is not None:
        proposal_trace = {}
        with args.proposal_trace.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                ids = tuple(int(value) for value in row["ids"])
                if ids in proposal_trace:
                    raise ValueError(f"duplicate proposal trace IDs: {ids}")
                proposal_trace[ids] = row
    ctx = CosimOracleContext(args.config, args.sim_input, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    with args.snapshots.open() as f:
        for line in f:
            if not line.strip():
                continue
            if exported >= args.limit:
                break
            snapshot = json.loads(line)
            sid = int(snapshot["snapshot_id"])
            out = args.output_dir / f"ds_{args.dataset_id_offset + sid:05d}"
            if out.exists():
                raise FileExistsError(f"refusing to overwrite dataset {out}")
            meta = export_snapshot(ctx, snapshot, workload, out, args.plans, args.sampling_seed, proposal_trace)
            exported += 1
            print(f"[{exported}/{args.limit}] snapshot={sid} slate={meta['slate_size']} rtt={meta['best_slate_rtt']:.6f}", flush=True)
    if exported != args.limit:
        raise RuntimeError(f"requested {args.limit} groups, found {exported}")
    print(f"Exported {exported} best_of_sampled_slate datasets to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
