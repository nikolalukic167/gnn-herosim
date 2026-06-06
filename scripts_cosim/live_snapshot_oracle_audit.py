#!/usr/bin/env python3
"""Audit live scheduling snapshots against the training co-sim RTT oracle."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.gnn_snapshot_inference import choose_gnn_live_decode
from scripts_cosim.live_snapshot_cosim_oracle import (
    CosimOracleContext,
    oracle_choice_cosim,
    policy_rtt_cosim,
    snapshot_tasks,
)
from src.placement.scheduling_cost import expected_completion_from_snapshot_candidate

Choice = Dict[str, Any]
Snapshot = Dict[str, Any]
Cost = Dict[str, float]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def placement_key(choice: Choice) -> str:
    return f"{choice['node_name']}:{choice['platform_id']}"


def candidate_cost(
    candidate: Choice,
    queued_before: int,
    combo_added_before: int,
) -> Cost:
    exec_time = _safe_float(candidate.get("execution_time"))
    queue_time = (
        _safe_float(candidate.get("current_task_remaining"))
        + _safe_float(candidate.get("comm_remaining"))
        + queued_before * exec_time
        + combo_added_before * exec_time
    )
    cold = 0.0 if candidate.get("initialized", True) else _safe_float(candidate.get("cold_start_time"))
    cost = {
        "queue": queue_time,
        "exec": exec_time,
        "network": _safe_float(candidate.get("network_latency")),
        "comm": _safe_float(candidate.get("communications_time")),
        "cold": cold,
    }
    cost["total"] = expected_completion_from_snapshot_candidate(
        candidate, queued_before, combo_added_before
    )
    return cost


def choose_knative(tasks: Sequence[Dict[str, Any]]) -> List[Choice]:
    """Queue-length baseline (shortest queue + batch roll-forward)."""
    added: Dict[str, int] = {}
    choices: List[Choice] = []
    for task in tasks:
        candidates = list(task.get("candidates", []))
        initialized = [c for c in candidates if c.get("initialized", True)]
        pool = initialized or candidates
        chosen = min(
            pool,
            key=lambda c: (int(c.get("queue_length", 0) or 0) + added.get(placement_key(c), 0), placement_key(c)),
        )
        choices.append(chosen)
        key = placement_key(chosen)
        added[key] = added.get(key, 0) + 1
    return choices


def choose_knative_ect(tasks: Sequence[Dict[str, Any]]) -> List[Choice]:
    """Expected-completion-time baseline (same roll-forward as choose_knative)."""
    added: Dict[str, int] = {}
    choices: List[Choice] = []
    for task in tasks:
        candidates = list(task.get("candidates", []))
        initialized = [c for c in candidates if c.get("initialized", True)]
        pool = initialized or candidates

        def ect_key(candidate: Choice) -> tuple:
            key = placement_key(candidate)
            queued_before = int(candidate.get("queue_length", 0) or 0)
            ect = expected_completion_from_snapshot_candidate(
                candidate, queued_before, added.get(key, 0)
            )
            return (ect, key)

        chosen = min(pool, key=ect_key)
        choices.append(chosen)
        key = placement_key(chosen)
        added[key] = added.get(key, 0) + 1
    return choices


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    denom = hi - lo if hi != lo else 1.0
    return [1.0 + ((v - lo) / denom) * 99.0 for v in values]


def choose_herocache_like(tasks: Sequence[Dict[str, Any]]) -> List[Choice]:
    added: Dict[str, int] = {}
    choices: List[Choice] = []
    for task in tasks:
        candidates = list(task.get("candidates", []))
        if not candidates:
            continue

        max_duration_deviation = _safe_float(
            (task.get("qos") or {}).get("maxDurationDeviation"),
            1.0,
        )
        max_exec = max((_safe_float(c.get("execution_time")) for c in candidates), default=0.0)
        deadline = max_exec * max_duration_deviation

        penalty_raw: List[float] = []
        energy_raw: List[float] = []
        consolidation_raw: List[float] = []
        for candidate in candidates:
            key = placement_key(candidate)
            queue_len = int(candidate.get("queue_length", 0) or 0) + added.get(key, 0)
            immediate = candidate_cost(candidate, int(candidate.get("queue_length", 0) or 0), added.get(key, 0))
            penalty_raw.append(1.0 if immediate["total"] > deadline else 0.0)
            energy_raw.append(_safe_float(candidate.get("energy")))
            consolidation_raw.append(math.exp(min(queue_len, 100)))

        penalty = _normalize(penalty_raw)
        energy = _normalize(energy_raw)
        consolidation = _normalize(consolidation_raw)
        scores = [
            (2.0 / 3.0) * penalty[i]
            + (0.5 / 6.0) * energy[i]
            + (1.5 / 6.0) * consolidation[i]
            for i in range(len(candidates))
        ]
        chosen = candidates[min(range(len(candidates)), key=lambda i: (scores[i], placement_key(candidates[i])))]
        choices.append(chosen)
        key = placement_key(chosen)
        added[key] = added.get(key, 0) + 1
    return choices


def load_gnn_model(model_path: Path):
    import torch
    from src.policy.gnn.gnn_model import TaskPlacementGNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskPlacementGNN(
        task_feature_dim=3,
        platform_feature_dim=13,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=3,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    return model, device


def read_snapshots(path: Path, limit: Optional[int]) -> Iterable[Snapshot]:
    with path.open() as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                yield json.loads(line)


def analyze(args: argparse.Namespace) -> List[Dict[str, Any]]:
    ctx = CosimOracleContext(
        config_path=args.config,
        sim_input_path=args.sim_input,
        seed=args.seed,
    )
    gnn = load_gnn_model(args.gnn_model) if args.gnn_model else None
    task_types_data = ctx._sim_inputs.get("task_types", {})
    infra_nodes = ctx._base_infrastructure.get("nodes", [])
    rows: List[Dict[str, Any]] = []
    skipped = 0
    sim_runs = 0
    start_time = time.monotonic()
    deadline = start_time + args.max_runtime_s if args.max_runtime_s else None
    timed_out = False

    for snapshot in read_snapshots(args.snapshots, args.limit):
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            print(
                f"[TIME] reached max runtime ({args.max_runtime_s}s); "
                f"stopping after {len(rows)} snapshots",
                flush=True,
            )
            break

        tasks = snapshot_tasks(snapshot, args.horizon)
        if not tasks:
            skipped += 1
            continue
        try:
            oracle = oracle_choice_cosim(ctx, snapshot, tasks, args.max_combos)
        except ValueError as exc:
            skipped += 1
            print(f"[WARN] snapshot {snapshot.get('snapshot_id')}: {exc}")
            continue
        sim_runs += oracle.combo_count

        policies: Dict[str, Optional[List[Choice]]] = {
            "knative": choose_knative(tasks),
            "knative_ect": choose_knative_ect(tasks),
            "herocache_like": choose_herocache_like(tasks),
            "gnn": (
                choose_gnn_live_decode(
                    tasks,
                    snapshot,
                    gnn[0],
                    gnn[1],
                    infra_nodes,
                    task_types_data,
                )
                if gnn
                else None
            ),
        }
        shortest_first = choose_knative(tasks)[0]
        row: Dict[str, Any] = {
            "snapshot_id": snapshot.get("snapshot_id"),
            "time": snapshot.get("time"),
            "horizon": len(tasks),
            "combo_count": oracle.combo_count,
            "oracle_rtt": oracle.rtt,
            "oracle_first": placement_key(oracle.combo[0]),
            "shortest_first": placement_key(shortest_first),
            "oracle_non_shortest": placement_key(oracle.combo[0]) != placement_key(shortest_first),
            "oracle_mode": "cosim_rtt",
        }

        for name, choices in policies.items():
            if not choices:
                row[f"{name}_rtt"] = ""
                row[f"{name}_regret"] = ""
                continue
            try:
                policy_rtt = policy_rtt_cosim(ctx, snapshot, tasks, choices)
            except Exception as exc:
                print(f"[WARN] snapshot {snapshot.get('snapshot_id')} policy {name}: {exc}")
                row[f"{name}_rtt"] = ""
                row[f"{name}_regret"] = ""
                continue
            sim_runs += 1
            row[f"{name}_rtt"] = policy_rtt
            row[f"{name}_regret"] = policy_rtt - oracle.rtt
        rows.append(row)
        gc.collect()

        if args.progress and len(rows) % max(1, args.progress) == 0:
            elapsed = time.monotonic() - start_time
            print(
                f"[progress] audited {len(rows)} snapshots ({sim_runs} co-sim runs, {elapsed:.0f}s elapsed)",
                flush=True,
            )

    if timed_out:
        print(f"[TIME] partial audit: {len(rows)}/500 snapshots within {args.max_runtime_s}s budget")
    if skipped:
        print(f"[WARN] skipped {skipped} snapshots above combo limit or without candidates")
    print(f"Co-sim runs executed: {sim_runs}", flush=True)
    return rows


def write_csv(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No audit rows produced.")
        return
    n = len(rows)
    non_shortest = sum(1 for row in rows if row.get("oracle_non_shortest"))
    print(f"Snapshots audited: {n}")
    print(f"Oracle non-shortest first choice: {non_shortest}/{n} ({non_shortest / n * 100:.1f}%)")
    for policy in ("knative", "knative_ect", "herocache_like", "gnn"):
        regrets = [_safe_float(row.get(f"{policy}_regret")) for row in rows if row.get(f"{policy}_regret") != ""]
        if not regrets:
            continue
        avg = sum(regrets) / len(regrets)
        wins = sum(1 for value in regrets if abs(value) < 1e-9)
        print(f"{policy}: avg_regret={avg:.6f}, oracle_ties={wins}/{len(regrets)}")

    hrc_regrets = [_safe_float(row.get("herocache_like_regret")) for row in rows if row.get("herocache_like_regret") != ""]
    kn_regrets = [_safe_float(row.get("knative_regret")) for row in rows if row.get("knative_regret") != ""]
    ect_regrets = [_safe_float(row.get("knative_ect_regret")) for row in rows if row.get("knative_ect_regret") != ""]
    if hrc_regrets and kn_regrets:
        hrc_vs_kn = sum(1 for h, k in zip(hrc_regrets, kn_regrets) if h < k - 1e-9)
        print(f"HRC wins vs knative (queue): {hrc_vs_kn}/{len(hrc_regrets)}")
    if hrc_regrets and ect_regrets:
        hrc_vs_ect = sum(1 for h, e in zip(hrc_regrets, ect_regrets) if h < e - 1e-9)
        print(f"HRC wins vs knative_ect: {hrc_vs_ect}/{len(hrc_regrets)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "simulation_data" / "space_with_network.json",
    )
    parser.add_argument(
        "--sim-input",
        type=Path,
        default=PROJECT_ROOT / "data" / "nofs-ids",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-combos", type=int, default=200000)
    parser.add_argument("--gnn-model", type=Path, default=None)
    parser.add_argument("--progress", type=int, default=10, help="Print progress every N snapshots")
    parser.add_argument(
        "--max-runtime-s",
        type=int,
        default=None,
        help="Stop audit after this many seconds (partial results written)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze(args)
    write_csv(rows, args.output)
    print_summary(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
