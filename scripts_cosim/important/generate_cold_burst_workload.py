#!/usr/bin/env python3
"""
Generate Regime B cold-burst workloads with mixed task types and burst_id tags.

Mixed dnn1/dnn2 bursts force distinct image pulls through one FilterStore under
node_disk_v2, defeating the single-image disk-cache shortcut.

Usage:
    pipenv run python3 scripts_cosim/important/generate_cold_burst_workload.py \\
        --output data/nofs-ids/traces/workload-cold-burst-mixed.json

    pipenv run python3 scripts_cosim/important/generate_cold_burst_workload.py \\
        --hub-config simulation_data/normal_sim_sweeps/regime_b_hub9/configs/hub_k6_seek50.json \\
        --burst-sizes 4,8 --burst-interval 180 --num-bursts 3
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK_TYPES = ("dnn1", "dnn2")
DEFAULT_BURST_SIZES = (4, 8, 16)
DEFAULT_BURST_INTERVAL_S = 180.0
DEFAULT_CLIENT_COUNT = 20


def _parse_int_list(raw: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"expected comma-separated integers, got {raw!r}")
    if any(v < 1 for v in values):
        raise ValueError(f"burst sizes must be >= 1, got {values}")
    return values


def _hub_seeker_flags_from_config(
    hub_config: Dict[str, Any],
    *,
    seed: int,
    client_count: int = DEFAULT_CLIENT_COUNT,
    server_count: int = DEFAULT_CLIENT_COUNT,
) -> List[bool]:
    """Replicate degree_skewed_core client hub-seeker RNG (matches infra topology gen)."""
    topo = hub_config.get("network", {}).get("topology", {})
    if topo.get("type") != "degree_skewed_core":
        return [i >= client_count // 2 for i in range(client_count)]

    k_core = int(topo.get("k_core", 4))
    hub_frac = float(topo.get("hub_seeker_fraction", 0.40))
    p_core = float(topo.get("p_core", 0.95))
    p_periphery = float(topo.get("p_periphery", 0.15))

    rng = random.Random(seed)
    flags: List[bool] = []
    for i in range(client_count):
        is_hub_seeker = rng.random() < hub_frac
        for s_idx in range(server_count):
            in_core = s_idx < k_core
            if is_hub_seeker and in_core:
                p_conn = p_core
            elif not is_hub_seeker and not in_core:
                p_conn = p_periphery
            elif is_hub_seeker and not in_core:
                p_conn = p_periphery * 0.5
            else:
                p_conn = p_core * 0.3
            if rng.random() < p_conn:
                pass
        flags.append(is_hub_seeker)
    return flags


def clients_from_hub_config(
    hub_config: Dict[str, Any],
    *,
    seed: int,
    client_count: int = DEFAULT_CLIENT_COUNT,
    server_count: int = DEFAULT_CLIENT_COUNT,
    mode: str = "periphery",
) -> List[str]:
    """Select client nodes by topology role: periphery, hub_seeker, or all."""
    flags = _hub_seeker_flags_from_config(
        hub_config, seed=seed, client_count=client_count, server_count=server_count
    )
    if mode == "hub_seeker":
        clients = [f"client_node{i}" for i, seek in enumerate(flags) if seek]
    elif mode == "periphery":
        clients = [f"client_node{i}" for i, seek in enumerate(flags) if not seek]
    elif mode == "all":
        clients = [f"client_node{i}" for i in range(client_count)]
    else:
        raise ValueError(f"unknown client mode {mode!r}; use periphery, hub_seeker, or all")
    if not clients:
        clients = [f"client_node{i}" for i in range(client_count)]
    return clients


def periphery_clients_from_hub_config(
    hub_config: Dict[str, Any],
    *,
    seed: int,
    client_count: int = DEFAULT_CLIENT_COUNT,
    server_count: int = DEFAULT_CLIENT_COUNT,
) -> List[str]:
    return clients_from_hub_config(
        hub_config,
        seed=seed,
        client_count=client_count,
        server_count=server_count,
        mode="periphery",
    )


def generate_cold_burst_workload(
    *,
    burst_sizes: Sequence[int],
    burst_times: Sequence[float],
    task_types: Sequence[str],
    client_nodes: Sequence[str],
    seed: int,
    qos_name: str = "medium",
    max_duration_deviation: int = 15,
) -> Dict[str, Any]:
    if not burst_sizes:
        raise ValueError("burst_sizes must not be empty")
    if len(burst_times) != len(burst_sizes):
        raise ValueError(
            f"burst_times length ({len(burst_times)}) must match burst_sizes ({len(burst_sizes)})"
        )
    if not task_types:
        raise ValueError("task_types must not be empty")
    if not client_nodes:
        raise ValueError("client_nodes must not be empty")

    rng = random.Random(seed)
    events: List[Dict[str, Any]] = []
    for t_burst, n in zip(burst_times, burst_sizes):
        burst_id = f"T{t_burst:g}_N{n}"
        for _ in range(n):
            tt = rng.choice(list(task_types))
            events.append(
                {
                    "timestamp": float(t_burst),
                    "application": {"name": f"nofs-{tt}", "dag": {tt: []}},
                    "qos": {"name": qos_name, "maxDurationDeviation": max_duration_deviation},
                    "node_name": rng.choice(list(client_nodes)),
                    "burst_id": burst_id,
                }
            )

    duration = max(burst_times) + max(burst_sizes) * 35.0 + 60.0
    return {
        "rps": 0,
        "duration": duration,
        "events": events,
        "regime_b_meta": {
            "seed": seed,
            "burst_sizes": list(burst_sizes),
            "burst_times": list(burst_times),
            "task_types": list(task_types),
            "client_nodes": list(client_nodes),
            "num_events": len(events),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/nofs-ids/traces/workload-cold-burst-mixed.json",
    )
    parser.add_argument("--num-bursts", type=int, default=3)
    parser.add_argument(
        "--single-burst",
        action="store_true",
        help="One burst at t=0 (overrides --num-bursts and --burst-interval)",
    )
    parser.add_argument(
        "--burst-size",
        type=int,
        default=None,
        help="Tasks in single-burst mode (default: first value from --burst-sizes)",
    )
    parser.add_argument(
        "--burst-sizes",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BURST_SIZES[:3]),
        help="Comma-separated tasks per burst (cycles if fewer than num-bursts)",
    )
    parser.add_argument(
        "--client-mode",
        choices=["periphery", "hub_seeker", "all"],
        default="periphery",
        help="Route bursts from periphery (default), hub-seeker, or all clients",
    )
    parser.add_argument(
        "--burst-interval",
        type=float,
        default=DEFAULT_BURST_INTERVAL_S,
        help="Seconds between burst start times (default: 180)",
    )
    parser.add_argument(
        "--task-types",
        type=str,
        default=",".join(DEFAULT_TASK_TYPES),
        help="Comma-separated task types to mix within each burst",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hub-config",
        type=Path,
        default=None,
        help="Hub space config JSON — selects periphery clients matching topology seed",
    )
    parser.add_argument(
        "--client-nodes",
        type=str,
        default=None,
        help="Override client list (comma-separated); skips hub-config periphery logic",
    )
    args = parser.parse_args()

    burst_size_cycle = _parse_int_list(args.burst_sizes)
    task_types = tuple(t.strip() for t in args.task_types.split(",") if t.strip())
    if args.single_burst:
        n = args.burst_size if args.burst_size is not None else burst_size_cycle[0]
        burst_times = [0.0]
        burst_sizes = [n]
    else:
        burst_times = [i * args.burst_interval for i in range(args.num_bursts)]
        burst_sizes = [burst_size_cycle[i % len(burst_size_cycle)] for i in range(args.num_bursts)]

    if args.client_nodes:
        client_nodes = [c.strip() for c in args.client_nodes.split(",") if c.strip()]
    elif args.hub_config is not None:
        if not args.hub_config.exists():
            raise SystemExit(f"hub config not found: {args.hub_config}")
        hub_config = json.loads(args.hub_config.read_text())
        seed = int(hub_config.get("network", {}).get("topology", {}).get("seed", args.seed))
        client_count = int(hub_config.get("nodes", {}).get("client_nodes", {}).get("count", 20))
        server_count = int(hub_config.get("nodes", {}).get("server_nodes", {}).get("count", 20))
        client_nodes = clients_from_hub_config(
            hub_config,
            seed=seed,
            client_count=client_count,
            server_count=server_count,
            mode=args.client_mode,
        )
    else:
        client_nodes = [f"client_node{i}" for i in range(10, 20)]

    workload = generate_cold_burst_workload(
        burst_sizes=burst_sizes,
        burst_times=burst_times,
        task_types=task_types,
        client_nodes=client_nodes,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(workload, indent=2) + "\n")
    print(f"Wrote {len(workload['events'])} events -> {args.output}")
    print(f"  bursts: {len(burst_sizes)}  sizes={burst_sizes}  times={burst_times}")
    print(f"  task_types={list(task_types)}  clients={len(client_nodes)} mode={args.client_mode}")


if __name__ == "__main__":
    main()
