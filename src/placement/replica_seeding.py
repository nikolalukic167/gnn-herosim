"""Shared helpers for integrating pre-seeded replicas into live orchestrators.

Knative/GNN historically ignored ``initial_replicas``, so free policies started
empty and piled onto a single cold replica (warm after first pull → near-oracle).
Regime B scarce-preinit stub requires these helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform


def integrate_initial_replicas(
    *,
    replicas: Dict[str, Set[Tuple["Node", "Platform"]]],
    available_resources: Dict["Node", Set["Platform"]],
    initial_replicas: Dict[str, Set[Tuple["Node", "Platform"]]],
    task_types: Dict[str, Any],
    average_contention: Dict[str, Dict[Tuple[int, int], float]],
    label: str,
) -> int:
    """
    Copy pre-seeded replicas into system state and remove them from available pool.

    Returns number of replica tuples integrated. Fail loud if a seed references an
    unknown task type or a platform missing from available_resources.
    """
    if not initial_replicas:
        return 0

    total = 0
    print(f"\n=== {label}: integrating {len(initial_replicas)} pre-seeded replica sets ===")
    for task_type, replica_set in initial_replicas.items():
        if not replica_set:
            continue
        if task_type not in replicas:
            raise KeyError(
                f"{label}: initial_replicas has unknown task_type={task_type!r}; "
                f"known={sorted(replicas)}"
            )
        replicas[task_type] = set(replica_set)
        total += len(replica_set)
        print(f"  {task_type}: {len(replica_set)} replicas")
        for node, platform in replica_set:
            if node not in available_resources or platform not in available_resources[node]:
                raise RuntimeError(
                    f"{label}: seeded {node.node_name}:{platform.id} for {task_type} "
                    f"missing from available_resources (double-book or create_nodes mismatch)"
                )
            available_resources[node].remove(platform)
            node.available_platforms -= 1
            memory_required = task_types[task_type]["memoryRequirements"][
                platform.type["shortName"]
            ]
            node.available_memory -= memory_required
            average_contention[task_type][(node.id, platform.id)] = 0.0
    print(f"=== {label}: {total} initial replicas integrated ===\n")
    return total


def start_deferred_cold_init(
    env: Any,
    autoscaler: Any,
    node: "Node",
    platform: "Platform",
    replicas_for_type: Set[Tuple["Node", "Platform"]],
    task_type: Any,
    system_state: Any,
) -> None:
    """Kick image pull for a deferred-cold replica at placement time (FilterStore path)."""
    if platform.initialized.triggered:
        return
    env.process(
        autoscaler.initialize_replica(
            (node, platform),
            replicas_for_type,
            task_type,
            system_state,
        )
    )
