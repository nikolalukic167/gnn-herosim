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
    # peer_affinity_v1 (2026-09-10): a corpus generated with `replica_overlap` (equivalently
    # `--allow-non-unique-replicas`) legitimately hosts replicas of SEVERAL task types on one
    # platform, so the same (node, platform) appears in more than one seeded set. Availability
    # and memory are properties of the physical slot and must be charged ONCE; contention is
    # tracked per (task_type, node, platform) and must be seeded for EVERY type. Claiming
    # unconditionally made the second task type to reach a shared slot find it already gone and
    # raise, which is why every reactive/GNN live arm refused such a corpus while
    # `determined_determined` replayed it fine -- `DeterminedOrchestrator` already carries this
    # exact guard (`src/policy/determined/orchestrator.py:121-134`), and commit dcb8e12 fixed
    # three other disjoint-platform assumptions in the live path but not this shared helper.
    #
    # Behaviour on a NON-overlap corpus is unchanged, structurally: the generator refuses
    # duplicate platform assignments when `replica_overlap` is off
    # (`src/generate_infrastructure.py:744-757`), so every (node, platform) is seen exactly
    # once, the guard is true on every iteration, and `claimed` is never consulted. The raise is
    # narrowed to the case its message names -- a slot that never existed (create_nodes
    # mismatch) -- rather than removed.
    claimed: Set[Tuple["Node", "Platform"]] = set()
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
            if node in available_resources and platform in available_resources[node]:
                available_resources[node].remove(platform)
                node.available_platforms -= 1
                memory_required = task_types[task_type]["memoryRequirements"][
                    platform.type["shortName"]
                ]
                node.available_memory -= memory_required
                claimed.add((node, platform))
            elif (node, platform) not in claimed:
                raise RuntimeError(
                    f"{label}: seeded {node.node_name}:{platform.id} for {task_type} "
                    f"missing from available_resources (double-book or create_nodes mismatch)"
                )
            # Always: contention is keyed by (task_type, node, platform), so a shared slot
            # needs an entry per hosting type or the autoscaler KeyErrors later.
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
