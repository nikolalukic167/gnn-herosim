"""accel_replica_v1: which compatible platform carries a function's replica (preinit.replica_placement_rule).

One definition, used at every site that picks a platform for a replica:
  generator (generate_infrastructure.py)       list order of the node's platforms          -> order_platforms
  precreate_replicas fallback (simulation.py)  list order of the node's platforms          -> order_platforms
  autoscaler create_first_replica              alphabetical platform type, first that fits -> order_platform_types
  autoscaler create_replica                    most available node, then platform id       -> restrict_to_fastest + the existing key

`first_compatible` (the default) leaves every site exactly as it was. `fastest_compatible` ranks by the task type's
executionTime table (ties: platform id for platforms, platform name for types). The rule of a run is set once per simulation from the
infrastructure config (`replica_placement_rule`, written only when it is not the default) and recorded in the run provenance.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Mapping, Sequence, TypeVar

FIRST = "first_compatible"
FASTEST = "fastest_compatible"
RULES = (FIRST, FASTEST)

_rule = FIRST
T = TypeVar("T")


def validate(rule: str) -> str:
    if rule not in RULES:
        raise ValueError(f"preinit.replica_placement_rule must be one of {RULES}, got {rule!r}")
    return rule


def set_rule(rule: str | None) -> str:
    global _rule
    _rule = validate(rule or FIRST)
    return _rule


def current_rule() -> str:
    return _rule


def _exec_s(task_type: Mapping[str, Any], short_name: str) -> float:
    table = task_type.get("executionTime") or {}
    if short_name not in table:
        raise ValueError(
            f"replica_placement_rule={FASTEST}: task type {task_type.get('name')!r} has no executionTime for platform {short_name!r}"
        )
    return float(table[short_name])


def order_platforms(task_type: Mapping[str, Any], items: Sequence[T], short_name: Callable[[T], str], platform_id: Callable[[T], int],
                    rule: str | None = None) -> List[T]:
    """Candidates in the order a replica walk takes them. first_compatible: as given (the node's list order)."""
    rule = validate(rule or _rule)
    if rule == FIRST:
        return list(items)
    return sorted(items, key=lambda p: (_exec_s(task_type, short_name(p)), platform_id(p)))


def order_platform_types(task_type: Mapping[str, Any], short_names: Iterable[str], rule: str | None = None) -> List[str]:
    """Platform types in the order create_first_replica tries them. first_compatible: alphabetical (today)."""
    rule = validate(rule or _rule)
    names = sorted(short_names)
    if rule == FIRST:
        return names
    return sorted(names, key=lambda n: (_exec_s(task_type, n), n))


def restrict_to_fastest(task_type: Mapping[str, Any], candidates: Sequence[T], short_name: Callable[[T], str], rule: str | None = None) -> List[T]:
    """create_replica: under fastest_compatible keep only the candidates of the fastest platform type present; first_compatible: unchanged."""
    rule = validate(rule or _rule)
    cands = list(candidates)
    if rule == FIRST or not cands:
        return cands
    best = min(_exec_s(task_type, short_name(c)) for c in cands)
    return [c for c in cands if _exec_s(task_type, short_name(c)) == best]
