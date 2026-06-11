"""
Platform warmth predicates — single source of truth for pull vs sandbox gates.

Tier 1: node disk cache can skip image pull (node_disk_v2).
Tier 2: sandbox cold-start uses previous_task only; pull uses disk only in v2.
Tier 3 stubs: see comments at bottom of file.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

PLATFORM_REUSE_V1 = "platform_reuse_v1"
NODE_DISK_V2 = "node_disk_v2"
VALID_WARMTH_PHYSICS = frozenset({PLATFORM_REUSE_V1, NODE_DISK_V2})

# TIER3_STUB: node_pulls_in_flight counter → estimated_pull_remaining_sec feature
# TIER3_STUB: storage_busy = len(node.storage.items) < expected_local_count
# See memory/storage_contention.md § fair feature candidates


class InvalidWarmthPhysicsError(ValueError):
    """Raised when warmth_physics is not a recognized value."""


def resolve_warmth_physics(
    infra_value: Optional[str] = None,
    env_override: Optional[str] = None,
) -> str:
    """
    Resolve warmth physics version from infrastructure config and/or env override.

    Priority: explicit infra_value, then HEROSIM_WARMTH_PHYSICS env, then default v1.
    Fail-loud on unknown values.
    """
    raw = infra_value
    if raw is None or raw == "":
        raw = env_override if env_override is not None else os.environ.get("HEROSIM_WARMTH_PHYSICS")
    if raw is None or raw == "":
        return PLATFORM_REUSE_V1
    if raw not in VALID_WARMTH_PHYSICS:
        raise InvalidWarmthPhysicsError(
            f"Invalid warmth_physics={raw!r}; expected one of {sorted(VALID_WARMTH_PHYSICS)}"
        )
    return raw


def _task_type_name(task_type: Any) -> str:
    if isinstance(task_type, dict):
        return task_type["name"]
    return task_type.type["name"]


def sandbox_is_warm(platform: "Platform", task_type: Any) -> bool:
    """True when the immediately previous task on this platform matches task_type (sandbox reuse)."""
    if platform.previous_task is None:
        return False
    prev_name = platform.previous_task.type.get("name")
    return prev_name is not None and prev_name == _task_type_name(task_type)


def node_has_cached_image(
    node: "Node",
    platform_short_name: str,
    task_type: Any,
    *,
    active_storage: Any = None,
) -> bool:
    """True if any local storage on the node has the image (including checked-out devices)."""
    seen: set[int] = set()
    if active_storage is not None and not active_storage.type.get("remote"):
        seen.add(id(active_storage))
        if active_storage.has_function(platform_short_name, task_type):
            return True
    for node_storage in node.storage.items:
        if id(node_storage) in seen:
            continue
        if node_storage.type.get("remote"):
            continue
        if node_storage.has_function(platform_short_name, task_type):
            return True
    return False


def needs_image_pull(
    physics: str,
    platform: "Platform",
    node: "Node",
    task_type: Any,
    *,
    active_storage: Any = None,
) -> bool:
    """
    True when initialize_replica must run the full image pull path.

    v1 (platform_reuse_v1): coupled — same predicate as sandbox (previous_task match skips pull).
    v2 (node_disk_v2): disk-scoped only — previous_task never skips pull (eviction-safe).
    """
    if physics not in VALID_WARMTH_PHYSICS:
        raise InvalidWarmthPhysicsError(
            f"Invalid warmth_physics={physics!r}; expected one of {sorted(VALID_WARMTH_PHYSICS)}"
        )
    if physics == PLATFORM_REUSE_V1:
        return not sandbox_is_warm(platform, task_type)
    return not node_has_cached_image(
        node,
        platform.type["shortName"],
        task_type,
        active_storage=active_storage,
    )


def image_pull_disk_hit(
    physics: str,
    platform: "Platform",
    node: "Node",
    task_type: Any,
    *,
    active_storage: Any = None,
) -> bool:
    """True when pull is skipped due to a node disk cache hit (v2 only; always False in v1)."""
    if physics != NODE_DISK_V2:
        return False
    return node_has_cached_image(
        node,
        platform.type["shortName"],
        task_type,
        active_storage=active_storage,
    )
