"""
Platform warmth predicates — single source of truth for pull vs sandbox gates.

Tier 1: node disk cache can skip image pull (node_disk_v2).
Tier 2: sandbox cold-start uses previous_task only; pull uses disk only in v2.
Tier 3: node_cold_count / estimated_pull_remaining_sec (FilterStore depth observables).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform

PLATFORM_REUSE_V1 = "platform_reuse_v1"
NODE_DISK_V2 = "node_disk_v2"
VALID_WARMTH_PHYSICS = frozenset({PLATFORM_REUSE_V1, NODE_DISK_V2})

# FilterStore pull observables (storage_contention.md § fair feature candidates).
# shared_fate = cold/total saturates at 1.0; absolute cold_count distinguishes N=1 vs N=12.
DEFAULT_T_PULL_S = 31.3038  # dnn1 @ flashCard write ∩ 100 MB/s network
DEFAULT_STORAGE_WRITE_MBPS = 171.0
DEFAULT_NETWORK_BANDWIDTH_MBPS = 100.0
DEFAULT_STORAGE_WRITE_LATENCY_S = 0.00012
# Normalize estimated_pull_remaining_sec into ~O(1) for GNN/MLP (31s→0.31, 375s→3.75).
ESTIMATED_PULL_REMAINING_NORM_S = 100.0


WARMTH_PHYSICS_SOURCE_CONFIG = "config"
WARMTH_PHYSICS_SOURCE_ENV = "env"
WARMTH_PHYSICS_SOURCE_DEFAULT = "default"


class InvalidWarmthPhysicsError(ValueError):
    """Raised when warmth_physics is not a recognized value."""


class ImplicitWarmthPhysicsError(ValueError):
    """Raised when a run relies on the default warmth physics instead of declaring it."""


def estimate_unit_pull_sec(
    image_size_gb: Optional[float] = None,
    *,
    storage_write_mbps: float = DEFAULT_STORAGE_WRITE_MBPS,
    network_bandwidth_mbps: float = DEFAULT_NETWORK_BANDWIDTH_MBPS,
    write_latency_s: float = DEFAULT_STORAGE_WRITE_LATENCY_S,
) -> float:
    """Unit cold-image pull duration (seconds), matching autoscaler FilterStore math."""
    if image_size_gb is None or float(image_size_gb) <= 0.0:
        return float(DEFAULT_T_PULL_S)
    speed = min(float(storage_write_mbps), float(network_bandwidth_mbps))
    if speed <= 0.0:
        raise ValueError(
            f"Invalid pull speed={speed} "
            f"(storage_write_mbps={storage_write_mbps}, network_bandwidth_mbps={network_bandwidth_mbps})"
        )
    return float(image_size_gb) / (speed / 1024.0) + float(write_latency_s)


def estimated_pull_remaining_sec(node_cold_count: float, unit_pull_sec: float) -> float:
    """Schedule-time estimate of FilterStore serialization wait ≈ cold_count × T_pull."""
    if node_cold_count < 0:
        raise ValueError(f"node_cold_count must be >= 0, got {node_cold_count}")
    if unit_pull_sec < 0:
        raise ValueError(f"unit_pull_sec must be >= 0, got {unit_pull_sec}")
    return float(node_cold_count) * float(unit_pull_sec)


def normalize_estimated_pull_remaining_sec(remaining_sec: float) -> float:
    """Scale seconds for platform feature vectors (fail-loud on bad norm constant)."""
    if ESTIMATED_PULL_REMAINING_NORM_S <= 0.0:
        raise ValueError(
            f"ESTIMATED_PULL_REMAINING_NORM_S must be > 0, got {ESTIMATED_PULL_REMAINING_NORM_S}"
        )
    return float(remaining_sec) / float(ESTIMATED_PULL_REMAINING_NORM_S)


def unit_pull_sec_from_task_priors(
    task_priors: Optional[Any],
    platform_type: str,
    *,
    preferred_task_types: Optional[tuple] = None,
    network_bandwidth_mbps: float = DEFAULT_NETWORK_BANDWIDTH_MBPS,
) -> float:
    """
    Resolve T_pull from task-type imageSize priors for a platform type.

    Prefers dnn1 then dnn2; falls back to DEFAULT_T_PULL_S when priors lack imageSize.
    Storage write speed/latency use flashCard defaults (gate physics).
    """
    prefs = preferred_task_types or ("dnn1", "dnn2")
    if not task_priors:
        return float(DEFAULT_T_PULL_S)
    for task_name in prefs:
        priors = task_priors.get(str(task_name)) if hasattr(task_priors, "get") else None
        if not isinstance(priors, dict):
            continue
        image_map = priors.get("imageSize")
        if not isinstance(image_map, dict):
            continue
        raw = image_map.get(platform_type)
        if raw is None:
            continue
        try:
            image_gb = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Non-numeric imageSize for task={task_name!r} platform={platform_type!r}: {raw!r}"
            ) from exc
        if image_gb > 0.0:
            return estimate_unit_pull_sec(
                image_gb,
                storage_write_mbps=DEFAULT_STORAGE_WRITE_MBPS,
                network_bandwidth_mbps=network_bandwidth_mbps,
                write_latency_s=DEFAULT_STORAGE_WRITE_LATENCY_S,
            )
    return float(DEFAULT_T_PULL_S)


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


def describe_warmth_physics(
    infra_value: Optional[str] = None,
    env_override: Optional[str] = None,
) -> dict:
    """
    Resolve warmth physics and report which layer supplied the value.

    The two physics versions differ by ~100x in live total RTT on identical
    configs, so a run that silently takes the default is not comparable to one
    that declared it. Callers must record the source alongside the value.
    """
    if infra_value is not None and infra_value != "":
        return {
            "warmth_physics": resolve_warmth_physics(infra_value),
            "warmth_physics_source": WARMTH_PHYSICS_SOURCE_CONFIG,
        }
    env_raw = env_override if env_override is not None else os.environ.get("HEROSIM_WARMTH_PHYSICS")
    if env_raw is not None and env_raw != "":
        return {
            "warmth_physics": resolve_warmth_physics(None, env_raw),
            "warmth_physics_source": WARMTH_PHYSICS_SOURCE_ENV,
        }
    return {
        "warmth_physics": PLATFORM_REUSE_V1,
        "warmth_physics_source": WARMTH_PHYSICS_SOURCE_DEFAULT,
    }


def require_explicit_warmth_physics(descriptor: dict) -> None:
    """
    Fail loudly when HEROSIM_REQUIRE_EXPLICIT_PHYSICS is set and physics is implicit.

    Every sweep script must set this so cross-sweep tables cannot mix regimes.
    """
    if descriptor["warmth_physics_source"] != WARMTH_PHYSICS_SOURCE_DEFAULT:
        return
    raw = os.environ.get("HEROSIM_REQUIRE_EXPLICIT_PHYSICS", "")
    if raw.strip().lower() not in ("1", "true", "yes", "on"):
        return
    raise ImplicitWarmthPhysicsError(
        "warmth_physics was not declared: set it in the space config "
        "(\"warmth_physics\": \"node_disk_v2\") or export HEROSIM_WARMTH_PHYSICS. "
        f"Implicit default is {PLATFORM_REUSE_V1!r}, which is not the Regime A physics."
    )


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
