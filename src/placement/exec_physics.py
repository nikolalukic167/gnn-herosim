"""Execution-time physics. See docs/lineages/hidden_exec_s0_v1.md.

`table_v0` (default): a task runs exactly `executionTime[platform]` from the task-type table,
bit-identical to every run before this module existed.

`hidden_node_v1`: realized duration = table x node speed x (1 + beta x co-executing) x noise, where
  * node speed is a mean-one lognormal (sigma NODE_SIGMA) per node name,
  * beta ~ U(0, BETA_MAX) per node name, applied per OTHER platform on the node executing when this
    one starts (counted once, at start),
  * noise is a mean-one lognormal with coefficient of variation NOISE_CV per invocation.
Every draw is a hash of (HEROSIM_EXEC_SEED, key), never a shared RNG consumed in event order, so two
runs, two policies or two co-sim plans see the same node constants and the same noise per
(task, node, platform). Schedulers keep reading the table; only an arm that asks
`expected_factor` sees the hidden constants (the oracle).
"""

from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from statistics import NormalDist
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.placement.infrastructure import Platform

TABLE_V0 = "table_v0"
HIDDEN_NODE_V1 = "hidden_node_v1"
VALID_EXEC_PHYSICS = frozenset({TABLE_V0, HIDDEN_NODE_V1})

NODE_SIGMA = 0.5
BETA_MAX = 0.5
NOISE_CV = 0.2
_NOISE_SIGMA = math.sqrt(math.log(1.0 + NOISE_CV ** 2))
_STD_NORMAL = NormalDist()


class InvalidExecPhysicsError(ValueError):
    pass


def resolve_exec_physics() -> str:
    raw = os.environ.get("HEROSIM_EXEC_PHYSICS", "") or TABLE_V0
    if raw not in VALID_EXEC_PHYSICS:
        raise InvalidExecPhysicsError(
            f"Invalid HEROSIM_EXEC_PHYSICS={raw!r}; expected one of {sorted(VALID_EXEC_PHYSICS)}"
        )
    return raw


def exec_seed() -> int:
    raw = os.environ.get("HEROSIM_EXEC_SEED", "")
    if resolve_exec_physics() != TABLE_V0 and raw == "":
        raise InvalidExecPhysicsError("hidden_node_v1 needs HEROSIM_EXEC_SEED; an implicit seed is not an environment")
    return int(raw or 0)


def describe_exec_physics() -> dict:
    physics = resolve_exec_physics()
    if physics == TABLE_V0:
        return {"exec_physics": physics}
    return {"exec_physics": physics, "exec_seed": exec_seed(), "exec_node_sigma": NODE_SIGMA,
            "exec_beta_max": BETA_MAX, "exec_noise_cv": NOISE_CV}


def _uniform(key: str) -> float:
    digest = hashlib.sha256(f"{exec_seed()}|{key}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2.0 ** 64


def _mean_one_lognormal(sigma: float, key: str) -> float:
    return math.exp(sigma * _STD_NORMAL.inv_cdf(_uniform(key)) - sigma * sigma / 2.0)


@lru_cache(maxsize=None)
def node_speed(node_name: str) -> float:
    return _mean_one_lognormal(NODE_SIGMA, f"speed|{node_name}")


@lru_cache(maxsize=None)
def node_beta(node_name: str) -> float:
    return BETA_MAX * _uniform(f"beta|{node_name}")


def co_executing(platform: "Platform") -> int:
    return sum(1 for p in platform.node.platforms.items
               if p is not platform and getattr(p, "executing", False))


def expected_factor(platform: "Platform") -> float:
    """Mean realized/table ratio if a task started on `platform` now. 1.0 under table_v0."""
    if resolve_exec_physics() == TABLE_V0:
        return 1.0
    name = platform.node.node_name
    return node_speed(name) * (1.0 + node_beta(name) * co_executing(platform))


def realized_factor(platform: "Platform", task_id: int) -> float:
    if resolve_exec_physics() == TABLE_V0:
        return 1.0
    noise = _mean_one_lognormal(
        _NOISE_SIGMA, f"noise|{task_id}|{platform.node.node_name}|{platform.id}")
    return expected_factor(platform) * noise
