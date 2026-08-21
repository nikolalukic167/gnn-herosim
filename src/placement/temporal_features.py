"""Temporal-remainder features (dims 9-11) — one formula, shared by cache and live.

What these are
--------------
Platform feature dims 9-11 describe how much work the platform is already committed to:

    dim 9   current_task_remaining   seconds left on the task in flight
    dim 10  cold_start_remaining     seconds left of a cold start
    dim 11  comm_remaining           seconds left of input/output transfer

All three are read from the captured system state when it recorded them, and *estimated
from queue depth* when it did not. That estimate is where two bugs lived, and it took four
independent copies of the formula to hide them:

  src/policy/tabular/feature_builder.py      live inference (GNN + MLP)
  src/notebooks/prepare_graphs_cache.py      main training cache
  src/notebooks/prepare_graphs_cache_seq.py  sequential-decode cache
  src/notebooks/prepare_graphs_ram.py        in-RAM cache variant

**Bug 1 — the estimate was gated at the wrong granularity.** All three cache builders wrote
`if temporal_state: <use recorded> else: <estimate>`, i.e. they decided per *snapshot*,
while live decides per *platform*. So on any snapshot carrying some recorded temporal data
but a queued platform with no remainder, the cache trained on 0.0 while live served an
estimate. Measured 2026-08-19 over 8 collections (including `shallow_v1`), 100% of datasets,
up to 75 of ~200 platforms each. That is a real train/serve divergence, and a *signed* one —
the cache side is a floor, so the bias does not average out.

**Bug 2 — live averaged over the wrong task types.** Live iterated every key of
`task-types.json` (`for _name, priors in task_types_data.items()`), which pulls in `rf` and
`cnn`. No corpus in this repo dispatches either, and `cnn` costs 3.09s on `rpiCpu` — a 9.5x
outlier that dominated the mean. Live served 0.0815 where the correct estimate is 0.0086.
The cache builders already restricted to `TASK_TYPES_VOCAB`, which is also the rule dim 12
(`target_concurrency`) follows for exactly this reason.

Both are fixed here, once. Returns raw **seconds**; every caller divides by 10.0 itself
because that normalization is part of their own feature layout.

Sibling modules with the same purpose and the same history: `queue_features.py`,
`topology_features.py`.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

# Cold start and comm are modelled as fixed fractions of the execution estimate. Preserved
# exactly as all four copies had them -- these numbers are load-bearing for every existing
# checkpoint, so they are pinned here rather than tuned.
COLD_START_FRACTION = 0.1
COMM_FRACTION = 0.05


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Tolerant float conversion. The cache builders used `_safe_float`, live used bare
    `float()`; the tolerant version is correct for both since captured state can carry
    nulls."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result and abs(result) != float("inf") else default


def mean_execution_time(
    platform_type: str,
    task_types_data: Optional[Mapping[str, Any]],
    task_types_vocab: Sequence[str],
) -> float:
    """Mean positive execution time on `platform_type`, over the vocab task types only.

    Restricted to the vocab on purpose -- see Bug 2 in the module docstring. Task types with
    no positive entry for this platform are skipped rather than counted as zero, so the mean
    is over platforms that can actually run the type.
    """
    if not task_types_data:
        return 0.0
    total = 0.0
    count = 0
    for task_type_name in task_types_vocab:
        priors = task_types_data.get(str(task_type_name)) or {}
        exec_map = priors.get("executionTime")
        if not isinstance(exec_map, Mapping):
            continue
        exec_time = _safe_float(exec_map.get(platform_type, 0.0), 0.0)
        if exec_time > 0:
            total += exec_time
            count += 1
    return total / count if count else 0.0


def temporal_remainders(
    *,
    queue_depth: float,
    recorded: Optional[Mapping[str, Any]],
    platform_type: str,
    task_types_data: Optional[Mapping[str, Any]],
    task_types_vocab: Sequence[str],
) -> Tuple[float, float, float]:
    """`(current_task_remaining, cold_start_remaining, comm_remaining)` in seconds.

    Args:
        queue_depth: This platform's **raw** queue depth (not normalized) — the estimate
            triggers on there being queued work at all, not on how much.
        recorded: This platform's captured temporal entry, or None/{} when absent.
        platform_type: Short name (`rpiCpu`, `xavierGpu`, ...) for the priors lookup.
        task_types_data: Parsed `task-types.json`.
        task_types_vocab: The task types this corpus actually dispatches.

    The recorded value always wins. The estimate fills in only for a platform that has
    queued work and no recorded remainder — per platform, which is the whole point.
    """
    entry = recorded or {}
    current = _safe_float(entry.get("current_task_remaining", 0.0), 0.0)
    cold_start = _safe_float(entry.get("cold_start_remaining", 0.0), 0.0)
    comm = _safe_float(entry.get("comm_remaining", 0.0), 0.0)

    if queue_depth > 0 and current == 0.0:
        avg_exec = mean_execution_time(platform_type, task_types_data, task_types_vocab)
        if avg_exec > 0:
            current = avg_exec
            cold_start = current * COLD_START_FRACTION
            comm = current * COMM_FRACTION

    return current, cold_start, comm
