"""partial_state_v6: the platform's in-flight work as two appended platform columns.

Under HEROSIM_REPLICA_RELEASE=1 a task leaves `platform.queue` when its input stage starts and then waits on the replica's compute lock, so the queue columns
(dim 7, the usage ratio, the load-seconds backlog) cannot see the tasks that are already committed to a replica but not yet served. v6 appends, after the
four-type columns, per platform:

    inflight_count      `scaleout.platform_in_flight`: queued (plus virtual backlog) + admitted (in cold start or ingress) + in `platform.inflight`
                        + the current task when it is not in `inflight`; divided by the queue column's own divisor (the batch's adaptive queue norm), as the queue is
    inflight_work       log1p of the remaining service seconds of the tasks in `platform.inflight` (below); the accounting of the cd_ib0_inflightfull reference bar

Remaining service of one in-flight task, by stage, using the code's own estimates (no constant is invented):

    compute started     max(0, compute_start_time + execution - now) + output       (execution: the table value on this platform type)
    waiting for lock    execution + output
    input stage         execution + _approx_comm(task type)                          (its progress is unknown to a scheduler, so it is charged whole)

with output = `infrastructure._output_estimate_seconds`. A task still in cold start or ingress (`platform.admitted`) counts in `inflight_count` only. This is
`live_audit.inflight_full_seconds` (the drain of the `cd_ib0_inflightfull` bar) except for a task in its output write, which here owes the whole output
estimate: the stored corpus states record a task as done computing but not how far its output write has got, and the two sides must compute the same number.

The corpus states (`warm_snapshot.json` -> snapshot.fidelity.ghosts, one record per task in `inflight` or `admitted`) carry what the cache needs; the cache
builder computes the same two numbers from them with `state_from_ghosts`, and the live capture computes them from the platform objects with `live_state`, so
the two paths cannot define the columns differently (tests/test_inflight_features.py). The live capture requires the replica-release physics.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

INFLIGHT_DIM = 2
COUNT_KEY = "inflight_count"
WORK_KEY = "inflight_work_seconds"
_STAGES_WITH_WORK = ("compute", "lock_wait", "input_io", "rendezvous")
_ADMITTED_STAGES = ("cold", "ingress")


def inflight_enabled(contract: Optional[str] = None) -> bool:
    from src.policy.tabular.reduced_features import INFLIGHT_CONTRACTS, resolve_partial_state_contract

    return resolve_partial_state_contract(contract) in INFLIGHT_CONTRACTS


def _table_exec(task_type: Mapping[str, Any], platform_type: str) -> float:
    return float(task_type.get("executionTime", {}).get(platform_type, 0.0) or 0.0)


def _output_seconds(task: Any) -> float:
    from src.placement.infrastructure import _output_estimate_seconds

    return float(_output_estimate_seconds(task))


def work_seconds_live(platform: Any) -> float:
    """Remaining service seconds of the tasks in `platform.inflight` (module docstring)."""
    from src.placement.live_snapshot_seed import _approx_comm

    now = float(platform.env.now)
    plat_type = platform.type["shortName"]
    total = 0.0
    for task in list(platform.inflight):
        exec_s = _table_exec(task.type, plat_type)
        start = getattr(task, "compute_start_time", None)
        if start is not None:
            total += max(0.0, float(start) + exec_s - now) + _output_seconds(task)
        elif getattr(task, "io_end_time", None) is not None:
            total += exec_s + _output_seconds(task)
        else:
            total += exec_s + _approx_comm(task.type)
    return total


def live_state(platform: Any) -> Tuple[int, float]:
    """(inflight_count, remaining in-flight service seconds) of a live platform. Requires the replica-release physics."""
    from src.placement.infrastructure import REPLICA_RELEASE
    from src.placement.scaleout import platform_in_flight

    if not REPLICA_RELEASE:
        raise RuntimeError("FAIL LOUD: partial_state_v6 in-flight columns are defined for HEROSIM_REPLICA_RELEASE=1 only")
    return int(platform_in_flight(platform)), work_seconds_live(platform)


def state_from_ghosts(
    ghosts: Iterable[Mapping[str, Any]],
    queue_length: int,
    task_types: Mapping[str, Mapping[str, Any]],
    platform_type: str,
) -> Tuple[int, float]:
    """The same two numbers from a stored state: the fidelity ghost records of ONE platform and its queue length (queue + virtual backlog).

    Fails loud when a record is not one of the known stages, or when its execution is not the table value (an execution-physics factor other than 1 would make the
    live table-based estimate and the stored one differ)."""
    from src.placement.live_snapshot_seed import _approx_comm

    count = int(queue_length)
    total = 0.0
    for g in ghosts:
        stage = g.get("stage")
        count += 1
        if stage in _ADMITTED_STAGES:
            continue
        if stage not in _STAGES_WITH_WORK:
            raise ValueError(f"FAIL LOUD: unknown in-flight stage {stage!r} in record {g.get('tid')}")
        ttype = task_types[str(g["fn"])]
        table = _table_exec(ttype, platform_type)
        if abs(float(g["exec"]) - table) > 1e-12 * max(1.0, table):
            raise ValueError(f"FAIL LOUD: record {g.get('tid')} on {g.get('q')}: stored execution {g['exec']!r} is not the table value {table!r} "
                             "(execution physics is not 1; the in-flight columns would not match the live estimate)")
        out = float(g["output"])
        if stage == "compute":
            total += float(g["compute_remaining"]) + out
        elif stage == "lock_wait":
            total += table + out
        else:  # input_io, rendezvous
            total += table + _approx_comm(ttype)
    return count, total


def columns(count: float, work_seconds: float, queue_norm: float) -> np.ndarray:
    """The two appended columns, shared by the live and the cache builders: count over the queue column's divisor, log1p of the work seconds."""
    return np.asarray([float(count) / float(queue_norm), float(np.log1p(max(0.0, float(work_seconds))))], dtype=np.float64)


def widen_graph(graph: Any) -> None:
    """Cache side: append the held-apart in-flight columns to platform_features, in place, and drop them."""
    import torch

    pe = getattr(graph, "inflight_platform_extra", None)
    if pe is None:
        raise ValueError("partial_state_v6: graph carries no in-flight columns (built under another contract)")
    graph.platform_features = torch.cat([graph.platform_features, pe.to(graph.platform_features.dtype)], dim=1)
    del graph.inflight_platform_extra
