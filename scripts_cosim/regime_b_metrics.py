"""
Regime B metrics — transient cold-start burst scoring.

Uses max burst elapsed / last-task RTT instead of sum(elapsed) total_rtt,
which dilutes contended FilterStore catastrophes across cheap parallel tasks.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional


def attach_burst_ids_from_workload(
    task_rows: List[Dict[str, Any]],
    workload_events: List[Dict[str, Any]],
    *,
    burst_key: str = "burst_id",
    default_burst: str = "default",
) -> List[Dict[str, Any]]:
    """
    Copy burst_id from workload events onto task rows by task_id order.

    task_id is assumed to match the index of the corresponding workload event.
    """
    by_id = {int(r["task_id"]): dict(r) for r in task_rows}
    out: List[Dict[str, Any]] = []
    for idx, event in enumerate(workload_events):
        row = by_id.get(idx)
        if row is None:
            continue
        row[burst_key] = event.get(burst_key) or default_burst
        out.append(row)
    for row in task_rows:
        tid = int(row["task_id"])
        if tid not in {int(r["task_id"]) for r in out}:
            tagged = dict(row)
            tagged[burst_key] = default_burst
            out.append(tagged)
    out.sort(key=lambda r: int(r["task_id"]))
    return out


def burst_regime_summary(
    task_rows: List[Dict[str, Any]],
    *,
    burst_key: str = "burst_id",
    oracle_rtt: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Regime B metrics: evaluates transient cold-start intelligence.

    Never use sum(elapsed) for contention; uses max burst elapsed as primary score.
    """
    if not task_rows:
        raise ValueError("empty task_rows")

    by_burst: Dict[str, List[Dict[str, Any]]] = {}
    for row in task_rows:
        bid = row.get(burst_key) or "default"
        by_burst.setdefault(str(bid), []).append(row)

    burst_summaries: List[Dict[str, Any]] = []
    for bid, rows in sorted(by_burst.items()):
        ordered = sorted(rows, key=lambda r: int(r["task_id"]))
        elapsed = [float(r["elapsed_time"]) for r in ordered]
        queue = [float(r["queue_time"]) for r in ordered]
        burst_summaries.append(
            {
                "burst_id": bid,
                "n_tasks": len(ordered),
                "max_elapsed_s": max(elapsed),
                # Last-by-task_id (arrival order), not max — storage_contention trap
                # uses the final task's wall time under FilterStore serialization.
                "last_task_elapsed_s": elapsed[-1],
                "mean_elapsed_s": mean(elapsed),
                "mean_queue_time_s": mean(queue),
                "max_queue_time_s": max(queue),
            }
        )

    # Primary score: worst burst max-elapsed (transient catastrophe), with
    # last-task reported alongside for FilterStore / cold-start audits.
    policy_rtt = max(b["max_elapsed_s"] for b in burst_summaries)
    last_task_rtt = max(b["last_task_elapsed_s"] for b in burst_summaries)
    out: Dict[str, Any] = {
        "regime_b_primary_score_s": policy_rtt,
        "last_task_rtt_s": last_task_rtt,
        "burst_summaries": burst_summaries,
    }
    if oracle_rtt is not None:
        out["oracle_rtt_s"] = float(oracle_rtt)
        out["oracle_regret_s"] = policy_rtt - float(oracle_rtt)
        out["oracle_regret_ratio"] = (
            policy_rtt / float(oracle_rtt) if float(oracle_rtt) > 0 else float("inf")
        )
    return out


def workload_has_burst_ids(workload: Dict[str, Any], *, burst_key: str = "burst_id") -> bool:
    """True when any workload event carries a burst tag (Regime B trace)."""
    events = workload.get("events") or []
    return any(ev.get(burst_key) for ev in events)


def regime_b_metrics_from_simulation(
    workload: Dict[str, Any],
    stats: Optional[Dict[str, Any]],
    *,
    extract_task_metrics_fn: Any,
    burst_key: str = "burst_id",
) -> Dict[str, Any]:
    """
    Build Regime B summary from simulation stats + tagged workload events.

    Requires SIM_FORCE_FULL_STATS=1 so taskResults are populated.
    """
    if not workload_has_burst_ids(workload, burst_key=burst_key):
        raise ValueError("workload has no burst_id tags")
    if not stats:
        raise ValueError("empty simulation stats")

    task_rows = extract_task_metrics_fn(stats)
    if not task_rows:
        raise ValueError(
            "no task rows in stats — set SIM_FORCE_FULL_STATS=1 for Regime B workloads"
        )
    tagged = attach_burst_ids_from_workload(
        task_rows,
        workload.get("events") or [],
        burst_key=burst_key,
    )
    summary = burst_regime_summary(tagged, burst_key=burst_key)
    summary["total_rtt_trap"] = total_rtt_trap_stats(tagged, burst_key=burst_key)
    return summary


def total_rtt_trap_stats(
    task_rows: List[Dict[str, Any]],
    *,
    burst_key: str = "burst_id",
) -> Dict[str, Any]:
    """
    Contrast misleading sum(elapsed) with Regime B max-burst score on same rows.
    """
    if not task_rows:
        raise ValueError("empty task_rows")
    total_rtt = sum(float(r["elapsed_time"]) for r in task_rows)
    regime = burst_regime_summary(task_rows, burst_key=burst_key)
    primary = float(regime["regime_b_primary_score_s"])
    return {
        "total_rtt_s": total_rtt,
        "regime_b_primary_score_s": primary,
        "total_rtt_over_primary_ratio": total_rtt / primary if primary > 0 else float("inf"),
    }
