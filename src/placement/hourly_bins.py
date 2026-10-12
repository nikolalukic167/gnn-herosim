"""long_horizon_s0_v1: per-bin latency breakdown by ARRIVAL (dispatch) time, computed in-process from the task archive so the read
does not need the per-task raw file. Opt-in (HEROSIM_HOURLY_BINS=1) so every other run keeps its stats keys.

A task is counted as in the whole-run averages (orchestrator._stats_low_memory): not internal, dispatched and done. Bin b holds the
tasks with b * width <= dispatched < (b + 1) * width. The n-weighted bin means equal the whole-run means (tests/test_long_horizon.py).
"""
from __future__ import annotations

import bisect
import math
import os
from typing import Dict, Iterable, List, Optional

# per-task attribute -> bin key (mean over the bin's tasks)
FIELDS = (
    ("elapsed_time", "latency"),
    ("queue_time", "queue"),
    ("cold_start_time", "cold"),
    ("peer_exchange_time", "exchange"),
    ("execution_time", "exec"),
    ("pull_time", "pull"),
    ("communications_time", "comm"),
    ("network_latency", "network"),
    ("compute_time", "compute"),
)


def enabled() -> bool:
    return os.getenv("HEROSIM_HOURLY_BINS", "0") == "1"


def bin_width_s() -> float:
    w = float(os.getenv("HEROSIM_HOURLY_BIN_S", "3600"))
    if not w > 0:
        raise ValueError(f"HEROSIM_HOURLY_BIN_S must be > 0, got {w}")
    return w


def _val(task, attr: str) -> float:
    if attr == "elapsed_time":
        e = getattr(task, "elapsed_time", None)
        return float(e) if e is not None else float(task.done_time) - float(task.dispatched_time)
    return float(getattr(task, attr, 0.0) or 0.0)


def hourly_bins(tasks: Iterable, width_s: float) -> Dict[str, object]:
    rows = [t for t in tasks if not getattr(t, "is_internal", False)
            and getattr(t, "done_time", None) is not None and getattr(t, "dispatched_time", None) is not None]
    if not rows:
        raise ValueError("hourly_bins: no completed tasks")
    disp = sorted(float(t.dispatched_time) for t in rows)
    done = sorted(float(t.done_time) for t in rows)
    nbins = int(math.floor(disp[-1] / width_s)) + 1
    acc: List[Dict[str, float]] = [{k: 0.0 for _a, k in FIELDS} for _ in range(nbins)]
    n = [0] * nbins
    last_disp: List[Optional[float]] = [None] * nbins
    last_done: List[Optional[float]] = [None] * nbins
    for t in rows:
        d = float(t.dispatched_time)
        b = int(math.floor(d / width_s))
        n[b] += 1
        for attr, key in FIELDS:
            acc[b][key] += _val(t, attr)
        if last_disp[b] is None or d > last_disp[b] or (d == last_disp[b] and float(t.done_time) > last_done[b]):
            last_disp[b], last_done[b] = d, float(t.done_time)
    out = []
    for b in range(nbins):
        end = (b + 1) * width_s
        row = {"bin": b, "start_s": b * width_s, "end_s": end, "n": n[b],
               # tasks dispatched before the bin end (the bin is [start, end)) and not done by it
               "in_system_at_end": bisect.bisect_left(disp, end) - bisect.bisect_right(done, end)}
        for _a, key in FIELDS:
            row[key] = acc[b][key] / n[b] if n[b] else None
        # stability: the completion of the bin's last-arriving task, behind the bin end (> 0 = it finished after the bin closed)
        row["last_arrival_done_minus_end_s"] = (last_done[b] - end) if last_done[b] is not None else None
        out.append(row)
    return {"width_s": width_s, "n_tasks": len(rows), "bins": out,
            "last_dispatch_s": disp[-1], "last_done_s": done[-1]}
