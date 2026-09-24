#!/usr/bin/env python3
"""
Shared metric extraction for live-sim result JSONs.

Result files reach 120MB because of the stats blob, so nothing here parses the
whole document. `total_rtt` is peeked from the head/tail; the 99-quantile
response-time distribution is pulled out with a bounded streaming scan.

Two metrics matter and only one was ever read by the compare scripts:
  - total_rtt: sum of per-task elapsed time (the paper's Regime A primary)
  - taskResponseTimeDistribution: p50/p90/p99 of per-task elapsed time, which is
    where a collision-robustness advantage would show up if it existed
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCALAR_KEYS = (
    "total_rtt",
    "num_tasks",
    "averageQueueTime",
    "coldStartProportion",
    "averageElapsedTime",
)
_QUANTILE_KEY = "taskResponseTimeDistribution"
_CHUNK_BYTES = 4 << 20


class MetricExtractionError(ValueError):
    """Raised when a result JSON does not contain a required metric."""


def _peek_blob(path: Path, window: int = 65536) -> str:
    size = path.stat().st_size
    with open(path, "rb") as fh:
        head = fh.read(window)
        tail = b""
        if size > 2 * window:
            fh.seek(size - window)
            tail = fh.read()
    return head.decode("utf-8", "ignore") + "\n" + tail.decode("utf-8", "ignore")


def peek_scalar(path: Path, key: str, blob: Optional[str] = None) -> Optional[float]:
    """First numeric occurrence of `key` in the head/tail window, or None."""
    if blob is None:
        blob = _peek_blob(path)
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)', blob)
    if match is None:
        return None
    return float(match.group(1))


def extract_number_array(path: Path, key: str) -> Optional[List[float]]:
    """
    Stream the file looking for `"key": [ ... ]` and return the parsed array.

    Only the array text is held in memory, so this is safe on 120MB results.
    """
    needle = f'"{key}"'
    pending = ""
    collecting = False
    buffer = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        while True:
            chunk = fh.read(_CHUNK_BYTES)
            if not chunk:
                break
            if collecting:
                end = chunk.find("]")
                if end == -1:
                    buffer += chunk
                    if len(buffer) > _CHUNK_BYTES:
                        raise MetricExtractionError(
                            f"{key} array in {path} exceeds {_CHUNK_BYTES} bytes"
                        )
                    continue
                buffer += chunk[: end + 1]
                return json.loads(buffer)

            window = pending + chunk
            idx = window.find(needle)
            if idx == -1:
                pending = window[-len(needle) :]
                continue
            open_idx = window.find("[", idx)
            if open_idx == -1:
                pending = window[idx:]
                continue
            end = window.find("]", open_idx)
            if end != -1:
                return json.loads(window[open_idx : end + 1])
            collecting = True
            buffer = window[open_idx:]
            pending = ""
    if collecting:
        raise MetricExtractionError(f"unterminated {key} array in {path}")
    return None


def _quantile(dist: List[float], pct: int) -> float:
    """
    Value at percentile `pct` from statistics.quantiles(n=100) output.

    That call returns 99 cut points, so index pct-1 is the pct-th percentile.
    """
    if not 1 <= pct <= 99:
        raise ValueError(f"pct must be in 1..99, got {pct}")
    if len(dist) != 99:
        raise MetricExtractionError(
            f"expected 99 quantile cut points, got {len(dist)}"
        )
    return float(dist[pct - 1])


def load_metrics(path: Path, *, require_tail: bool = True) -> Dict[str, Any]:
    """
    Extract comparison metrics from one result JSON.

    Fails loudly on a missing or non-positive total_rtt, and on a missing
    response-time distribution unless `require_tail` is False.
    """
    path = Path(path)
    blob = _peek_blob(path)

    metrics: Dict[str, Any] = {"path": str(path), "name": path.stem}
    for key in _SCALAR_KEYS:
        metrics[key] = peek_scalar(path, key, blob=blob)

    rtt = metrics.get("total_rtt")
    if rtt is None:
        raise MetricExtractionError(f"missing total_rtt in {path}")
    if not math.isfinite(rtt) or rtt <= 0:
        raise MetricExtractionError(f"non-positive/non-finite total_rtt={rtt} in {path}")

    physics = re.search(r'"warmth_physics"\s*:\s*"([^"]+)"', blob)
    source = re.search(r'"warmth_physics_source"\s*:\s*"([^"]+)"', blob)
    metrics["warmth_physics"] = physics.group(1) if physics else None
    metrics["warmth_physics_source"] = source.group(1) if source else None

    dist = extract_number_array(path, _QUANTILE_KEY)
    if dist is None:
        if require_tail:
            raise MetricExtractionError(f"missing {_QUANTILE_KEY} in {path}")
        metrics["p50"] = metrics["p90"] = metrics["p99"] = None
        return metrics

    metrics["p50"] = _quantile(dist, 50)
    metrics["p90"] = _quantile(dist, 90)
    metrics["p99"] = _quantile(dist, 99)
    return metrics


def mean(values: List[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)
