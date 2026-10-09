"""Wall-clock cost of the scheduler's decision, per call and per task (r1_attribution_v1, descriptive).

Measurement only: it reads ``time.perf_counter`` around code the scheduler already runs and schedules nothing in SimPy, so event order and
every simulated field are unchanged. A call is one scheduling decision: a batch on the batch path (decode, refine and keep-warm, the
span ``inference_time`` already covers), a single task on the per-arrival path (the synchronous code of the policy's ``placement``).
Model loading is timed apart (``set_model_load``) and never enters a per-decision figure.
"""
from __future__ import annotations

import statistics as st
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

_MODEL_LOAD_S = 0.0


def set_model_load(seconds: float) -> None:
    global _MODEL_LOAD_S
    _MODEL_LOAD_S += float(seconds)


class DecisionLog:
    def __init__(self) -> None:
        self.calls: List[Tuple[int, float, int]] = []  # (tasks decided, seconds, refine moves)

    def add(self, n_tasks: int, seconds: float, refine_moves: int = 0) -> None:
        self.calls.append((int(n_tasks), float(seconds), int(refine_moves)))

    @staticmethod
    def _wq(pairs: List[Tuple[float, int]], q: float) -> float:
        """Weighted quantile of (value, weight) pairs, the nearest-rank rule."""
        pairs = sorted(pairs)
        total = sum(w for _, w in pairs)
        cut, acc = q * total, 0
        for v, w in pairs:
            acc += w
            if acc >= cut:
                return v
        return pairs[-1][0]

    def summary(self) -> Dict[str, Any]:
        secs = sorted(c[1] for c in self.calls)
        out: Dict[str, Any] = {"model_load_s": _MODEL_LOAD_S, "calls": len(self.calls)}
        if not self.calls:
            return out
        per_task = [(s / n, n) for n, s, _ in self.calls if n > 0]
        out.update({
            "tasks": sum(c[0] for c in self.calls), "total_s": sum(secs),
            "call_median_s": st.median(secs), "call_p95_s": secs[min(len(secs) - 1, int(0.95 * len(secs)))], "call_max_s": secs[-1],
            "per_task_median_s": self._wq(per_task, 0.5), "per_task_p95_s": self._wq(per_task, 0.95),
            "per_task_mean_s": sum(secs) / max(1, sum(c[0] for c in self.calls)),
            "refine_moves_total": sum(c[2] for c in self.calls), "refine_moves_per_call_mean": sum(c[2] for c in self.calls) / len(self.calls),
            "refine_moves_per_call_median": st.median(c[2] for c in self.calls),
        })
        return out


def log_of(scheduler) -> DecisionLog:
    return scheduler.__dict__.setdefault("decision_log", DecisionLog())


def timed_generator(gen: Generator, sink: List[float]) -> Generator:
    """Drive ``gen`` exactly as SimPy would and add the wall time spent inside its own code (not while it waits) to ``sink[0]``.
    It yields the same events in the same order and returns the same value, so the simulation cannot tell it from ``gen``."""
    value: Any = None
    exc: Optional[BaseException] = None
    while True:
        t0 = time.perf_counter()
        try:
            event = gen.throw(exc) if exc is not None else gen.send(value)
        except StopIteration as stop:
            sink[0] += time.perf_counter() - t0
            return stop.value
        sink[0] += time.perf_counter() - t0
        try:
            value = yield event
            exc = None
        except GeneratorExit:
            gen.close()
            raise
        except BaseException as e:  # a failed event or an interrupt: the inner generator sees it, as it would under SimPy
            exc = e
