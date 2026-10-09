"""One progress-rate watchdog for the long runs of the r1_attribution_v1 pipeline (capture, build sweep, live gate driver).

A run reports `progress` out of `total` (a trigger task id out of 50,000, sweep placements done out of `num_placements`). The watchdog
says "kill" when the run cannot finish inside its limit, instead of waiting for the limit:

  stall      progress has not moved for `stall_s` (default 300 s), at any time
  projected  after `warmup_s` (default 600 s), t + (total - progress) / rate > margin x limit_s, rate measured over the last `window_s`

The margin (default 1.5) is there because a run's rate is not constant: a cell can creep early and finish, and heavy cells slow as their
backlog grows. Time is passed in, so the rule is a pure function of the samples and testable without sleeping.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

STALL = "stall"
PROJECTED = "projected"


class ProgressWatchdog:
    def __init__(self, total: float, limit_s: float, *, warmup_s: float = 600.0, margin: float = 1.5, stall_s: float = 300.0,
                 window_s: float = 300.0):
        if total <= 0 or limit_s <= 0:
            raise ValueError("total and limit_s must be positive")
        self.total, self.limit_s = float(total), float(limit_s)
        self.warmup_s, self.margin, self.stall_s, self.window_s = warmup_s, margin, stall_s, window_s
        self.samples: List[Tuple[float, float]] = []
        self._moved_at: Optional[float] = None
        self.reason: Optional[str] = None

    def rate(self) -> Optional[float]:
        """progress per second over the last window (or since the first sample when the run is younger than the window)."""
        if len(self.samples) < 2:
            return None
        t1, p1 = self.samples[-1]
        t0, p0 = next(((t, p) for t, p in self.samples if t >= t1 - self.window_s), self.samples[0])
        return (p1 - p0) / (t1 - t0) if t1 > t0 else None

    def projected_finish(self) -> Optional[float]:
        r, (t, p) = self.rate(), self.samples[-1] if self.samples else (None, (0, 0))
        if r is None or r <= 0:
            return None
        return t + (self.total - p) / r

    def update(self, t: float, progress: float) -> Optional[str]:
        """Record a sample at run time `t` (seconds since start); returns STALL, PROJECTED or None."""
        previous = self.samples[-1][1] if self.samples else None
        if previous is not None and progress < previous:
            progress = previous
        self.samples.append((float(t), float(progress)))
        if previous is None or progress > previous:
            self._moved_at = float(t)
        if progress >= self.total:
            return None
        if t - self._moved_at >= self.stall_s:
            self.reason = f"{STALL}: no progress for {t - self._moved_at:.0f}s (at {progress:.0f}/{self.total:.0f})"
            return STALL
        if t >= self.warmup_s:
            fin = self.projected_finish()
            if fin is not None and fin > self.margin * self.limit_s:
                self.reason = (f"{PROJECTED}: {progress:.0f}/{self.total:.0f} at {t:.0f}s, rate {self.rate():.3g}/s, finish at {fin:.0f}s "
                               f"> {self.margin} x {self.limit_s:.0f}s")
                return PROJECTED
        return None
