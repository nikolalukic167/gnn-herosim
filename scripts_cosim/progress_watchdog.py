"""Gate-side wiring of the shared progress-rate rule (src/placement/progress_watchdog.py, S6) for the r1_attribution_v1 gate.

Each cell reports its simulated time (src/placement/progress.py); progress = simulated time out of the workload's last arrival (the run ends
at about its last arrival; a collapsing cell overruns it, and the hard limit catches that). The rule kills a cell whose clock has not moved for
300 s, or, after 600 s, whose projected finish (rate over the last 300 s) exceeds 1.5 x the limit. A killed cell is counted as failed.
Also the 5 % rung pause line, read from the shared output directory.
"""
from __future__ import annotations

import glob
import json
import os
import signal
import threading
import time
from typing import Callable, Dict, List, Optional

from src.placement.progress_watchdog import ProgressWatchdog

POLL_S = 15.0
PAUSE_SHARE = 0.05  # production pauses for a rung when more than 5 % of its decided cells hang
PAUSE_MIN_CELLS = 40  # ... once at least this many are decided (one failure in a handful is not a rate)


class Watchdog:
    """Polls one cell's progress file; kills the cell's process group when the shared rule says so. ``verdict`` is set on a kill."""

    def __init__(self, path: str, last_arrival_s: float, limit_s: float, poll_s: float = POLL_S, clock: Callable[[], float] = time.time,
                 **rule):
        self.path, self.last_arrival_s, self.limit_s, self.poll_s, self.clock = path, last_arrival_s, limit_s, poll_s, clock
        self.rule = ProgressWatchdog(last_arrival_s, limit_s, **rule)
        self.verdict: Optional[Dict[str, object]] = None
        self.start = None
        self._stop = threading.Event()
        self._thread = None

    def attach(self, proc) -> None:
        self.start = self.clock()
        self._thread = threading.Thread(target=self._loop, args=(proc,), daemon=True)
        self._thread.start()

    def read(self) -> float:
        try:
            return float(json.load(open(self.path))["sim_now"])
        except (OSError, ValueError, KeyError):
            return 0.0

    def step(self) -> Optional[Dict[str, object]]:
        t, sim = self.clock() - self.start, self.read()
        kind = self.rule.update(t, sim)
        if not kind:
            return None
        return {"reason": kind, "detail": self.rule.reason, "projected_s": self.rule.projected_finish(), "sim_now": sim, "elapsed_s": t}

    def _loop(self, proc) -> None:
        while not self._stop.wait(self.poll_s):
            if proc.poll() is not None:
                return
            v = self.step()
            if v:
                self.verdict = v
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                return

    def stop(self) -> None:
        self._stop.set()


def rung_state(out_dir: str, tag: str, exclude_kinds: tuple = ("reactive",)) -> Dict[str, int]:
    """Decided cells of one rung from the shared output directory (array shards see each other's files). Knative is context:
    its collapse is expected at the heavy rung and is not counted against the pause line."""
    decided = hung = 0
    for f in glob.glob(os.path.join(out_dir, f"cc40s*__g?{tag}__*")):
        if f.endswith(".summary.json"):
            failed = False
        elif f.endswith(".failed.json"):
            failed = True
        else:
            continue
        kind = os.path.basename(f).split("__")[2].rsplit("_s", 1)[0]
        if kind in exclude_kinds:
            continue
        decided += 1
        if failed:
            try:
                why = str(json.load(open(f)).get("why", ""))
            except (OSError, ValueError):
                why = "unreadable"
            hung += why.startswith(("timeout", "watchdog", "unreadable"))
    return {"decided": decided, "hung": hung}


def paused(state: Dict[str, int]) -> bool:
    return state["decided"] >= PAUSE_MIN_CELLS and state["hung"] / state["decided"] > PAUSE_SHARE
