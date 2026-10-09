"""Progress-rate watchdog and rung pause line for the r1_attribution_v1 gate.

Replaces fixed 3x reruns. A cell reports its simulated time (src/placement/progress.py). After ``grace_s`` of wall time the
projected finish is ``last_arrival / (sim_now / elapsed)`` (the run ends at about its last arrival; a collapsing cell
overruns that, and the hard timeout catches it). A cell projected past ``limit_s``, or whose simulated clock has not moved for
``stall_s`` (the starved-client spin), is killed and counted as failed.
"""
from __future__ import annotations

import glob
import json
import os
import signal
import threading
import time
from typing import Callable, Dict, List, Optional

GRACE_S = 600.0
STALL_S = 600.0
POLL_S = 15.0
PAUSE_SHARE = 0.05  # production pauses for a rung when more than 5 % of its decided cells hang
PAUSE_MIN_CELLS = 40  # ... once at least this many are decided (one failure in a handful is not a rate)


def decide(elapsed_s: float, sim_now: Optional[float], last_arrival_s: float, limit_s: float, stalled_s: float,
           grace_s: float = GRACE_S, stall_s: float = STALL_S) -> Optional[Dict[str, object]]:
    """None to keep running, or the reason to kill: {"reason", "projected_s", ...}."""
    if elapsed_s < grace_s:
        return None
    if stalled_s >= stall_s:
        return {"reason": "stall", "projected_s": None, "sim_now": sim_now, "elapsed_s": elapsed_s, "stalled_s": stalled_s}
    if not sim_now or sim_now <= 0:
        return {"reason": "no-progress", "projected_s": None, "sim_now": sim_now, "elapsed_s": elapsed_s, "stalled_s": stalled_s}
    projected = last_arrival_s * elapsed_s / sim_now
    if projected > limit_s:
        return {"reason": "projected", "projected_s": projected, "sim_now": sim_now, "elapsed_s": elapsed_s, "stalled_s": stalled_s}
    return None


class Watchdog:
    """Polls one cell's progress file; kills the cell's process group when ``decide`` says so. ``verdict`` is set on a kill."""

    def __init__(self, path: str, last_arrival_s: float, limit_s: float, grace_s: float = GRACE_S, stall_s: float = STALL_S,
                 poll_s: float = POLL_S, clock: Callable[[], float] = time.time):
        self.path, self.last_arrival_s, self.limit_s = path, last_arrival_s, limit_s
        self.grace_s, self.stall_s, self.poll_s, self.clock = grace_s, stall_s, poll_s, clock
        self.verdict: Optional[Dict[str, object]] = None
        self.last_sim: Optional[float] = None
        self.last_sim_moved = None
        self.start = None
        self._stop = threading.Event()
        self._thread = None

    def attach(self, proc) -> None:
        self.start = self.clock()
        self.last_sim_moved = self.start
        self._thread = threading.Thread(target=self._loop, args=(proc,), daemon=True)
        self._thread.start()

    def read(self) -> Optional[float]:
        try:
            return float(json.load(open(self.path))["sim_now"])
        except (OSError, ValueError, KeyError):
            return None

    def step(self) -> Optional[Dict[str, object]]:
        now = self.clock()
        sim = self.read()
        if sim is not None and sim != self.last_sim:
            self.last_sim, self.last_sim_moved = sim, now
        return decide(now - self.start, self.last_sim, self.last_arrival_s, self.limit_s, now - self.last_sim_moved,
                      self.grace_s, self.stall_s)

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
