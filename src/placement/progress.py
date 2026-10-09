"""Simulated-time progress file for the gate driver's watchdog (r1_attribution_v1).

A daemon thread writes ``{"sim_now", "wall"}`` to ``HEROSIM_PROGRESS_FILE`` every few wall seconds. It reads ``env.now`` and
schedules nothing in SimPy, so event order and every simulated field are unchanged; unset, nothing starts.
"""
from __future__ import annotations

import json
import os
import threading
import time

ENV = "HEROSIM_PROGRESS_FILE"
INTERVAL_S = 15.0


def start_progress_reporter(env, path: str | None = None, interval: float | None = None):
    path = path or os.environ.get(ENV)
    interval = interval or float(os.environ.get("HEROSIM_PROGRESS_INTERVAL_S", INTERVAL_S))
    if not path:
        return None
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            tmp = path + ".tmp"
            try:
                with open(tmp, "w") as fh:
                    json.dump({"sim_now": float(env.now), "wall": time.time()}, fh)
                os.replace(tmp, path)
            except OSError:
                pass  # the watchdog treats a missing file as no progress; a full disk must not kill the run
            stop.wait(interval)

    threading.Thread(target=loop, name="herosim-progress", daemon=True).start()
    return stop
