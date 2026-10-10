"""Run a command under the progress watchdog (src/placement/progress_watchdog.py) and a hard limit, like `timeout` but killing early.

    watchdog_run.py --progress-file snap.jsonl --progress-key trigger_task_id --total 50000 --limit 7200 -- python3 src/executesimulation.py ...

Progress = `--progress-key` of the last line of the JSONL file (a missing file is progress 0). Exit codes: the command's own; 137 when the
watchdog killed it (reason on stderr); 124 when the hard limit did. The process group is killed with SIGKILL, so a wrapper that maps rc 124
and 137 to a `hung` sentinel needs no other change.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.placement.progress_watchdog import ProgressWatchdog


def last_progress(path: Path, key: str) -> float:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            lines = [l for l in fh.read().splitlines() if l.strip()]
        return float(json.loads(lines[-1])[key]) if lines else 0.0
    except (OSError, ValueError, KeyError):
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress-file", type=Path, required=True)
    ap.add_argument("--progress-key", default="trigger_task_id")
    ap.add_argument("--total", type=float, required=True)
    ap.add_argument("--limit", type=float, required=True, help="hard limit in seconds")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--warmup", type=float, default=600.0)
    ap.add_argument("--stall", type=float, default=300.0)
    ap.add_argument("--window", type=float, default=300.0)
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--timeline", type=Path, help="append one JSON sample per poll (t, progress, rate): the data to calibrate the margin")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    wd = ProgressWatchdog(a.total, a.limit, warmup_s=a.warmup, margin=a.margin, stall_s=a.stall, window_s=a.window)
    proc = subprocess.Popen(cmd, start_new_session=True)
    t0 = time.time()
    while True:
        try:
            return proc.wait(timeout=a.poll)
        except subprocess.TimeoutExpired:
            pass
        t = time.time() - t0
        p = last_progress(a.progress_file, a.progress_key)
        # a run that has not written its first progress line yet (start-up, fast-forward) is not stalled; the hard limit still applies
        verdict = wd.update(t, p) if p > 0 else None
        if a.timeline:
            with open(a.timeline, "a") as fh:
                fh.write(json.dumps({"t": round(t, 1), "progress": p, "rate": wd.rate()}) + "\n")
        if verdict:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            print(f"watchdog: killed ({wd.reason})", file=sys.stderr)
            return 137
        if t >= a.limit:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            print(f"watchdog: hard limit {a.limit:.0f}s reached", file=sys.stderr)
            return 124


if __name__ == "__main__":
    raise SystemExit(main())
