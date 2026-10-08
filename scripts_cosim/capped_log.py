"""Run a command with its combined stdout/stderr written to a size-capped log.

A hung run (a starved-client spin logs ~900 lines per simulated second) wrote logs of several GB and threatened the
/home quota. The log keeps the first 80 % of the cap, then the last 20 % of the output, and says how many bytes were
dropped in between. The cap is HEROSIM_GATE_LOG_CAP_MB (default 50; 0 or negative = uncapped, the old behaviour).
Nothing but the log changes: the simulation reads its result from `--output`, not from the log.
"""
from __future__ import annotations

import collections
import os
import subprocess
from typing import Deque, Mapping, Optional, Sequence

CAP_ENV = "HEROSIM_GATE_LOG_CAP_MB"
DEFAULT_CAP_MB = 50.0
HEAD_SHARE = 0.8
CHUNK = 1 << 16


def cap_bytes() -> Optional[int]:
    mb = float(os.environ.get(CAP_ENV, DEFAULT_CAP_MB))
    return None if mb <= 0 else int(mb * 1024 * 1024)


def run_logged(cmd: Sequence[str], env: Mapping[str, str], cwd: str, log_path: str, cap: Optional[int] = None) -> int:
    """Return the command's exit code; the log is at ``log_path``."""
    if cap is None:
        with open(log_path, "w") as fh:
            return subprocess.run(list(cmd), env=dict(env), cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode
    head_limit = int(cap * HEAD_SHARE)
    tail_limit = cap - head_limit
    head_written = 0
    tail: Deque[bytes] = collections.deque()
    tail_size = 0
    dropped = 0
    proc = subprocess.Popen(list(cmd), env=dict(env), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(log_path, "wb") as fh:
        while True:
            chunk = proc.stdout.read(CHUNK)
            if not chunk:
                break
            if head_written < head_limit:
                room = head_limit - head_written
                fh.write(chunk[:room])
                head_written += min(len(chunk), room)
                chunk = chunk[room:]
                if not chunk:
                    continue
            tail.append(chunk)
            tail_size += len(chunk)
            while tail_size > tail_limit:
                over = tail_size - tail_limit
                first = tail[0]
                if len(first) <= over:
                    tail.popleft()
                    tail_size -= len(first)
                    dropped += len(first)
                else:
                    tail[0] = first[over:]
                    tail_size -= over
                    dropped += over
        if dropped:
            fh.write(f"\n[log capped at {cap} bytes: {dropped} bytes dropped after the first {head_written}; "
                     f"the last {tail_size} follow]\n".encode())
        for c in tail:
            fh.write(c)
    return proc.wait()
