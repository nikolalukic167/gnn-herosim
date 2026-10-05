#!/usr/bin/env python3
"""Write the second-failure record for each local_features_v1 rerun (job 827433) that produced no summary.

A target is recorded only if the rerun actually started it: its log (or .log.gz) was written at or after the rerun's
start. Existing non-empty records (memory cap, -9) are kept; empty ones (writes failed at the home quota, 05:14) are
replaced. Runs the rerun never started are listed and left without a record.
"""
import datetime as dt
import glob
import gzip
import json
import os
import re

LF = "/home/nikola.lukic/gnn-herosim/simulation_data/local_features_v1"
START = dt.datetime(2026, 10, 5, 3, 29).timestamp()
STAMP = re.compile(rb"\[ ([0-9]+\.[0-9]+) \]")


def last_clock(log):
    try:
        if log.endswith(".gz"):
            with gzip.open(log, "rb") as fh:
                tail = b""
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    tail = (tail + chunk)[-4000:]
        else:
            with open(log, "rb") as fh:
                fh.seek(max(0, os.path.getsize(log) - 4000))
                tail = fh.read()
    except OSError:
        return None
    m = STAMP.findall(tail)
    return float(m[-1]) if m else None


def log_mtime(log):
    """A .gz written by `gzip -c <file>` keeps the original file's mtime in its header (bytes 4-8)."""
    if log.endswith(".gz"):
        with open(log, "rb") as fh:
            h = fh.read(8)
        return float(int.from_bytes(h[4:8], "little"))
    return os.path.getmtime(log)


def targets():
    out = [(os.path.join(LF, "gate"), f[: -len(".failed-t2700.json")])
           for f in glob.glob(os.path.join(LF, "gate", "*.failed-t2700.json"))]
    ref = os.path.join(LF, "ref_retry")
    for f in glob.glob(os.path.join(ref, "*.log")) + glob.glob(os.path.join(ref, "*.log.gz")):
        base = re.sub(r"\.log(\.gz)?$", "", f)
        if not os.path.exists(base + ".summary.json"):
            out.append((ref, base))
    return sorted(set(out))


written, kept, unstarted, ok = [], [], [], []
for d, base in targets():
    name = os.path.basename(base)
    if os.path.exists(base + ".summary.json"):
        ok.append(name)
        continue
    log = next((p for p in (base + ".log", base + ".log.gz") if os.path.exists(p)), None)
    if log is None or log_mtime(log) < START:
        unstarted.append(name)
        continue
    rec = base + ".failed.json"
    if os.path.exists(rec) and os.path.getsize(rec) > 0:
        kept.append(name)
        continue
    last = dt.datetime.fromtimestamp(log_mtime(log)).strftime("%H:%M")
    body = {"arm": name, "returncode": None, "rerun_job": 827433, "rerun_timeout_s": 8100,
            "why": f"hung on the rerun: no simulation progress in its log after {last}; the run was stopped by the home "
                   "quota (05:14) or the job cancel (05:35) before the 8100 s timeout",
            "log_last_write": last, "sim_clock_at_last_line": last_clock(log)}
    with open(rec + ".partial", "w") as fh:
        json.dump(body, fh)
    os.replace(rec + ".partial", rec)
    written.append(name)

print(json.dumps({"written": written, "kept_memory_cap": kept, "never_rerun": unstarted, "rerun_succeeded": ok},
                 indent=1))
