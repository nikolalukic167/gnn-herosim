#!/usr/bin/env python3
"""scale_d_v1 smoke read (v3 item 3): reads ONLY wall seconds, decisionTiming (per-task mean / p95, call stats), peak memory (sacct MaxRSS) and completion
(Rulings item 7). No latency, queue, exchange or any outcome field is opened or printed.
usage: scale_d_smoke_read.py <cells root, e.g. .../scale_d/SA/cells> <servers> <clients>
"""
import glob
import json
import math
import os
import subprocess
import sys

root, servers, clients = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "*.summary.json"))):
    d = json.load(open(f))
    base = f[: -len(".summary.json")]
    dt = d.get("decisionTiming") or {}
    done = (d.get("num_tasks") == 50000 and isinstance(d.get("endTime"), (int, float)) and math.isfinite(d["endTime"])
            and not os.path.exists(base + ".failed.json"))
    rows.append({
        "cell": os.path.basename(base), "servers": d.get("servers"), "clients": d.get("clients"),
        "size_ok": (d.get("servers"), d.get("clients")) == (servers, clients),
        "wall_s": d.get("wallclock_s"), "complete": done,
        "dt_calls": dt.get("calls"), "dt_total_s": dt.get("total_s"), "dt_per_task_median_s": dt.get("per_task_median_s"),
        "dt_per_task_p95_s": dt.get("per_task_p95_s"), "dt_per_task_mean_s": dt.get("per_task_mean_s"),
        "dt_call_median_s": dt.get("call_median_s"), "dt_call_p95_s": dt.get("call_p95_s"), "dt_call_max_s": dt.get("call_max_s"),
    })
failed = sorted(os.path.basename(x) for x in glob.glob(os.path.join(root, "*", "*.failed.json")))
mem = {}
for line in subprocess.run(["sacct", "-n", "-P", "-X", "--name", "sdsmk-smoke_g0moderate_s0,sdsmk-smoke_g0moderate_static_s1", "-o", "JobID,JobName,Elapsed,State"],
                           capture_output=True, text=True).stdout.splitlines():
    jid = line.split("|")[0]
    out = subprocess.run(["sacct", "-n", "-P", "-j", jid, "-o", "JobID,MaxRSS"], capture_output=True, text=True).stdout
    mem[jid] = {"job": line, "steps": [x for x in out.splitlines() if x.split("|")[1]]}
print(json.dumps({"root": root, "expect_servers": servers, "expect_clients": clients, "cells": rows, "failed_json": failed, "sacct": mem}, indent=1))
bad = [r["cell"] for r in rows if not (r["size_ok"] and r["complete"])]
print("SMOKE", "OK" if rows and not bad and not failed else f"NOT-OK {bad} {failed}", "cells", len(rows))
