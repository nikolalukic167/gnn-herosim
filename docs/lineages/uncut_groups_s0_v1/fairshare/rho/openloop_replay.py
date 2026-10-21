#!/usr/bin/env python3
"""SCIENTIST's open-loop monotonicity test (a), uncut_groups_s0_v1 fair share. Replays one filed heavy bar cell's peer exchanges (cc40s19537 / g0heavy,
switch off, 24 servers; xfers.jsonl from openloop_trace.sh) through the exchange model with NO autoscaler or scheduler response: every task starts its exchange
at its filed time; a task's transfers run one after another (serial partners); a transfer = start-up delay (0 under (i)), then all its bytes through the
processor-shared links (directed hops recovered from the route), then the route latency. Models: (i) fair share alone; (ii) fair share + TCP, delay
d = tcp_transfer(size, solo bottleneck, lat) - size / solo bottleneck. Per transfer (task, k-th partner) it compares, under (ii) against (i): its duration and
its absolute completion time; reports the share with (ii) < (i) (a violation of 'adding a start-up delay that holds no link cannot make a transfer finish
sooner'), by how much, and the same for the task's exchange end. Usage: openloop_replay.py <xfers.jsonl> <out.txt>; needs PYTHONPATH at the repo pin."""
import json, os, statistics as st, sys
import simpy
from src.placement.network_fabric import FairShare, tcp_startup_delay

xf, out_p = sys.argv[1:3]
rows = [json.loads(l) for l in open(xf)]
by_task = {}
for r in rows:
    by_task.setdefault(r["task"], []).append(r)


def directed(r):
    cur, hops = r["src"], []
    for key, bw in r["route"]:
        a, b = key.split("|")
        nxt = b if cur == a else a
        assert cur in (a, b)
        hops.append((f"{cur}>{nxt}", bw)); cur = nxt
    assert cur == r["dst"]
    return hops


def replay(tcp):
    env = simpy.Environment(); fs = FairShare(env)
    res = {}
    def task_proc(tid, trs):
        yield env.timeout(trs[0]["t"])
        for k, r in enumerate(trs):
            hops = directed(r); bn = min(bw for _k, bw in hops)
            start = env.now
            d = tcp_startup_delay(r["bytes"], bn, r["latency"]) if tcp else 0.0
            if d > 0:
                yield env.timeout(d)
            yield fs.start(hops, r["bytes"], 1.0)
            yield env.timeout(r["latency"])
            res[(tid, k)] = (start, env.now)
    for tid, trs in by_task.items():
        env.process(task_proc(tid, trs))
    env.run()
    return res, fs.peak_flows_on_a_link


a, pa = replay(False); b, pb = replay(True)
assert a.keys() == b.keys() and len(a) == len(rows)
dur_viol = [(k, (b[k][1] - b[k][0]) - (a[k][1] - a[k][0])) for k in a if (b[k][1] - b[k][0]) < (a[k][1] - a[k][0]) - 1e-9]
end_viol = [(k, b[k][1] - a[k][1]) for k in a if b[k][1] < a[k][1] - 1e-9]
tasks = {}
for (tid, k), (s, e) in a.items():
    tasks[tid] = max(tasks.get(tid, (0, 0)) , (k, e))
task_a = {tid: max(a[(tid, k)][1] for k in range(len(by_task[tid]))) for tid in by_task}
task_b = {tid: max(b[(tid, k)][1] for k in range(len(by_task[tid]))) for tid in by_task}
task_viol = [tid for tid in by_task if task_b[tid] < task_a[tid] - 1e-9]
n = len(a)
txt = [__doc__.strip(), "",
       f"transfers {n}, tasks {len(by_task)}; peak flows on a link: (i) {pa}, (ii) {pb}",
       f"(i)  duration: mean {st.mean(a[k][1]-a[k][0] for k in a):.3f} s, median {st.median(a[k][1]-a[k][0] for k in a):.3f} s",
       f"(ii) duration: mean {st.mean(b[k][1]-b[k][0] for k in b):.3f} s, median {st.median(b[k][1]-b[k][0] for k in b):.3f} s",
       f"transfers whose DURATION under (ii) is shorter than under (i): {len(dur_viol)} of {n} ({100*len(dur_viol)/n:.2f} %)"
       + (f"; largest shortfall {max(-v for _, v in dur_viol):.3f} s, median {st.median(-v for _, v in dur_viol):.3f} s" if dur_viol else ""),
       f"transfers whose COMPLETION time under (ii) is earlier than under (i): {len(end_viol)} of {n} ({100*len(end_viol)/n:.2f} %)"
       + (f"; largest {max(-v for _, v in end_viol):.3f} s, median {st.median(-v for _, v in end_viol):.3f} s" if end_viol else ""),
       f"tasks whose whole exchange ends earlier under (ii) than under (i): {len(task_viol)} of {len(by_task)} ({100*len(task_viol)/len(by_task):.2f} %)",
       f"task exchange end, (ii) minus (i): mean {st.mean(task_b[t]-task_a[t] for t in by_task):+.3f} s, median {st.median(task_b[t]-task_a[t] for t in by_task):+.3f} s"]
s = "\n".join(txt); print(s)
if os.path.exists(out_p):
    raise SystemExit("FAIL LOUD: written once")
open(out_p + ".partial", "w").write(s + "\n"); os.replace(out_p + ".partial", out_p)
