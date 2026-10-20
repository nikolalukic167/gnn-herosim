#!/usr/bin/env python3
"""Companion to openloop_replay.py (SCIENTIST's open-loop test (a)): the pointwise comparison of models (i) and (ii) changes every flow's start-up at once, so a flow can
finish sooner because the flows it shares links with are delayed. The clean monotonicity property is on ONE flow: delaying only that flow (its TCP start-up delay,
all others at 0) must not make it finish sooner. For a seeded sample of N transfers of the same filed cell, replay once with only that transfer carrying its start-up
delay and compare its own completion time with the baseline (fair share alone). Usage: openloop_perturb.py <xfers.jsonl> <N> <out.txt>; PYTHONPATH at the repo pin."""
import json, os, random, statistics as st, sys, time
import simpy
from src.placement.network_fabric import FairShare, tcp_startup_delay

xf, N, out_p = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = [json.loads(l) for l in open(xf)]
by_task = {}
for r in rows:
    by_task.setdefault(r["task"], []).append(r)


def directed(r):
    cur, hops = r["src"], []
    for key, bw in r["route"]:
        a, b = key.split("|"); nxt = b if cur == a else a
        hops.append((f"{cur}>{nxt}", bw)); cur = nxt
    return hops


def replay(only):
    env = simpy.Environment(); fs = FairShare(env); res = {}
    def task_proc(tid, trs):
        yield env.timeout(trs[0]["t"])
        for k, r in enumerate(trs):
            hops = directed(r); bn = min(bw for _k, bw in hops)
            d = tcp_startup_delay(r["bytes"], bn, r["latency"]) if only == (tid, k) else 0.0
            if d > 0:
                yield env.timeout(d)
            yield fs.start(hops, r["bytes"], 1.0)
            yield env.timeout(r["latency"])
            res[(tid, k)] = env.now
    for tid, trs in by_task.items():
        env.process(task_proc(tid, trs))
    env.run()
    return res


t0 = time.time(); base = replay(None); one = time.time() - t0
keys = sorted(base)
random.Random(20261020).shuffle(keys)
sample = keys[:N]
bad, gaps, other_faster = [], [], 0
for key in sample:
    r = replay(key)
    gaps.append(r[key] - base[key])
    if r[key] < base[key] - 1e-9:
        bad.append((key, r[key] - base[key]))
    other_faster += sum(1 for k in r if k != key and r[k] < base[k] - 1e-9)
txt = [__doc__.strip(), "", f"transfers {len(rows)}; one replay {one:.1f} s; sample {N} (seed 20261020)",
       f"own completion with only that flow delayed minus baseline: min {min(gaps):+.4f} s, median {st.median(gaps):+.4f} s, max {max(gaps):+.4f} s",
       f"sampled flows that finish SOONER when only they are delayed: {len(bad)} of {N}" + (f"; worst {min(v for _, v in bad):+.6f} s" if bad else ""),
       f"flows other than the delayed one that finish sooner (summed over the sample): {other_faster}"]
s = "\n".join(txt); print(s)
if os.path.exists(out_p):
    raise SystemExit("FAIL LOUD: written once")
open(out_p + ".partial", "w").write(s + "\n"); os.replace(out_p + ".partial", out_p)
