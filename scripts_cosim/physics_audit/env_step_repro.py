#!/usr/bin/env python3
"""Minimal reproduction: one extra scheduled event changes a result, because the autoscaler loop calls env.step().

`Autoscaler.autoscaler_process` / `_kpa_autoscaler_process` (src/placement/autoscaler.py) end every iteration with

    self.env.step()                              # "Next event"
    yield self.env.timeout(self.reconcile_interval)

`Environment.step()` pops and processes the NEXT event in the queue, whatever it is and whenever it is scheduled,
and moves the clock to it. So what the loop consumes depends on what else is in the queue. A process that only
reads state but owns a Timeout therefore changes the run. This script has two parts:

  1. standalone SimPy: the loop shape, with and without ONE extra Timeout (a process that reads and writes nothing);
  2. (--sim) the real simulator on a small cell, with 0, 1, 100 and 1000 idle timeouts. In the simulator one event
     is usually harmless: the loop's step() takes a bystander's event only when it happens to be next in the queue,
     and a handful of events are consumed early without reordering anything that matters. A stream of them is not
     harmless (a 0.1 s sampler moved a cell's total RTT by 16 %).

usage: env_step_repro.py            # part 1
       env_step_repro.py --sim CFG WORKLOAD [--events N]   # part 2 (R1 environment variables must be set)
"""
from __future__ import annotations

import sys


def standalone() -> int:
    import simpy

    def run(with_bystander: bool):
        env = simpy.Environment()
        queue = simpy.Store(env)
        done = {}

        def worker():                      # takes jobs one at a time, 1.0 s each
            while True:
                job = yield queue.get()
                yield env.timeout(1.0)
                done[job] = env.now

        def arrivals():                    # a job every 0.4 s
            for job in range(8):
                yield env.timeout(0.4)
                yield queue.put(job)

        def autoscaler():                  # the loop's shape: act, step(), sleep; it adds a worker when work waits
            workers = 1
            while True:
                if len(queue.items) >= 2 and workers < 3:
                    workers += 1
                    env.process(worker())
                env.step()                 # <- consumes whatever event is next, and moves the clock to it
                yield env.timeout(0.5)

        def bystander():                   # reads nothing, writes nothing: it owns ONE Timeout
            yield env.timeout(0.1)

        env.process(worker())
        env.process(arrivals())
        env.process(autoscaler())
        if with_bystander:
            env.process(bystander())
        env.run(until=40.0)
        return done

    a, b = run(False), run(True)
    print("job completion times without the bystander:", {k: round(v, 2) for k, v in sorted(a.items())})
    print("job completion times with ONE extra Timeout:", {k: round(v, 2) for k, v in sorted(b.items())})
    print("IDENTICAL" if a == b else "DIFFERENT: one Timeout owned by a process that changes nothing changed when jobs finished")
    return 0 if a != b else 1


def in_simulator(cfg: str, workload: str, events: int) -> int:
    import json
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    driver = f'''
import sys
sys.path.insert(0, {str(repo)!r})
import runpy
if {{extra}}:
    import src.placement.orchestrator as o
    orig = o.Orchestrator.initializer_process
    def patched(self):
        def idle():                        # {{extra}} idle timeouts, 0.37 s apart: no state read, no state written
            for _ in range({{extra}}):
                yield self.env.timeout(0.37)
        self.env.process(idle())
        yield from orig(self)
    o.Orchestrator.initializer_process = patched
sys.argv = ["executesimulation.py", "--config", {cfg!r}, "--workload", {workload!r}, "--policy", "peer_greedy_network_cd",
            "--output", sys.argv[1]]
runpy.run_module("src.executesimulation", run_name="__main__")
'''
    out = {}
    for extra in (0, 1, 100, 1000):
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "drive.py"
            script.write_text(driver.replace("{extra}", str(extra)))
            res_path = Path(d) / "res.json"
            env = dict(os.environ, HEROSIM_MAX_EVENTS=str(events), PYTHONPATH=str(repo))
            proc = subprocess.run([sys.executable, str(script), str(res_path)], env=env, cwd=str(repo),
                                  capture_output=True, text=True)
            if proc.returncode != 0 or not res_path.exists():
                print(proc.stderr[-800:])
                return 2
            stats = json.load(open(res_path))["stats"]
            out[extra] = (stats["total_rtt"], stats["num_tasks"], stats["endTime"])
    for k, (rtt, tasks, end) in out.items():
        print(f"{k:5d} idle timeouts: total_rtt={rtt:.6f}  tasks={tasks}  endTime={end:.6f}"
              + ("" if k == 0 else "   " + ("same as none" if out[k] == out[0] else "DIFFERENT from none")))
    return 0 if any(out[k] != out[0] for k in out if k) else 1


if __name__ == "__main__":
    if "--sim" in sys.argv:
        i = sys.argv.index("--sim")
        n = int(sys.argv[sys.argv.index("--events") + 1]) if "--events" in sys.argv else 2000
        sys.exit(in_simulator(sys.argv[i + 1], sys.argv[i + 2], n))
    sys.exit(standalone())
