"""Independent ready-set dispatch used by state/action learning experiments."""
import numpy as np


DYNAMIC_FEATURES = 13


def _candidate_features(problem, assignment, next_op, active, last, ready_time, now):
    jobs, operations, hosts = problem["p"].shape
    durations = np.take_along_axis(problem["p"], assignment[..., None], axis=-1)[..., 0]
    tail = np.flip(np.cumsum(np.flip(durations, axis=1), axis=1), axis=1)
    ready_jobs = [job for job in range(jobs) if job not in {v[0] for v in active.values()}
                  and next_op[job] < operations]
    result = np.zeros((jobs * operations, DYNAMIC_FEATURES), dtype=np.float32)
    feasible = np.zeros(jobs * operations, dtype=bool)
    for job in ready_jobs:
        operation = int(next_op[job])
        index = job * operations + operation
        host = int(assignment[job, operation])
        lock = int(problem["locks"][job, operation])
        blocked = host in active
        blocked |= any(lock & int(problem["locks"][other, op])
                       for other, op, _ in active.values())
        blocked |= any(problem["domains"][host, domain] and
                       sum(problem["domains"][other_host, domain] for other_host in active) >= 2
                       for domain in range(problem["domains"].shape[1]))
        if blocked:
            continue
        previous = last.get(host, -1)
        setup = float(problem["setup"][previous, problem["types"][job, operation]]) if previous >= 0 else 0.0
        overlap = sum(bool(lock & int(problem["locks"][other, next_op[other]]))
                      for other in ready_jobs if other != job)
        domain_load = max((sum(problem["domains"][other_host, domain] for other_host in active)
                           for domain in range(problem["domains"].shape[1])
                           if problem["domains"][host, domain]), default=0)
        previous_one_hot = [float(previous == value) for value in range(4)]
        result[index] = [now / 100, (now - ready_time[job]) / 100,
                         durations[job, operation] / 20, setup / 20,
                         tail[job, operation] / 100,
                         (operations - operation) / operations,
                         overlap / max(1, jobs - 1), len(active) / hosts,
                         domain_load / 2, *previous_one_hot]
        feasible[index] = True
    return result, feasible


def materialize(problem, assignment, choose, record=False):
    """Run greedy ready-set choices and return a behaviorally complete priority."""
    jobs, operations, _ = problem["p"].shape
    next_op = np.zeros(jobs, dtype=int)
    ready_time = np.zeros(jobs, dtype=float)
    active = {}
    last = {}
    rank = np.empty((jobs, operations), dtype=np.int64)
    states = []
    now = 0.0
    started = done = 0
    while done < jobs * operations:
        for host, (job, operation, end) in list(active.items()):
            if end <= now + 1e-9:
                next_op[job] += 1
                ready_time[job] = now
                last[host] = int(problem["types"][job, operation])
                del active[host]
                done += 1
        if done == jobs * operations:
            break
        while True:
            dynamic, feasible = _candidate_features(
                problem, assignment, next_op, active, last, ready_time, now)
            candidates = np.flatnonzero(feasible)
            if not len(candidates):
                break
            selected = int(choose(dynamic, feasible))
            if selected not in candidates:
                raise ValueError("chooser selected an infeasible operation")
            job, operation = divmod(selected, operations)
            host = int(assignment[job, operation])
            previous = last.get(host, -1)
            setup = float(problem["setup"][previous, problem["types"][job, operation]]) if previous >= 0 else 0.0
            end = now + float(problem["p"][job, operation, host]) + setup
            active[host] = (job, operation, end)
            rank[job, operation] = started
            started += 1
            if record:
                states.append({"dynamic": dynamic, "feasible": feasible,
                               "target": selected, "time": now})
        if not active:
            raise RuntimeError("ready-set dispatch deadlock")
        now = min(value[2] for value in active.values())
    return rank, states


def trace_priority(problem, assignment, priority):
    flat = np.asarray(priority).reshape(-1)
    return materialize(problem, assignment,
                       lambda dynamic, feasible: np.flatnonzero(feasible)[
                           np.argmin(flat[feasible])], record=True)
