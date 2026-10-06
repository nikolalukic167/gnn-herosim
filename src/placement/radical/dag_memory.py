"""Opt-in DAG output residency with capacity, remote reads and durable-store reads."""
import numpy as np
import simpy

from .dag_reservation import checked, problem as dag_problem


def problem(seed, jobs=8, ops=8, hosts=8):
    b = dag_problem(seed, jobs, ops, hosts)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2001]))
    b['output_size_mb'] = rng.integers(8, 25, size=(jobs, ops)).astype(float)
    b['host_memory_mb'] = np.full(hosts, 32., dtype=float)
    b['store_read_ms_per_mb'] = .75
    b['peer_read_ms_per_mb'] = .25
    b['memory_contract'] = 'durable_output_cache_v1'
    return b


def checked_memory(b, assignment, rank, retain):
    _, pred, a, r, _ = checked(b, assignment, rank)
    keep = np.asarray(retain)
    sizes = np.asarray(b['output_size_mb'], dtype=float)
    caps = np.asarray(b['host_memory_mb'], dtype=float)
    if keep.shape != a.shape or not np.isin(keep, [0, 1]).all():
        raise ValueError('retention must be a binary operation array')
    if sizes.shape != a.shape or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError('invalid output sizes')
    if caps.shape != (b['p'].shape[2],) or not np.isfinite(caps).all() or np.any(caps <= 0):
        raise ValueError('invalid host memory')
    for key in ('store_read_ms_per_mb', 'peer_read_ms_per_mb'):
        if not np.isfinite(b[key]) or b[key] < 0:
            raise ValueError('invalid read rate')
    return pred, a, r, keep.astype(bool), sizes, caps


def replay(b, assignment, rank, retain):
    pred, a, r, keep, sizes, caps = checked_memory(b, assignment, rank, retain)
    jobs, ops = a.shape
    parents = {(j, k): [(j, z) for z in range(k) if int(pred[j, k]) & (1 << z)]
               for j in range(jobs) for k in range(ops)}
    remaining = {node: 0 for node in parents}
    for group in parents.values():
        for node in group:
            remaining[node] += 1
    todo = set(parents)
    done, active, last, cached = {}, {}, {}, {}
    host_cache = {h: [] for h in range(b['p'].shape[2])}
    events, starts, ends = [], np.empty(a.shape), np.empty(a.shape)
    counts = {'cache_hits': 0, 'peer_reads': 0, 'store_reads': 0,
              'capacity_evictions': 0, 'binding_completions': 0}
    env = simpy.Environment()

    def release_cache(node):
        if node in cached:
            host = cached.pop(node)
            host_cache[host].remove(node)

    def process():
        while len(done) < a.size:
            now = float(env.now)
            completed = sorted((host for host, rec in active.items() if rec['end'] <= now + 1e-9))
            for host in completed:
                rec = active.pop(host)
                node = (rec['job'], rec['operation'])
                done[node] = now
                ends[node] = now
                last[host] = int(b['types'][node])
                events.append({'event': 'done', **rec})
                if keep[node] and remaining[node] and sizes[node] <= caps[host]:
                    if sum(sizes[value] for value in host_cache[host]) + sizes[node] > caps[host]:
                        counts['binding_completions'] += 1
                    while host_cache[host] and sum(sizes[value] for value in host_cache[host]) + sizes[node] > caps[host]:
                        victim = host_cache[host][0]
                        release_cache(victim)
                        counts['capacity_evictions'] += 1
                        events.append({'event': 'evict', 'job': victim[0], 'operation': victim[1],
                                       'host': host, 'time': now})
                    host_cache[host].append(node)
                    cached[node] = host
                    events.append({'event': 'cache', 'job': node[0], 'operation': node[1],
                                   'host': host, 'time': now, 'size_mb': float(sizes[node])})
            if len(done) == a.size:
                break
            ready = sorted((node for node in todo if all(parent in done for parent in parents[node])),
                           key=lambda node: int(r[node]))
            for node in ready:
                host = int(a[node])
                lock = int(b['locks'][node])
                if host in active:
                    continue
                if any(lock & int(b['locks'][rec['job'], rec['operation']]) for rec in active.values()):
                    continue
                if any(b['domains'][host, g] and sum(b['domains'][h, g] for h in active) >= 2
                       for g in range(b['domains'].shape[1])):
                    continue
                data_ms = 0.
                for parent in parents[node]:
                    if parent in cached:
                        if cached[parent] == host:
                            counts['cache_hits'] += 1
                        else:
                            data_ms += sizes[parent] * b['peer_read_ms_per_mb']
                            counts['peer_reads'] += 1
                    else:
                        data_ms += sizes[parent] * b['store_read_ms_per_mb']
                        counts['store_reads'] += 1
                    remaining[parent] -= 1
                    if remaining[parent] == 0:
                        release_cache(parent)
                setup = float(b['setup'][last[host], b['types'][node]]) if host in last else 0.
                rec = {'job': node[0], 'operation': node[1], 'host': host, 'start': now,
                       'end': now + float(b['p'][node][host]) + setup + data_ms,
                       'setup': setup, 'data_ms': float(data_ms)}
                starts[node] = now
                active[host] = rec
                todo.remove(node)
                events.append({'event': 'start', **rec})
            if not active:
                raise RuntimeError('memory DAG deadlock')
            yield env.timeout(min(rec['end'] for rec in active.values()) - now)

    env.process(process())
    env.run()
    return {'objective': float(ends[:, -1].sum()), 'starts': starts, 'ends': ends,
            'events': events, 'counts': counts}


def hand_retention(b, assignment, rule):
    if rule == 'spill_all':
        return np.zeros(b['p'].shape[:2], dtype=np.int8)
    if rule == 'keep_all':
        return np.ones(b['p'].shape[:2], dtype=np.int8)
    pred = b['predecessors']
    jobs, ops = pred.shape
    consumers = np.zeros((jobs, ops), dtype=int)
    local = np.zeros((jobs, ops), dtype=int)
    for j in range(jobs):
        for k in range(ops):
            for z in range(k):
                if int(pred[j, k]) & (1 << z):
                    consumers[j, z] += 1
                    local[j, z] += int(assignment[j, z] == assignment[j, k])
    if rule == 'reuse':
        return (consumers >= 2).astype(np.int8)
    if rule == 'local_reuse':
        return (local > 0).astype(np.int8)
    if rule == 'benefit_per_mb':
        return ((local + .333 * (consumers - local)) * b['output_size_mb'] > 10).astype(np.int8)
    raise ValueError('unknown hand retention rule')
