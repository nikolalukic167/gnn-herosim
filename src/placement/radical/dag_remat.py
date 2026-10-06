"""Opt-in one-level recomputation of evicted DAG outputs on consumer hosts."""
import numpy as np
import simpy

from .dag_memory import checked_memory, hand_retention, problem as memory_problem


def problem(seed, jobs=8, ops=8, hosts=8):
    b = memory_problem(seed, jobs, ops, hosts)
    b['memory_contract'] = 'durable_output_cache_remat_v1'
    return b


def checked_remat(b, assignment, rank, retain, recompute):
    pred, a, r, keep, sizes, caps = checked_memory(b, assignment, rank, retain)
    action = np.asarray(recompute)
    if action.shape != a.shape or not np.isin(action, [0, 1]).all():
        raise ValueError('recompute must be a binary operation array')
    for j, k in np.argwhere(action):
        for child in range(k + 1, a.shape[1]):
            if int(pred[j, child]) & (1 << k):
                if not np.isfinite(b['p'][j, k, int(a[j, child])]):
                    raise ValueError('recompute operation cannot execute on consumer host')
    return pred, a, r, keep, sizes, caps, action.astype(bool)


def effective_lock(b, node, parents, recompute, cached):
    lock = int(b['locks'][node])
    for parent in parents[node]:
        if recompute[parent] and parent not in cached:
            lock |= int(b['locks'][parent])
    return lock


def replay(b, assignment, rank, retain, recompute):
    pred, a, r, keep, sizes, caps, action = checked_remat(b, assignment, rank, retain, recompute)
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
              'capacity_evictions': 0, 'binding_completions': 0,
              'rematerializations': 0, 'remat_input_store_reads': 0,
              'remat_input_peer_reads': 0, 'remat_input_cache_hits': 0}
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
                lock = effective_lock(b, node, parents, action, cached)
                if host in active:
                    continue
                if any(lock & rec['lock'] for rec in active.values()):
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
                    elif action[parent]:
                        for grandparent in parents[parent]:
                            if grandparent in cached:
                                if cached[grandparent] == host:
                                    counts['remat_input_cache_hits'] += 1
                                else:
                                    data_ms += sizes[grandparent] * b['peer_read_ms_per_mb']
                                    counts['remat_input_peer_reads'] += 1
                            else:
                                data_ms += sizes[grandparent] * b['store_read_ms_per_mb']
                                counts['remat_input_store_reads'] += 1
                        data_ms += float(b['p'][parent][host])
                        counts['rematerializations'] += 1
                        events.append({'event': 'remat', 'job': parent[0], 'operation': parent[1],
                                       'consumer_job': node[0], 'consumer_operation': node[1],
                                       'host': host, 'time': now})
                    else:
                        data_ms += sizes[parent] * b['store_read_ms_per_mb']
                        counts['store_reads'] += 1
                    remaining[parent] -= 1
                    if remaining[parent] == 0:
                        release_cache(parent)
                setup = float(b['setup'][last[host], b['types'][node]]) if host in last else 0.
                rec = {'job': node[0], 'operation': node[1], 'host': host, 'start': now,
                       'end': now + float(b['p'][node][host]) + setup + data_ms,
                       'setup': setup, 'data_ms': float(data_ms), 'lock': lock}
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
