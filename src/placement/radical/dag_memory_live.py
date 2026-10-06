"""Opt-in HeROsim execution of the durable-output cache contract."""
import copy
import random
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from src.placement.availability import service_phase
from . import coordinator
from .dag_memory import checked_memory
from .dispatch_live import PriorityExecution
from scripts_cosim.peer_lookahead_live_probe import deadline
from scripts_cosim.workflow_live import substrate


class MemoryExecution(PriorityExecution):
    def __init__(self, env, config, b, rank, keep):
        super().__init__(env, config)
        self.priority = rank.copy()
        self.b = b
        self.keep = keep
        jobs, ops = keep.shape
        self.parents = {(j, k): [(j, z) for z in range(k)
                        if int(b['predecessors'][j, k]) & (1 << z)]
                        for j in range(jobs) for k in range(ops)}
        self.remaining = {node: 0 for node in self.parents}
        for group in self.parents.values():
            for node in group:
                self.remaining[node] += 1
        self.cached = {}
        self.host_cache = {f'node{h}': [] for h in range(b['p'].shape[2])}
        self.memory_events = []
        self.counts = {'cache_hits': 0, 'peer_reads': 0, 'store_reads': 0,
                       'capacity_evictions': 0, 'binding_completions': 0}

    def _release(self, node):
        if node in self.cached:
            host = self.cached.pop(node)
            self.host_cache[host].remove(node)

    def _charge(self, node, host):
        data_ms = 0.
        sizes = self.b['output_size_mb']
        for parent in self.parents[node]:
            if parent in self.cached:
                if self.cached[parent] == host:
                    self.counts['cache_hits'] += 1
                else:
                    data_ms += sizes[parent] * self.b['peer_read_ms_per_mb']
                    self.counts['peer_reads'] += 1
            else:
                data_ms += sizes[parent] * self.b['store_read_ms_per_mb']
                self.counts['store_reads'] += 1
            self.remaining[parent] -= 1
            if self.remaining[parent] == 0:
                self._release(parent)
        return float(data_ms)

    def _complete_cache(self, node, host):
        size = float(self.b['output_size_mb'][node])
        cap = float(self.b['host_memory_mb'][int(host[4:])])
        if not self.keep[node] or not self.remaining[node] or size > cap:
            return
        resident = self.host_cache[host]
        if sum(self.b['output_size_mb'][value] for value in resident) + size > cap:
            self.counts['binding_completions'] += 1
        while resident and sum(self.b['output_size_mb'][value] for value in resident) + size > cap:
            victim = resident[0]
            self._release(victim)
            self.counts['capacity_evictions'] += 1
            self.memory_events.append({'event': 'evict', 'job': victim[0], 'operation': victim[1],
                                       'host': host, 'time': float(self.env.now)})
        resident.append(node)
        self.cached[node] = host
        self.memory_events.append({'event': 'cache', 'job': node[0], 'operation': node[1],
                                   'host': host, 'time': float(self.env.now), 'size_mb': size})

    def dispatch(self, event):
        self.barrier = None
        ordered = sorted(self.pending, key=lambda rec: (self.priority[rec['rule']['job'], rec['rule']['operation']],
                                                         rec['rule']['job']))
        for rec in ordered:
            host, rule = rec['host'], rec['rule']
            if host in self.active:
                continue
            if any(rule['locks'] & value['rule']['locks'] for value in self.active.values()):
                continue
            if any(member and sum(self.domains[h][g] for h in self.active) >= 2
                   for g, member in enumerate(self.domains[host])):
                continue
            node = (rule['job'], rule['operation'])
            setup = self.setup[self.last[host]][rule['type']] if host in self.last else 0.
            data_ms = self._charge(node, host)
            rec['setup'] = setup
            rec['data_ms'] = data_ms
            rec['start'] = float(self.env.now)
            self.active[host] = rec
            self.pending.remove(rec)
            self.events.append({'event': 'start', 'job': node[0], 'operation': node[1],
                                'host': host, 'time': float(self.env.now), 'setup': setup,
                                'data_ms': data_ms, 'base': rec['base']})
            rec['grant'].succeed(rec['base'] + setup + data_ms / 1000)

    def execute(self, platform, task, base_duration):
        host, name = platform.node.node_name, task.type['name']
        if host not in self.domains or name not in self.rules:
            raise ValueError('task outside memory manifest')
        grant = self.env.event()
        rec = {'host': host, 'rule': self.rules[name], 'ready': float(self.env.now),
               'grant': grant, 'base': base_duration}
        self.pending.append(rec)
        self.wake()
        service_phase(platform, 'execution_wait', event=grant, unresolved=('mixed_host_lock_power',))
        duration = yield grant
        wait = self.env.now - rec['ready']
        task.node_contention_time = wait
        platform.node.contention_time += wait
        end = round(float(self.env.now) + duration, 12)
        service_phase(platform, 'execution', seconds=end - self.env.now)
        yield self.env.timeout(end - self.env.now)
        self.last[host] = rec['rule']['type']
        del self.active[host]
        node = (rec['rule']['job'], rec['rule']['operation'])
        self.events.append({'event': 'done', 'job': node[0], 'operation': node[1],
                            'host': host, 'time': float(self.env.now), 'setup': rec['setup'],
                            'data_ms': rec['data_ms'], 'base': base_duration, 'wait': float(wait)})
        self._complete_cache(node, host)
        self.wake()
        return duration

    def stats(self):
        return {**super().stats(), 'dispatch_contract': 'mixed_dag_memory_v1',
                'memory_contract': self.b['memory_contract'],
                'output_size_mb': self.b['output_size_mb'].tolist(),
                'host_memory_mb': self.b['host_memory_mb'].tolist(),
                'counts': self.counts, 'memory_events': self.memory_events}


def herosim(b, assignment, rank, retain, log):
    from src.executecosimulation import execute_simulation
    from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkScheduler

    pred, a, r, keep, _, _ = checked_memory(b, assignment, rank, retain)
    config, inputs = substrate(b['p'] / 1000)
    jobs, ops = a.shape
    for job in range(jobs):
        inputs['application_types'][f'nofs-job{job}']['dag'] = {
            f'op{job}_{op}': [f'op{job}_{parent}' for parent in range(op)
                             if int(pred[job, op]) & (1 << parent)]
            for op in range(ops)}
    config['infrastructure']['mixed_execution'] = coordinator.config_for(b)
    old_execution, old_placement = coordinator.MixedExecution, PeerGreedyNetworkScheduler.placement
    decisions = []

    class BoundMemory(MemoryExecution):
        def __init__(self, env, settings):
            super().__init__(env, settings, b, r, keep)

    def place(self, state, task):
        if False:
            yield
        job, op = map(int, task.type['name'][2:].split('_'))
        host = int(a[job, op])
        matches = [(node, platform) for node, platform in
                   self._get_valid_replicas(state.replicas[task.type['name']], task)
                   if node.node_name == f'node{host}']
        if len(matches) != 1:
            raise RuntimeError('memory assignment has no unique eligible replica')
        decisions.append({'job': job, 'operation': op, 'host': host, 'time': float(self.env.now)})
        return matches[0]

    coordinator.MixedExecution = BoundMemory
    PeerGreedyNetworkScheduler.placement = place
    random.seed(int(b['seed']))
    np.random.seed(int(b['seed']))
    try:
        with redirect_stdout(log), redirect_stderr(log), deadline(30):
            result = execute_simulation(config, copy.deepcopy(inputs),
                                        'peer_greedy_network_peer_greedy_network',
                                        keep_alive=1000000, queue_length=100)
    finally:
        coordinator.MixedExecution = old_execution
        PeerGreedyNetworkScheduler.placement = old_placement
    stats = result['stats']
    rows = stats['taskResults']
    if len(rows) != a.size or len(decisions) != a.size:
        raise RuntimeError('incomplete actual memory DAG execution')
    completions = {}
    for task in rows:
        job, op = map(int, task['taskType']['name'][2:].split('_'))
        if op == ops - 1:
            completions[job] = task['doneTime']
    if set(completions) != set(range(jobs)):
        raise RuntimeError('missing memory DAG sink')
    return {'job_completion_sum': sum(completions.values()), 'job_completions': completions,
            'tasks': rows, 'decisions': decisions, 'mixed_execution': stats['mixedExecution']}
