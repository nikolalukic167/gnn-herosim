"""Actual-HeROsim adapter for resource-accounted one-level DAG rematerialization."""
import copy
import random
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from . import coordinator
from .dag_memory_live import MemoryExecution
from .dag_remat import checked_remat, effective_lock
from scripts_cosim.peer_lookahead_live_probe import deadline
from scripts_cosim.workflow_live import substrate


class RematExecution(MemoryExecution):
    def __init__(self, env, config, b, rank, keep, recompute):
        super().__init__(env, config, b, rank, keep)
        self.recompute = recompute
        self.counts.update(rematerializations=0, remat_input_store_reads=0,
                           remat_input_peer_reads=0, remat_input_cache_hits=0)

    def dispatch(self, event):
        self.barrier = None
        ordered = sorted(self.pending, key=lambda rec: (self.priority[rec['rule']['job'], rec['rule']['operation']],
                                                         rec['rule']['job']))
        for rec in ordered:
            host, rule = rec['host'], rec['rule']
            node = (rule['job'], rule['operation'])
            lock = effective_lock(self.b, node, self.parents, self.recompute, self.cached)
            if host in self.active:
                continue
            if any(lock & value['effective_lock'] for value in self.active.values()):
                continue
            if any(member and sum(self.domains[h][g] for h in self.active) >= 2
                   for g, member in enumerate(self.domains[host])):
                continue
            setup = self.setup[self.last[host]][rule['type']] if host in self.last else 0.
            data_ms = self._charge(node, host)
            rec['setup'] = setup
            rec['data_ms'] = data_ms
            rec['start'] = float(self.env.now)
            rec['effective_lock'] = lock
            self.active[host] = rec
            self.pending.remove(rec)
            self.events.append({'event': 'start', 'job': node[0], 'operation': node[1],
                                'host': host, 'time': float(self.env.now), 'setup': setup,
                                'data_ms': data_ms, 'base': rec['base'], 'effective_lock': lock})
            rec['grant'].succeed(rec['base'] + setup + data_ms / 1000)

    def _charge(self, node, host):
        data_ms = 0.
        sizes = self.b['output_size_mb']
        host_id = int(host[4:])
        for parent in self.parents[node]:
            if parent in self.cached:
                if self.cached[parent] == host:
                    self.counts['cache_hits'] += 1
                else:
                    data_ms += sizes[parent] * self.b['peer_read_ms_per_mb']
                    self.counts['peer_reads'] += 1
            elif self.recompute[parent]:
                for grandparent in self.parents[parent]:
                    if grandparent in self.cached:
                        if self.cached[grandparent] == host:
                            self.counts['remat_input_cache_hits'] += 1
                        else:
                            data_ms += sizes[grandparent] * self.b['peer_read_ms_per_mb']
                            self.counts['remat_input_peer_reads'] += 1
                    else:
                        data_ms += sizes[grandparent] * self.b['store_read_ms_per_mb']
                        self.counts['remat_input_store_reads'] += 1
                data_ms += float(self.b['p'][parent][host_id])
                self.counts['rematerializations'] += 1
                self.memory_events.append({'event': 'remat', 'job': parent[0],
                                           'operation': parent[1], 'consumer_job': node[0],
                                           'consumer_operation': node[1], 'host': host,
                                           'time': float(self.env.now)})
            else:
                data_ms += sizes[parent] * self.b['store_read_ms_per_mb']
                self.counts['store_reads'] += 1
            self.remaining[parent] -= 1
            if self.remaining[parent] == 0:
                self._release(parent)
        return float(data_ms)

    def stats(self):
        return {**super().stats(), 'dispatch_contract': 'mixed_dag_remat_v1',
                'recompute': self.recompute.astype(int).tolist()}


def herosim(b, assignment, rank, retain, recompute, log):
    from src.executecosimulation import execute_simulation
    from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkScheduler

    pred, a, r, keep, _, _, action = checked_remat(b, assignment, rank, retain, recompute)
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

    class BoundRemat(RematExecution):
        def __init__(self, env, settings):
            super().__init__(env, settings, b, r, keep, action)

    def place(self, state, task):
        if False:
            yield
        job, op = map(int, task.type['name'][2:].split('_'))
        host = int(a[job, op])
        matches = [(node, platform) for node, platform in
                   self._get_valid_replicas(state.replicas[task.type['name']], task)
                   if node.node_name == f'node{host}']
        if len(matches) != 1:
            raise RuntimeError('remat assignment has no unique eligible replica')
        decisions.append({'job': job, 'operation': op, 'host': host, 'time': float(self.env.now)})
        return matches[0]

    coordinator.MixedExecution = BoundRemat
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
        raise RuntimeError('incomplete actual remat DAG execution')
    completions = {}
    for task in rows:
        job, op = map(int, task['taskType']['name'][2:].split('_'))
        if op == ops - 1:
            completions[job] = task['doneTime']
    if set(completions) != set(range(jobs)):
        raise RuntimeError('missing remat DAG sink')
    return {'job_completion_sum': sum(completions.values()), 'job_completions': completions,
            'tasks': rows, 'decisions': decisions, 'mixed_execution': stats['mixedExecution']}
