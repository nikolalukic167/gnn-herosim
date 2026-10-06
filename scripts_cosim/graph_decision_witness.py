"""Construct a distant-information witness, then try to erase it with cheap controls.

Two hosts, private replicas, two incompatible terminal tasks. This is a deliberately
small constructed example, not a sample of production workloads or a GNN result.
Every complete placement is also replayed through the live per-arrival engine.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_cosim.peer_lookahead_live_probe import deadline, min_sum, summarize_stats


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edges_for(cfg, family, variant):
    edges = [(0, 1, 4), (1, 2, 4), (4, 5, 4), (5, 6, 4)]
    edges += [(2, 3, 4), (6, 7, 4)] if variant == 0 else [(2, 7, 4), (3, 6, 4)]
    if family in ('bridge', 'loop'):
        edges.append((1, 5, 1))
    if family == 'loop':
        edges.append((2, 6, 1))
    if family not in ('chains', 'bridge', 'loop'):
        raise ValueError(family)
    return [(i, j, cfg['chain_weight'] if w == 4 else cfg['bridge_weight']) for i, j, w in sorted(edges)]


def domains_for(cfg):
    return [[int(cfg['anchors'][str(i)])] if str(i) in cfg['anchors'] else list(range(cfg.get('hosts', 2)))
            for i in range(cfg['tasks'])]


def exchange_cost(cfg, weight):
    return 2 * (weight * cfg['payload_unit_bytes'] / (cfg['bandwidth_mbps'] * 1024**2)
                + cfg['peer_latency_s'])


def energy(cfg, edges, plan):
    unary = cfg.get('unary_seconds')
    return (sum(exchange_cost(cfg, w) for i, j, w in edges if plan[i] != plan[j])
            + (sum(unary[i][h] for i, h in enumerate(plan)) if unary is not None else 0.))


def beliefs(cfg, edges, domains, rounds):
    unary = cfg.get('unary_seconds')
    costs = [np.array([unary[i][h] for h in d], dtype=float) if unary is not None else np.zeros(len(d))
             for i, d in enumerate(domains)]
    pairs = {(i, j): (np.array(domains[i])[:, None] != np.array(domains[j])[None, :])
             * exchange_cost(cfg, w) for i, j, w in edges}
    return min_sum(costs, pairs, rounds)


def control_plan(cfg, edges, control):
    if control not in ('immediate_paper', 'peer_mass') and control not in {f'min_sum_{r}' for r in cfg['message_rounds']}:
        raise ValueError(f'unknown control: {control}')
    domains = domains_for(cfg)
    plan = []
    for i, choices in enumerate(domains):
        if control.startswith('min_sum_'):
            conditional = [[plan[j]] if j < i else d for j, d in enumerate(domains)]
            score = beliefs(cfg, edges, conditional, int(control.rsplit('_', 1)[1]))[i]
        else:
            unary = cfg.get('unary_seconds')
            score = np.array([unary[i][h] for h in choices], dtype=float) if unary is not None else np.zeros(len(choices))
            for a, b, w in edges:
                if i not in (a, b):
                    continue
                other = b if a == i else a
                if other < i:
                    score += [exchange_cost(cfg, w) * (c != plan[other]) for c in choices]
                elif control == 'peer_mass':
                    score += [exchange_cost(cfg, w) * np.mean([c != d for d in domains[other]]) for c in choices]
        plan.append(choices[int(np.argmin(score))])
    return plan


def cut_plan(cfg, edges):
    """Exact binary attractive-pair objective; solve without enumerating placements."""
    n = cfg['tasks']; source = n; sink = n + 1
    cap = np.zeros((n + 2, n + 2), dtype=np.int64)
    for i, j, weight in edges:
        cost = exchange_cost(cfg, weight) * 10000
        if not np.isclose(cost, round(cost), rtol=0, atol=1e-7):
            raise ValueError('cut scaling would round an edge cost')
        cap[i, j] = cap[j, i] = round(cost)
    pin = int(cap.sum()) + 1
    for tid, host in cfg['anchors'].items():
        if int(host) == 0:
            cap[source, int(tid)] = pin
        else:
            cap[int(tid), sink] = pin
    flow = maximum_flow(csr_matrix(cap), source, sink)
    residual = cap - flow.flow.toarray()
    reachable = {source}; stack = [source]
    while stack:
        node = stack.pop()
        for other in np.flatnonzero(residual[node] > 0):
            if int(other) not in reachable:
                reachable.add(int(other)); stack.append(int(other))
    return [0 if i in reachable else 1 for i in range(n)]


def local_signature(cfg, edges):
    visible = {0}
    for _ in range(2):
        visible |= {j if i in visible else i for i, j, _ in edges if i in visible or j in visible}
    domains = domains_for(cfg)
    return {'nodes': [(i, domains[i]) for i in sorted(visible)],
            'edges': [(i, j, w) for i, j, w in edges if i in visible and j in visible],
            'weighted_degrees': [sum(w for a, b, w in edges if i in (a, b)) for i in sorted(visible)],
            'two_hop_root_columns': beliefs(cfg, edges, domains, 2)[0].tolist()}


def substrate(cfg):
    hosts = range(cfg.get('hosts', 2))
    inputs = {key: json.loads((ROOT / 'data/nofs-ids' / filename).read_text()) for key, filename in
              [('platform_types', 'platform-types.json'), ('storage_types', 'storage-types.json'),
               ('qos_types', 'qos-types.json')]}
    platform_base = inputs['platform_types']['rpiCpu']
    inputs['platform_types'] = {}; inputs['task_types'] = {}; inputs['application_types'] = {}
    domains = domains_for(cfg)
    for i in range(cfg['tasks']):
        name, hardware, app = f'w{i}', f'wcpu{i}', f'nofs-w{i}'
        inputs['platform_types'][hardware] = {**platform_base, 'shortName': hardware, 'name': hardware}
        inputs['application_types'][app] = {'name': app, 'dag': {name: []}}
        inputs['task_types'][name] = {
            'name': name, 'platforms': [hardware], 'memoryRequirements': {hardware: .01},
            'coldStartDuration': {hardware: 0.}, 'executionTime': {hardware: cfg['execution_s']},
            'energy': {hardware: 0.}, 'imageSize': {hardware: .001},
            'stateSize': {app: {'input': 0, 'output': 0}}}
    nodes = [{'node_name': 'client_node0', 'memory': 32, 'platforms': [],
              'storage': ['flashCard', 'someRemote'], 'type': 'rpi', 'network_map': {f'node{h}': 0. for h in hosts}}]
    placements = {f'w{i}': [] for i in range(cfg['tasks'])}
    pid = 0
    for host in hosts:
        types = []
        for i, choices in enumerate(domains):
            if host in choices:
                types.append(f'wcpu{i}')
                placements[f'w{i}'].append({'node_name': f'node{host}', 'platform_id': pid, 'warm': True})
                pid += 1
        nodes.append({'node_name': f'node{host}', 'memory': 32, 'platforms': types,
                      'storage': ['flashCard', 'someRemote'], 'type': 'rpi',
                      'network_map': {'client_node0': 0., **{f'node{h}': cfg['peer_latency_s'] for h in hosts if h != host}}})
    replica_cfg = {name: {'per_client': 0, 'per_server': 1} for name in placements}
    infra = {'network': {'bandwidth': cfg['bandwidth_mbps']}, 'nodes': nodes,
             'warmth_physics': 'node_disk_v2', 'preinitialize_platforms': True,
             'defer_cold_replica_init': False, 'replicas': replica_cfg,
             'replica_plan': {'preinit_clients': [], 'preinit_servers': [f'node{h}' for h in hosts],
                              'preinit_task_types': [], 'replicas_config': replica_cfg},
             'deterministic_replica_placements': placements,
             'deterministic_queue_distributions': {name: {} for name in placements}}
    return infra, inputs


def live(cfg, infra, inputs, edges, gap, plan, log, *, exchange=True, immediate=False):
    from src.executecosimulation import execute_simulation
    from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkScheduler
    original = PeerGreedyNetworkScheduler.placement
    decisions = []
    def place(self, state, task):
        if immediate:
            result = yield from original(self, state, task)
        else:
            candidates = self._get_valid_replicas(state.replicas[task.type['name']], task)
            matches = [(node, platform) for node, platform in candidates if node.node_name == f'node{plan[int(task.id)]}']
            if len(matches) != 1:
                raise RuntimeError('declared private candidate is not uniquely available')
            result = matches[0]
        decisions.append({'task': int(task.id), 'host': int(result[0].node_name[-1]),
                          'observed_ids': sorted(self._pg_orchestrator().task_by_id)})
        return result
    events = [{'timestamp': i * gap, 'application': inputs['application_types'][f'nofs-w{i}'],
               'qos': inputs['qos_types']['medium'], 'node_name': 'client_node0'} for i in range(cfg['tasks'])]
    config = {'infrastructure': copy.deepcopy(infra),
              'workload': {'rps': 1 / gap if gap else cfg['tasks'], 'duration': max(1., cfg['tasks'] * gap),
                           'events': events,
                           'peer_exchange': [[i, j, w * cfg['payload_unit_bytes']] for i, j, w in edges] if exchange else []}}
    random.seed(cfg['seed']); np.random.seed(cfg['seed'])
    PeerGreedyNetworkScheduler.placement = place
    try:
        with redirect_stdout(log), redirect_stderr(log), deadline(cfg['run_timeout_s']):
            result = execute_simulation(config, copy.deepcopy(inputs), 'peer_greedy_network_peer_greedy_network',
                                        keep_alive=1000000, queue_length=100)
        stats = summarize_stats(result['stats'], cfg['tasks'])
        # Unlike the original probe's eight-task prefix, this construction is one complete group.
        stats['group_rtt'] = stats['total_rtt']
        if len(decisions) != cfg['tasks']:
            raise RuntimeError('missing decisions')
        if gap > 0 and any(d['observed_ids'] != list(range(d['task'] + 1)) for d in decisions):
            raise RuntimeError('partial-arrival run exposed unexpected task instances')
        stats['decisions'] = decisions
        stats['drain_contract'] = result['stats']['schedulerCounters']['pg_drain_contract']
        return stats
    finally:
        PeerGreedyNetworkScheduler.placement = original


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'experiments/graph_decision_witness_v1.json')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(); cfg = json.loads(args.config.read_text())
    if cfg['tasks'] != 8 or cfg['anchors'] != {'3': 0, '7': 1}:
        raise ValueError('this construction requires eight tasks and the registered anchors')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError('output directory must be empty; preserve earlier artifacts')
    for name in list(os.environ):
        if name.startswith(('HEROSIM_', 'GNN_', 'LIVE_AUDIT_', 'COSIM_')):
            os.environ.pop(name)
    os.environ.update(HEROSIM_PEER_EXCHANGE='1', HEROSIM_PG_DRAIN_CONTRACT='availability_v2',
                      HEROSIM_COSIM_KEEP_ALIVE='1000000', COSIM_SUPPRESS_SIM_PRINTS='1', SIM_FORCE_FULL_STATS='1')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = {'config': cfg, 'smoke': args.smoke, 'config_sha256': digest(args.config),
                'script_sha256': digest(__file__), 'dependencies': {p: digest(ROOT / p) for p in
                 ['src/placement/infrastructure.py', 'src/placement/availability.py',
                  'src/policy/peer_greedy_network/scheduler.py', 'scripts_cosim/peer_lookahead_live_probe.py',
                  'src/executecosimulation.py', 'src/eventgenerator.py',
                  'data/nofs-ids/platform-types.json', 'data/nofs-ids/storage-types.json',
                  'data/nofs-ids/qos-types.json']}}
    (args.out_dir / 'protocol_before_run.json').write_text(json.dumps(protocol, indent=2)+'\n')
    infra, inputs = substrate(cfg)
    (args.out_dir / 'substrate.json').write_text(json.dumps({'infrastructure': infra, 'sim_inputs': inputs}, indent=2)+'\n')
    plans = list(itertools.product(*domains_for(cfg)))
    families = cfg['families'][:1] if args.smoke else cfg['families']
    gaps = cfg['arrival_gaps_s'][-1:] if args.smoke else cfg['arrival_gaps_s']
    results = []; started = time.monotonic(); runs = 0
    for family in families:
        signatures = [local_signature(cfg, edges_for(cfg, family, v)) for v in (0, 1)]
        if signatures[0] != signatures[1]:
            raise RuntimeError('root two-hop information differs')
        for gap in gaps:
            for variant in (0, 1):
                edges = edges_for(cfg, family, variant)
                controls = {name: control_plan(cfg, edges, name) for name in
                            ['immediate_paper', 'peer_mass'] + [f'min_sum_{r}' for r in cfg['message_rounds']]}
                t0 = time.perf_counter(); controls['min_cut'] = cut_plan(cfg, edges)
                cut_wall = time.perf_counter() - t0
                cell = args.out_dir / f'{family}_g{gap}_v{variant}'
                (cell / 'placements').mkdir(parents=True, exist_ok=True)
                rows = []
                with (cell / 'simulation.log').open('w') as log, (cell / 'placements/placements.jsonl').open('w') as sweep_file:
                    for plan in plans:
                        if time.monotonic() - started > cfg['total_timeout_s']:
                            raise TimeoutError('total CPU-screen wall budget exceeded')
                        stat = live(cfg, infra, inputs, edges, gap, plan, log)
                        runs += 1
                        row = {'placement_plan': list(plan), 'rtt': stat['total_rtt'], 'exchange': stat['exchange'],
                               'queue': stat['queue'], 'rendezvous': stat['rendezvous'], 'tasks': stat['tasks']}
                        sweep_file.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
                    rule = live(cfg, infra, inputs, edges, gap, None, log, immediate=True); runs += 1
                    off = [live(cfg, infra, inputs, edges, gap, plan, log, exchange=False) for plan in (plans[0], plans[-1])]
                    runs += 2
                values = np.array([r['rtt'] for r in rows]); opt = float(values.min())
                root_q = [min(r['rtt'] for r in rows if r['placement_plan'][0] == host) for host in (0, 1)]
                by_plan = {tuple(r['placement_plan']): r for r in rows}
                costs = np.array([energy(cfg, edges, p) for p in plans])
                residual = values - costs
                if not np.allclose(residual, residual[0], rtol=0, atol=1e-8):
                    raise RuntimeError('live RTT has placement-dependent costs absent from cut objective')
                if any(abs(r['exchange'] - c) > 1e-8 or r['queue'] > 1e-8 for r, c in zip(rows, costs)):
                    raise RuntimeError('exchange/queue accounting does not match isolated construction')
                if any(s['exchange'] or s['rendezvous'] for s in off) or not np.isclose(off[0]['total_rtt'], off[1]['total_rtt']):
                    raise RuntimeError('exchange-off control failed')
                control_read = {name: {'plan': p, 'rtt': by_plan[tuple(p)]['rtt'],
                                      'regret_pct': 100 * (by_plan[tuple(p)]['rtt'] / opt - 1)} for name, p in controls.items()}
                control_read['immediate_live'] = {'plan': [d['host'] for d in rule['decisions']],
                                                   'rtt': rule['total_rtt'], 'regret_pct': 100 * (rule['total_rtt'] / opt - 1)}
                row = {'family': family, 'gap': gap, 'variant': variant, 'edges': edges,
                       'local_signature': signatures[variant], 'optimum': opt, 'root_completion_rtt': root_q,
                       'optimal_root_hosts': [h for h in (0, 1) if abs(root_q[h] - opt) < 1e-8],
                       'controls': control_read, 'cut_wall_s': cut_wall, 'live_runs': len(plans)+3,
                       'rtt_minus_exchange': float(residual[0]), 'exchange_off_rtt': off[0]['total_rtt'],
                       'max_queue': max(r['queue'] for r in rows), 'placement_sweep_sha256': digest(cell / 'placements/placements.jsonl'),
                       'rule_decisions': rule['decisions']}
                (cell / 'read.json').write_text(json.dumps(row, indent=2)+'\n'); results.append(row)
                print(f'{family} gap={gap} variant={variant}: best root={row["optimal_root_hosts"]}, cut regret={control_read["min_cut"]["regret_pct"]:.3g}%', flush=True)
    pairs = []
    for a, b in zip(results[::2], results[1::2]):
        regrets = np.array([np.array(x['root_completion_rtt']) / x['optimum'] - 1 for x in (a, b)]) * 100
        pairs.append({'family': a['family'], 'gap': a['gap'],
                      'best_shared_root_mean_regret_pct': float(regrets.mean(0).min()),
                      'different_optimal_root': not bool(set(a['optimal_root_hosts']) & set(b['optimal_root_hosts']))})
    outcome = {'protocol': protocol, 'wall_s': time.monotonic()-started, 'live_runs': runs,
               'cells': results, 'pairs': pairs,
               'verdict': 'NO-TRAINING: exact hand min-cut closes this constructed binary objective'
                          if all(abs(r['controls']['min_cut']['regret_pct']) < 1e-8 for r in results)
                          else 'AUDIT-FAILURE: cut and live exhaustive optimum disagree'}
    (args.out_dir / 'read.json').write_text(json.dumps(outcome, indent=2, allow_nan=False)+'\n')
    print(outcome['verdict'], flush=True)
    if outcome['verdict'].startswith('AUDIT-FAILURE'):
        raise RuntimeError(outcome['verdict'])


if __name__ == '__main__':
    main()
