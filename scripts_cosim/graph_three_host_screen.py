"""Bounded CPU challenge of a three-host attractive peer-exchange objective."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts_cosim import graph_decision_witness as witness


def make_edges(cfg, seed):
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(cfg['tasks']):
        for j in range(i + 1, cfg['tasks']):
            if j == i + 1 or rng.random() < cfg['edge_probability']:
                edges.append((i, j, int(rng.integers(1, cfg['weight_max'] + 1))))
    return edges


def exhaustive(cfg, edges):
    plans = np.array(list(itertools.product(*witness.domains_for(cfg))), dtype=np.int64)
    values = np.zeros(len(plans))
    if 'unary_seconds' in cfg:
        unary = np.asarray(cfg['unary_seconds'])
        values += unary[np.arange(cfg['tasks'])[None, :], plans].sum(axis=1)
    for i, j, w in edges:
        values += witness.exchange_cost(cfg, w) * (plans[:, i] != plans[:, j])
    return plans, values


def coordinate_descent(cfg, edges, starts):
    domains = witness.domains_for(cfg)
    best = None
    for start in starts:
        plan = list(start)
        while True:
            changed = False
            for i, choices in enumerate(domains):
                current = witness.energy(cfg, edges, plan)
                for host in choices:
                    proposal = plan.copy(); proposal[i] = host
                    value = witness.energy(cfg, edges, proposal)
                    if value < current - 1e-10:
                        plan = proposal; current = value; changed = True
            if not changed:
                break
        score = witness.energy(cfg, edges, plan)
        if best is None or score < best[0]:
            best = (score, plan)
    return best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'experiments/graph_three_host_screen_v1.json')
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args(); cfg = json.loads(args.config.read_text())
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError('preserve prior artifacts: output must be empty')
    if cfg['hosts'] != 3 or cfg['tasks'] != 9 or not set(cfg['live_seeds']) <= set(cfg['graph_seeds']):
        raise ValueError('unexpected diagnostic configuration')
    for name in list(os.environ):
        if name.startswith(('HEROSIM_', 'GNN_', 'LIVE_AUDIT_', 'COSIM_')):
            os.environ.pop(name)
    os.environ.update(HEROSIM_PEER_EXCHANGE='1', HEROSIM_PG_DRAIN_CONTRACT='availability_v2',
                      HEROSIM_COSIM_KEEP_ALIVE='1000000', COSIM_SUPPRESS_SIM_PRINTS='1', SIM_FORCE_FULL_STATS='1')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [__file__, str(args.config), 'scripts_cosim/graph_decision_witness.py',
             'scripts_cosim/peer_lookahead_live_probe.py', 'src/executecosimulation.py',
             'src/eventgenerator.py', 'src/placement/infrastructure.py', 'src/placement/availability.py',
             'src/policy/peer_greedy_network/scheduler.py', 'src/policy/gnn/prefix_serving.py',
             'data/nofs-ids/platform-types.json', 'data/nofs-ids/storage-types.json', 'data/nofs-ids/qos-types.json']
    protocol = {'config': cfg, 'source_sha256': {str(Path(p).resolve().relative_to(ROOT)): witness.digest(p) for p in paths}}
    infra, inputs = witness.substrate(cfg)
    graphs = {seed: make_edges(cfg, seed) for seed in cfg['graph_seeds']}
    if len({json.dumps(e) for e in graphs.values()}) != len(graphs):
        raise ValueError('duplicate graph draws')
    (args.out_dir / 'protocol_before_run.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (args.out_dir / 'substrate.json').write_text(json.dumps({'infrastructure': infra, 'inputs': inputs}, indent=2)+'\n')
    started = time.monotonic(); results = []; runs = 0
    for seed, edges in graphs.items():
        controls = {}; timings = {}
        for name in ('immediate_paper', 'peer_mass', 'min_sum_3', 'min_sum_9'):
            t = time.perf_counter(); controls[name] = witness.control_plan(cfg, edges, name)
            timings[name] = (time.perf_counter() - t) * 1000
        starts = list(controls.values()) + [[d[0] if len(d) == 1 else h for d in witness.domains_for(cfg)] for h in range(3)]
        t = time.perf_counter(); controls['multistart_cd'] = coordinate_descent(cfg, edges, starts)
        timings['multistart_cd'] = (time.perf_counter() - t) * 1000 + sum(timings.values())
        t = time.perf_counter(); plans, values = exhaustive(cfg, edges)
        controls['exhaustive'] = plans[int(np.argmin(values))].tolist()
        timings['exhaustive'] = (time.perf_counter() - t) * 1000
        row = {'seed': seed, 'edges': edges, 'optimum_exchange': float(values.min()),
               'controls': {name: {'plan': p, 'exchange': witness.energy(cfg, edges, p), 'wall_ms': timings[name]}
                            for name, p in controls.items()}}
        cell = args.out_dir / str(seed); cell.mkdir()
        if seed in cfg['live_seeds']:
            (cell / 'placements').mkdir(); rtts = []
            with (cell / 'simulation.log').open('w') as log, (cell / 'placements/placements.jsonl').open('w') as out:
                for plan, expected in zip(plans, values):
                    if time.monotonic() - started > cfg['total_timeout_s']:
                        raise TimeoutError('diagnostic budget exceeded')
                    stat = witness.live(cfg, infra, inputs, edges, cfg['arrival_gap_s'], plan, log)
                    runs += 1
                    if abs(stat['exchange'] - expected) > 1e-8 or stat['queue'] > 1e-8 or stat['drain_contract'] != 'availability_v2':
                        raise RuntimeError('live cost or contract mismatch')
                    rtts.append(stat['total_rtt'])
                    out.write(json.dumps({'placement_plan': plan.tolist(), 'rtt': stat['total_rtt'], 'stats': stat}, allow_nan=False)+'\n')
                rule = witness.live(cfg, infra, inputs, edges, cfg['arrival_gap_s'], None, log, immediate=True); runs += 1
                off = [witness.live(cfg, infra, inputs, edges, cfg['arrival_gap_s'], p, log, exchange=False) for p in (plans[0], plans[-1])]; runs += 2
            residual = np.array(rtts) - values
            if not np.allclose(residual, residual[0], atol=1e-8, rtol=0):
                raise RuntimeError('analytical objective is not the live objective')
            if any(s['exchange'] or s['rendezvous'] for s in off) or abs(off[0]['total_rtt'] - off[1]['total_rtt']) > 1e-8:
                raise RuntimeError('exchange-off control failed')
            by_plan = {tuple(p): r for p, r in zip(plans, rtts)}
            for ctrl in row['controls'].values():
                ctrl['live_rtt'] = by_plan[tuple(ctrl['plan'])]
            row.update(live_optimum=min(rtts), immediate_live=rule, exchange_off=off,
                       constant_rtt=float(residual[0]), sweep_sha256=witness.digest(cell / 'placements/placements.jsonl'))
        (cell / 'read.json').write_text(json.dumps(row, indent=2)+'\n'); results.append(row)
        print(f'seed={seed} exact search={timings["exhaustive"]:.2f} ms; CD excess={row["controls"]["multistart_cd"]["exchange"]-row["optimum_exchange"]:.3f} s', flush=True)
    for path, expected in protocol['source_sha256'].items():
        if witness.digest(ROOT / path) != expected:
            raise RuntimeError(f'source changed during run: {path}')
    within_budget = all(r['controls']['exhaustive']['wall_ms'] <= cfg['diagnostic_search_budget_ms'] for r in results)
    outcome = {'protocol': protocol, 'rows': results, 'live_runs': runs, 'wall_s': time.monotonic()-started,
               'verdict': 'NO-TRAINING: exact search fits the diagnostic budget' if within_budget else 'UNRESOLVED: exact search exceeded diagnostic budget'}
    (args.out_dir / 'read.json').write_text(json.dumps(outcome, indent=2, allow_nan=False)+'\n')
    print(outcome['verdict'], flush=True)


if __name__ == '__main__':
    main()
