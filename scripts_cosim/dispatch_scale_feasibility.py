"""CPU-only S0 timing calibration; never reads a corpus or trained checkpoint."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.placement.radical.dispatch import DispatchNative, priorities
from src.placement.radical.dispatch_live import herosim, replay
from src.placement.radical.environment import initial, pack, problem, serialized
from src.policy.dispatch_priority.model import PriorityNet, features
from scripts_cosim.workflow_proposal_gate import configure_environment

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    'scope': 'S0 calibration only; no block-move lineage or training registration',
    'scales': [[8, 6, 4], [16, 8, 8], [32, 8, 16], [48, 8, 24]],
    'seed_starts': [138000, 138100, 138200, 138300],
    'cases_per_scale': 8,
    'repeats': 7,
    'candidate_count': 32,
    'budget_ladder_ms': [5, 10, 25, 50, 100, 250],
    'calibration_fraction': 0.85,
    'selection': 'largest scale whose full proxy p95 fits 85% of 250 ms; smallest fitting ladder budget',
    'proxy': 'srpt:128 including placement + base features + untrained hidden16/layers2 encoder + 32 independent exact-scored rank swaps',
    'limitations': 'Not block proposals, training evidence, or a served latency guarantee; fixed six locks and three power domains.',
    'parity': 'two 48-op fixtures before sweep; first case each larger scale live; all cases independent replay',
    'future_comparison': 'Both matched proposal count and equal end-to-end wall time, charging common placement, are mandatory.',
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured(fn):
    expected = np.asarray(fn())
    times = []
    for _ in range(PROTOCOL['repeats']):
        start = time.perf_counter_ns()
        value = fn()
        times.append((time.perf_counter_ns() - start) / 1e6)
        if not np.array_equal(np.asarray(value), expected):
            raise ValueError('nondeterministic benchmark')
    return {'median_ms': float(np.median(times)), 'samples_ms': times}


def candidate_scores(engine, b, a, rank):
    result = []
    for i in range(PROTOCOL['candidate_count']):
        candidate = rank.copy()
        x, y = i % rank.size, (i * 17 + 13) % rank.size
        candidate.flat[x], candidate.flat[y] = candidate.flat[y], candidate.flat[x]
        result.append(engine.score_priority(b, a, candidate)[0])
    return np.array(result)


def verify_case(engine, b, a, rank, cell, live):
    cost, ends = engine.score_priority(b, a, rank)
    independent = replay(b, a, rank)
    if cost != independent['objective'] or not np.array_equal(ends, independent['ends']):
        raise ValueError('native/SimPy mismatch')
    save(cell / 'replay.json', {**independent, 'ends': ends.tolist(), 'assignment': a.tolist(), 'priority': rank.tolist()})
    if live:
        with (cell / 'simulation.log').open('w') as log:
            actual = herosim(b, a, rank, log, b['seed'])
        if abs(actual['total_rtt'] * 1000 - cost) > 1e-6:
            raise ValueError('live objective mismatch')
        seen = set()
        for task in actual['tasks']:
            j, k = map(int, task['taskType']['name'][2:].split('_'))
            if (j, k) in seen or task['executionNode'] != f'node{a[j, k]}' or abs(task['doneTime'] * 1000 - ends[j, k]) > 1e-6:
                raise ValueError('live operation mismatch')
            seen.add((j, k))
        if len(seen) != a.size or actual['mixed_execution']['dispatch_contract'] != 'mixed_ready_priority_v1':
            raise ValueError('incomplete live replay')
        save(cell / 'live.json', actual)
    starts = [e for e in independent['events'] if e['event'] == 'start']
    makespan = float(ends.max())
    occupied = sum(e['end'] - e['start'] for e in starts)
    assigned = np.take_along_axis(b['p'], a[..., None], -1)[..., 0]
    return {'live': live, 'operations': a.size, 'objective': cost,
            'mean_active_hosts': occupied / makespan,
            'host_utilization': occupied / makespan / b['p'].shape[-1],
            'mean_job_stretch_vs_assigned_service': float(np.mean(ends[:, -1] / assigned.sum(1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    configure_environment()
    sources = sorted(set([Path(__file__).resolve(), *ROOT.glob('src/placement/radical/*.py'),
                          *ROOT.glob('src/placement/radical/*.cpp'),
                          *ROOT.glob('src/policy/dispatch_priority/*.py'),
                          ROOT / 'src/policy/mixed/model.py',
                          *[ROOT / p for p in ('scripts_cosim/workflow_proposal_gate.py', 'scripts_cosim/workflow_live.py',
                             'scripts_cosim/mixed_physics_live.py', 'src/placement/infrastructure.py',
                             'src/placement/executor.py', 'src/placement/simulation.py', 'src/executecosimulation.py')]]))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    save(out / 'protocol_before_run.json', {'protocol': PROTOCOL, 'sources': hashes,
         'python': sys.executable, 'platform': platform.platform(), 'numpy': np.__version__,
         'torch': torch.__version__, 'threads': {k: os.getenv(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
         'cpu': Path('/proc/cpuinfo').read_text().split('model name', 1)[-1].split('\n', 1)[0]})
    engine = DispatchNative(out / 'build')
    save(out / 'native.json', engine.provenance)
    identities = set()
    rows = []
    # These two fixtures gate entry to the timing sweep itself.
    for seed in (138900, 138901):
        b = problem(seed)
        cell = out / str(seed)
        cell.mkdir()
        save(cell / 'input.json', serialized(b))
        _, plan = engine.plan(b, 'srpt:128')
        verify_case(engine, b, *plan, cell, live=True)
    print('SMALL THREE-WAY PARITY PASS', flush=True)
    with torch.inference_mode():
        for (jobs, ops, hosts), start_seed in zip(PROTOCOL['scales'], PROTOCOL['seed_starts']):
            for seed in range(start_seed, start_seed + PROTOCOL['cases_per_scale']):
                b = problem(seed, jobs=jobs, ops=ops, hosts=hosts)
                identity = hashlib.sha256(pack(b).tobytes()).hexdigest()
                if identity in identities or not np.array_equal(pack(b), pack(problem(seed, jobs, ops, hosts))):
                    raise ValueError('duplicate or nondeterministic physical input')
                identities.add(identity)
                cell = out / str(seed)
                cell.mkdir()
                save(cell / 'input.json', serialized(b))
                cost, (a, rank) = engine.plan(b, 'srpt:128')
                parity = verify_case(engine, b, a, rank, cell, live=seed == start_seed and jobs > 8)
                x, adj = features(b, a)
                torch.manual_seed(0)
                model = PriorityNet(x.shape[-1], hidden=16, layers=2, mp=True).eval()
                xt, at = torch.from_numpy(x)[None], torch.from_numpy(adj)[None]

                def encode():
                    return model(xt, at).numpy()

                def full_proxy():
                    _, (assignment, r) = engine.plan(b, 'srpt:128')
                    xx, aa = features(b, assignment)
                    model(torch.from_numpy(xx)[None], torch.from_numpy(aa)[None])
                    return candidate_scores(engine, b, assignment, r)

                checks = {
                    'dispatch_score': lambda: engine.score_priority(b, a, rank)[1],
                    'placement_descent64': lambda: engine.search(b, initial(b, 'affinity'), 64)[1],
                    'srpt128_total': lambda: engine.plan(b, 'srpt:128')[1],
                    'features_base': lambda: features(b, a)[0],
                    'features_hand12': lambda: features(b, a, hand=True)[0],
                    'encoder_forward': encode,
                    'exact_candidates32': lambda: candidate_scores(engine, b, a, rank),
                    'full_proxy': full_proxy,
                }
                timings = {name: measured(fn) for name, fn in checks.items()}
                row = {'seed': seed, 'operations': jobs * ops, 'hosts': hosts, 'physical_sha256': identity,
                       'input_sha256': sha(cell / 'input.json'), 'parity': parity, 'timings': timings,
                       'feature_dim': x.shape[-1], 'adjacency_bytes': adj.nbytes}
                save(cell / 'read.json', row)
                rows.append(row)
                print('CASE', seed, jobs * ops, 'proxy_ms', round(timings['full_proxy']['median_ms'], 3), flush=True)
    summary = []
    for jobs, ops, hosts in PROTOCOL['scales']:
        group = [r for r in rows if r['operations'] == jobs * ops]
        p95 = {name: float(np.quantile([r['timings'][name]['median_ms'] for r in group], .95)) for name in rows[0]['timings']}
        budgets = [b for b in PROTOCOL['budget_ladder_ms'] if p95['full_proxy'] <= b * PROTOCOL['calibration_fraction']]
        summary.append({'operations': jobs * ops, 'hosts': hosts, 'p95_ms': p95,
                        'provisional_budget_ms': min(budgets) if budgets else None,
                        'median_active_hosts': float(np.median([r['parity']['mean_active_hosts'] for r in group])),
                        'median_job_stretch': float(np.median([r['parity']['mean_job_stretch_vs_assigned_service'] for r in group]))})
    eligible = [r for r in summary if r['provisional_budget_ms'] is not None]
    selected = max(eligible, key=lambda r: r['operations']) if eligible else None
    for p, digest in hashes.items():
        if sha(ROOT / p) != digest:
            raise ValueError('source changed during S0: ' + p)
    save(out / 'read.json', {'status': 'S0-TIMING-CALIBRATION-COMPLETE', 'scales': summary, 'selected': selected,
                           'source_hashes_unchanged': True, 'unique_inputs': len(identities),
                           'live_runs': 5, 'live_operations': 2 * 48 + 128 + 256 + 384,
                           'training_authorized': False, 'environment_qualification': 'PENDING; fixed global resource pools',
                           'required_next': 'S1 move definitions and S2 headroom/search-hardness/nonlocal residual gates; no GPU'})
    save(out / 'artifacts.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*.json'))})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
