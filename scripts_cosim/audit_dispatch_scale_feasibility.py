"""Recompute S0 parity, coverage, fingerprints and timing selection from artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts_cosim.audit_radical_physics import load_problem
from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.dispatch_live import replay
from src.placement.radical.environment import pack

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(out):
    pre = read(out / 'protocol_before_run.json')
    protocol = pre['protocol']
    for name, digest in pre['sources'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in read(out / 'artifacts.json').items():
        assert sha(out / name) == digest, name
    engine = DispatchNative(out / 'build')
    assert engine.provenance == read(out / 'native.json')
    seeds = [seed for start in protocol['seed_starts'] for seed in range(start, start + protocol['cases_per_scale'])]
    assert sorted(int(p.name) for p in out.iterdir() if p.is_dir() and p.name.isdigit()) == sorted(seeds + [138900, 138901])
    identities, live_runs, live_ops = set(), 0, 0
    for seed in seeds + [138900, 138901]:
        cell = out / str(seed)
        b = load_problem(cell / 'input.json')
        identity = hashlib.sha256(pack(b).tobytes()).hexdigest()
        assert identity not in identities
        identities.add(identity)
        rec = read(cell / 'replay.json')
        a, rank = np.array(rec['assignment']), np.array(rec['priority'])
        cost, ends = engine.score_priority(b, a, rank)
        independent = replay(b, a, rank)
        assert cost == rec['objective'] == independent['objective']
        assert np.array_equal(ends, rec['ends']) and np.array_equal(ends, independent['ends'])
        assert independent['events'] == rec['events']
        if seed in seeds:
            row = read(cell / 'read.json')
            assert row['physical_sha256'] == identity and row['input_sha256'] == sha(cell / 'input.json')
            scale_index = next(i for i, start in enumerate(protocol['seed_starts']) if start <= seed < start + protocol['cases_per_scale'])
            assert tuple(b['p'].shape) == tuple(protocol['scales'][scale_index])
            assert row['seed'] == b['seed'] == seed
            starts = [event for event in independent['events'] if event['event'] == 'start']
            occupancy = sum(event['end'] - event['start'] for event in starts) / float(ends.max())
            assigned = np.take_along_axis(b['p'], a[..., None], -1)[..., 0]
            assert row['parity']['mean_active_hosts'] == occupancy
            assert row['parity']['host_utilization'] == occupancy / b['p'].shape[-1]
            assert row['parity']['mean_job_stretch_vs_assigned_service'] == float(np.mean(ends[:, -1] / assigned.sum(1)))
            c, plan = engine.plan(b, 'srpt:128')
            assert c == cost and np.array_equal(plan, [a, rank])
            for timing in row['timings'].values():
                assert len(timing['samples_ms']) == protocol['repeats']
                assert min(timing['samples_ms']) > 0
                assert timing['median_ms'] == float(np.median(timing['samples_ms']))
        expected_live = seed in [138900, 138901, *protocol['seed_starts'][1:]]
        assert (cell / 'live.json').exists() == expected_live
        if expected_live:
            actual = read(cell / 'live.json')
            assert abs(actual['total_rtt'] * 1000 - cost) < 1e-6
            seen = set()
            for task in actual['tasks']:
                j, k = map(int, task['taskType']['name'][2:].split('_'))
                assert (j, k) not in seen
                seen.add((j, k))
                assert task['executionNode'] == f'node{a[j, k]}'
                assert abs(task['doneTime'] * 1000 - ends[j, k]) < 1e-6
            assert len(seen) == a.size
            live_runs += 1
            live_ops += a.size
    report = read(out / 'read.json')
    for scale in report['scales']:
        rows = [read(out / str(s) / 'read.json') for s in seeds]
        rows = [r for r in rows if r['operations'] == scale['operations']]
        assert len(rows) == protocol['cases_per_scale']
        for metric, value in scale['p95_ms'].items():
            assert value == float(np.quantile([r['timings'][metric]['median_ms'] for r in rows], .95))
        assert scale['median_active_hosts'] == float(np.median([r['parity']['mean_active_hosts'] for r in rows]))
        assert scale['median_job_stretch'] == float(np.median([r['parity']['mean_job_stretch_vs_assigned_service'] for r in rows]))
        budgets = [b for b in protocol['budget_ladder_ms'] if scale['p95_ms']['full_proxy'] <= b * protocol['calibration_fraction']]
        assert scale['provisional_budget_ms'] == (min(budgets) if budgets else None)
    eligible = [r for r in report['scales'] if r['provisional_budget_ms'] is not None]
    assert report['selected'] == (max(eligible, key=lambda r: r['operations']) if eligible else None)
    assert report['live_runs'] == live_runs and report['live_operations'] == live_ops
    assert report['unique_inputs'] == len(seeds)
    return {'status': 'PASS', 'unique_inputs_including_parity_fixtures': len(identities),
            'live_runs': live_runs, 'live_operations': live_ops, 'report_sha256': sha(out / 'read.json'),
            'audit_source_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    result = audit(out)
    (out / 'AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
