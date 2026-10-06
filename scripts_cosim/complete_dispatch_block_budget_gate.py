"""Correct a frozen screen's live-control selection without overwriting its artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.mixed_dispatch_block_screen import parity
from scripts_cosim.workflow_proposal_gate import configure_environment
from src.placement.radical.dispatch_adaptive import AdaptiveDispatch
from src.placement.radical.dispatch_live import replay

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def selection(out):
    pre, report = read(out / 'protocol_before_run.json'), read(out / 'read.json')
    rows = [read(p) for p in sorted(out.glob('qualification/*/read.json'))]
    names = [name for name in report['capture'] if name.startswith('wall_')]
    eligible = [name for name in names if all(q['time_ms'] <= pre['protocol']['budget_ms']
                 for row in rows for q in row['arms'][name]['repeats'])]
    if not eligible:
        raise ValueError('no budget-valid hand arm; cannot manufacture an equal-budget gate')
    best = max(eligible, key=lambda name: report['capture'][name])
    return rows, eligible, best


def audit(out):
    corrected = out / 'budget_control_correction'
    protocol = read(corrected / 'protocol_before_replay.json')
    assert sha(out / 'read.json') == protocol['original_report_sha256']
    for name, digest in protocol['sources'].items():
        assert sha(ROOT / name) == digest
    for name, digest in read(corrected / 'artifacts.json').items():
        assert sha(corrected / name) == digest
    rows, eligible, selected = selection(out)
    assert eligible == protocol['eligible_arms'] and selected == protocol['selected']
    engine = AdaptiveDispatch(out / 'build')
    operations = 0
    for row in rows:
        cell = corrected / str(row['seed'])
        b = load_problem(out / 'qualification' / str(row['seed']) / 'input.json')
        arm = row['arms'][selected]
        a, rank = np.array(arm['assignment']), np.array(arm['priority'])
        value, ends = engine.score_priority(b, a, rank)
        independent = replay(b, a, rank)
        rec, actual = read(cell / 'control_replay.json'), read(cell / 'control_live.json')
        assert value == arm['cost'] == rec['objective'] == independent['objective']
        assert np.array_equal(ends, rec['ends']) and np.array_equal(ends, independent['ends'])
        assert rec['events'] == independent['events']
        assert abs(actual['total_rtt'] * 1000 - value) < 1e-6
        seen = set()
        for task in actual['tasks']:
            j, k = map(int, task['taskType']['name'][2:].split('_'))
            assert (j, k) not in seen
            seen.add((j, k))
            assert task['executionNode'] == f'node{a[j,k]}'
            assert abs(task['doneTime'] * 1000 - ends[j,k]) < 1e-6
        assert len(seen) == a.size
        operations += a.size
    result = {'status': 'PASS', 'selected': selected, 'eligible_arms': eligible,
              'capture_fraction': read(out / 'read.json')['capture'][selected],
              'live_runs': len(rows), 'live_operations': operations,
              'original_headroom_verdict_unchanged': True, 'training_allowed': False,
              'audit_source_sha256': sha(Path(__file__))}
    save(corrected / 'AUDIT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    out, corrected = args.out, args.out / 'budget_control_correction'
    if args.audit_only:
        print(json.dumps(audit(out), indent=2))
        return
    corrected.mkdir(exist_ok=False)
    rows, eligible, selected = selection(out)
    sources = read(out / 'protocol_before_run.json')['sources']
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError('source changed: ' + name)
    save(corrected / 'protocol_before_replay.json', {
        'reason': 'Original live-control selector did not filter timing-invalid arms. Preserve original negative screen; replay the strongest fixed arm meeting every repeat budget.',
        'selection': 'Max capture among fixed arms with all 48 timing repeats at or below 250 ms; no new objective optimization',
        'eligible_arms': eligible, 'selected': selected, 'sources': sources,
        'original_report_sha256': sha(out / 'read.json'), 'expected_live_runs': len(rows)})
    configure_environment()
    engine = AdaptiveDispatch(out / 'build')
    for row in rows:
        cell = corrected / str(row['seed'])
        cell.mkdir()
        input_path = out / 'qualification' / str(row['seed']) / 'input.json'
        b = load_problem(input_path)
        arm = row['arms'][selected]
        save(cell / 'input_reference.json', {'path': str(input_path.resolve()), 'sha256': sha(input_path)})
        parity(engine, b, np.array(arm['assignment']), np.array(arm['priority']), cell, 'control', live=True)
        print('CORRECTED LIVE', row['seed'], selected, flush=True)
    save(corrected / 'artifacts.json', {str(p.relative_to(corrected)): sha(p) for p in sorted(corrected.rglob('*.json'))})
    print(json.dumps(audit(out), indent=2), flush=True)


if __name__ == '__main__':
    main()
