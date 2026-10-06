"""Audit DAG memory S0 sources, inputs, searched plans and actual task completions."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.placement.radical.dag_memory import hand_retention, problem, replay
from src.placement.radical.dag_reservation_bridge import BridgedDag
from src.placement.radical.environment import serialized
from scripts_cosim.dag_memory_lifetime_s0 import RULES, identity, search
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha


def audit(out):
    registration = read(out / 'protocol_before_run.json')
    protocol = registration['protocol']
    for name, digest in registration['sources'].items():
        assert sha(ROOT / name) == digest, name
    engine = BridgedDag(out / 'build')
    assert engine.provenance == read(out / 'native.json')
    calibration = read(out / 'calibration.json')
    seen = set()
    rows = []
    runs = operations = searches = 0
    for group in ('calibration', 'fresh'):
        for cell in sorted((out / group).iterdir()):
            row = read(cell / 'read.json')
            b = problem(row['seed'], *protocol['shape'])
            assert serialized(b) == read(cell / 'input.json')
            assert row['identity'] == identity(b) and row['identity'] not in seen
            seen.add(row['identity'])
            a, rank = np.array(row['assignment']), np.array(row['priority'])
            for rule in RULES:
                expected = replay(b, a, rank, hand_retention(b, a, rule))
                assert expected['objective'] == row['simple'][rule]['cost']
                assert expected['counts'] == row['simple'][rule]['counts']
            assert row['initial_rule'] == min(RULES, key=lambda rule: row['simple'][rule]['cost'])
            plans = list(row['arms'].values()) if group == 'calibration' else [row['hand'], *row['references'], row['reference']]
            for p in plans:
                expected = replay(b, a, rank, p['retain'])
                assert expected['objective'] == p['cost'] and expected['counts'] == p['counts']
                assert expected['starts'].tolist() == p['starts']
                assert expected['ends'].tolist() == p['ends']
            if group == 'fresh':
                rows.append(row)
                for offset, source in enumerate((row['simple'][row['initial_rule']], row['hand'])):
                    expected = search(b, a, rank, source, 'anneal', row['seed'] + 71 + offset,
                                      steps=protocol['reference_steps'])
                    actual = row['references'][offset]
                    assert all(expected[key] == actual[key] for key in ('cost', 'retain', 'counts', 'scores'))
                    searches += expected['scores']
                assert row['reference']['cost'] == min(
                    [row['simple'][row['initial_rule']], row['hand'], *row['references']],
                    key=lambda p: p['cost'])['cost']
            for path in cell.glob('*_live.json'):
                name = path.name.removesuffix('_live.json')
                p = row['simple']['keep_all'] if group == 'calibration' else (
                    row['simple'][row['initial_rule']] if name == 'initial' else row[name])
                actual = read(path)
                expected = replay(b, a, rank, p['retain'])
                assert abs(actual['job_completion_sum'] * 1000 - expected['objective']) < 1e-6
                assert actual['mixed_execution']['counts'] == expected['counts']
                assert len(actual['tasks']) == a.size
                for task in actual['tasks']:
                    j, k = map(int, task['taskType']['name'][2:].split('_'))
                    assert abs(task['doneTime'] * 1000 - expected['ends'][j, k]) < 1e-6
                runs += 1
                operations += a.size
    eligible = [mode for mode in protocol['hand_modes']
                if all(row['arms'][mode]['time_ms'] <= protocol['budget_ms']
                       for row in calibration['rows'])]
    selected = min(eligible, key=lambda mode: np.mean(
        [row['arms'][mode]['cost'] for row in calibration['rows']]))
    assert calibration['eligible'] == eligible and calibration['selected'] == selected
    assert {row['seed'] for row in rows} == set(range(protocol['fresh_seeds'][0], protocol['fresh_seeds'][1] + 1))
    report = read(out / 'read.json')
    gains = [(r['hand']['cost'] - r['reference']['cost']) / r['hand']['cost'] for r in rows]
    assert report['median_reference_headroom_pct'] == float(np.median(gains) * 100)
    assert report['selected_hand'] == selected and report['live_runs'] == runs
    assert report['live_operations'] == operations
    result = {'status': 'PASS', 'inputs': len(seen), 'live_runs': runs,
              'live_operations': operations, 'reference_search_proposals_reexecuted': searches,
              'report_sha256': sha(out / 'read.json')}
    save(out / 'AUDIT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.out)
