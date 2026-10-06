"""Verify and summarize the repaired proposal-frontier live gate."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / 'docs/lineages/proposal_frontier_v1/gate_freeze_repaired_2026-09-23.json'
ARMS = ('gnn', 'mpoff', 'handmulti')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/tmp/proposal-frontier-v1-repair'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(FREEZE.read_text())
    for name, expected in freeze['artifacts'].items():
        if digest(Path(name).read_bytes()) != expected:
            raise RuntimeError(f'frozen artifact changed: {name}')
    per_seed = []
    files = {}
    code_fingerprints = set()
    for seed in freeze['seeds']:
        audit_path = args.root / f'seed-{seed}-audit.json'
        archives_path = args.root / f'seed-{seed}-archives.json'
        audit = json.loads(audit_path.read_text())
        archives = json.loads(archives_path.read_text())
        if audit['status'] != 'pass' or set(audit['arms']) != set(ARMS):
            raise RuntimeError(f'incomplete audit: {seed}')
        row = {'seed': seed, 'arms': {}}
        for arm in ARMS:
            archived = archives[arm]
            zipped = Path(archived['path'])
            zipped_data = zipped.read_bytes()
            raw = gzip.decompress(zipped_data)
            result = json.loads(raw)
            saved = audit['arms'][arm]
            if digest(zipped_data) != archived['sha256'] or digest(raw) != archived['raw_sha256']:
                raise RuntimeError(f'archive checksum mismatch: {zipped}')
            if result['status'] != 'success' or result['num_tasks'] != 5120:
                raise RuntimeError(f'incomplete run: {seed} {arm}')
            if result['total_rtt'] != saved['total_rtt_s']:
                raise RuntimeError(f'RTT audit mismatch: {seed} {arm}')
            if result['rtt_overview']['total_inference_time_s'] != saved['inference_s']:
                raise RuntimeError(f'inference audit mismatch: {seed} {arm}')
            code_fingerprints.add((saved['code']['commit'], saved['code']['diff_sha256']))
            row['arms'][arm] = {
                'elapsed_s_per_task': saved['total_rtt_s'] / 5120,
                'inference_ms_per_task': saved['mean_inference_ms'],
                'peer_exchange_s': saved['peer_exchange_s'],
                'unique_proposals': saved['unique_proposals'],
                'winner_counts': saved['winner_counts'],
            }
            files[zipped.name] = {'sha256': archived['sha256'], 'bytes': zipped.stat().st_size}
        for control in ('mpoff', 'handmulti'):
            gnn = row['arms']['gnn']['elapsed_s_per_task']
            other = row['arms'][control]['elapsed_s_per_task']
            row[f'gnn_gain_vs_{control}_pct'] = (other - gnn) / other * 100
        per_seed.append(row)
    if len(code_fingerprints) != 1:
        raise RuntimeError('code fingerprints differ across runs')
    medians = {arm: {
        'elapsed_s_per_task': median(row['arms'][arm]['elapsed_s_per_task'] for row in per_seed),
        'inference_ms_per_task': median(row['arms'][arm]['inference_ms_per_task'] for row in per_seed),
    } for arm in ARMS}
    comparisons = {}
    for control in ('mpoff', 'handmulti'):
        gains = [row[f'gnn_gain_vs_{control}_pct'] for row in per_seed]
        comparisons[control] = {
            'median_gain_pct': median(gains),
            'positive_topologies': sum(gain > 0 for gain in gains),
            'time_feasible': medians[control]['inference_ms_per_task'] <= medians['gnn']['inference_ms_per_task'],
        }
    passed = all(not result['time_feasible'] or
                 (result['median_gain_pct'] >= 2 and result['positive_topologies'] >= 7)
                 for result in comparisons.values())
    summary = {
        'contract': 'proposal_frontier_v1_repaired_read',
        'freeze_sha256': digest(FREEZE.read_bytes()),
        'seeds': freeze['seeds'], 'runs_audited': len(per_seed) * len(ARMS),
        'code_fingerprint': {'commit': next(iter(code_fingerprints))[0],
                             'diff_sha256': next(iter(code_fingerprints))[1]},
        'medians': medians, 'comparisons': comparisons,
        'registered_pass': passed,
        'per_seed': per_seed, 'raw_archives': files,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'registered_pass': passed, 'comparisons': comparisons,
                      'medians': medians, 'runs_audited': summary['runs_audited']},
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
